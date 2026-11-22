//! PRISM proximity-stress overlap removal using Delaunay triangulation or KD-tree fallback.
//!
//! Implements Gansner & Hu (2010), "Efficient Node Overlap Removal Using a Proximity Stress
//! Model", using `Delaunay2d` (under `feature = "delaunay"`) or `StaticKdTree` $k$-NN fallback
//! (under `--no-default-features`), combined with an allocation-free Sweep-and-Prune (SAP)
//! overlap detector to guarantee `0` overlaps at convergence, and a zero-dense-allocation split
//! `L+D+U` sparse CSR fused 2-RHS Symmetric SSOR-PCG solver.

use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
#[cfg(feature = "delaunay")]
use egraph_spatial::Delaunay2d;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial::{KdQueryScratch, StaticKdTree};

#[cfg(not(feature = "delaunay"))]
const K_FALLBACK_NEIGHBORS: usize = 12;

const MIN_EDGE_DIST: f64 = 1e-9;
const MIN_WEIGHT_DIST: f64 = 1e-6;
const INNER_MAJORIZATION_ITERS: usize = 1;
const OVERLAP_WEIGHT_BOOST: f64 = 16.0;
const SSOR_OMEGA: f64 = 1.2;
const OVERLAP_MICRO_DIST_RATIO: f64 = 1e-4;

/// Hyperparameters for [`PrismOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrismConfig {
    /// Maximum number of PRISM proximity-stress rounds (`default: 50`).
    pub max_iterations: usize,
    /// Maximum per-round edge expansion factor $s_{\max} > 1.0$ (`default: 1.5`).
    pub max_scale_factor: f64,
    /// Maximum inner PCG iterations per linear solve (`default: 80`).
    pub cg_max_iters: usize,
    /// Relative residual tolerance for inner PCG (`default: 1e-3`).
    pub cg_tolerance: f64,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed used for coincident-point jitter (`default: 42`).
    pub seed: u64,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_scale_factor: 1.5,
            cg_max_iters: 80,
            cg_tolerance: 1e-3,
            padding: 0.0,
            seed: 42,
        }
    }
}

impl PrismConfig {
    /// Validates all PRISM configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.max_scale_factor.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::max_scale_factor"));
        }
        if self.max_scale_factor <= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "max_scale_factor must be > 1.0, got {}",
                self.max_scale_factor
            )));
        }
        if self.cg_max_iters == 0 {
            return Err(CoreError::invalid_input("cg_max_iters must be >= 1"));
        }
        if !self.cg_tolerance.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::cg_tolerance"));
        }
        if self.cg_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "cg_tolerance must be >= 0.0, got {}",
                self.cg_tolerance
            )));
        }
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}

/// Reusable sparse CSR Preconditioned Conjugate Gradient (PCG) workspace.
///
/// Executes each CG iteration in $O(\mathrm{nnz})$ time with zero dense $N \times N$
/// matrix allocations and zero per-solve heap allocations once sized.
#[derive(Debug, Clone, Default)]
pub struct SparsePcgWorkspace {
    diag_inv: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    r_y: Vec<f64>,
    z_y: Vec<f64>,
    p_y: Vec<f64>,
    ap_y: Vec<f64>,
}

impl SparsePcgWorkspace {
    /// Creates a [`SparsePcgWorkspace`] pre-sized for `n` unknowns.
    pub fn new(n: usize) -> Self {
        Self {
            diag_inv: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
            r_y: vec![0.0; n],
            z_y: vec![0.0; n],
            p_y: vec![0.0; n],
            ap_y: vec![0.0; n],
        }
    }

    /// Ensures internal scratch vectors hold at least `n` elements.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.diag_inv.len() < n {
            self.diag_inv.resize(n, 0.0);
            self.r.resize(n, 0.0);
            self.z.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.ap.resize(n, 0.0);
        }
        if self.r_y.len() < n {
            self.r_y.resize(n, 0.0);
            self.z_y.resize(n, 0.0);
            self.p_y.resize(n, 0.0);
            self.ap_y.resize(n, 0.0);
        }
    }

    /// Solves the sparse symmetric positive semi-definite CSR system `A x = b` via Jacobi PCG.
    ///
    /// Each row `i` in `(indptr, indices, values)` stores its off-diagonal and diagonal entries.
    /// Safely exits early when the initial/updated residual is zero or when `denom <= 1e-30`
    /// or `r^T z <= 1e-30` (such as on singular or disconnected Laplacian components).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_csr(
        &mut self,
        indptr: &[u32],
        indices: &[u32],
        values: &[f64],
        b: &[f64],
        x: &mut [f64],
        tol: f64,
        max_iters: usize,
    ) -> CoreResult<usize> {
        let n = b.len();
        if indptr.len() != n + 1 {
            return Err(CoreError::dimension_mismatch(
                n + 1,
                indptr.len(),
                "SparsePcgWorkspace::solve_csr::indptr",
            ));
        }
        if indices.len() != values.len() {
            return Err(CoreError::dimension_mismatch(
                indices.len(),
                values.len(),
                "SparsePcgWorkspace::solve_csr::values",
            ));
        }
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "SparsePcgWorkspace::solve_csr::x",
            ));
        }
        if indptr[n] as usize != indices.len() {
            return Err(CoreError::invalid_csr(
                "SparsePcgWorkspace::solve_csr indptr[n] != indices.len()",
            ));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PCG tolerance must be finite and >= 0.0, got {tol}"
            )));
        }
        if n == 0 || max_iters == 0 {
            return Ok(0);
        }

        self.ensure_capacity(n);

        // Extract diagonal entries and compute initial residual r = b - A x.
        let mut rr0 = 0.0_f64;
        for i in 0..n {
            let start = indptr[i] as usize;
            let end = indptr[i + 1] as usize;
            let mut diag = 0.0_f64;
            let mut ax = 0.0_f64;
            let i_u32 = i as u32;
            for pos in start..end {
                let j = indices[pos];
                let val = values[pos];
                if j == i_u32 {
                    diag += val;
                }
                ax += val * x[j as usize];
            }
            self.diag_inv[i] = if diag.is_finite() && diag > 1e-30 {
                1.0 / diag
            } else {
                0.0
            };
            let ri = b[i] - ax;
            self.r[i] = ri;
            rr0 += ri * ri;
        }

        if !rr0.is_finite() || rr0 <= 1e-30 {
            return Ok(0);
        }
        let stop_rr = (tol * tol) * rr0;
        if rr0 <= stop_rr {
            return Ok(0);
        }

        let mut rz_old = 0.0_f64;
        for i in 0..n {
            let zi = self.diag_inv[i] * self.r[i];
            self.z[i] = zi;
            self.p[i] = zi;
            rz_old += self.r[i] * zi;
        }
        if !rz_old.is_finite() || rz_old <= 1e-30 {
            return Ok(0);
        }

        for iter in 0..max_iters {
            let mut denom = 0.0_f64;
            for i in 0..n {
                let start = indptr[i] as usize;
                let end = indptr[i + 1] as usize;
                let mut sum = 0.0_f64;
                for pos in start..end {
                    sum += values[pos] * self.p[indices[pos] as usize];
                }
                self.ap[i] = sum;
                denom += self.p[i] * sum;
            }
            if !denom.is_finite() || denom <= 1e-30 {
                return Ok(iter);
            }

            let alpha = rz_old / denom;
            if !alpha.is_finite() {
                return Ok(iter);
            }

            let mut rr = 0.0_f64;
            for i in 0..n {
                let cand_x = x[i] + alpha * self.p[i];
                if !cand_x.is_finite() {
                    return Ok(iter);
                }
                x[i] = cand_x;
                let ri = self.r[i] - alpha * self.ap[i];
                self.r[i] = ri;
                rr += ri * ri;
            }

            if !rr.is_finite() || rr == 0.0 || rr <= stop_rr {
                return Ok(iter + 1);
            }

            let mut rz_new = 0.0_f64;
            for i in 0..n {
                let zi = self.diag_inv[i] * self.r[i];
                self.z[i] = zi;
                rz_new += self.r[i] * zi;
            }
            if !rz_new.is_finite() || rz_new <= 1e-30 {
                return Ok(iter + 1);
            }

            let beta = rz_new / rz_old;
            if !beta.is_finite() {
                return Ok(iter + 1);
            }
            rz_old = rz_new;
            for i in 0..n {
                self.p[i] = self.z[i] + beta * self.p[i];
            }
        }

        Ok(max_iters)
    }

    /// Solves two sparse symmetric positive semi-definite systems `A x = bx` and `A y = by`
    /// simultaneously using split `L + D + U` CSR storage and Symmetric SSOR preconditioning.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_ldu_2rhs_ssor(
        &mut self,
        diag: &[f64],
        diag_inv: &[f64],
        l_indptr: &[u32],
        l_indices: &[u32],
        l_values: &[f64],
        u_indptr: &[u32],
        u_indices: &[u32],
        u_values: &[f64],
        bx: &[f64],
        by: &[f64],
        x: &mut [f64],
        y: &mut [f64],
        tol: f64,
        max_iters: usize,
        omega: f64,
    ) -> CoreResult<usize> {
        let n = bx.len();
        if n == 0 || max_iters == 0 {
            return Ok(0);
        }
        self.ensure_capacity(n);

        // Compute initial residuals r_x = bx - A x and r_y = by - A y in a single pass over A.
        let mut rr0_x = 0.0_f64;
        let mut rr0_y = 0.0_f64;
        for i in 0..n {
            let di = diag[i];
            let mut ax = di * x[i];
            let mut ay = di * y[i];
            let ls = l_indptr[i] as usize;
            let le = l_indptr[i + 1] as usize;
            for pos in ls..le {
                let j = l_indices[pos] as usize;
                let v = l_values[pos];
                ax += v * x[j];
                ay += v * y[j];
            }
            let us = u_indptr[i] as usize;
            let ue = u_indptr[i + 1] as usize;
            for pos in us..ue {
                let j = u_indices[pos] as usize;
                let v = u_values[pos];
                ax += v * x[j];
                ay += v * y[j];
            }
            let rxi = bx[i] - ax;
            let ryi = by[i] - ay;
            self.r[i] = rxi;
            self.r_y[i] = ryi;
            rr0_x += rxi * rxi;
            rr0_y += ryi * ryi;
        }

        let tol_sq = tol * tol;
        let stop_rr_x = tol_sq * rr0_x;
        let stop_rr_y = tol_sq * rr0_y;
        let mut active_x = rr0_x.is_finite() && rr0_x > 1e-30 && rr0_x > stop_rr_x;
        let mut active_y = rr0_y.is_finite() && rr0_y > 1e-30 && rr0_y > stop_rr_y;
        if !active_x && !active_y {
            return Ok(0);
        }

        let mut rz_old_x = 0.0_f64;
        let mut rz_old_y = 0.0_f64;
        if active_x && active_y {
            ssor_apply_2rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r[..n],
                &self.r_y[..n],
                &mut self.z[..n],
                &mut self.z_y[..n],
            );
            for i in 0..n {
                let zx = self.z[i];
                let zy = self.z_y[i];
                self.p[i] = zx;
                self.p_y[i] = zy;
                rz_old_x += self.r[i] * zx;
                rz_old_y += self.r_y[i] * zy;
            }
        } else if active_x {
            ssor_apply_1rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r[..n],
                &mut self.z[..n],
            );
            for i in 0..n {
                let zx = self.z[i];
                self.p[i] = zx;
                rz_old_x += self.r[i] * zx;
            }
        } else {
            ssor_apply_1rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r_y[..n],
                &mut self.z_y[..n],
            );
            for i in 0..n {
                let zy = self.z_y[i];
                self.p_y[i] = zy;
                rz_old_y += self.r_y[i] * zy;
            }
        }

        if !rz_old_x.is_finite() || rz_old_x <= 1e-30 {
            active_x = false;
        }
        if !rz_old_y.is_finite() || rz_old_y <= 1e-30 {
            active_y = false;
        }
        if !active_x && !active_y {
            return Ok(0);
        }

        for iter in 0..max_iters {
            if active_x && active_y {
                let mut denom_x = 0.0_f64;
                let mut denom_y = 0.0_f64;
                for i in 0..n {
                    let di = diag[i];
                    let mut sx = di * self.p[i];
                    let mut sy = di * self.p_y[i];
                    let ls = l_indptr[i] as usize;
                    let le = l_indptr[i + 1] as usize;
                    for pos in ls..le {
                        let j = l_indices[pos] as usize;
                        let v = l_values[pos];
                        sx += v * self.p[j];
                        sy += v * self.p_y[j];
                    }
                    let us = u_indptr[i] as usize;
                    let ue = u_indptr[i + 1] as usize;
                    for pos in us..ue {
                        let j = u_indices[pos] as usize;
                        let v = u_values[pos];
                        sx += v * self.p[j];
                        sy += v * self.p_y[j];
                    }
                    self.ap[i] = sx;
                    self.ap_y[i] = sy;
                    denom_x += self.p[i] * sx;
                    denom_y += self.p_y[i] * sy;
                }

                let mut alpha_x = 0.0_f64;
                let mut alpha_y = 0.0_f64;
                if !denom_x.is_finite() || denom_x <= 1e-30 {
                    active_x = false;
                } else {
                    alpha_x = rz_old_x / denom_x;
                    if !alpha_x.is_finite() {
                        active_x = false;
                    }
                }
                if !denom_y.is_finite() || denom_y <= 1e-30 {
                    active_y = false;
                } else {
                    alpha_y = rz_old_y / denom_y;
                    if !alpha_y.is_finite() {
                        active_y = false;
                    }
                }
                if !active_x && !active_y {
                    return Ok(iter);
                }

                let mut rr_x = 0.0_f64;
                let mut rr_y = 0.0_f64;
                if active_x && active_y {
                    for i in 0..n {
                        let cand_x = x[i] + alpha_x * self.p[i];
                        let cand_y = y[i] + alpha_y * self.p_y[i];
                        if !cand_x.is_finite() {
                            active_x = false;
                        } else {
                            x[i] = cand_x;
                            let rxi = self.r[i] - alpha_x * self.ap[i];
                            self.r[i] = rxi;
                            rr_x += rxi * rxi;
                        }
                        if !cand_y.is_finite() {
                            active_y = false;
                        } else {
                            y[i] = cand_y;
                            let ryi = self.r_y[i] - alpha_y * self.ap_y[i];
                            self.r_y[i] = ryi;
                            rr_y += ryi * ryi;
                        }
                    }
                } else if active_x {
                    for i in 0..n {
                        let cand_x = x[i] + alpha_x * self.p[i];
                        if !cand_x.is_finite() {
                            active_x = false;
                            break;
                        }
                        x[i] = cand_x;
                        let rxi = self.r[i] - alpha_x * self.ap[i];
                        self.r[i] = rxi;
                        rr_x += rxi * rxi;
                    }
                } else {
                    for i in 0..n {
                        let cand_y = y[i] + alpha_y * self.p_y[i];
                        if !cand_y.is_finite() {
                            active_y = false;
                            break;
                        }
                        y[i] = cand_y;
                        let ryi = self.r_y[i] - alpha_y * self.ap_y[i];
                        self.r_y[i] = ryi;
                        rr_y += ryi * ryi;
                    }
                }

                if active_x && (!rr_x.is_finite() || rr_x == 0.0 || rr_x <= stop_rr_x) {
                    active_x = false;
                }
                if active_y && (!rr_y.is_finite() || rr_y == 0.0 || rr_y <= stop_rr_y) {
                    active_y = false;
                }
                if !active_x && !active_y {
                    return Ok(iter + 1);
                }

                if active_x && active_y {
                    ssor_apply_2rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r[..n],
                        &self.r_y[..n],
                        &mut self.z[..n],
                        &mut self.z_y[..n],
                    );
                    let mut rz_new_x = 0.0_f64;
                    let mut rz_new_y = 0.0_f64;
                    for i in 0..n {
                        rz_new_x += self.r[i] * self.z[i];
                        rz_new_y += self.r_y[i] * self.z_y[i];
                    }
                    let beta_x = if rz_new_x.is_finite() && rz_new_x > 1e-30 {
                        rz_new_x / rz_old_x
                    } else {
                        active_x = false;
                        0.0
                    };
                    let beta_y = if rz_new_y.is_finite() && rz_new_y > 1e-30 {
                        rz_new_y / rz_old_y
                    } else {
                        active_y = false;
                        0.0
                    };
                    if !beta_x.is_finite() {
                        active_x = false;
                    }
                    if !beta_y.is_finite() {
                        active_y = false;
                    }
                    if !active_x && !active_y {
                        return Ok(iter + 1);
                    }
                    rz_old_x = rz_new_x;
                    rz_old_y = rz_new_y;
                    if active_x && active_y {
                        for i in 0..n {
                            self.p[i] = self.z[i] + beta_x * self.p[i];
                            self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                        }
                    } else if active_x {
                        for i in 0..n {
                            self.p[i] = self.z[i] + beta_x * self.p[i];
                        }
                    } else {
                        for i in 0..n {
                            self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                        }
                    }
                } else if active_x {
                    ssor_apply_1rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r[..n],
                        &mut self.z[..n],
                    );
                    let mut rz_new_x = 0.0_f64;
                    for i in 0..n {
                        rz_new_x += self.r[i] * self.z[i];
                    }
                    if !rz_new_x.is_finite() || rz_new_x <= 1e-30 {
                        return Ok(iter + 1);
                    }
                    let beta_x = rz_new_x / rz_old_x;
                    if !beta_x.is_finite() {
                        return Ok(iter + 1);
                    }
                    rz_old_x = rz_new_x;
                    for i in 0..n {
                        self.p[i] = self.z[i] + beta_x * self.p[i];
                    }
                } else {
                    ssor_apply_1rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r_y[..n],
                        &mut self.z_y[..n],
                    );
                    let mut rz_new_y = 0.0_f64;
                    for i in 0..n {
                        rz_new_y += self.r_y[i] * self.z_y[i];
                    }
                    if !rz_new_y.is_finite() || rz_new_y <= 1e-30 {
                        return Ok(iter + 1);
                    }
                    let beta_y = rz_new_y / rz_old_y;
                    if !beta_y.is_finite() {
                        return Ok(iter + 1);
                    }
                    rz_old_y = rz_new_y;
                    for i in 0..n {
                        self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                    }
                }
            } else {
                // Single active axis remaining.
                let (r_slice, z_slice, p_slice, ap_slice, sol_slice, rz_old_ref, stop_rr) =
                    if active_x {
                        (
                            &mut self.r[..n],
                            &mut self.z[..n],
                            &mut self.p[..n],
                            &mut self.ap[..n],
                            &mut x[..n],
                            &mut rz_old_x,
                            stop_rr_x,
                        )
                    } else {
                        (
                            &mut self.r_y[..n],
                            &mut self.z_y[..n],
                            &mut self.p_y[..n],
                            &mut self.ap_y[..n],
                            &mut y[..n],
                            &mut rz_old_y,
                            stop_rr_y,
                        )
                    };

                let mut denom = 0.0_f64;
                for i in 0..n {
                    let mut sum = diag[i] * p_slice[i];
                    let ls = l_indptr[i] as usize;
                    let le = l_indptr[i + 1] as usize;
                    for pos in ls..le {
                        sum += l_values[pos] * p_slice[l_indices[pos] as usize];
                    }
                    let us = u_indptr[i] as usize;
                    let ue = u_indptr[i + 1] as usize;
                    for pos in us..ue {
                        sum += u_values[pos] * p_slice[u_indices[pos] as usize];
                    }
                    ap_slice[i] = sum;
                    denom += p_slice[i] * sum;
                }
                if !denom.is_finite() || denom <= 1e-30 {
                    return Ok(iter);
                }
                let alpha = *rz_old_ref / denom;
                if !alpha.is_finite() {
                    return Ok(iter);
                }
                let mut rr = 0.0_f64;
                for i in 0..n {
                    let cand = sol_slice[i] + alpha * p_slice[i];
                    if !cand.is_finite() {
                        return Ok(iter);
                    }
                    sol_slice[i] = cand;
                    let ri = r_slice[i] - alpha * ap_slice[i];
                    r_slice[i] = ri;
                    rr += ri * ri;
                }
                if !rr.is_finite() || rr == 0.0 || rr <= stop_rr {
                    return Ok(iter + 1);
                }
                ssor_apply_1rhs(
                    n, diag_inv, l_indptr, l_indices, l_values, u_indptr, u_indices, u_values,
                    omega, r_slice, z_slice,
                );
                let mut rz_new = 0.0_f64;
                for i in 0..n {
                    rz_new += r_slice[i] * z_slice[i];
                }
                if !rz_new.is_finite() || rz_new <= 1e-30 {
                    return Ok(iter + 1);
                }
                let beta = rz_new / *rz_old_ref;
                if !beta.is_finite() {
                    return Ok(iter + 1);
                }
                *rz_old_ref = rz_new;
                for i in 0..n {
                    p_slice[i] = z_slice[i] + beta * p_slice[i];
                }
            }
        }

        Ok(max_iters)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ssor_apply_2rhs(
    n: usize,
    diag_inv: &[f64],
    l_indptr: &[u32],
    l_indices: &[u32],
    l_values: &[f64],
    u_indptr: &[u32],
    u_indices: &[u32],
    u_values: &[f64],
    omega: f64,
    rx: &[f64],
    ry: &[f64],
    zx: &mut [f64],
    zy: &mut [f64],
) {
    for i in 0..n {
        let ls = l_indptr[i] as usize;
        let le = l_indptr[i + 1] as usize;
        let mut acc_x = 0.0_f64;
        let mut acc_y = 0.0_f64;
        for pos in ls..le {
            let j = l_indices[pos] as usize;
            let v = l_values[pos];
            acc_x += v * zx[j];
            acc_y += v * zy[j];
        }
        let dinv = diag_inv[i];
        zx[i] = dinv * (rx[i] - omega * acc_x);
        zy[i] = dinv * (ry[i] - omega * acc_y);
    }

    for i in (0..n).rev() {
        let us = u_indptr[i] as usize;
        let ue = u_indptr[i + 1] as usize;
        let mut acc_x = 0.0_f64;
        let mut acc_y = 0.0_f64;
        for pos in us..ue {
            let j = u_indices[pos] as usize;
            let v = u_values[pos];
            acc_x += v * zx[j];
            acc_y += v * zy[j];
        }
        let dinv = diag_inv[i];
        zx[i] -= omega * dinv * acc_x;
        zy[i] -= omega * dinv * acc_y;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ssor_apply_1rhs(
    n: usize,
    diag_inv: &[f64],
    l_indptr: &[u32],
    l_indices: &[u32],
    l_values: &[f64],
    u_indptr: &[u32],
    u_indices: &[u32],
    u_values: &[f64],
    omega: f64,
    r: &[f64],
    z: &mut [f64],
) {
    for i in 0..n {
        let ls = l_indptr[i] as usize;
        let le = l_indptr[i + 1] as usize;
        let mut acc = 0.0_f64;
        for pos in ls..le {
            acc += l_values[pos] * z[l_indices[pos] as usize];
        }
        z[i] = diag_inv[i] * (r[i] - omega * acc);
    }

    for i in (0..n).rev() {
        let us = u_indptr[i] as usize;
        let ue = u_indptr[i + 1] as usize;
        let mut acc = 0.0_f64;
        for pos in us..ue {
            acc += u_values[pos] * z[u_indices[pos] as usize];
        }
        z[i] -= omega * diag_inv[i] * acc;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SapRect {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    idx: u32,
}

#[allow(clippy::too_many_arguments)]
fn sap_find_overlaps<T: Scalar>(
    xs: &[T],
    ys: &[T],
    hw: &[T],
    hh: &[T],
    sap_bounds: &mut Vec<SapRect>,
    sap_order: &mut Vec<u32>,
    sap_sorted: &mut Vec<SapRect>,
    out: &mut Vec<(u32, u32)>,
) {
    out.clear();
    let n = xs.len();
    if n < 2 {
        return;
    }
    sap_bounds.resize(n, SapRect::default());
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        let hwi = hw[i];
        let hhi = hh[i];
        sap_bounds[i] = SapRect {
            x_min: (xi - hwi).to_f64(),
            x_max: (xi + hwi).to_f64(),
            y_min: (yi - hhi).to_f64(),
            y_max: (yi + hhi).to_f64(),
            idx: i as u32,
        };
    }

    if sap_order.len() != n {
        sap_order.clear();
        sap_order.extend(0..n as u32);
    }
    sap_order.sort_unstable_by(|&a, &b| {
        sap_bounds[a as usize]
            .x_min
            .total_cmp(&sap_bounds[b as usize].x_min)
            .then_with(|| a.cmp(&b))
    });

    sap_sorted.clear();
    for &idx in sap_order.iter() {
        sap_sorted.push(sap_bounds[idx as usize]);
    }

    for p in 0..n {
        let a = sap_sorted[p];
        for b in sap_sorted.iter().skip(p + 1) {
            if b.x_min >= a.x_max {
                break;
            }
            if a.x_min < b.x_max && a.y_min < b.y_max && b.y_min < a.y_max {
                if a.idx < b.idx {
                    out.push((a.idx, b.idx));
                } else {
                    out.push((b.idx, a.idx));
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PrismEdge {
    u: u32,
    v: u32,
    target_dist: f64,
    weight: f64,
}

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress overlap removal solver.
///
/// Each [`IterativeSolver::step`] performs one PRISM proximity-graph + fused 2-RHS sparse
/// Symmetric SSOR-PCG stress-majorization round and updates
/// [`PrismOverlapRemoval::last_overlap_count`].
#[derive(Debug, Clone)]
pub struct PrismOverlapRemoval<S: Scalar = f64> {
    config: PrismConfig,
    num_nodes: usize,
    eff_hw_f64: Vec<f64>,
    eff_hh_f64: Vec<f64>,
    eff_hw_s: Vec<S>,
    eff_hh_s: Vec<S>,
    rng: DeterministicRng,
    pcg: SparsePcgWorkspace,
    // Reusable per-round scratch buffers.
    work_layout: Layout2d<f64>,
    x_next: Vec<f64>,
    y_next: Vec<f64>,
    rhs_x: Vec<f64>,
    rhs_y: Vec<f64>,
    last_out_x: Vec<f64>,
    last_out_y: Vec<f64>,
    active_overlaps: Vec<(u32, u32)>,
    sap_bounds: Vec<SapRect>,
    sap_order: Vec<u32>,
    sap_sorted: Vec<SapRect>,
    raw_edges: Vec<(u32, u32)>,
    prism_edges: Vec<PrismEdge>,
    diag: Vec<f64>,
    diag_inv: Vec<f64>,
    l_deg: Vec<u32>,
    u_deg: Vec<u32>,
    l_pos: Vec<u32>,
    u_pos: Vec<u32>,
    l_indptr: Vec<u32>,
    l_indices: Vec<u32>,
    l_values: Vec<f64>,
    u_indptr: Vec<u32>,
    u_indices: Vec<u32>,
    u_values: Vec<f64>,
    sort_pts: Vec<(f64, f64, u32)>,
    #[cfg(not(feature = "delaunay"))]
    kd_scratch: KdQueryScratch,
    #[cfg(not(feature = "delaunay"))]
    knn_out: Vec<(usize, f64)>,
    iter: usize,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> PrismOverlapRemoval<S> {
    /// Creates a [`PrismOverlapRemoval`] solver for rectangular nodes with per-node half-widths
    /// and half-heights.
    pub fn new(half_widths: Vec<S>, half_heights: Vec<S>, config: PrismConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "PrismOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "PrismOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite("PrismOverlapRemoval::new::extents"));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(S::from_f64(hw_f));
            eff_hh_s.push(S::from_f64(hh_f));
        }

        let status = if config.max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_f64,
            eff_hh_f64,
            eff_hw_s,
            eff_hh_s,
            rng: DeterministicRng::from_seed_u64(config.seed),
            pcg: SparsePcgWorkspace::new(n),
            work_layout: Layout2d::zeros(n),
            x_next: vec![0.0; n],
            y_next: vec![0.0; n],
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            last_out_x: vec![0.0; n],
            last_out_y: vec![0.0; n],
            active_overlaps: Vec::new(),
            sap_bounds: Vec::with_capacity(n),
            sap_order: (0..n as u32).collect(),
            sap_sorted: Vec::with_capacity(n),
            raw_edges: Vec::new(),
            prism_edges: Vec::new(),
            diag: vec![0.0; n],
            diag_inv: vec![0.0; n],
            l_deg: vec![0; n],
            u_deg: vec![0; n],
            l_pos: vec![0; n],
            u_pos: vec![0; n],
            l_indptr: vec![0; n + 1],
            l_indices: Vec::new(),
            l_values: Vec::new(),
            u_indptr: vec![0; n + 1],
            u_indices: Vec::new(),
            u_values: Vec::new(),
            sort_pts: Vec::with_capacity(n),
            #[cfg(not(feature = "delaunay"))]
            kd_scratch: KdQueryScratch::with_capacity(K_FALLBACK_NEIGHBORS),
            #[cfg(not(feature = "delaunay"))]
            knn_out: Vec::with_capacity(K_FALLBACK_NEIGHBORS),
            iter: 0,
            last_overlap_count: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`PrismOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: PrismConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &PrismConfig {
        &self.config
    }

    /// Returns the expected number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent step.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }

    fn jitter_coincident_points(&mut self) -> bool {
        let n = self.num_nodes;
        if n < 2 {
            return false;
        }
        self.sort_pts.clear();
        {
            let xs = self.work_layout.x();
            let ys = self.work_layout.y();
            for i in 0..n {
                self.sort_pts.push((xs[i], ys[i], i as u32));
            }
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut moved = false;
        let (xs_mut, ys_mut) = self.work_layout.xy_mut();
        let mut i = 0usize;
        while i < n {
            let (base_x, base_y, _) = self.sort_pts[i];
            let mut j = i + 1;
            while j < n {
                let (nx, ny, _) = self.sort_pts[j];
                if (nx - base_x).abs() <= MIN_EDGE_DIST && (ny - base_y).abs() <= MIN_EDGE_DIST {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                moved = true;
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = extent * 0.05;
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    xs_mut[idx] += radius * cos_a;
                    ys_mut[idx] += radius * sin_a;
                }
            }
            i = j;
        }
        moved
    }

    fn build_proximity_edges(&mut self) -> CoreResult<()> {
        self.raw_edges.clear();

        #[cfg(feature = "delaunay")]
        {
            Delaunay2d::triangulate_edges_into(&self.work_layout, &mut self.raw_edges)?;
        }

        #[cfg(not(feature = "delaunay"))]
        {
            let n = self.num_nodes;
            let tree = StaticKdTree::<2, f64>::from_layout(&self.work_layout)?;
            let k = K_FALLBACK_NEIGHBORS.min(n.saturating_sub(1));
            for u in 0..n {
                tree.k_nearest_excluding_self_with_scratch(
                    u,
                    k,
                    &mut self.kd_scratch,
                    &mut self.knn_out,
                )?;
                let u32_u = u as u32;
                for &(v, _) in &self.knn_out {
                    let v32 = v as u32;
                    if u32_u < v32 {
                        self.raw_edges.push((u32_u, v32));
                    } else {
                        self.raw_edges.push((v32, u32_u));
                    }
                }
            }
        }

        // Merge in any non-Delaunay / non-kNN overlapping rectangle pairs so overlaps always
        // converge to 0.
        self.raw_edges.extend_from_slice(&self.active_overlaps);
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();
        Ok(())
    }

    fn build_laplacian_csr(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        self.prism_edges.clear();
        self.l_deg.fill(0);
        self.u_deg.fill(0);

        let s_max = self.config.max_scale_factor;
        // Tiny clearance margin ensures separated boxes cross the exact touching boundary cleanly
        // even under neighbor tension and f32/f64 rounding.
        let sep_eps = 1.0 + 1e-5_f64.max(16.0 * S::EPSILON.to_f64());

        let xs = self.work_layout.x();
        let ys = self.work_layout.y();

        for &(raw_u, raw_v) in &self.raw_edges {
            if raw_u == raw_v {
                continue;
            }
            let (u32_u, u32_v) = if raw_u < raw_v {
                (raw_u, raw_v)
            } else {
                (raw_v, raw_u)
            };
            let u = u32_u as usize;
            let v = u32_v as usize;

            let mut dx = xs[u] - xs[v];
            let mut dy = ys[u] - ys[v];
            let mut dist = dx.hypot(dy);
            let extent = self.eff_hw_f64[u]
                .max(self.eff_hh_f64[u])
                .max(self.eff_hw_f64[v])
                .max(self.eff_hh_f64[v]);

            if dist < MIN_EDGE_DIST {
                let angle = self.rng.next_f64() * std::f64::consts::TAU;
                let (sin_a, cos_a) = angle.sin_cos();
                let eps = extent.max(1e-2) * 1e-3;
                dx = eps * cos_a;
                dy = eps * sin_a;
                dist = eps;
            }

            let req_x = self.eff_hw_f64[u] + self.eff_hw_f64[v];
            let req_y = self.eff_hh_f64[u] + self.eff_hh_f64[v];
            let adx = dx.abs();
            let ady = dy.abs();

            let (s_ij, is_overlapping) = if req_x > 0.0 && req_y > 0.0 && adx < req_x && ady < req_y
            {
                let tx = if adx > 1e-12 {
                    (req_x * sep_eps) / adx
                } else {
                    f64::INFINITY
                };
                let ty = if ady > 1e-12 {
                    (req_y * sep_eps) / ady
                } else {
                    f64::INFINITY
                };
                (tx.min(ty).max(1.0).min(s_max), true)
            } else {
                (1.0, false)
            };

            // Collapsed micro-distance guard: overlapping near-coincident vertices (e.g., from
            // t-FDP leaf collapse) start at least `extent * 1e-4` instead of spending ~50 rounds
            // expanding geometrically from ~1e-9.
            let eff_dist = if is_overlapping {
                dist.max(extent * OVERLAP_MICRO_DIST_RATIO)
            } else {
                dist
            };
            let target_dist = s_ij * eff_dist;
            let d_eff = target_dist.max(MIN_WEIGHT_DIST);
            let mut w = 1.0 / (d_eff * d_eff);
            if is_overlapping {
                w *= OVERLAP_WEIGHT_BOOST;
            }

            self.prism_edges.push(PrismEdge {
                u: u32_u,
                v: u32_v,
                target_dist,
                weight: w,
            });
            // Canonical u < v places (u, v) in U[u] and (v, u) in L[v].
            self.u_deg[u] += 1;
            self.l_deg[v] += 1;
        }

        // Build split L + D + U CSR for symmetric weighted Laplacian L_w.
        self.l_indptr[0] = 0;
        self.u_indptr[0] = 0;
        for i in 0..n {
            self.l_indptr[i + 1] = self.l_indptr[i]
                .checked_add(self.l_deg[i])
                .ok_or_else(|| CoreError::invalid_csr("PRISM lower CSR nnz overflow"))?;
            self.u_indptr[i + 1] = self.u_indptr[i]
                .checked_add(self.u_deg[i])
                .ok_or_else(|| CoreError::invalid_csr("PRISM upper CSR nnz overflow"))?;
        }
        let l_nnz = self.l_indptr[n] as usize;
        let u_nnz = self.u_indptr[n] as usize;
        self.l_indices.resize(l_nnz, 0);
        self.l_values.resize(l_nnz, 0.0);
        self.u_indices.resize(u_nnz, 0);
        self.u_values.resize(u_nnz, 0.0);

        self.l_pos.copy_from_slice(&self.l_indptr[..n]);
        self.u_pos.copy_from_slice(&self.u_indptr[..n]);
        self.diag.fill(0.0);

        for edge in &self.prism_edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let w = edge.weight;

            self.diag[u] += w;
            self.diag[v] += w;

            let pu = self.u_pos[u] as usize;
            self.u_indices[pu] = edge.v;
            self.u_values[pu] = -w;
            self.u_pos[u] += 1;

            let pv = self.l_pos[v] as usize;
            self.l_indices[pv] = edge.u;
            self.l_values[pv] = -w;
            self.l_pos[v] += 1;
        }

        for i in 0..n {
            let d = self.diag[i];
            self.diag_inv[i] = if d.is_finite() && d > 1e-30 {
                1.0 / d
            } else {
                0.0
            };
        }

        Ok(())
    }

    fn run_majorization_round(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        let xs = self.work_layout.x();
        let ys = self.work_layout.y();
        self.x_next.copy_from_slice(xs);
        self.y_next.copy_from_slice(ys);

        let mean_x_before = xs.iter().sum::<f64>() / (n as f64);
        let mean_y_before = ys.iter().sum::<f64>() / (n as f64);

        for _inner in 0..INNER_MAJORIZATION_ITERS {
            self.rhs_x.fill(0.0);
            self.rhs_y.fill(0.0);

            for edge in &self.prism_edges {
                let u = edge.u as usize;
                let v = edge.v as usize;
                let dx = self.x_next[u] - self.x_next[v];
                let dy = self.y_next[u] - self.y_next[v];
                let norm = dx.hypot(dy);
                if norm >= MIN_EDGE_DIST && norm.is_finite() {
                    let f = edge.weight * edge.target_dist / norm;
                    let fx = f * dx;
                    let fy = f * dy;
                    self.rhs_x[u] += fx;
                    self.rhs_x[v] -= fx;
                    self.rhs_y[u] += fy;
                    self.rhs_y[v] -= fy;
                } else {
                    // Deterministic unit fallback if coincident nodes survived sorted jitter.
                    let angle = ((edge.u.wrapping_mul(2_654_435_761)
                        ^ edge.v.wrapping_mul(2_246_822_519))
                        as f64)
                        * 1e-9;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let f = edge.weight * edge.target_dist;
                    let fx = f * cos_a;
                    let fy = f * sin_a;
                    self.rhs_x[u] += fx;
                    self.rhs_x[v] -= fx;
                    self.rhs_y[u] += fy;
                    self.rhs_y[v] -= fy;
                }
            }

            self.pcg.solve_ldu_2rhs_ssor(
                &self.diag,
                &self.diag_inv,
                &self.l_indptr,
                &self.l_indices,
                &self.l_values,
                &self.u_indptr,
                &self.u_indices,
                &self.u_values,
                &self.rhs_x,
                &self.rhs_y,
                &mut self.x_next,
                &mut self.y_next,
                self.config.cg_tolerance,
                self.config.cg_max_iters,
                SSOR_OMEGA,
            )?;
        }

        // Preserve layout centroid to avoid numerical drift across rounds.
        let mean_x_after = self.x_next.iter().sum::<f64>() / (n as f64);
        let mean_y_after = self.y_next.iter().sum::<f64>() / (n as f64);
        let shift_x = mean_x_before - mean_x_after;
        let shift_y = mean_y_before - mean_y_after;
        for i in 0..n {
            self.x_next[i] += shift_x;
            self.y_next[i] += shift_y;
        }

        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S> for PrismOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "PrismOverlapRemoval::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let cache_hit = self.iter > 0 && {
            let xs = layout.x();
            let ys = layout.y();
            xs.iter()
                .zip(self.last_out_x.iter())
                .all(|(&a, &b)| a.to_f64() == b)
                && ys
                    .iter()
                    .zip(self.last_out_y.iter())
                    .all(|(&a, &b)| a.to_f64() == b)
        };

        if !cache_hit {
            sap_find_overlaps(
                layout.x(),
                layout.y(),
                &self.eff_hw_s,
                &self.eff_hh_s,
                &mut self.sap_bounds,
                &mut self.sap_order,
                &mut self.sap_sorted,
                &mut self.active_overlaps,
            );
        }

        if self.active_overlaps.is_empty() {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            let (wx, wy) = self.work_layout.xy_mut();
            for i in 0..n {
                wx[i] = xs[i].to_f64();
                wy[i] = ys[i].to_f64();
            }
        }

        if self.jitter_coincident_points() {
            sap_find_overlaps(
                self.work_layout.x(),
                self.work_layout.y(),
                &self.eff_hw_f64,
                &self.eff_hh_f64,
                &mut self.sap_bounds,
                &mut self.sap_order,
                &mut self.sap_sorted,
                &mut self.active_overlaps,
            );
        }

        self.build_proximity_edges()?;
        self.build_laplacian_csr()?;
        self.run_majorization_round()?;

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.x_next[i]);
                let ny = S::from_f64(self.y_next[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
                self.last_out_x[i] = xs_mut[i].to_f64();
                self.last_out_y[i] = ys_mut[i].to_f64();
            }
        }

        sap_find_overlaps(
            layout.x(),
            layout.y(),
            &self.eff_hw_s,
            &self.eff_hh_s,
            &mut self.sap_bounds,
            &mut self.sap_order,
            &mut self.sap_sorted,
            &mut self.active_overlaps,
        );
        self.last_overlap_count = self.active_overlaps.len();
        self.iter += 1;

        self.status = if self.last_overlap_count == 0 {
            SolverStatus::Converged
        } else if self.iter >= self.config.max_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };

        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.max_iterations
    }
}
