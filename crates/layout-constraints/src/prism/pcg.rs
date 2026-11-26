use egraph_core::{CoreError, CoreResult};

/// Reusable sparse CSR Preconditioned Conjugate Gradient (PCG) workspace.
///
/// Executes each CG iteration in $O(\mathrm{nnz})$ time with zero dense $N \times N$
/// matrix allocations and zero per-solve heap allocations once sized.
#[derive(Debug, Clone, Default)]
pub struct SparsePcgWorkspace {
    pub(super) diag_inv: Vec<f64>,
    pub(super) r: Vec<f64>,
    pub(super) z: Vec<f64>,
    pub(super) p: Vec<f64>,
    pub(super) ap: Vec<f64>,
    pub(super) r_y: Vec<f64>,
    pub(super) z_y: Vec<f64>,
    pub(super) p_y: Vec<f64>,
    pub(super) ap_y: Vec<f64>,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_ldu_1rhs_tail(
        &mut self,
        start_iter: usize,
        max_iters: usize,
        active_x: bool,
        diag: &[f64],
        diag_inv: &[f64],
        l_indptr: &[u32],
        l_indices: &[u32],
        l_values: &[f64],
        u_indptr: &[u32],
        u_indices: &[u32],
        u_values: &[f64],
        x: &mut [f64],
        y: &mut [f64],
        mut rz_old_x: f64,
        mut rz_old_y: f64,
        stop_rr_x: f64,
        stop_rr_y: f64,
        omega: f64,
    ) -> CoreResult<usize> {
        let n = x.len();
        let (r_slice, z_slice, p_slice, ap_slice, sol_slice, rz_old_ref, stop_rr) = if active_x {
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

        for iter in start_iter..max_iters {
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
                n, diag_inv, l_indptr, l_indices, l_values, u_indptr, u_indices, u_values, omega,
                r_slice, z_slice,
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

        Ok(max_iters)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn ssor_apply_2rhs(
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
pub(super) fn ssor_apply_1rhs(
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
