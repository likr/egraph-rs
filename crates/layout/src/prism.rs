//! PRISM proximity-stress overlap removal using Delaunay triangulation or KD-tree fallback.
//!
//! Implements Gansner & Hu (2010), "Efficient Node Overlap Removal Using a Proximity Stress
//! Model", using `Delaunay2d` (under `feature = "delaunay"`) or `StaticKdTree` $k$-NN fallback
//! (under `--no-default-features`), combined with `RectangleOverlapSweep` to guarantee `0`
//! overlaps at convergence, and a zero-dense-allocation sparse CSR Jacobi-PCG solver.

use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
#[cfg(feature = "delaunay")]
use egraph_spatial::Delaunay2d;
use egraph_spatial::RectangleOverlapSweep;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial::{KdQueryScratch, StaticKdTree};

#[cfg(not(feature = "delaunay"))]
const K_FALLBACK_NEIGHBORS: usize = 12;

const MIN_EDGE_DIST: f64 = 1e-9;
const MIN_WEIGHT_DIST: f64 = 1e-6;
const INNER_MAJORIZATION_ITERS: usize = 3;
const OVERLAP_WEIGHT_BOOST: f64 = 16.0;

/// Hyperparameters for [`PrismOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrismConfig {
    /// Maximum number of PRISM proximity-stress rounds (`default: 50`).
    pub max_iterations: usize,
    /// Maximum per-round edge expansion factor $s_{\max} > 1.0$ (`default: 1.5`).
    pub max_scale_factor: f64,
    /// Maximum inner Jacobi-PCG iterations per linear solve (`default: 200`).
    pub cg_max_iters: usize,
    /// Relative residual tolerance for inner Jacobi-PCG (`default: 1e-5`).
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
            cg_max_iters: 200,
            cg_tolerance: 1e-5,
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

/// Reusable sparse CSR Jacobi Preconditioned Conjugate Gradient (PCG) workspace.
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
/// Each [`IterativeSolver::step`] performs one PRISM proximity-graph + sparse Jacobi-PCG
/// stress-majorization round and updates [`PrismOverlapRemoval::last_overlap_count`].
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
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    x_next: Vec<f64>,
    y_next: Vec<f64>,
    rhs_x: Vec<f64>,
    rhs_y: Vec<f64>,
    raw_edges: Vec<(u32, u32)>,
    prism_edges: Vec<PrismEdge>,
    deg: Vec<u32>,
    diag_pos: Vec<u32>,
    csr_indptr: Vec<u32>,
    csr_indices: Vec<u32>,
    csr_values: Vec<f64>,
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
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            x_next: vec![0.0; n],
            y_next: vec![0.0; n],
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            raw_edges: Vec::new(),
            prism_edges: Vec::new(),
            deg: vec![0; n],
            diag_pos: vec![0; n],
            csr_indptr: vec![0; n + 1],
            csr_indices: Vec::new(),
            csr_values: Vec::new(),
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

    fn jitter_coincident_points(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        self.sort_pts.clear();
        for i in 0..n {
            self.sort_pts.push((self.x_buf[i], self.y_buf[i], i as u32));
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut i = 0usize;
        while i < n {
            let (base_x, base_y, _) = self.sort_pts[i];
            let mut j = i + 1;
            while j < n {
                let (nx, ny, _) = self.sort_pts[j];
                if (nx - base_x).abs() <= 1e-10 && (ny - base_y).abs() <= 1e-10 {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                // Perturb every site in the coincident cluster (preserving the first if desired,
                // or perturbing all subsequent sites deterministically with scale proportional to box size).
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = extent * 0.05;
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    self.x_buf[idx] += radius * cos_a;
                    self.y_buf[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    fn build_proximity_edges(&mut self) -> CoreResult<()> {
        self.raw_edges.clear();

        let work_layout = Layout2d::from_xy_slices(&self.x_buf, &self.y_buf)?;

        #[cfg(feature = "delaunay")]
        {
            let dt = Delaunay2d::triangulate(&work_layout)?;
            self.raw_edges.extend_from_slice(dt.edges());
        }

        #[cfg(not(feature = "delaunay"))]
        {
            let n = self.num_nodes;
            let tree = StaticKdTree::<2, f64>::from_layout(&work_layout)?;
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
        let active_overlaps = RectangleOverlapSweep::find_layout_overlaps(
            &work_layout,
            &self.eff_hw_f64,
            &self.eff_hh_f64,
        )?;
        for (u, v) in active_overlaps {
            self.raw_edges.push((u as u32, v as u32));
        }

        // Also ensure graph connectivity if Delaunay was collinear or disconnected: chain
        // consecutive sorted nodes if needed? Delaunay2d already chains collinear points,
        // and overlap pairs are directly connected. Deduplicate edges cleanly.
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();
        Ok(())
    }

    fn build_laplacian_csr(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        self.prism_edges.clear();
        self.deg.fill(0);

        let s_max = self.config.max_scale_factor;
        // Tiny clearance margin ensures separated boxes cross the exact touching boundary cleanly
        // even under neighbor tension and f32/f64 rounding.
        let sep_eps = 1.0 + 1e-5_f64.max(16.0 * S::EPSILON.to_f64());

        for &(u32_u, u32_v) in &self.raw_edges {
            let u = u32_u as usize;
            let v = u32_v as usize;
            let mut dx = self.x_buf[u] - self.x_buf[v];
            let mut dy = self.y_buf[u] - self.y_buf[v];
            let mut dist = dx.hypot(dy);
            if dist < MIN_EDGE_DIST {
                let angle = self.rng.next_f64() * std::f64::consts::TAU;
                let (sin_a, cos_a) = angle.sin_cos();
                let eps = self.eff_hw_f64[u]
                    .max(self.eff_hh_f64[u])
                    .max(self.eff_hw_f64[v])
                    .max(self.eff_hh_f64[v])
                    .max(1e-2)
                    * 1e-3;
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

            let target_dist = s_ij * dist;
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
            self.deg[u] += 1;
            self.deg[v] += 1;
        }

        // Build CSR for symmetric weighted Laplacian L_w (including diagonal entry per row).
        self.csr_indptr[0] = 0;
        for i in 0..n {
            // Each row has `deg[i]` off-diagonal entries + 1 diagonal entry.
            let row_len = self.deg[i] + 1;
            self.csr_indptr[i + 1] = self.csr_indptr[i]
                .checked_add(row_len)
                .ok_or_else(|| CoreError::invalid_csr("PRISM CSR nnz overflow"))?;
        }
        let nnz = self.csr_indptr[n] as usize;
        self.csr_indices.resize(nnz, 0);
        self.csr_values.resize(nnz, 0.0);

        // Place diagonal at the start of each row, followed by off-diagonal entries.
        for i in 0..n {
            let start = self.csr_indptr[i] as usize;
            self.csr_indices[start] = i as u32;
            self.csr_values[start] = 0.0;
            self.diag_pos[i] = (start + 1) as u32;
        }

        for edge in &self.prism_edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let w = edge.weight;

            let du = self.csr_indptr[u] as usize;
            let dv = self.csr_indptr[v] as usize;
            self.csr_values[du] += w;
            self.csr_values[dv] += w;

            let pu = self.diag_pos[u] as usize;
            self.csr_indices[pu] = edge.v;
            self.csr_values[pu] = -w;
            self.diag_pos[u] += 1;

            let pv = self.diag_pos[v] as usize;
            self.csr_indices[pv] = edge.u;
            self.csr_values[pv] = -w;
            self.diag_pos[v] += 1;
        }

        Ok(())
    }

    fn run_majorization_round(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        self.x_next.copy_from_slice(&self.x_buf);
        self.y_next.copy_from_slice(&self.y_buf);

        let mean_x_before = self.x_buf.iter().sum::<f64>() / (n as f64);
        let mean_y_before = self.y_buf.iter().sum::<f64>() / (n as f64);

        for inner in 0..INNER_MAJORIZATION_ITERS {
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
                }
            }

            // Save previous iterate to measure inner convergence.
            let mut max_prev_disp = 0.0_f64;
            if inner > 0 {
                self.x_buf.copy_from_slice(&self.x_next);
                self.y_buf.copy_from_slice(&self.y_next);
            }

            self.pcg.solve_csr(
                &self.csr_indptr,
                &self.csr_indices,
                &self.csr_values,
                &self.rhs_x,
                &mut self.x_next,
                self.config.cg_tolerance,
                self.config.cg_max_iters,
            )?;
            self.pcg.solve_csr(
                &self.csr_indptr,
                &self.csr_indices,
                &self.csr_values,
                &self.rhs_y,
                &mut self.y_next,
                self.config.cg_tolerance,
                self.config.cg_max_iters,
            )?;

            if inner > 0 {
                for i in 0..n {
                    let d = (self.x_next[i] - self.x_buf[i]).hypot(self.y_next[i] - self.y_buf[i]);
                    if d > max_prev_disp {
                        max_prev_disp = d;
                    }
                }
                if max_prev_disp <= 1e-4 {
                    break;
                }
            }
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

        let initial_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if initial_overlaps.is_empty() {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.x_buf[i] = xs[i].to_f64();
                self.y_buf[i] = ys[i].to_f64();
            }
        }

        self.jitter_coincident_points();
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
            }
        }

        let remaining =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        self.last_overlap_count = remaining.len();
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
