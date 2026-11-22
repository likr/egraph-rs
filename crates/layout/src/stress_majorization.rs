//! Pure-Rust Gansner-Koren-North (2004) Stress Majorization solver backed by a
//! zero-per-iteration-allocation Jacobi Preconditioned Conjugate Gradient [`PcgWorkspace`].

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

/// Reusable, zero-per-solve-allocation Jacobi Preconditioned Conjugate Gradient (PCG) workspace.
#[derive(Debug, Clone, Default)]
pub struct PcgWorkspace {
    diag_inv: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

impl PcgWorkspace {
    /// Creates a [`PcgWorkspace`] pre-sized for `n` unknowns.
    pub fn new(n: usize) -> Self {
        Self {
            diag_inv: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        }
    }

    /// Ensures internal scratch buffers hold at least `n` elements.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.diag_inv.len() < n {
            self.diag_inv.resize(n, 0.0);
            self.r.resize(n, 0.0);
            self.z.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.ap.resize(n, 0.0);
        }
    }

    /// Solves the dense symmetric positive semi-definite system `A x = b` via Jacobi PCG.
    ///
    /// Safely exits early when the initial/updated residual is zero, when `denom = p^T A p <= 1e-30`,
    /// or when `r^T z <= 1e-30` (such as on singular or disconnected Laplacian components).
    /// Returns the number of CG iterations executed.
    pub fn solve_dense_symmetric(
        &mut self,
        a: &[f64],
        b: &[f64],
        x: &mut [f64],
        tol: f64,
        max_iters: usize,
    ) -> CoreResult<usize> {
        let n = b.len();
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "PcgWorkspace::solve_dense_symmetric::x",
            ));
        }
        let expected_a = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("PCG matrix dimension overflow"))?;
        if a.len() != expected_a {
            return Err(CoreError::dimension_mismatch(
                expected_a,
                a.len(),
                "PcgWorkspace::solve_dense_symmetric::a",
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

        for i in 0..n {
            let diag = a[i * n + i];
            self.diag_inv[i] = if diag.is_finite() && diag > 1e-30 {
                1.0 / diag
            } else {
                0.0
            };
        }

        let mut rr0 = 0.0_f64;
        for i in 0..n {
            let row = &a[i * n..(i + 1) * n];
            let mut ax = 0.0_f64;
            for j in 0..n {
                ax += row[j] * x[j];
            }
            let ri = b[i] - ax;
            self.r[i] = ri;
            rr0 += ri * ri;
        }

        if !rr0.is_finite() || rr0 == 0.0 {
            return Ok(0);
        }
        let stop_rr = (tol * tol) * rr0.max(1.0);
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
                let row = &a[i * n..(i + 1) * n];
                let mut sum = 0.0_f64;
                for j in 0..n {
                    sum += row[j] * self.p[j];
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

/// Hyperparameters for [`StressMajorizationSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressMajorizationConfig {
    /// Relative stress convergence tolerance (`default: 1e-4`).
    pub epsilon: f64,
    /// Maximum outer stress majorization iterations (`default: 100`).
    pub max_iterations: usize,
    /// Inner Jacobi-PCG residual tolerance (`default: 1e-6`).
    pub pcg_tolerance: f64,
    /// Optional explicit inner PCG iteration cap (`None` uses `(2 * n).max(16)`).
    pub pcg_max_iterations: Option<usize>,
    /// Minimum pair separation below which majorizing forces are skipped (`default: 1e-4`).
    pub min_norm: f64,
}

impl Default for StressMajorizationConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            max_iterations: 100,
            pcg_tolerance: 1e-6,
            pcg_max_iterations: None,
            min_norm: 1e-4,
        }
    }
}

impl StressMajorizationConfig {
    /// Validates stress majorization parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "epsilon must be finite and >= 0.0, got {}",
                self.epsilon
            )));
        }
        if !self.pcg_tolerance.is_finite() || self.pcg_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "pcg_tolerance must be finite and >= 0.0, got {}",
                self.pcg_tolerance
            )));
        }
        if let Some(m) = self.pcg_max_iterations
            && m == 0
        {
            return Err(CoreError::invalid_input(
                "pcg_max_iterations must be >= 1 when specified",
            ));
        }
        if !self.min_norm.is_finite() || self.min_norm <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_norm must be finite and > 0.0, got {}",
                self.min_norm
            )));
        }
        Ok(())
    }
}

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver.
#[derive(Debug, Clone)]
pub struct StressMajorizationSolver<S: Scalar = f64> {
    config: StressMajorizationConfig,
    num_nodes: usize,
    dists: Vec<f64>,
    weights: Vec<f64>,
    l_w: Vec<f64>,
    rhs_x: Vec<f64>,
    rhs_y: Vec<f64>,
    sol_x: Vec<f64>,
    sol_y: Vec<f64>,
    pcg: PcgWorkspace,
    prev_stress: Option<f64>,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> StressMajorizationSolver<S> {
    /// Constructs a [`StressMajorizationSolver`] by computing all-pairs shortest paths via
    /// [`BfsWorkspace`] (unweighted) or [`DijkstraWorkspace`] (weighted).
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite(
                        "StressMajorizationSolver::from_graph::weights",
                    ));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "graph edge weights must be non-negative",
                    ));
                }
            }
        }

        let n = graph.num_nodes();
        let total = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
        let mut dists_flat = vec![f64::INFINITY; total];

        if n > 0 {
            let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
            let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
            for u in 0..n {
                let row = &mut dists_flat[u * n..(u + 1) * n];
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, u, row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, u, row)?;
                }
            }
        }

        Self::from_distance_matrix(n, &dists_flat, config)
    }

    /// Constructs a [`StressMajorizationSolver`] from a flat row-major `(n, n)` distance matrix.
    ///
    /// Positive `f64::INFINITY` marks unreachable pairs (weight `0.0`); `NaN` and negative
    /// distances are rejected.
    pub fn from_distance_matrix(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let expected = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("distance matrix dimension overflow"))?;
        if dists_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                dists_flat.len(),
                "StressMajorizationSolver::from_distance_matrix",
            ));
        }

        for &d in dists_flat {
            if d.is_nan() {
                return Err(CoreError::non_finite(
                    "StressMajorizationSolver::from_distance_matrix",
                ));
            }
            if d < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "distance matrix entries must be >= 0.0, got {d}"
                )));
            }
        }

        let mut dists = vec![0.0_f64; expected];
        let mut weights = vec![0.0_f64; expected];
        let mut l_w = vec![0.0_f64; expected];

        for i in 0..n {
            for j in (i + 1)..n {
                let dij_raw = dists_flat[i * n + j];
                let dji_raw = dists_flat[j * n + i];
                let ok_ij = dij_raw.is_finite() && dij_raw > 0.0;
                let ok_ji = dji_raw.is_finite() && dji_raw > 0.0;
                let d = match (ok_ij, ok_ji) {
                    (true, true) => 0.5 * (dij_raw + dji_raw),
                    (true, false) => dij_raw,
                    (false, true) => dji_raw,
                    (false, false) => f64::INFINITY,
                };
                if d.is_finite() && d > 0.0 {
                    let w = 1.0 / (d * d);
                    if w.is_finite() && w > 0.0 {
                        dists[i * n + j] = d;
                        dists[j * n + i] = d;
                        weights[i * n + j] = w;
                        weights[j * n + i] = w;
                        l_w[i * n + j] = -w;
                        l_w[j * n + i] = -w;
                        l_w[i * n + i] += w;
                        l_w[j * n + j] += w;
                    }
                }
            }
        }

        let status = if config.max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            dists,
            weights,
            l_w,
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            sol_x: vec![0.0; n],
            sol_y: vec![0.0; n],
            pcg: PcgWorkspace::new(n),
            prev_stress: None,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &StressMajorizationConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the stress value recorded after the most recent step, if any.
    #[inline]
    pub fn last_stress(&self) -> Option<f64> {
        self.prev_stress
    }

    /// Evaluates total weighted stress $\sum_{i < j} w_{ij} (\|x_i - x_j\| - d_{ij})^2$ for `layout`.
    pub fn compute_stress(&self, layout: &Layout2d<S>) -> CoreResult<f64> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "StressMajorizationSolver::compute_stress",
            ));
        }
        layout.validate_finite()?;
        Ok(self.compute_stress_slices(layout.x(), layout.y()))
    }

    fn compute_stress_slices(&self, xs: &[S], ys: &[S]) -> f64 {
        let n = self.num_nodes;
        let mut total = 0.0_f64;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let row_off = i * n;
            for j in (i + 1)..n {
                let w = self.weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                let d_target = self.dists[row_off + j];
                let norm = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
                let diff = norm - d_target;
                total += w * diff * diff;
            }
        }
        total
    }
}

impl<S: Scalar> IterativeSolver<S> for StressMajorizationSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "StressMajorizationSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.prev_stress = Some(0.0);
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let s_old = match self.prev_stress {
            Some(s) => s,
            None => self.compute_stress_slices(layout.x(), layout.y()),
        };

        self.rhs_x.fill(0.0);
        self.rhs_y.fill(0.0);
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.sol_x[i] = xs[i].to_f64();
                self.sol_y[i] = ys[i].to_f64();
            }
        }

        let min_norm = self.config.min_norm;
        for i in 0..n {
            let xi = self.sol_x[i];
            let yi = self.sol_y[i];
            let row_off = i * n;
            for j in (i + 1)..n {
                let w = self.weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                let d_target = self.dists[row_off + j];
                let dx = xi - self.sol_x[j];
                let dy = yi - self.sol_y[j];
                let norm = dx.hypot(dy);
                if norm >= min_norm && norm.is_finite() {
                    let f = w * d_target / norm;
                    let fx = f * dx;
                    let fy = f * dy;
                    self.rhs_x[i] += fx;
                    self.rhs_x[j] -= fx;
                    self.rhs_y[i] += fy;
                    self.rhs_y[j] -= fy;
                }
            }
        }

        let pcg_max = self.config.pcg_max_iterations.unwrap_or((2 * n).max(16));
        self.pcg.solve_dense_symmetric(
            &self.l_w,
            &self.rhs_x,
            &mut self.sol_x,
            self.config.pcg_tolerance,
            pcg_max,
        )?;
        self.pcg.solve_dense_symmetric(
            &self.l_w,
            &self.rhs_y,
            &mut self.sol_y,
            self.config.pcg_tolerance,
            pcg_max,
        )?;

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.sol_x[i]);
                let ny = S::from_f64(self.sol_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        }

        let new_stress = self.compute_stress_slices(layout.x(), layout.y());
        self.prev_stress = Some(new_stress);
        self.iter += 1;

        let converged = if s_old > 1e-30 {
            (s_old - new_stress).abs() / s_old < self.config.epsilon
        } else {
            new_stress <= 1e-30
        };

        self.status = if converged {
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

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
            y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    #[test]
    fn test_pcg_workspace_exact_zero_residual_and_singular_safety() {
        let mut pcg = PcgWorkspace::new(3);
        let a = [3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let b = [5.0, 5.0, 0.0];
        let mut x = [0.0, 0.0, 7.0];
        let iters = pcg
            .solve_dense_symmetric(&a, &b, &mut x, 1e-10, 20)
            .unwrap();
        assert!(iters > 0);
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
        assert_eq!(x[2], 7.0);

        // Zero residual exits immediately with 0 iterations (no 0/0 NaN).
        let iters_zero = pcg
            .solve_dense_symmetric(&a, &b, &mut x, 1e-10, 20)
            .unwrap();
        assert_eq!(iters_zero, 0);

        // Zero matrix (all singular) exits safely in 0 iterations.
        let zero_a = [0.0; 4];
        let zero_b = [0.0; 2];
        let mut x2 = [1.0, -2.0];
        assert_eq!(
            pcg.solve_dense_symmetric(&zero_a, &zero_b, &mut x2, 1e-6, 10)
                .unwrap(),
            0
        );
        assert_eq!(x2, [1.0, -2.0]);
    }

    #[test]
    fn test_stress_majorization_step_vs_run_bit_exact_f64_and_f32() {
        let grid = grid_graph(3, 3);
        let cfg = StressMajorizationConfig {
            max_iterations: 30,
            epsilon: 1e-6,
            ..Default::default()
        };

        // f64 bit-exact check
        let mut s_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
        let mut s_step = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
        let mut l_run = seeded_layout::<f64>(9, 1.0, 11);
        let mut l_step = l_run.clone();

        let stress_before = s_run.compute_stress(&l_run).unwrap();
        let st_run = s_run.run(&mut l_run).unwrap();
        assert!(st_run.is_finished());
        let stress_after = s_run.compute_stress(&l_run).unwrap();
        assert!(stress_after < stress_before * 0.2);

        let _ = s_step.run_steps(&mut l_step, 4).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_run.status(), s_step.status());
        assert_eq!(s_run.iteration(), s_step.iteration());
        for i in 0..9 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact check
        let mut s32_run = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
        let mut s32_step = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(9, 1.0, 11);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..9 {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_stress_majorization_disconnected_weighted_singleton_empty_and_errors() {
        let cfg = StressMajorizationConfig::default();

        // Empty and singleton graphs
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = StressMajorizationSolver::<f64>::from_graph(&empty, cfg).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = StressMajorizationSolver::<f64>::from_graph(&single, cfg).unwrap();
        let mut l1 = Layout2d::from_xy(vec![3.0], vec![-4.0]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x(), &[3.0]);
        assert_eq!(l1.y(), &[-4.0]);

        // Disconnected components + isolated vertex remain finite and minimize component stress.
        let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let mut s_disc = StressMajorizationSolver::<f64>::from_graph(&disc, cfg).unwrap();
        let mut l_disc = Layout2d::from_xy(
            vec![0.0, 0.2, 5.0, 8.0, -3.0],
            vec![0.0, 0.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        let status = s_disc.run(&mut l_disc).unwrap();
        assert!(status.is_finished());
        assert!(l_disc.is_all_finite());
        let d01 = (l_disc.x()[0] - l_disc.x()[1]).hypot(l_disc.y()[0] - l_disc.y()[1]);
        let d23 = (l_disc.x()[2] - l_disc.x()[3]).hypot(l_disc.y()[2] - l_disc.y()[3]);
        assert!((d01 - 1.0).abs() < 1e-2, "d01={d01}");
        assert!((d23 - 1.0).abs() < 1e-2, "d23={d23}");
        assert_eq!(l_disc.x()[4], -3.0);
        assert_eq!(l_disc.y()[4], 2.0);

        // Weighted graph via DijkstraWorkspace
        let wg = CsrGraph::<f64>::from_edges(
            3,
            &[(0, 1), (1, 2), (0, 2)],
            Some(&[1.0, 2.0, 10.0]),
            false,
            false,
        )
        .unwrap();
        let mut sw = StressMajorizationSolver::<f64>::from_graph(&wg, cfg).unwrap();
        let mut lw = Layout2d::from_xy(vec![0.0, 1.0, 5.0], vec![0.0, 0.2, 0.0]).unwrap();
        sw.run(&mut lw).unwrap();
        let d02 = (lw.x()[0] - lw.x()[2]).hypot(lw.y()[0] - lw.y()[2]);
        assert!((d02 - 3.0).abs() < 0.05, "d02={d02}");

        // Validation checks: NaN, negative distance, dimension mismatch
        assert!(matches!(
            StressMajorizationSolver::<f64>::from_distance_matrix(
                2,
                &[0.0, f64::NAN, 1.0, 0.0],
                cfg
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            StressMajorizationSolver::<f64>::from_distance_matrix(2, &[0.0, -1.0, -1.0, 0.0], cfg),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            StressMajorizationSolver::<f64>::from_distance_matrix(2, &[0.0, 1.0], cfg),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut wrong_l = Layout2d::<f64>::zeros(4);
        assert!(matches!(
            sw.step(&mut wrong_l),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }
}
