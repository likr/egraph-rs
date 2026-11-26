//! Pure-Rust Gansner-Koren-North (2004) Stress Majorization solver backed by a
//! zero-per-iteration-allocation Jacobi Preconditioned Conjugate Gradient [`PcgWorkspace`].

mod config;
mod pcg;
#[cfg(test)]
mod tests;

pub use config::StressMajorizationConfig;
pub use pcg::PcgWorkspace;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

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
