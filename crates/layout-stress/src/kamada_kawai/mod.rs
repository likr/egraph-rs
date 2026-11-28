//! Pure-Rust Kamada-Kawai (1989) spring layout solver with positive-definite Hessian
//! guard and bounded inner/outer iteration limits over any [`RiemannianLayout`].

mod solver;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, Scalar,
    SolverStatus,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use solver::{KamadaKawaiState, step_unified, validate_riemannian};

/// Hyperparameters for [`KamadaKawaiSolver`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct KamadaKawaiConfig {
    /// Outer gradient-norm convergence threshold $\varepsilon > 0$ (`default: 1e-2`).
    pub eps: f64,
    /// Maximum Newton-Raphson inner iterations per selected node (`default: 50`).
    pub max_inner_iters: usize,
    /// Optional cap on outer node-selection steps (`None` uses `(10 * n).max(100)`).
    pub max_iterations: Option<usize>,
}

impl Default for KamadaKawaiConfig {
    fn default() -> Self {
        Self {
            eps: 1e-2,
            max_inner_iters: 50,
            max_iterations: None,
        }
    }
}

impl KamadaKawaiConfig {
    /// Creates a new [`KamadaKawaiConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the outer gradient-norm convergence threshold `eps`.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the maximum Newton-Raphson inner iterations per selected node.
    pub fn with_max_inner_iters(mut self, max_inner_iters: usize) -> Self {
        self.max_inner_iters = max_inner_iters;
        self
    }

    /// Sets the optional cap on outer node-selection steps.
    pub fn with_max_iterations(mut self, max_iterations: Option<usize>) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Validates Kamada-Kawai hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("KamadaKawaiConfig::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eps must be > 0.0, got {}",
                self.eps
            )));
        }
        if self.max_inner_iters == 0 {
            return Err(CoreError::invalid_input("max_inner_iters must be >= 1"));
        }
        Ok(())
    }
}

/// Cooperative pure-Rust Kamada-Kawai (1989) layout solver implementing [`IterativeSolver`]
/// over any [`RiemannianLayout`] (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, and
/// external Riemannian targets).
///
/// Each [`IterativeSolver::step`] selects the vertex with largest gradient norm and moves it
/// via Newton-Raphson (up to `max_inner_iters`), guarding singular or indefinite tangent
/// Hessians with a bounded steepest-descent fallback so degenerate or saddle configurations
/// never step uphill or enter infinite loops.
#[derive(Debug, Clone)]
pub struct KamadaKawaiSolver<L: RiemannianLayout = Layout2d<f64>> {
    state: KamadaKawaiState,
    _layout: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout> KamadaKawaiSolver<L> {
    /// Constructs a [`KamadaKawaiSolver`] from a [`CsrGraph`] using [`BfsWorkspace`]
    /// (unweighted) or [`DijkstraWorkspace`] (weighted).
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: KamadaKawaiConfig) -> CoreResult<Self> {
        Self::from_graph(graph, config)
    }

    /// Constructs a [`KamadaKawaiSolver`] from a [`CsrGraph`] using [`BfsWorkspace`]
    /// (unweighted) or [`DijkstraWorkspace`] (weighted).
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: KamadaKawaiConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite(
                        "KamadaKawaiSolver::from_graph::weights",
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

    /// Constructs a [`KamadaKawaiSolver`] from an unweighted [`CsrGraph`] (ignoring edge
    /// weights and accepting `CsrGraph<()>` directly) via [`BfsWorkspace`].
    pub fn from_unweighted_graph<W>(
        graph: &CsrGraph<W>,
        config: KamadaKawaiConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let total = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
        let mut dists_flat = vec![f64::INFINITY; total];

        if n > 0 {
            let mut bfs = BfsWorkspace::new(n);
            for u in 0..n {
                let row = &mut dists_flat[u * n..(u + 1) * n];
                bfs.bfs_from(graph, u, row)?;
            }
        }

        Self::from_distance_matrix(n, &dists_flat, config)
    }

    /// Constructs a [`KamadaKawaiSolver`] from a flat row-major `(n, n)` distance matrix.
    ///
    /// Positive `f64::INFINITY` denotes unreachable node pairs (`k_ij = 0.0`); `NaN` and
    /// negative values are rejected.
    pub fn from_distance_matrix(
        n: usize,
        dists_flat: &[f64],
        config: KamadaKawaiConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let expected = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("distance matrix dimension overflow"))?;
        if dists_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                dists_flat.len(),
                "KamadaKawaiSolver::from_distance_matrix",
            ));
        }

        for &d in dists_flat {
            if d.is_nan() {
                return Err(CoreError::non_finite(
                    "KamadaKawaiSolver::from_distance_matrix",
                ));
            }
            if d < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "distance matrix entries must be >= 0.0, got {d}"
                )));
            }
        }

        let mut l = vec![0.0_f64; expected];
        let mut k = vec![0.0_f64; expected];
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
                    let k_ij = 1.0 / (d * d);
                    if k_ij.is_finite() && k_ij > 0.0 {
                        l[i * n + j] = d;
                        l[j * n + i] = d;
                        k[i * n + j] = k_ij;
                        k[j * n + i] = k_ij;
                    }
                }
            }
        }

        let max_iterations = config
            .max_iterations
            .unwrap_or_else(|| n.saturating_mul(10).max(100));
        let status = if max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            state: KamadaKawaiState {
                config,
                num_nodes: n,
                max_iterations,
                l,
                k,
                log_buf: Vec::new(),
                grad: Vec::new(),
                hessian: Vec::new(),
                delta: Vec::new(),
                locked_dim: None,
                iter: 0,
                status,
            },
            _layout: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &KamadaKawaiConfig {
        &self.state.config
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.state.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.state.iter
    }

    /// Returns the configured maximum iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.max_iterations)
    }

    #[cfg(test)]
    pub(super) fn target_distances(&self) -> &[f64] {
        &self.state.l
    }

    #[cfg(test)]
    pub(super) fn spring_weights(&self) -> &[f64] {
        &self.state.k
    }
}

impl<L: RiemannianLayout> IterativeSolver for KamadaKawaiSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_riemannian(&self.state, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if let Err(err) = step_unified(&mut self.state, target) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.state.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        Ok(self.state.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.state.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.state.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.max_iterations)
    }
}
