//! Pure-Rust Gansner-Koren-North (2004) Stress Majorization solver backed by a
//! zero-per-iteration-allocation Jacobi Preconditioned Conjugate Gradient [`PcgWorkspace`],
//! plus Riemannian geodesic SMACOF support over any [`RiemannianLayout`].

mod config;
mod euclidean;
mod pcg;
mod riemannian;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

pub use config::StressMajorizationConfig;
pub use pcg::PcgWorkspace;

use std::any::TypeId;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, IterativeSolver, Layout2d, ManifoldKind,
    ManifoldLayout2d, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use euclidean::{compute_stress_euclidean, step_euclidean, validate_euclidean};
use riemannian::{
    compute_stress_riemannian, step_riemannian, validate_manifold, validate_riemannian,
};

#[derive(Debug, Clone)]
pub(super) struct StressMajorizationState {
    pub(super) config: StressMajorizationConfig,
    pub(super) num_nodes: usize,
    pub(super) dists: Vec<f64>,
    pub(super) weights: Vec<f64>,
    pub(super) l_w: Vec<f64>,
    pub(super) rhs_x: Vec<f64>,
    pub(super) rhs_y: Vec<f64>,
    pub(super) sol_x: Vec<f64>,
    pub(super) sol_y: Vec<f64>,
    pub(super) nd_rhs: Vec<f64>,
    pub(super) nd_sol: Vec<f64>,
    pub(super) log_buf: Vec<f64>,
    pub(super) step_buf: Vec<f64>,
    pub(super) pcg: PcgWorkspace,
    pub(super) is_manifold: bool,
    pub(super) locked_dim: Option<usize>,
    pub(super) locked_manifold: Option<ManifoldKind>,
    pub(super) prev_stress: Option<f64>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

type StressValidateFn<L> = fn(&StressMajorizationState, &L) -> CoreResult<()>;
type StressStepFn<L> = fn(&mut StressMajorizationState, &mut L) -> CoreResult<()>;
type StressComputeFn<L> = fn(&StressMajorizationState, &L) -> CoreResult<f64>;

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver implementing
/// [`IterativeSolver`].
///
/// Construct via [`StressMajorizationSolver::new`] / [`from_graph`](Self::from_graph) /
/// [`from_distance_matrix`](Self::from_distance_matrix) for [`EuclideanLayout`] targets
/// (`Layout2d`, `Layout3d`, `LayoutNd` using Jacobi PCG), or via
/// [`from_graph_riemannian`](Self::from_graph_riemannian) /
/// [`from_distance_matrix_riemannian`](Self::from_distance_matrix_riemannian) for any
/// [`RiemannianLayout`] (`ManifoldLayout2d` and general curved spaces using damped geodesic SMACOF).
#[derive(Debug, Clone)]
pub struct StressMajorizationSolver<L: RiemannianLayout = Layout2d<f64>> {
    state: StressMajorizationState,
    validate_fn: StressValidateFn<L>,
    step_fn: StressStepFn<L>,
    stress_fn: StressComputeFn<L>,
    _layout: PhantomData<fn() -> L>,
}

fn apsp_from_weighted_graph<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &StressMajorizationConfig,
) -> CoreResult<(usize, Vec<f64>)> {
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
    Ok((n, dists_flat))
}

fn apsp_from_unweighted_graph<W>(
    graph: &CsrGraph<W>,
    config: &StressMajorizationConfig,
) -> CoreResult<(usize, Vec<f64>)> {
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
    Ok((n, dists_flat))
}

fn build_state_from_distance_matrix(
    n: usize,
    dists_flat: &[f64],
    config: StressMajorizationConfig,
    is_manifold: bool,
) -> CoreResult<StressMajorizationState> {
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

    Ok(StressMajorizationState {
        config,
        num_nodes: n,
        dists,
        weights,
        l_w,
        rhs_x: Vec::new(),
        rhs_y: Vec::new(),
        sol_x: Vec::new(),
        sol_y: Vec::new(),
        nd_rhs: Vec::new(),
        nd_sol: Vec::new(),
        log_buf: Vec::new(),
        step_buf: Vec::new(),
        pcg: PcgWorkspace::new(n),
        is_manifold,
        locked_dim: None,
        locked_manifold: None,
        prev_stress: None,
        iter: 0,
        status,
    })
}

impl<L: EuclideanLayout + 'static> StressMajorizationSolver<L> {
    /// Constructs a Euclidean PCG [`StressMajorizationSolver`] by computing all-pairs shortest
    /// paths via [`BfsWorkspace`] (unweighted) or [`DijkstraWorkspace`] (weighted).
    #[inline]
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_graph(graph, config)
    }

    /// Constructs a Euclidean PCG [`StressMajorizationSolver`] by computing all-pairs shortest
    /// paths via [`BfsWorkspace`] (unweighted) or [`DijkstraWorkspace`] (weighted).
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let (n, dists_flat) = apsp_from_weighted_graph(graph, &config)?;
        Self::from_distance_matrix(n, &dists_flat, config)
    }

    /// Constructs a Euclidean PCG [`StressMajorizationSolver`] from an unweighted [`CsrGraph`]
    /// (ignoring edge weights and accepting `CsrGraph<()>` directly) via [`BfsWorkspace`].
    pub fn from_unweighted_graph<W>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let (n, dists_flat) = apsp_from_unweighted_graph(graph, &config)?;
        Self::from_distance_matrix(n, &dists_flat, config)
    }

    /// Constructs a Euclidean PCG [`StressMajorizationSolver`] from a flat row-major `(n, n)`
    /// distance matrix.
    ///
    /// Positive `f64::INFINITY` marks unreachable pairs (weight `0.0`); `NaN` and negative
    /// distances are rejected.
    pub fn from_distance_matrix(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let state = build_state_from_distance_matrix(n, dists_flat, config, false)?;
        Ok(Self {
            state,
            validate_fn: validate_euclidean::<L>,
            step_fn: step_euclidean::<L>,
            stress_fn: compute_stress_euclidean::<L>,
            _layout: PhantomData,
        })
    }
}

impl<L: RiemannianLayout + Clone + 'static> StressMajorizationSolver<L> {
    /// Constructs a Riemannian geodesic SMACOF [`StressMajorizationSolver`] for curved spaces
    /// (such as [`ManifoldLayout2d`]) by computing all-pairs shortest paths via [`BfsWorkspace`]
    /// (unweighted) or [`DijkstraWorkspace`] (weighted).
    pub fn from_graph_riemannian<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let (n, dists_flat) = apsp_from_weighted_graph(graph, &config)?;
        Self::from_distance_matrix_riemannian(n, &dists_flat, config)
    }

    /// Constructs a Riemannian geodesic SMACOF [`StressMajorizationSolver`] from an unweighted
    /// [`CsrGraph`] (ignoring edge weights and accepting `CsrGraph<()>` directly).
    pub fn from_unweighted_graph_riemannian<W>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let (n, dists_flat) = apsp_from_unweighted_graph(graph, &config)?;
        Self::from_distance_matrix_riemannian(n, &dists_flat, config)
    }

    /// Constructs a Riemannian geodesic SMACOF [`StressMajorizationSolver`] from a flat row-major
    /// `(n, n)` distance matrix.
    pub fn from_distance_matrix_riemannian(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        let is_manifold = TypeId::of::<L>() == TypeId::of::<ManifoldLayout2d<L::Scalar>>();
        let state = build_state_from_distance_matrix(n, dists_flat, config, is_manifold)?;
        let validate_fn: StressValidateFn<L> = if is_manifold {
            validate_manifold::<L>
        } else {
            validate_riemannian::<L>
        };
        Ok(Self {
            state,
            validate_fn,
            step_fn: step_riemannian::<L>,
            stress_fn: compute_stress_riemannian::<L>,
            _layout: PhantomData,
        })
    }
}

impl<L: RiemannianLayout> StressMajorizationSolver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &StressMajorizationConfig {
        &self.state.config
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the stress value recorded after the most recent step, if any.
    #[inline]
    pub fn last_stress(&self) -> Option<f64> {
        self.state.prev_stress
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
        Some(self.state.config.max_iterations)
    }

    /// Evaluates total weighted stress $\sum_{i < j} w_{ij} (d(x_i, x_j) - d_{ij})^2$ for `layout`.
    pub fn compute_stress(&self, layout: &L) -> CoreResult<f64> {
        self.validate_target(layout)?;
        (self.stress_fn)(&self.state, layout)
    }
}

impl<L: RiemannianLayout> IterativeSolver for StressMajorizationSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (self.validate_fn)(&self.state, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if let Err(err) = (self.step_fn)(&mut self.state, target) {
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
        Some(self.state.config.max_iterations)
    }
}
