//! Pure-Rust Gansner-Koren-North (2004) Stress Majorization solver backed by a
//! zero-per-iteration-allocation Jacobi Preconditioned Conjugate Gradient ([`EuclideanPcg`] /
//! [`PcgWorkspace`]) on [`EuclideanLayout`](egraph_core::EuclideanLayout) targets and damped
//! geodesic SMACOF ([`GeodesicGradient`]) over any [`RiemannianLayout`] target.

mod config;
mod euclidean;
mod graph_prep;
mod pcg;
mod riemannian;
mod strategy;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

pub use config::StressMajorizationConfig;
pub use pcg::PcgWorkspace;
pub use strategy::{
    DefaultStressStrategy, EuclideanPcg, GeodesicGradient, StressMajorizationStrategy,
};

use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, Scalar,
    SolverStatus,
};

use graph_prep::{apsp_from_unweighted_graph, apsp_from_weighted_graph, build_dists_and_weights};

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver implementing
/// [`IterativeSolver`].
///
/// The optimization strategy `B: StressMajorizationStrategy<L>` is resolved at compile time via
/// [`DefaultStressStrategy`]:
/// - [`Layout2d`](egraph_core::Layout2d), [`Layout3d`](egraph_core::Layout3d), and
///   [`LayoutNd`](egraph_core::LayoutNd) default to [`EuclideanPcg`].
/// - [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d) defaults to [`GeodesicGradient`].
/// - [`GeodesicGradient`] is open for any [`RiemannianLayout`], including Euclidean layouts when
///   selected explicitly via `StressMajorizationSolver::<Layout2d<f64>, GeodesicGradient>` or
///   [`new_with_strategy`](Self::new_with_strategy).
///
/// Attempting to pair a non-Euclidean manifold layout with [`EuclideanPcg`] fails at compile time:
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, ManifoldLayout2d};
/// use egraph_layout_stress::{EuclideanPcg, StressMajorizationConfig, StressMajorizationSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _solver = StressMajorizationSolver::<ManifoldLayout2d<f64>, EuclideanPcg>::from_graph(
///     &g,
///     StressMajorizationConfig::default(),
/// );
/// ```
///
/// Using the default strategy on [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d) automatically
/// selects [`GeodesicGradient`]:
///
/// ```
/// use egraph_core::{CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d};
/// use egraph_layout_stress::{StressMajorizationConfig, StressMajorizationSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let mut solver = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph(
///     &g,
///     StressMajorizationConfig::default(),
/// )
/// .unwrap();
/// let mut layout = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 2, 7, 1.0).unwrap();
/// assert!(solver.run(&mut layout).unwrap().is_finished());
/// ```
pub struct StressMajorizationSolver<
    L: RiemannianLayout + DefaultStressStrategy = Layout2d<f64>,
    B: StressMajorizationStrategy<L> = <L as DefaultStressStrategy>::Strategy,
> {
    config: StressMajorizationConfig,
    num_nodes: usize,
    dists: Vec<f64>,
    weights: Vec<f64>,
    strategy: B,
    strategy_state: B::State,
    prev_stress: Option<f64>,
    iter: usize,
    status: SolverStatus,
    _layout: PhantomData<fn() -> L>,
}

impl<L, B> Clone for StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            num_nodes: self.num_nodes,
            dists: self.dists.clone(),
            weights: self.weights.clone(),
            strategy: self.strategy.clone(),
            strategy_state: self.strategy_state.clone(),
            prev_stress: self.prev_stress,
            iter: self.iter,
            status: self.status,
            _layout: PhantomData,
        }
    }
}

impl<L, B> fmt::Debug for StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L> + fmt::Debug,
    B::State: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StressMajorizationSolver")
            .field("config", &self.config)
            .field("num_nodes", &self.num_nodes)
            .field("strategy", &self.strategy)
            .field("prev_stress", &self.prev_stress)
            .field("iter", &self.iter)
            .field("status", &self.status)
            .finish()
    }
}

impl<L, B> StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L> + Default,
{
    /// Constructs a [`StressMajorizationSolver`] using `B::default()` by computing all-pairs
    /// shortest paths via `BfsWorkspace` (unweighted) or `DijkstraWorkspace` (weighted).
    #[inline]
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_with_strategy(graph, config, B::default())
    }

    /// Constructs a [`StressMajorizationSolver`] using `B::default()` by computing all-pairs
    /// shortest paths via `BfsWorkspace` (unweighted) or `DijkstraWorkspace` (weighted).
    #[inline]
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_with_strategy(graph, config, B::default())
    }

    /// Constructs a [`StressMajorizationSolver`] from an unweighted [`CsrGraph`] (ignoring edge
    /// weights and accepting `CsrGraph<()>` directly) using `B::default()`.
    #[inline]
    pub fn from_unweighted_graph<W>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_unweighted_graph_with_strategy(graph, config, B::default())
    }

    /// Constructs a [`StressMajorizationSolver`] from a flat row-major `(n, n)` distance matrix
    /// using `B::default()`.
    ///
    /// Positive `f64::INFINITY` marks unreachable pairs (weight `0.0`); `NaN` and negative
    /// distances are rejected.
    #[inline]
    pub fn from_distance_matrix(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_distance_matrix_with_strategy(n, dists_flat, config, B::default())
    }
}

impl<L, B> StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L>,
{
    /// Constructs a [`StressMajorizationSolver`] with an explicit strategy `strategy` by computing
    /// all-pairs shortest paths via `BfsWorkspace` or `DijkstraWorkspace`.
    #[inline]
    pub fn new_with_strategy<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
        strategy: B,
    ) -> CoreResult<Self> {
        Self::from_graph_with_strategy(graph, config, strategy)
    }

    /// Constructs a [`StressMajorizationSolver`] with an explicit strategy `strategy` by computing
    /// all-pairs shortest paths via `BfsWorkspace` or `DijkstraWorkspace`.
    pub fn from_graph_with_strategy<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
        strategy: B,
    ) -> CoreResult<Self> {
        strategy.validate()?;
        let (n, dists_flat) = apsp_from_weighted_graph(graph, &config)?;
        Self::from_distance_matrix_with_strategy(n, &dists_flat, config, strategy)
    }

    /// Constructs a [`StressMajorizationSolver`] from an unweighted [`CsrGraph`] with an explicit
    /// strategy `strategy`.
    pub fn from_unweighted_graph_with_strategy<W>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
        strategy: B,
    ) -> CoreResult<Self> {
        strategy.validate()?;
        let (n, dists_flat) = apsp_from_unweighted_graph(graph, &config)?;
        Self::from_distance_matrix_with_strategy(n, &dists_flat, config, strategy)
    }

    /// Constructs a [`StressMajorizationSolver`] from a flat row-major `(n, n)` distance matrix
    /// with an explicit strategy `strategy`.
    pub fn from_distance_matrix_with_strategy(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
        strategy: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        strategy.validate()?;
        let (dists, weights) = build_dists_and_weights(n, dists_flat)?;
        let strategy_state = strategy.init_state(n, &weights)?;
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
            strategy,
            strategy_state,
            prev_stress: None,
            iter: 0,
            status,
            _layout: PhantomData,
        })
    }
}

impl<L> StressMajorizationSolver<L, GeodesicGradient>
where
    L: RiemannianLayout + DefaultStressStrategy,
    GeodesicGradient: StressMajorizationStrategy<L>,
{
    /// Convenience alias for [`from_graph_with_strategy`](Self::from_graph_with_strategy) with
    /// [`GeodesicGradient`].
    #[inline]
    pub fn from_graph_riemannian<W: Scalar>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_with_strategy(graph, config, GeodesicGradient::new())
    }

    /// Convenience alias for
    /// [`from_unweighted_graph_with_strategy`](Self::from_unweighted_graph_with_strategy) with
    /// [`GeodesicGradient`].
    #[inline]
    pub fn from_unweighted_graph_riemannian<W>(
        graph: &CsrGraph<W>,
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_unweighted_graph_with_strategy(graph, config, GeodesicGradient::new())
    }

    /// Convenience alias for
    /// [`from_distance_matrix_with_strategy`](Self::from_distance_matrix_with_strategy) with
    /// [`GeodesicGradient`].
    #[inline]
    pub fn from_distance_matrix_riemannian(
        n: usize,
        dists_flat: &[f64],
        config: StressMajorizationConfig,
    ) -> CoreResult<Self> {
        Self::from_distance_matrix_with_strategy(n, dists_flat, config, GeodesicGradient::new())
    }
}

impl<L, B> StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L>,
{
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &StressMajorizationConfig {
        &self.config
    }

    /// Returns the active stress majorization strategy.
    #[inline]
    pub fn strategy(&self) -> &B {
        &self.strategy
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the stress value recorded after the most recent step, if any.
    #[inline]
    pub fn last_stress(&self) -> Option<f64> {
        self.prev_stress
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the configured maximum iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.config.max_iterations)
    }

    /// Evaluates total weighted stress $\sum_{i < j} w_{ij} (d(x_i, x_j) - d_{ij})^2$ for `layout`.
    pub fn compute_stress(&self, layout: &L) -> CoreResult<f64> {
        self.validate_target(layout)?;
        self.strategy.compute_stress(
            &self.strategy_state,
            self.num_nodes,
            &self.dists,
            &self.weights,
            layout,
        )
    }
}

impl<L, B> IterativeSolver for StressMajorizationSolver<L, B>
where
    L: RiemannianLayout + DefaultStressStrategy,
    B: StressMajorizationStrategy<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        self.strategy
            .validate_target(&self.strategy_state, self.num_nodes, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let s_old = match self.prev_stress {
            Some(s) => s,
            None => self.strategy.compute_stress(
                &self.strategy_state,
                self.num_nodes,
                &self.dists,
                &self.weights,
                target,
            )?,
        };

        match self.strategy.step(
            &mut self.strategy_state,
            self.num_nodes,
            &self.dists,
            &self.weights,
            &self.config,
            s_old,
            target,
        ) {
            Ok(None) => {
                self.prev_stress = Some(0.0);
                self.status = SolverStatus::Converged;
            }
            Ok(Some(new_stress)) => {
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
            }
            Err(err) => {
                if matches!(
                    err,
                    CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
                ) {
                    self.status = SolverStatus::Failed;
                }
                return Err(err);
            }
        }

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
    fn total_iterations(&self) -> Option<usize> {
        Some(self.config.max_iterations)
    }
}
