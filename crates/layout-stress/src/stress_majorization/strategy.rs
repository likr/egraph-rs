use egraph_core::{
    CoreResult, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d, RiemannianLayout, Scalar,
};

use super::StressMajorizationConfig;

/// Jacobi Preconditioned Conjugate Gradient (PCG) stress majorization strategy for
/// [`EuclideanLayout`](egraph_core::EuclideanLayout) targets (`Layout2d`, `Layout3d`, `LayoutNd`).
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EuclideanPcg {}

impl EuclideanPcg {
    /// Creates a new [`EuclideanPcg`] stress majorization strategy.
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

/// Damped geodesic SMACOF gradient strategy for any [`RiemannianLayout`] target
/// (including [`ManifoldLayout2d`] and external curved spaces).
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeodesicGradient {}

impl GeodesicGradient {
    /// Creates a new [`GeodesicGradient`] stress majorization strategy.
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

/// Open optimization strategy contract for [`StressMajorizationSolver`](super::StressMajorizationSolver).
///
/// External crates (such as `egraph-surface`) can implement this trait for custom layouts and
/// strategies without modifying `egraph-layout-stress`.
pub trait StressMajorizationStrategy<L: RiemannianLayout + ?Sized> {
    /// Reusable per-solver workspace state owned by
    /// [`StressMajorizationSolver`](super::StressMajorizationSolver).
    type State: Clone;

    /// Validates strategy-specific hyperparameters before solver construction.
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    /// Initializes strategy-specific scratch buffers and precomputed operators for `num_nodes`
    /// vertices and symmetric row-major `(num_nodes, num_nodes)` stress weights `weights`.
    fn init_state(&self, num_nodes: usize, weights: &[f64]) -> CoreResult<Self::State>;

    /// Validates `target` compatibility without mutating solver or target state.
    fn validate_target(&self, state: &Self::State, num_nodes: usize, target: &L) -> CoreResult<()>;

    /// Evaluates total weighted stress $\sum_{i < j} w_{ij} (d(x_i, x_j) - d_{ij})^2$ on `layout`.
    fn compute_stress(
        &self,
        state: &Self::State,
        num_nodes: usize,
        dists: &[f64],
        weights: &[f64],
        layout: &L,
    ) -> CoreResult<f64>;

    /// Executes one majorization step on `layout` from previous stress `prev_stress`.
    ///
    /// Returns `Ok(None)` when `num_nodes <= 1` (trivially converged without incrementing
    /// `iteration`), or `Ok(Some(new_stress))` after advancing one step.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        state: &mut Self::State,
        num_nodes: usize,
        dists: &[f64],
        weights: &[f64],
        config: &StressMajorizationConfig,
        prev_stress: f64,
        layout: &mut L,
    ) -> CoreResult<Option<f64>>;
}

/// Associates each [`RiemannianLayout`] target type with its default
/// [`StressMajorizationStrategy`].
pub trait DefaultStressStrategy: RiemannianLayout {
    /// Default strategy selected by [`StressMajorizationSolver`](super::StressMajorizationSolver)
    /// when `B` is not explicitly specified.
    type Strategy: StressMajorizationStrategy<Self>;
}

impl<S: Scalar> DefaultStressStrategy for Layout2d<S> {
    type Strategy = EuclideanPcg;
}

impl<S: Scalar> DefaultStressStrategy for Layout3d<S> {
    type Strategy = EuclideanPcg;
}

impl<S: Scalar> DefaultStressStrategy for LayoutNd<S> {
    type Strategy = EuclideanPcg;
}

impl<S: Scalar> DefaultStressStrategy for ManifoldLayout2d<S> {
    type Strategy = GeodesicGradient;
}
