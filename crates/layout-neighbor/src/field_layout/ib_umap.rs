use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, SparseProbabilities};

use super::config::FieldFftStats;
use super::ib_umap_config::IbUmapConfig;
use super::ib_umap_repulsion::{
    DefaultIbUmapBackend, IbUmapCore, IbUmapRepulsion, IbUmapWorkspace,
};

/// Default per-node 2-hop cap used across 1+2-hop priority CSR builders (`32`).
pub const DEFAULT_MAX_2HOP_PER_NODE: usize = 32;

/// Cooperative, zero-per-step-allocation `ibUMAP` (interpolation-based batch Graph UMAP)
/// solver.
///
/// The repulsion backend `B` is bound at the type level via [`IbUmapRepulsion<L>`], rejecting
/// unsupported `(layout, backend)` combinations at compile time:
/// - [`Layout2d`](egraph_core::Layout2d): [`IbFft`](crate::IbFft) (default),
///   [`BarnesHut`](crate::BarnesHut), [`Exact`](crate::Exact), and
///   [`IbUmapBackend2d`](crate::IbUmapBackend2d).
///
/// # Compile-time Space/Backend Guards
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{IbFft, IbUmapConfig, IbUmapSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = IbUmapSolver::<Layout3d<f64>, IbFft>::new(&g, IbUmapConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{BarnesHut, IbUmapConfig, IbUmapSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = IbUmapSolver::<Layout3d<f64>, BarnesHut>::new(&g, IbUmapConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_neighbor::{IbFft, IbUmapConfig, IbUmapSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = IbUmapSolver::<LayoutNd<f64>, IbFft>::new(&g, IbUmapConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_neighbor::{BarnesHut, IbUmapConfig, IbUmapSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = IbUmapSolver::<LayoutNd<f64>, BarnesHut>::new(&g, IbUmapConfig::new());
/// ```
pub struct IbUmapSolver<
    L: EuclideanLayout + DefaultIbUmapBackend = Layout2d<f64>,
    B: IbUmapRepulsion<L> = <L as DefaultIbUmapBackend>::Backend,
> {
    pub(super) core: IbUmapCore,
    pub(super) space: IbUmapWorkspace,
    pub(super) backend: B,
    pub(super) backend_state: B::State,
    pub(super) _target: PhantomData<fn() -> L>,
}

impl<L, B> Clone for IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            space: self.space.clone(),
            backend: self.backend.clone(),
            backend_state: self.backend_state.clone(),
            _target: PhantomData,
        }
    }
}

impl<L, B> fmt::Debug for IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L> + fmt::Debug,
    B::State: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IbUmapSolver")
            .field("core", &self.core)
            .field("space", &self.space)
            .field("backend", &self.backend)
            .field("backend_state", &self.backend_state)
            .finish()
    }
}

impl<L, B> IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L> + Default,
{
    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using default regularized DoG
    /// Varadhan parameters, [`DEFAULT_MAX_2HOP_PER_NODE`], and `B::default()`.
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: IbUmapConfig) -> CoreResult<Self> {
        Self::new_with_backend(graph, config, B::default())
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using `B::default()`.
    #[inline]
    pub fn from_landmarks<W>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        Self::from_landmarks_with_backend(graph, landmarks, max_2hop_per_node, config, B::default())
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using $O(|E|)$ 1+2-hop pair
    /// sampling, regularized DoG Varadhan landmark ALT distances, and `B::default()`.
    #[inline]
    pub fn from_graph_dog<W: Scalar>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_cfg: &DogVaradhanConfig,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_dog_with_backend(graph, max_2hop_per_node, dog_cfg, config, B::default())
    }

    /// Constructs an [`IbUmapSolver`] from a validated [`KnnGraph`] via
    /// [`compute_fuzzy_simplicial_set`](egraph_neighborhood::compute_fuzzy_simplicial_set)
    /// using `B::default()`.
    #[inline]
    pub fn from_knn(
        knn: &KnnGraph,
        sigma_iters: usize,
        sigma_tolerance: f64,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        Self::from_knn_with_backend(knn, sigma_iters, sigma_tolerance, config, B::default())
    }

    /// Constructs an [`IbUmapSolver`] directly from a precomputed fuzzy simplicial set
    /// [`SparseProbabilities`] and optional per-node degrees using `B::default()`.
    #[inline]
    pub fn from_fuzzy_set(
        probs: SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        Self::from_fuzzy_set_with_backend(probs, degrees, config, B::default())
    }

    /// Constructs an [`IbUmapSolver`] from a borrowed [`SparseProbabilities`] without cloning
    /// its `f64` values vector, using `B::default()`.
    #[inline]
    pub fn from_fuzzy_set_ref(
        probs: &SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        Self::from_fuzzy_set_ref_with_backend(probs, degrees, config, B::default())
    }
}

impl<L, B> IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L>,
{
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &IbUmapConfig {
        &self.core.config
    }

    /// Returns the repulsion backend configuration.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the number of vertices expected in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.core.num_nodes
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.core.indices.len()
    }

    /// Returns the symmetrized positive attraction edges `(indptr, indices, edge_weights)` in CSR format.
    #[inline]
    pub fn edge_csr(&self) -> (&[u32], &[u32], &[f32]) {
        (
            &self.core.indptr,
            &self.core.indices,
            &self.core.edge_weights,
        )
    }

    /// Returns the per-vertex inverse-degree hub damping factors `eta`.
    #[inline]
    pub fn hub_eta(&self) -> &[f64] {
        &self.core.hub_eta
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.backend.total_fft_2d_calls(&self.backend_state)
    }

    /// Returns telemetry counters from the most recent `IbFft` step, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FieldFftStats> {
        self.backend.last_fft_stats(&self.backend_state)
    }

    /// Returns the resolved UMAP low-dimensional curve parameters `(a, b)`.
    #[inline]
    pub fn ab(&self) -> (f64, f64) {
        (self.core.a, self.core.b)
    }

    /// Returns the resolved UMAP low-dimensional curve parameters `(a, b)`.
    #[inline]
    pub fn ab_params(&self) -> (f64, f64) {
        self.ab()
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[inline]
    pub fn a(&self) -> f64 {
        self.core.a
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[inline]
    pub fn b(&self) -> f64 {
        self.core.b
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.core.status
    }

    /// Returns the number of completed iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.core.iteration
    }

    /// Returns the total scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations)
    }

    #[cfg(test)]
    pub(super) fn backend_state(&self) -> &B::State {
        &self.backend_state
    }
}

impl<L, B> IterativeSolver for IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        target.validate_ib_umap_target(&self.core, &self.space)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.core.status.is_finished() {
            return Ok(self.core.status);
        }

        if self.core.num_nodes < 2 {
            self.core.status = SolverStatus::Converged;
            return Ok(self.core.status);
        }

        if let Err(err) = target.step_ib_umap_target(
            &mut self.core,
            &mut self.space,
            &self.backend,
            &mut self.backend_state,
        ) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_)
                    | CoreError::NonFiniteValue { .. }
                    | CoreError::BackendUnavailable(_)
                    | CoreError::DeviceLimitExceeded(_)
            ) {
                self.core.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.core.iteration += 1;
        self.core.status = if self.core.iteration >= self.core.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.core.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.core.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.core.iteration
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations)
    }
}
