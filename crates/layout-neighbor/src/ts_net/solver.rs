use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, SolverStatus,
};
use egraph_neighborhood::{KnnGraph, SparseProbabilities, TwoHopSamplingMode};
use egraph_spatial_fft::FitFieldStats;

use super::config::{TsNetConfig, TsNetStage};
use super::repulsion::{DefaultTsNetBackend, TsNetCore, TsNetRepulsion, TsNetWorkspace};

/// Cooperative, zero-allocation-per-step unified tsNET solver implementing [`IterativeSolver`].
///
/// The repulsion backend `B` is bound at the type level via [`TsNetRepulsion<L>`], rejecting
/// unsupported `(layout, backend)` combinations at compile time:
/// - [`Layout2d`](egraph_core::Layout2d): [`Exact`](crate::Exact),
///   [`BarnesHut`](crate::BarnesHut), [`FitFft`](crate::FitFft),
///   [`LinearFft`](crate::LinearFft), [`Auto2d`](crate::Auto2d) (default), and
///   [`TsNetBackend2d`](crate::TsNetBackend2d).
/// - [`Layout3d`](egraph_core::Layout3d): [`Exact`](crate::Exact),
///   [`BarnesHut`](crate::BarnesHut), [`Auto3d`](crate::Auto3d) (default), and
///   [`TsNetBackend3d`](crate::TsNetBackend3d).
/// - [`LayoutNd`](egraph_core::LayoutNd) / [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d):
///   [`Exact`](crate::Exact) (default).
/// - External [`RiemannianLayout`] targets (e.g., surface layouts): [`Exact`](crate::Exact) or
///   custom external [`TsNetRepulsion`] implementations.
///
/// # Compile-time Space/Backend Guards
///
/// Unsupported `(L, B)` pairs fail to compile:
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{FitFft, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<Layout3d<f64>, FitFft>::new(&g, TsNetConfig::pure_2hop());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{LinearFft, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<Layout3d<f64>, LinearFft>::new(&g, TsNetConfig::pure_2hop());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{Auto2d, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<Layout3d<f64>, Auto2d>::new(&g, TsNetConfig::pure_2hop());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_neighbor::{BarnesHut, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<LayoutNd<f64>, BarnesHut>::new(&g, TsNetConfig::pure_2hop());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, ManifoldLayout2d};
/// use egraph_layout_neighbor::{BarnesHut, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<ManifoldLayout2d<f64>, BarnesHut>::new(&g, TsNetConfig::pure_2hop());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout2d};
/// use egraph_layout_neighbor::{Auto3d, TsNetConfig, TsNetSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
/// let _ = TsNetSolver::<Layout2d<f64>, Auto3d>::new(&g, TsNetConfig::pure_2hop());
/// ```
pub struct TsNetSolver<
    L: RiemannianLayout + DefaultTsNetBackend = Layout2d<f64>,
    B: TsNetRepulsion<L> = <L as DefaultTsNetBackend>::Backend,
> {
    pub(super) core: TsNetCore,
    pub(super) space: TsNetWorkspace,
    pub(super) backend: B,
    pub(super) backend_state: B::State,
    pub(super) _target: PhantomData<fn() -> L>,
}

impl<L, B> Clone for TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L> + Clone,
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

impl<L, B> fmt::Debug for TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L> + fmt::Debug,
    B::State: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TsNetSolver")
            .field("core", &self.core)
            .field("space", &self.space)
            .field("backend", &self.backend)
            .field("backend_state", &self.backend_state)
            .finish()
    }
}

impl<L, B> TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L> + Default,
{
    /// Constructs a [`TsNetSolver`] directly from a [`CsrGraph`] using the Pure 1+2-Hop
    /// adaptive perplexity preset (`TsNetConfig::pure_2hop()`) and `B::default()`.
    #[inline]
    pub fn new<W>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        Self::new_with_backend(graph, config, B::default())
    }

    /// Alias for [`from_graph_pure_2hop`](Self::from_graph_pure_2hop) using `B::default()`.
    #[inline]
    pub fn pure_2hop<W>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        Self::from_graph_pure_2hop(graph, config)
    }

    /// Constructs a [`TsNetSolver`] from precomputed sparse symmetric joint probabilities `probs`,
    /// `config`, and `B::default()`.
    #[inline]
    pub fn from_probabilities(probs: SparseProbabilities, config: TsNetConfig) -> CoreResult<Self> {
        Self::from_probabilities_with_backend(probs, config, B::default())
    }

    /// Constructs a [`TsNetSolver`] from a validated [`KnnGraph`] and optional per-node
    /// perplexities using `B::default()`.
    #[inline]
    pub fn from_knn(
        knn: &KnnGraph,
        perplexities: Option<&[f64]>,
        uniform_perplexity: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_knn_with_backend(knn, perplexities, uniform_perplexity, config, B::default())
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using Pure 1+2-Hop CSR (`{1.0, 2.0}`),
    /// node-wise 2-hop adaptive perplexities (`clip(c * k2, min_p, max_p)`), and `B::default()`.
    #[inline]
    pub fn from_graph_pure_2hop<W>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        Self::from_graph_pure_2hop_with_backend(graph, config, B::default())
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ and `B::default()`.
    #[inline]
    pub fn from_graph_row_normalized<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_backend(
            graph,
            w2,
            max_2hop_per_node,
            config,
            B::default(),
        )
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with symmetric degree decay $(d_i d_j)^{-\mathrm{degree\_beta}}$ and `B::default()`.
    #[inline]
    pub fn from_graph_row_normalized_with_degree_decay<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_degree_decay_with_backend(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            config,
            B::default(),
        )
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with configurable [`TwoHopSamplingMode`], optional `max_bridge_degree`,
    /// symmetric degree decay, and `B::default()`.
    #[inline]
    pub fn from_graph_row_normalized_with_options<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        sampling_mode: TwoHopSamplingMode,
        max_bridge_degree: Option<usize>,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_options_with_backend(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            sampling_mode,
            max_bridge_degree,
            config,
            B::default(),
        )
    }

    /// Constructs a [`TsNetSolver`] from a graph using Partial BFS $k$-NN (`config.seed`), a
    /// uniform perplexity, and `B::default()`.
    #[inline]
    pub fn from_graph_partial_bfs<W>(
        graph: &CsrGraph<W>,
        k: usize,
        uniform_perplexity: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_partial_bfs_with_backend(
            graph,
            k,
            uniform_perplexity,
            config,
            B::default(),
        )
    }
}

impl<L, B> TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L>,
{
    /// Returns the telemetry counters from the most recent FFT field evaluation, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FitFieldStats> {
        self.backend.last_fft_stats(&self.backend_state)
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.backend.total_fft_2d_calls(&self.backend_state)
    }

    /// Returns the index of the active (or final completed) stage.
    #[inline]
    pub fn stage_index(&self) -> usize {
        self.core.stage_idx
    }

    /// Returns the stage definitions used by this solver.
    #[inline]
    pub fn stages(&self) -> &[TsNetStage] {
        &self.core.config.stages
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &TsNetConfig {
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

    /// Returns the symmetric high-dimensional joint probabilities `(indptr, indices, p_vals)` in CSR format.
    #[inline]
    pub fn probabilities_csr(&self) -> (&[u32], &[u32], &[f64]) {
        (&self.core.indptr, &self.core.indices, &self.core.p_vals)
    }

    /// Returns the stability-capped learning rate of the current stage.
    #[inline]
    pub fn current_stage_lr(&self) -> f64 {
        self.core.active_stage_lr
    }

    /// Returns the precomputed per-vertex inverse Jacobi preconditioning factors
    /// `1.0 / max(1.0, N * P_{i.})`, or an empty slice when disabled.
    #[inline]
    pub fn inv_jacobi(&self) -> &[f64] {
        &self.core.inv_jacobi
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.core.status
    }

    /// Returns the number of completed optimization iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.core.iteration
    }

    /// Returns the total configured optimization iterations across all stages.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations)
    }
}

impl<L, B> IterativeSolver for TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        target.validate_tsnet_target(&self.core, &self.space)
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

        if let Err(err) = target.step_tsnet_target(
            &mut self.core,
            &mut self.space,
            &self.backend,
            &mut self.backend_state,
        ) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.core.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.core.advance_stage_counters();
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
