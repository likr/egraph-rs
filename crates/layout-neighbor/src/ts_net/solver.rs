use std::any::TypeId;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd,
    ManifoldLayout2d, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_neighborhood::{
    KnnGraph, SparseProbabilities, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options, partial_bfs_knn,
};
use egraph_spatial_fft::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::validation::validate_sparse_probabilities;

use super::config::{TsNetBackend, TsNetConfig, TsNetStage, compute_stage_lr};
pub(super) use super::dispatch::TsNetState;
use super::dispatch::{
    TsNetStepFn, TsNetValidateFn, resolve_backend_for_target, step_any_2d, step_any_3d,
    step_any_manifold, step_any_nd, validate_basic, validate_manifold, validate_nd,
    validate_riemannian,
};
use super::riemannian::step_riemannian;

/// Cooperative, zero-allocation-per-step unified tsNET solver implementing [`IterativeSolver`].
///
/// Supports:
/// - [`Layout2d`]: `Exact`, `BarnesHut` (`ArenaQuadtree2d`), `FftInterpolation`, `LinearFft`, `Auto`.
/// - [`Layout3d`]: `Exact`, `BarnesHut` (`ArenaOctree3d`), `Auto`.
/// - [`LayoutNd`] / [`ManifoldLayout2d`] / any [`RiemannianLayout`]: `Exact` (`Auto` resolves to `Exact`).
#[derive(Debug, Clone)]
pub struct TsNetSolver<L: RiemannianLayout = Layout2d<f64>> {
    pub(super) state: TsNetState<L::Scalar>,
    validate_fn: TsNetValidateFn<L>,
    step_fn: TsNetStepFn<L>,
}

impl<L: RiemannianLayout + 'static> TsNetSolver<L> {
    /// Constructs a [`TsNetSolver`] directly from a [`CsrGraph`] using the Pure 1+2-Hop
    /// adaptive perplexity preset (`TsNetConfig::pure_2hop()`).
    #[inline]
    pub fn new<W>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        Self::from_graph_pure_2hop(graph, config)
    }

    /// Constructs a [`TsNetSolver`] from precomputed sparse symmetric joint probabilities `probs`
    /// and `config` (including `config.stages`).
    pub fn from_probabilities(probs: SparseProbabilities, config: TsNetConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;

        let n = probs.num_nodes();
        let total_iterations: usize = config
            .stages
            .iter()
            .try_fold(0usize, |acc, s| acc.checked_add(s.iterations))
            .ok_or_else(|| CoreError::invalid_input("total stage iterations overflow usize"))?;

        let first_active = config.stages.iter().position(|s| s.iterations > 0);
        let stage_idx = first_active.unwrap_or(0);
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();
        let is_manifold = tid == TypeId::of::<ManifoldLayout2d<L::Scalar>>();

        let resolved_backend =
            resolve_backend_for_target(config.backend, n, is_2d, is_3d, is_nd, is_manifold)?;

        let (validate_fn, step_fn): (TsNetValidateFn<L>, TsNetStepFn<L>) = if is_2d {
            (validate_basic::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_basic::<L>, step_any_3d::<L>)
        } else if is_nd {
            (validate_nd::<L>, step_any_nd::<L>)
        } else if is_manifold {
            (validate_manifold::<L>, step_any_manifold::<L>)
        } else {
            (validate_riemannian::<L>, step_riemannian::<L>)
        };

        let mut fit_field = None;
        let mut fit_out = FitFieldOutput::new();
        if is_2d
            && matches!(
                resolved_backend,
                TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
            )
        {
            let init_entropy = first_active.is_some_and(|idx| config.stages[idx].lambda_r > 0.0);
            fit_field = Some(FitField2d::new(
                FitFieldConfig::default()
                    .with_ibfft(config.effective_ibfft_for(resolved_backend))
                    .with_epsilon_r(config.epsilon_r)
                    .with_compute_entropy(init_entropy),
            )?);
            fit_out.kl_repulsion_x.reserve(n);
            fit_out.kl_repulsion_y.reserve(n);
            if config
                .stages
                .iter()
                .any(|s| s.iterations > 0 && s.lambda_r > 0.0)
            {
                fit_out.entropy_repulsion_x.reserve(n);
                fit_out.entropy_repulsion_y.reserve(n);
            }
        }

        let (xy_len, rep_len, ent_len) = if is_2d {
            let (r_len, e_len) = match resolved_backend {
                TsNetBackend::Exact | TsNetBackend::BarnesHut => {
                    let any_entropy = config
                        .stages
                        .iter()
                        .any(|s| s.iterations > 0 && s.lambda_r > 0.0);
                    (n, if any_entropy { n } else { 0 })
                }
                _ => (0, 0),
            };
            (n, r_len, e_len)
        } else if is_manifold {
            (n, 0, 0)
        } else {
            (0, 0, 0)
        };

        let base_lr = config.resolve_learning_rate(n);
        let initial_stage_lr = first_active
            .map(|idx| compute_stage_lr(&config.stages[idx], n, base_lr, config.momentum))
            .unwrap_or(base_lr);

        let (_n, indptr, indices, values) = probs.into_parts();
        let inv_jacobi = if config.jacobi_precondition {
            let n_f64 = n as f64;
            let mut diag = vec![1.0_f64; n];
            for i in 0..n {
                let start = indptr[i] as usize;
                let end = indptr[i + 1] as usize;
                let mut row_sum = 0.0_f64;
                for pos in start..end {
                    if indices[pos] as usize != i {
                        row_sum += values[pos];
                    }
                }
                let scale = (n_f64 * row_sum).max(1.0);
                if scale.is_finite() {
                    diag[i] = 1.0 / scale;
                }
            }
            diag
        } else {
            Vec::new()
        };

        let state = TsNetState {
            config,
            resolved_backend,
            num_nodes: n,
            indptr,
            indices,
            p_vals: values,
            inv_jacobi,
            attr_x: vec![0.0; xy_len],
            attr_y: vec![0.0; xy_len],
            vel_x: vec![0.0; xy_len],
            vel_y: vec![0.0; xy_len],
            rep_x: vec![L::Scalar::ZERO; rep_len],
            rep_y: vec![L::Scalar::ZERO; rep_len],
            ent_x: vec![L::Scalar::ZERO; ent_len],
            ent_y: vec![L::Scalar::ZERO; ent_len],
            fit_field,
            fit_out,
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
            dim3_scratch: None,
            nd_scratch: None,
            manifold_scratch: None,
            riemannian_scratch: None,
            locked_nd_dims: None,
            locked_manifold: None,
            status,
            iteration: 0,
            total_iterations,
            stage_idx,
            stage_iter: 0,
            stage_entered: false,
            prev_stage_lr: base_lr,
            active_stage_lr: initial_stage_lr,
            last_fft_stats: None,
            total_fft_2d_calls: 0,
        };

        Ok(Self {
            state,
            validate_fn,
            step_fn,
        })
    }

    /// Constructs a [`TsNetSolver`] from a validated [`KnnGraph`] and optional per-node
    /// perplexities (falling back to `uniform_perplexity` when `perplexities` is `None`).
    pub fn from_knn(
        knn: &KnnGraph,
        perplexities: Option<&[f64]>,
        uniform_perplexity: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = knn.num_nodes();
        let perp_cfg = config.perplexity_config();
        let probs = match perplexities {
            Some(perps) => compute_knn_probabilities(knn, perps, &perp_cfg)?,
            None => {
                if !uniform_perplexity.is_finite() {
                    return Err(CoreError::non_finite(
                        "TsNetSolver::from_knn::uniform_perplexity",
                    ));
                }
                if uniform_perplexity <= 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "uniform_perplexity must be > 0.0, got {uniform_perplexity}"
                    )));
                }
                let perps = vec![uniform_perplexity; n];
                compute_knn_probabilities(knn, &perps, &perp_cfg)?
            }
        };
        Self::from_probabilities(probs, config)
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using Pure 1+2-Hop CSR (`{1.0, 2.0}`)
    /// and node-wise 2-hop adaptive perplexities (`clip(c * k2, min_p, max_p)`).
    pub fn from_graph_pure_2hop<W>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        config.validate()?;
        let (knn, _k2_deg, _stats) = build_pure_2hop_knn(graph)?;
        let perplexities = compute_nodewise_2hop_perplexities(
            graph,
            config.perplexity_c,
            config.min_perplexity,
            config.max_perplexity,
        )?;
        Self::from_knn(&knn, Some(&perplexities), 1.0, config)
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ with
    /// $A = A_1 + w_2 A_2$.
    pub fn from_graph_row_normalized<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_degree_decay(graph, w2, max_2hop_per_node, 0.0, config)
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with symmetric degree decay $(d_i d_j)^{-\mathrm{degree\_beta}}$ and
    /// [`TwoHopSamplingMode::TwoStepRandomWalk`].
    pub fn from_graph_row_normalized_with_degree_decay<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_options(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            TwoHopSamplingMode::TwoStepRandomWalk,
            None,
            config,
        )
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with configurable [`TwoHopSamplingMode`], optional `max_bridge_degree`,
    /// and symmetric degree decay $(d_i d_j)^{-\mathrm{degree\_beta}}$.
    pub fn from_graph_row_normalized_with_options<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        sampling_mode: TwoHopSamplingMode,
        max_bridge_degree: Option<usize>,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let probs = graph_12hop_row_normalized_probabilities_with_options(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            sampling_mode,
            max_bridge_degree,
            config.seed,
        )?;
        Self::from_probabilities(probs, config)
    }

    /// Constructs a [`TsNetSolver`] from a graph using Partial BFS $k$-NN (`config.seed`) and a
    /// uniform perplexity.
    pub fn from_graph_partial_bfs<W>(
        graph: &CsrGraph<W>,
        k: usize,
        uniform_perplexity: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        if k == 0 {
            return Err(CoreError::invalid_input("k must be >= 1"));
        }
        let knn = partial_bfs_knn(graph, k, config.seed)?;
        Self::from_knn(&knn, None, uniform_perplexity, config)
    }
}

impl<L: RiemannianLayout> TsNetSolver<L> {
    /// Returns the telemetry counters from the most recent FFT field evaluation, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FitFieldStats> {
        self.state.last_fft_stats.as_ref()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.state.total_fft_2d_calls
    }

    /// Returns the index of the active (or final completed) stage.
    #[inline]
    pub fn stage_index(&self) -> usize {
        self.state.stage_idx
    }

    /// Returns the stage definitions used by this solver.
    #[inline]
    pub fn stages(&self) -> &[TsNetStage] {
        &self.state.config.stages
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &TsNetConfig {
        &self.state.config
    }

    /// Returns the concrete backend resolved at construction (expanding [`TsNetBackend::Auto`]).
    #[inline]
    pub fn resolved_backend(&self) -> TsNetBackend {
        self.state.resolved_backend
    }

    /// Returns the number of vertices expected in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the stability-capped learning rate of the current stage.
    #[inline]
    pub fn current_stage_lr(&self) -> f64 {
        self.state.active_stage_lr
    }

    /// Returns the precomputed per-vertex inverse Jacobi preconditioning factors
    /// `1.0 / max(1.0, N * P_{i.})`, or an empty slice when disabled.
    #[inline]
    pub fn inv_jacobi(&self) -> &[f64] {
        &self.state.inv_jacobi
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.state.status
    }

    /// Returns the number of completed optimization iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.state.iteration
    }

    /// Returns the total configured optimization iterations across all stages.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.total_iterations)
    }
}

impl<L: RiemannianLayout> IterativeSolver for TsNetSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (self.validate_fn)(&self.state, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if self.state.num_nodes < 2 {
            self.state.status = SolverStatus::Converged;
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

        self.state.advance_stage_counters();
        Ok(self.state.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.state.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.state.iteration
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.total_iterations)
    }
}
