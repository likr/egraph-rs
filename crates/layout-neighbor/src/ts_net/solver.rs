use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, SolverStatus};
use egraph_neighborhood::{
    KnnGraph, SparseProbabilities, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options, partial_bfs_knn,
};
use egraph_spatial_fft::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats};
use egraph_spatial_trees::ArenaQuadtree2d;

use crate::validation::validate_sparse_probabilities;

use super::config::{TsNetBackend, TsNetConfig, TsNetStage, compute_stage_lr};
use super::nd::TsNetNdScratch;

/// Cooperative, zero-allocation-per-step unified tsNET solver implementing [`IterativeSolver`](egraph_core::IterativeSolver).
#[derive(Debug, Clone)]
pub struct TsNetSolver<S: Scalar = f64> {
    pub(super) config: TsNetConfig,
    pub(super) resolved_backend: TsNetBackend,
    pub(super) stages: Vec<TsNetStage>,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) p_vals: Vec<f64>,
    pub(super) inv_jacobi: Vec<f64>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) rep_x: Vec<S>,
    pub(super) rep_y: Vec<S>,
    pub(super) ent_x: Vec<S>,
    pub(super) ent_y: Vec<S>,
    pub(super) fit_field: Option<FitField2d<S>>,
    pub(super) fit_out: FitFieldOutput<S>,
    pub(super) quadtree: ArenaQuadtree2d<S>,
    pub(super) nd_scratch: Option<TsNetNdScratch<S>>,
    pub(super) status: SolverStatus,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) stage_idx: usize,
    pub(super) stage_iter: usize,
    pub(super) stage_entered: bool,
    pub(super) prev_stage_lr: f64,
    pub(super) active_stage_lr: f64,
    pub(super) last_fft_stats: Option<FitFieldStats>,
    pub(super) total_fft_2d_calls: u64,
}

impl<S: Scalar> TsNetSolver<S> {
    /// Constructs a [`TsNetSolver`] directly from a [`CsrGraph`] using the Pure 1+2-Hop
    /// adaptive perplexity preset (`TsNetConfig::pure_2hop()`).
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: TsNetConfig) -> CoreResult<Self> {
        Self::from_graph_pure_2hop(graph, config)
    }

    /// Constructs a [`TsNetSolver`] from precomputed sparse symmetric joint probabilities `probs`.
    pub fn from_probabilities(probs: SparseProbabilities, config: TsNetConfig) -> CoreResult<Self> {
        let stages = config.stages().to_vec();
        Self::from_probabilities_with_stages(probs, config, stages)
    }

    /// Constructs a [`TsNetSolver`] from `probs` with an explicit list of optimization stages.
    pub fn from_probabilities_with_stages(
        probs: SparseProbabilities,
        config: TsNetConfig,
        stages: Vec<TsNetStage>,
    ) -> CoreResult<Self> {
        config.validate()?;
        for stage in &stages {
            stage.validate()?;
        }
        validate_sparse_probabilities(&probs)?;

        let n = probs.n_nodes;
        let total_iterations: usize = stages
            .iter()
            .try_fold(0usize, |acc, s| acc.checked_add(s.iterations))
            .ok_or_else(|| CoreError::invalid_input("total stage iterations overflow usize"))?;

        let first_active = stages.iter().position(|s| s.iterations > 0);
        let stage_idx = first_active.unwrap_or(0);
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let resolved_backend = config.resolve_backend(n);
        let mut fit_field = None;
        let mut fit_out = FitFieldOutput::new();
        if matches!(
            resolved_backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        ) {
            let init_entropy = first_active.is_some_and(|idx| stages[idx].lambda_r > 0.0);
            fit_field = Some(FitField2d::new(FitFieldConfig {
                ibfft: config.effective_ibfft_for(resolved_backend),
                epsilon_r: config.epsilon_r,
                compute_entropy: init_entropy,
            })?);
            fit_out.kl_repulsion_x.reserve(n);
            fit_out.kl_repulsion_y.reserve(n);
            if stages.iter().any(|s| s.iterations > 0 && s.lambda_r > 0.0) {
                fit_out.entropy_repulsion_x.reserve(n);
                fit_out.entropy_repulsion_y.reserve(n);
            }
        }

        let (rep_len, ent_len) = match resolved_backend {
            TsNetBackend::Exact | TsNetBackend::BarnesHut => {
                let any_entropy = stages.iter().any(|s| s.iterations > 0 && s.lambda_r > 0.0);
                (n, if any_entropy { n } else { 0 })
            }
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft | TsNetBackend::Auto => (0, 0),
        };

        let base_lr = config.resolve_learning_rate(n);
        let initial_stage_lr = first_active
            .map(|idx| compute_stage_lr(&stages[idx], n, base_lr, config.momentum))
            .unwrap_or(base_lr);

        let inv_jacobi = if config.jacobi_precondition {
            let n_f64 = n as f64;
            let mut diag = vec![1.0_f64; n];
            for i in 0..n {
                let start = probs.indptr[i] as usize;
                let end = probs.indptr[i + 1] as usize;
                let mut row_sum = 0.0_f64;
                for pos in start..end {
                    if probs.indices[pos] as usize != i {
                        row_sum += probs.values[pos];
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

        Ok(Self {
            config,
            resolved_backend,
            stages,
            num_nodes: n,
            indptr: probs.indptr,
            indices: probs.indices,
            p_vals: probs.values,
            inv_jacobi,
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            vel_x: vec![0.0; n],
            vel_y: vec![0.0; n],
            rep_x: vec![S::ZERO; rep_len],
            rep_y: vec![S::ZERO; rep_len],
            ent_x: vec![S::ZERO; ent_len],
            ent_y: vec![S::ZERO; ent_len],
            fit_field,
            fit_out,
            quadtree: ArenaQuadtree2d::new(),
            nd_scratch: None,
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
    pub fn from_graph_pure_2hop<W: Scalar>(
        graph: &CsrGraph<W>,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
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
    ///
    /// When `max_2hop_per_node` is `None`, exact uncapped 1+2-hop neighborhoods are used;
    /// when `Some(cap)`, 2-hop expansion is hub-safely capped using `config.seed`.
    pub fn from_graph_row_normalized<W: Scalar>(
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
    pub fn from_graph_row_normalized_with_degree_decay<W: Scalar>(
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
    pub fn from_graph_row_normalized_with_options<W: Scalar>(
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
    pub fn from_graph_partial_bfs<W: Scalar>(
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

    /// Returns the telemetry counters from the most recent FFT field evaluation, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<FitFieldStats> {
        self.last_fft_stats
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.total_fft_2d_calls
    }

    /// Returns the index of the active (or final completed) stage.
    #[inline]
    pub fn stage_index(&self) -> usize {
        self.stage_idx
    }

    /// Returns the stage definitions used by this solver.
    #[inline]
    pub fn stages(&self) -> &[TsNetStage] {
        &self.stages
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &TsNetConfig {
        &self.config
    }

    /// Returns the concrete backend resolved at construction (expanding [`TsNetBackend::Auto`]).
    #[inline]
    pub fn resolved_backend(&self) -> TsNetBackend {
        self.resolved_backend
    }

    /// Returns the number of vertices expected in [`Layout2d`](egraph_core::Layout2d).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the stability-capped learning rate of the current stage.
    #[inline]
    pub fn current_stage_lr(&self) -> f64 {
        self.active_stage_lr
    }

    /// Returns the precomputed per-vertex inverse Jacobi preconditioning factors
    /// `1.0 / max(1.0, N * P_{i.})`, or an empty slice when disabled.
    #[inline]
    pub fn inv_jacobi(&self) -> &[f64] {
        &self.inv_jacobi
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of completed optimization iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Returns the total configured optimization iterations across all stages.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}
