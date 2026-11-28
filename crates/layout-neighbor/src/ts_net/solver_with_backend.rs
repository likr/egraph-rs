use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, RiemannianLayout, SolverStatus};
use egraph_neighborhood::{
    KnnGraph, SparseProbabilities, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options, partial_bfs_knn,
};

use crate::validation::validate_sparse_probabilities;

use super::config::TsNetConfig;
use super::repulsion::{DefaultTsNetBackend, TsNetCore, TsNetRepulsion};
use super::solver::TsNetSolver;

impl<L, B> TsNetSolver<L, B>
where
    L: RiemannianLayout + DefaultTsNetBackend,
    B: TsNetRepulsion<L>,
{
    /// Constructs a [`TsNetSolver`] directly from a [`CsrGraph`] using Pure 1+2-Hop adaptive
    /// perplexity and an explicit repulsion `backend`.
    #[inline]
    pub fn new_with_backend<W>(
        graph: &CsrGraph<W>,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_pure_2hop_with_backend(graph, config, backend)
    }

    /// Alias for [`from_graph_pure_2hop_with_backend`](Self::from_graph_pure_2hop_with_backend).
    #[inline]
    pub fn pure_2hop_with_backend<W>(
        graph: &CsrGraph<W>,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_pure_2hop_with_backend(graph, config, backend)
    }

    /// Constructs a [`TsNetSolver`] from precomputed sparse symmetric joint probabilities `probs`,
    /// `config`, and an explicit repulsion `backend`.
    pub fn from_probabilities_with_backend(
        probs: SparseProbabilities,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
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

        let any_entropy = config
            .stages
            .iter()
            .any(|s| s.iterations > 0 && s.lambda_r > 0.0);
        let init_entropy = first_active.is_some_and(|idx| config.stages[idx].lambda_r > 0.0);

        let backend_state = backend.init_state(n, config.epsilon_r, any_entropy, init_entropy)?;
        let space = L::init_tsnet_workspace(n, any_entropy);

        let base_lr = config.resolve_learning_rate(n);
        let initial_stage_lr = first_active
            .map(|idx| config.compute_stage_lr(&config.stages[idx], n))
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

        let core = TsNetCore {
            config,
            num_nodes: n,
            indptr,
            indices,
            p_vals: values,
            inv_jacobi,
            status,
            iteration: 0,
            total_iterations,
            stage_idx,
            stage_iter: 0,
            stage_entered: false,
            prev_stage_lr: base_lr,
            active_stage_lr: initial_stage_lr,
        };

        Ok(Self {
            core,
            space,
            backend,
            backend_state,
            _target: PhantomData,
        })
    }

    /// Constructs a [`TsNetSolver`] from a validated [`KnnGraph`], optional per-node
    /// perplexities, `config`, and an explicit repulsion `backend`.
    pub fn from_knn_with_backend(
        knn: &KnnGraph,
        perplexities: Option<&[f64]>,
        uniform_perplexity: f64,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let n = knn.num_nodes();
        let perp_cfg = config.perplexity_config();
        let probs = match perplexities {
            Some(perps) => compute_knn_probabilities(knn, perps, &perp_cfg)?,
            None => {
                if !uniform_perplexity.is_finite() {
                    return Err(CoreError::non_finite(
                        "TsNetSolver::from_knn_with_backend::uniform_perplexity",
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
        Self::from_probabilities_with_backend(probs, config, backend)
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using Pure 1+2-Hop CSR (`{1.0, 2.0}`),
    /// node-wise 2-hop adaptive perplexities, and an explicit repulsion `backend`.
    pub fn from_graph_pure_2hop_with_backend<W>(
        graph: &CsrGraph<W>,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let (knn, _k2_deg, _stats) = build_pure_2hop_knn(graph)?;
        let perplexities = compute_nodewise_2hop_perplexities(
            graph,
            config.perplexity_c,
            config.min_perplexity,
            config.max_perplexity,
        )?;
        Self::from_knn_with_backend(&knn, Some(&perplexities), 1.0, config, backend)
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities and an explicit repulsion `backend`.
    pub fn from_graph_row_normalized_with_backend<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_degree_decay_with_backend(
            graph,
            w2,
            max_2hop_per_node,
            0.0,
            config,
            backend,
        )
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with symmetric degree decay and an explicit repulsion `backend`.
    pub fn from_graph_row_normalized_with_degree_decay_with_backend<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_row_normalized_with_options_with_backend(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            TwoHopSamplingMode::TwoStepRandomWalk,
            None,
            config,
            backend,
        )
    }

    /// Constructs a [`TsNetSolver`] directly from a graph using row-normalized 1+2-hop
    /// affinities with configurable [`TwoHopSamplingMode`], optional `max_bridge_degree`,
    /// symmetric degree decay, and an explicit repulsion `backend`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_graph_row_normalized_with_options_with_backend<W>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        sampling_mode: TwoHopSamplingMode,
        max_bridge_degree: Option<usize>,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let probs = graph_12hop_row_normalized_probabilities_with_options(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            sampling_mode,
            max_bridge_degree,
            config.seed,
        )?;
        Self::from_probabilities_with_backend(probs, config, backend)
    }

    /// Constructs a [`TsNetSolver`] from a graph using Partial BFS $k$-NN (`config.seed`), a
    /// uniform perplexity, and an explicit repulsion `backend`.
    pub fn from_graph_partial_bfs_with_backend<W>(
        graph: &CsrGraph<W>,
        k: usize,
        uniform_perplexity: f64,
        config: TsNetConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        if k == 0 {
            return Err(CoreError::invalid_input("k must be >= 1"));
        }
        let knn = partial_bfs_knn(graph, k, config.seed)?;
        Self::from_knn_with_backend(&knn, None, uniform_perplexity, config, backend)
    }
}
