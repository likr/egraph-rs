//! Pure-Rust unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET layout solver.
//!
//! Consolidates exact $O(N^2)$ tsNET, Barnes-Hut quadtree repulsion, and fused 2D FFT
//! interpolation (`FitField2d`) into a single allocation-free cooperative [`IterativeSolver`]
//! with a validated Pure 1+2-Hop adaptive perplexity preset ([`TsNetConfig::pure_2hop`]).

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_neighborhood::{
    KnnGraph, PerplexityConfig, SparseProbabilities, SymmetrizationMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_degree_decay, partial_bfs_knn,
};
use egraph_spatial::{
    ArenaQuadtree2d, FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats, IbFftConfig,
    InterpolationOrder,
};

const MIN_LAMBDA_SUM: f64 = 1e-12;
const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

/// Repulsion approximation backend used by [`TsNetSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TsNetBackend {
    /// Exact $O(N^2)$ pairwise Student-$t$ KL repulsion and entropy repulsion.
    Exact,
    /// $O(N \log N)$ Barnes-Hut quadtree approximation for KL and entropy repulsion.
    BarnesHut,
    /// Fused 2D FFT interpolation for KL and entropy repulsion via [`FitField2d`].
    #[default]
    FftInterpolation,
    /// Linear-time fused 2D FFT interpolation enforcing [`InterpolationOrder::Bilinear`] ($p=1$).
    LinearFft,
}

/// Hyperparameters governing a single optimization stage of [`TsNetSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsNetStage {
    /// Number of gradient descent iterations in this stage.
    pub iterations: usize,
    /// Early exaggeration multiplier applied to high-dimensional probabilities $P_{ij}$.
    pub exaggeration: f64,
    /// Weight of the KL divergence gradient term.
    pub lambda_kl: f64,
    /// Weight of the quadratic origin-compression gradient term.
    pub lambda_c: f64,
    /// Weight of the repulsive entropy gradient term.
    pub lambda_r: f64,
}

impl TsNetStage {
    /// Validates stage weights and exaggeration multiplier.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.exaggeration.is_finite() || self.exaggeration < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "stage exaggeration must be finite and >= 0.0, got {}",
                self.exaggeration
            )));
        }
        for (name, val) in [
            ("lambda_kl", self.lambda_kl),
            ("lambda_c", self.lambda_c),
            ("lambda_r", self.lambda_r),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "stage {name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        Ok(())
    }
}

/// Configuration for [`TsNetSolver`], defaulting to the Pure 1+2-Hop adaptive preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsNetConfig {
    /// Repulsion computation backend (`FftInterpolation` by default).
    pub backend: TsNetBackend,
    /// Multiplier $c$ for node-wise 2-hop adaptive perplexity (`0.9`).
    pub perplexity_c: f64,
    /// Lower clamp for node-wise 2-hop adaptive perplexity (`5.0`).
    pub min_perplexity: f64,
    /// Upper clamp for node-wise 2-hop adaptive perplexity (`1200.0`).
    pub max_perplexity: f64,
    /// Exponent applied to normalized neighbor distances `(d / max_d)^power` (`2.0`).
    pub power: f64,
    /// Maximum binary-search iterations for conditional entropy matching (`50`).
    pub sigma_iters: usize,
    /// Convergence tolerance for perplexity binary search (`1e-5`).
    pub sigma_tolerance: f64,
    /// Symmetrization strategy for high-dimensional probabilities (`StandardTsne`).
    pub symmetrization: SymmetrizationMode,
    /// Number of iterations in Stage 1 (early exaggeration, `250`).
    pub iterations_stage1: usize,
    /// Number of iterations in Stage 2 (compression relaxation, `250`).
    pub iterations_stage2: usize,
    /// Number of iterations in Stage 3 (fine-tuning / entropy repulsion, `500`).
    pub iterations_stage3: usize,
    /// Early exaggeration factor applied in Stage 1 (`4.0`).
    pub exaggeration: f64,
    /// Compression weight $\lambda_c$ in Stage 1 (`0.1`).
    pub lambda_c_stage1: f64,
    /// Compression weight $\lambda_c$ in Stage 2 (`0.01`).
    pub lambda_c_stage2: f64,
    /// Compression weight $\lambda_c$ in Stage 3 (`0.0`).
    pub lambda_c_stage3: f64,
    /// Repulsion weight $\lambda_r$ in Stage 3 (`0.0`).
    pub lambda_r_stage3: f64,
    /// Optional explicit base gradient descent learning rate (`None` uses $O(N)$ scaling `max(200.0, N / 3.0)`).
    pub learning_rate: Option<f64>,
    /// Enables per-vertex Jacobi preconditioning of the learning rate
    /// `lr_i = stage_lr / max(1.0, N * P_{i.})` (`true` by default).
    pub jacobi_precondition: bool,
    /// Momentum factor in `[0.0, 1.0)` (`0.8`).
    pub momentum: f64,
    /// Entropy softening constant $\varepsilon_r > 0$ (`0.05`).
    pub epsilon_r: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`0.5`).
    pub theta: f64,
    /// Interpolation-based 2D FFT grid configuration (`Bilinear` $p=1$, `intervals=25`, `64..=512`).
    pub ibfft: IbFftConfig,
    /// Deterministic explicit RNG seed (`42`).
    pub seed: u64,
}

impl Default for TsNetConfig {
    fn default() -> Self {
        Self::pure_2hop()
    }
}

impl TsNetConfig {
    /// Returns the Pure 1+2-Hop node-wise adaptive Fit-tsNET preset configuration.
    pub fn pure_2hop() -> Self {
        Self {
            backend: TsNetBackend::FftInterpolation,
            perplexity_c: 0.9,
            min_perplexity: 5.0,
            max_perplexity: 1200.0,
            power: 2.0,
            sigma_iters: 50,
            sigma_tolerance: 1e-5,
            symmetrization: SymmetrizationMode::StandardTsne,
            iterations_stage1: 250,
            iterations_stage2: 250,
            iterations_stage3: 500,
            exaggeration: 4.0,
            lambda_c_stage1: 0.1,
            lambda_c_stage2: 0.01,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.0,
            learning_rate: None,
            jacobi_precondition: true,
            momentum: 0.8,
            epsilon_r: 0.05,
            theta: 0.5,
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }

    /// Resolves the effective base learning rate for a graph with `num_nodes` vertices:
    /// explicit `Some(lr)` or automatic $O(N)$ scaling `200.0_f64.max((num_nodes as f64) / 3.0)`.
    #[inline]
    pub fn resolve_learning_rate(&self, num_nodes: usize) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| 200.0_f64.max((num_nodes as f64) / 3.0))
    }

    /// Returns the 3 optimization stages defined by this configuration.
    pub fn stages(&self) -> [TsNetStage; 3] {
        [
            TsNetStage {
                iterations: self.iterations_stage1,
                exaggeration: self.exaggeration,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage1,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: self.iterations_stage2,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage2,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: self.iterations_stage3,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage3,
                lambda_r: self.lambda_r_stage3,
            },
        ]
    }

    /// Validates all hyperparameters and nested `ibfft` settings.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.perplexity_c.is_finite() || self.perplexity_c <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "perplexity_c must be finite and > 0.0, got {}",
                self.perplexity_c
            )));
        }
        if !self.min_perplexity.is_finite() || self.min_perplexity <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_perplexity must be finite and > 0.0, got {}",
                self.min_perplexity
            )));
        }
        if !self.max_perplexity.is_finite() || self.max_perplexity < self.min_perplexity {
            return Err(CoreError::invalid_input(format!(
                "max_perplexity ({}) must be finite and >= min_perplexity ({})",
                self.max_perplexity, self.min_perplexity
            )));
        }
        if !self.power.is_finite() || self.power <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "power must be finite and > 0.0, got {}",
                self.power
            )));
        }
        if self.sigma_iters == 0 {
            return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
        }
        if !self.sigma_tolerance.is_finite() || self.sigma_tolerance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_tolerance must be finite and > 0.0, got {}",
                self.sigma_tolerance
            )));
        }
        if !self.exaggeration.is_finite() || self.exaggeration < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "exaggeration must be finite and >= 1.0, got {}",
                self.exaggeration
            )));
        }
        for (name, val) in [
            ("lambda_c_stage1", self.lambda_c_stage1),
            ("lambda_c_stage2", self.lambda_c_stage2),
            ("lambda_c_stage3", self.lambda_c_stage3),
            ("lambda_r_stage3", self.lambda_r_stage3),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        if let Some(lr) = self.learning_rate
            && (!lr.is_finite() || lr <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be finite and > 0.0, got {lr}"
            )));
        }
        if !self.momentum.is_finite() || self.momentum < 0.0 || self.momentum >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "momentum must be finite and in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.epsilon_r.is_finite() || self.epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "epsilon_r must be finite and > 0.0, got {}",
                self.epsilon_r
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        self.ibfft.validate()
    }

    fn effective_ibfft(&self) -> IbFftConfig {
        let mut cfg = self.ibfft;
        if self.backend == TsNetBackend::LinearFft {
            cfg.order = InterpolationOrder::Bilinear;
        }
        cfg
    }

    fn perplexity_config(&self) -> PerplexityConfig {
        PerplexityConfig {
            power: self.power,
            sigma_iters: self.sigma_iters,
            sigma_tolerance: self.sigma_tolerance,
            symmetrization: self.symmetrization,
        }
    }
}

/// Cooperative, zero-allocation-per-step unified tsNET solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct TsNetSolver<S: Scalar = f64> {
    config: TsNetConfig,
    stages: Vec<TsNetStage>,
    num_nodes: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    p_vals: Vec<f64>,
    inv_jacobi: Vec<f64>,
    attr_x: Vec<f64>,
    attr_y: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    ent_x: Vec<S>,
    ent_y: Vec<S>,
    fit_field: Option<FitField2d<S>>,
    fit_out: FitFieldOutput<S>,
    quadtree: ArenaQuadtree2d<S>,
    status: SolverStatus,
    iteration: usize,
    total_iterations: usize,
    stage_idx: usize,
    stage_iter: usize,
    stage_entered: bool,
    prev_stage_lr: f64,
    active_stage_lr: f64,
    last_fft_stats: Option<FitFieldStats>,
    total_fft_2d_calls: u64,
}

impl<S: Scalar> TsNetSolver<S> {
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

        let mut fit_field = None;
        let mut fit_out = FitFieldOutput::new();
        if matches!(
            config.backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        ) {
            let init_entropy = first_active.is_some_and(|idx| stages[idx].lambda_r > 0.0);
            fit_field = Some(FitField2d::new(FitFieldConfig {
                ibfft: config.effective_ibfft(),
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

        let (rep_len, ent_len) = match config.backend {
            TsNetBackend::Exact | TsNetBackend::BarnesHut => {
                let any_entropy = stages.iter().any(|s| s.iterations > 0 && s.lambda_r > 0.0);
                (n, if any_entropy { n } else { 0 })
            }
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => (0, 0),
        };

        let base_lr = config.resolve_learning_rate(n);
        let initial_stage_lr = first_active
            .map(|idx| compute_stage_lr(&stages[idx], n, base_lr, config.momentum))
            .unwrap_or(base_lr);

        let inv_jacobi = if config.jacobi_precondition {
            let n_f64 = n as f64;
            let mut diag = vec![1.0_f64; n];
            for i in 0..n {
                let start = probs.indptr[i];
                let end = probs.indptr[i + 1];
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
    /// affinities with symmetric degree decay $(d_i d_j)^{-\mathrm{degree\_beta}}$.
    pub fn from_graph_row_normalized_with_degree_decay<W: Scalar>(
        graph: &CsrGraph<W>,
        w2: f64,
        max_2hop_per_node: Option<usize>,
        degree_beta: f64,
        config: TsNetConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let probs = graph_12hop_row_normalized_probabilities_with_degree_decay(
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
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

    /// Returns the number of vertices expected in [`Layout2d`].
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

    fn enter_current_stage(&mut self) -> CoreResult<()> {
        let stage = self.stages[self.stage_idx];
        let base_lr = self.config.resolve_learning_rate(self.num_nodes);
        let stage_lr = compute_stage_lr(&stage, self.num_nodes, base_lr, self.config.momentum);
        if stage_lr < self.prev_stage_lr {
            let vel_scale = stage_lr / self.prev_stage_lr;
            for v in &mut self.vel_x {
                *v *= vel_scale;
            }
            for v in &mut self.vel_y {
                *v *= vel_scale;
            }
        }
        self.prev_stage_lr = stage_lr;
        self.active_stage_lr = stage_lr;

        let need_entropy = stage.lambda_r > 0.0;
        if matches!(
            self.config.backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        ) {
            let needs_rebuild = match &self.fit_field {
                Some(ff) => ff.config().compute_entropy != need_entropy,
                None => true,
            };
            if needs_rebuild {
                self.fit_field = Some(FitField2d::new(FitFieldConfig {
                    ibfft: self.config.effective_ibfft(),
                    epsilon_r: self.config.epsilon_r,
                    compute_entropy: need_entropy,
                })?);
            }
        }

        self.stage_entered = true;
        Ok(())
    }

    fn compute_attractions(&mut self, xs: &[S], ys: &[S], exaggeration: f64) {
        let n = self.num_nodes;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            for pos in start..end {
                let j = self.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dist_sq = dx * dx + dy * dy;
                let q_unnorm = 1.0 / (1.0 + dist_sq);
                let mult = (self.p_vals[pos] * exaggeration) * q_unnorm;
                ax += mult * dx;
                ay += mult * dy;
            }
            self.attr_x[i] = ax;
            self.attr_y[i] = ay;
        }
    }

    fn compute_exact_repulsion(&mut self, xs: &[S], ys: &[S], compute_entropy: bool) {
        let n = self.num_nodes;
        let eps_r = self.config.epsilon_r;
        let mut total_z = 0.0_f64;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let mut z_i = 0.0_f64;
            let mut rfx = 0.0_f64;
            let mut rfy = 0.0_f64;
            let mut efx = 0.0_f64;
            let mut efy = 0.0_f64;

            for j in 0..n {
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dist_sq = dx * dx + dy * dy;
                let w = 1.0 / (1.0 + dist_sq);
                z_i += w;
                let w2 = w * w;
                rfx += w2 * dx;
                rfy += w2 * dy;
                if compute_entropy {
                    let dist = dist_sq.sqrt();
                    if dist > EXACT_ENTROPY_MIN_DIST {
                        let mult = 1.0 / (dist * (dist + eps_r));
                        efx += mult * dx;
                        efy += mult * dy;
                    }
                }
            }
            total_z += z_i;
            self.rep_x[i] = S::from_f64(rfx);
            self.rep_y[i] = S::from_f64(rfy);
            if compute_entropy {
                self.ent_x[i] = S::from_f64(efx);
                self.ent_y[i] = S::from_f64(efy);
            }
        }

        let denom = total_z.max(MIN_TOTAL_Z);
        for i in 0..n {
            self.rep_x[i] = S::from_f64(self.rep_x[i].to_f64() / denom);
            self.rep_y[i] = S::from_f64(self.rep_y[i].to_f64() / denom);
        }
    }

    fn compute_barnes_hut_repulsion(
        &mut self,
        layout: &Layout2d<S>,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        self.quadtree.build(layout)?;
        let theta_s = S::from_f64(self.config.theta);
        self.quadtree
            .compute_kl_repulsion(theta_s, &mut self.rep_x, &mut self.rep_y)?;
        if compute_entropy {
            let eps_s = S::from_f64(self.config.epsilon_r).max(S::MIN_POSITIVE);
            self.quadtree.compute_entropy_repulsion(
                theta_s,
                eps_s,
                &mut self.ent_x,
                &mut self.ent_y,
            )?;
        }
        Ok(())
    }

    fn advance_stage_counters(&mut self) {
        self.iteration += 1;
        self.stage_iter += 1;
        if self.stage_iter < self.stages[self.stage_idx].iterations {
            self.status = SolverStatus::Running;
            return;
        }

        self.stage_iter = 0;
        self.stage_entered = false;
        if let Some(rel_next) = self.stages[self.stage_idx + 1..]
            .iter()
            .position(|s| s.iterations > 0)
        {
            self.stage_idx += 1 + rel_next;
            self.status = SolverStatus::Running;
        } else {
            self.status = SolverStatus::Completed;
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for TsNetSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "TsNetSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if !self.stage_entered {
            self.enter_current_stage()?;
        }

        let stage = self.stages[self.stage_idx];
        let compute_entropy = stage.lambda_r > 0.0;

        self.compute_attractions(layout.x(), layout.y(), stage.exaggeration);

        match self.config.backend {
            TsNetBackend::Exact => {
                self.compute_exact_repulsion(layout.x(), layout.y(), compute_entropy);
            }
            TsNetBackend::BarnesHut => {
                self.compute_barnes_hut_repulsion(layout, compute_entropy)?;
            }
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                let fit_field = self.fit_field.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("FitField2d workspace missing")
                })?;
                fit_field.evaluate(layout, &mut self.fit_out)?;
                self.last_fft_stats = Some(self.fit_out.stats);
                self.total_fft_2d_calls += self.fit_out.stats.total_fft_2d_calls;
            }
        }

        let n = self.num_nodes;
        let n_f64 = n as f64;
        let n_sq_f64 = n_f64 * n_f64;
        let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
        let stage_lr = self.active_stage_lr;
        let momentum = self.config.momentum;
        let use_jacobi = self.config.jacobi_precondition;

        let use_fft_buffers = matches!(
            self.config.backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        );

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            let (rep_x_slice, rep_y_slice) = if use_fft_buffers {
                (
                    self.fit_out.kl_repulsion_x.as_slice(),
                    self.fit_out.kl_repulsion_y.as_slice(),
                )
            } else {
                (self.rep_x.as_slice(), self.rep_y.as_slice())
            };
            let (ent_x_slice, ent_y_slice) = if compute_entropy {
                if use_fft_buffers {
                    (
                        self.fit_out.entropy_repulsion_x.as_slice(),
                        self.fit_out.entropy_repulsion_y.as_slice(),
                    )
                } else {
                    (self.ent_x.as_slice(), self.ent_y.as_slice())
                }
            } else {
                (&[][..], &[][..])
            };

            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();

                let grad_kl_x = 4.0 * (self.attr_x[i] - rep_x_slice[i].to_f64());
                let grad_kl_y = 4.0 * (self.attr_y[i] - rep_y_slice[i].to_f64());

                let grad_c_x = xi / n_f64;
                let grad_c_y = yi / n_f64;

                let (grad_r_x, grad_r_y) = if compute_entropy {
                    (
                        -ent_x_slice[i].to_f64() / n_sq_f64,
                        -ent_y_slice[i].to_f64() / n_sq_f64,
                    )
                } else {
                    (0.0, 0.0)
                };

                let raw_gx = (stage.lambda_kl * grad_kl_x
                    + stage.lambda_c * grad_c_x
                    + stage.lambda_r * grad_r_x)
                    / lambda_sum;
                let raw_gy = (stage.lambda_kl * grad_kl_y
                    + stage.lambda_c * grad_c_y
                    + stage.lambda_r * grad_r_y)
                    / lambda_sum;
                let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
                let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };

                let lr_i = if use_jacobi {
                    stage_lr * self.inv_jacobi[i]
                } else {
                    stage_lr
                };
                let mut vx = momentum * self.vel_x[i] - lr_i * gx;
                let mut vy = momentum * self.vel_y[i] - lr_i * gy;
                if !vx.is_finite() {
                    vx = 0.0;
                }
                if !vy.is_finite() {
                    vy = 0.0;
                }
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;

                let cand_x = xi + vx;
                let cand_y = yi + vy;
                let new_x = if cand_x.is_finite() { cand_x } else { xi };
                let new_y = if cand_y.is_finite() { cand_y } else { yi };

                // Reuse `attr_x` / `attr_y` as temporary `f64` coordinate buffers before recentering.
                self.attr_x[i] = new_x;
                self.attr_y[i] = new_y;
                mean_x += new_x;
                mean_y += new_y;
            }
        }

        mean_x /= n_f64;
        mean_y /= n_f64;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.attr_x[i] - shift_x);
            let cy = S::from_f64(self.attr_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }

        self.advance_stage_counters();
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}

#[inline]
fn compute_stage_lr(stage: &TsNetStage, n: usize, learning_rate: f64, momentum: f64) -> f64 {
    let n_f64 = n as f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let mut stage_lr = if stage.lambda_c > 0.0 {
        let max_stable_lr = (1.0 + momentum) * 0.85 * lambda_sum * n_f64 / stage.lambda_c;
        learning_rate.min(max_stable_lr)
    } else {
        learning_rate
    };
    if stage.lambda_r > 0.0 {
        let tsnet_lr_cap = 50.0_f64.min(2.0 * n_f64);
        stage_lr = stage_lr.min(tsnet_lr_cap);
    } else if stage.lambda_c == 0.0 || n < 100 {
        stage_lr = stage_lr.min(2.0 * n_f64);
    }
    stage_lr
}

fn validate_sparse_probabilities(probs: &SparseProbabilities) -> CoreResult<()> {
    let n = probs.n_nodes;
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if probs.indptr.len() != n + 1 {
        return Err(CoreError::dimension_mismatch(
            n + 1,
            probs.indptr.len(),
            "SparseProbabilities::indptr",
        ));
    }
    if probs.indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "SparseProbabilities indptr[0] must be 0, got {}",
            probs.indptr[0]
        )));
    }
    for row in 0..n {
        if probs.indptr[row + 1] < probs.indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "SparseProbabilities indptr must be non-decreasing at row {row}: {} > {}",
                probs.indptr[row],
                probs.indptr[row + 1]
            )));
        }
    }
    let nnz = probs.indptr[n];
    if probs.indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.indices.len(),
            "SparseProbabilities::indices",
        ));
    }
    if probs.values.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.values.len(),
            "SparseProbabilities::values",
        ));
    }
    for (&idx, &val) in probs.indices.iter().zip(probs.values.iter()) {
        if (idx as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n,
                "SparseProbabilities::indices",
            ));
        }
        if !val.is_finite() {
            return Err(CoreError::non_finite("SparseProbabilities::values"));
        }
        if val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SparseProbabilities values must be >= 0.0, got {val}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn make_barbell_graph(clique_size: usize) -> CsrGraph<f64> {
        let n = 2 * clique_size;
        let mut edges = Vec::new();
        for c in 0..2 {
            let off = (c * clique_size) as u32;
            for i in 0..clique_size as u32 {
                for j in (i + 1)..clique_size as u32 {
                    edges.push((off + i, off + j));
                }
            }
        }
        edges.push((0, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
            ys.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        }
        Layout2d::from_xy(xs, ys).unwrap()
    }

    fn rms_radius<S: Scalar>(layout: &Layout2d<S>) -> f64 {
        let n = layout.len();
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
            let xf = x.to_f64();
            let yf = y.to_f64();
            sum_sq += xf * xf + yf * yf;
        }
        (sum_sq / (n as f64)).sqrt()
    }

    #[test]
    fn test_tsnet_config_pure_2hop_defaults_and_validation() {
        let cfg = TsNetConfig::pure_2hop();
        assert_eq!(cfg, TsNetConfig::default());
        assert_eq!(cfg.backend, TsNetBackend::FftInterpolation);
        assert_eq!(cfg.perplexity_c, 0.9);
        assert_eq!(cfg.min_perplexity, 5.0);
        assert_eq!(cfg.max_perplexity, 1200.0);
        assert_eq!(cfg.power, 2.0);
        assert_eq!(cfg.sigma_iters, 50);
        assert_eq!(cfg.sigma_tolerance, 1e-5);
        assert_eq!(cfg.symmetrization, SymmetrizationMode::StandardTsne);
        assert_eq!(cfg.iterations_stage1, 250);
        assert_eq!(cfg.iterations_stage2, 250);
        assert_eq!(cfg.iterations_stage3, 500);
        assert_eq!(cfg.exaggeration, 4.0);
        assert_eq!(cfg.lambda_c_stage1, 0.1);
        assert_eq!(cfg.lambda_c_stage2, 0.01);
        assert_eq!(cfg.lambda_c_stage3, 0.0);
        assert_eq!(cfg.lambda_r_stage3, 0.0);
        assert_eq!(cfg.learning_rate, None);
        assert!(cfg.jacobi_precondition);
        assert_eq!(cfg.resolve_learning_rate(16), 200.0);
        assert_eq!(cfg.resolve_learning_rate(600), 200.0);
        assert!((cfg.resolve_learning_rate(3000) - 1000.0).abs() < 1e-12);
        assert_eq!(
            TsNetConfig {
                learning_rate: Some(42.0),
                ..cfg
            }
            .resolve_learning_rate(3000),
            42.0
        );
        assert_eq!(cfg.momentum, 0.8);
        assert_eq!(cfg.epsilon_r, 0.05);
        assert_eq!(cfg.theta, 0.5);
        assert_eq!(cfg.ibfft.order, InterpolationOrder::Bilinear);
        assert_eq!(cfg.ibfft.intervals, 25);
        assert_eq!(cfg.ibfft.min_grid_dim, 64);
        assert_eq!(cfg.ibfft.max_grid_dim, 512);
        assert_eq!(cfg.seed, 42);
        cfg.validate().unwrap();

        let invalid_cases = [
            TsNetConfig {
                perplexity_c: 0.0,
                ..cfg
            },
            TsNetConfig {
                min_perplexity: -1.0,
                ..cfg
            },
            TsNetConfig {
                min_perplexity: 10.0,
                max_perplexity: 5.0,
                ..cfg
            },
            TsNetConfig { power: 0.0, ..cfg },
            TsNetConfig {
                sigma_iters: 0,
                ..cfg
            },
            TsNetConfig {
                sigma_tolerance: f64::NAN,
                ..cfg
            },
            TsNetConfig {
                exaggeration: 0.5,
                ..cfg
            },
            TsNetConfig {
                lambda_c_stage1: -0.1,
                ..cfg
            },
            TsNetConfig {
                lambda_c_stage2: -0.1,
                ..cfg
            },
            TsNetConfig {
                lambda_c_stage3: -0.1,
                ..cfg
            },
            TsNetConfig {
                lambda_r_stage3: f64::INFINITY,
                ..cfg
            },
            TsNetConfig {
                learning_rate: Some(0.0),
                ..cfg
            },
            TsNetConfig {
                learning_rate: Some(f64::NAN),
                ..cfg
            },
            TsNetConfig {
                momentum: 1.0,
                ..cfg
            },
            TsNetConfig {
                momentum: -0.1,
                ..cfg
            },
            TsNetConfig {
                epsilon_r: 0.0,
                ..cfg
            },
            TsNetConfig { theta: -0.1, ..cfg },
            TsNetConfig {
                ibfft: IbFftConfig {
                    intervals: 0,
                    ..cfg.ibfft
                },
                ..cfg
            },
        ];
        for bad in invalid_cases {
            assert!(bad.validate().is_err(), "expected invalid config: {bad:?}");
        }
    }

    #[test]
    fn test_step_vs_run_bit_exact_f64_and_f32() {
        let grid = make_grid_graph(4, 4);
        let n = grid.num_nodes();

        for backend in [
            TsNetBackend::Exact,
            TsNetBackend::BarnesHut,
            TsNetBackend::FftInterpolation,
            TsNetBackend::LinearFft,
        ] {
            let cfg = TsNetConfig {
                backend,
                iterations_stage1: 8,
                iterations_stage2: 8,
                iterations_stage3: 8,
                lambda_r_stage3: 0.6,
                ..TsNetConfig::pure_2hop()
            };

            // f64 bit-exact check
            let mut l_run = seeded_layout::<f64>(n, 0.5, 7);
            let mut l_step = l_run.clone();
            let mut s_run = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
            let mut s_step = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();

            assert_eq!(s_run.status(), SolverStatus::Initialized);
            assert_eq!(s_run.total_iterations(), 24);
            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

            assert_eq!(
                s_step.run_steps(&mut l_step, 5).unwrap(),
                SolverStatus::Running
            );
            assert_eq!(s_step.iteration(), 5);
            assert_eq!(s_step.stage_index(), 0);
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            assert_eq!(s_step.status(), SolverStatus::Completed);
            assert_eq!(s_step.iteration(), 24);
            assert_eq!(s_step.stage_index(), 2);
            assert_eq!(s_step.total_fft_2d_calls(), s_run.total_fft_2d_calls());

            for i in 0..n {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact check
            let mut l32_run = seeded_layout::<f32>(n, 0.5, 7);
            let mut l32_step = l32_run.clone();
            let mut s32_run = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
            let mut s32_step = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
            assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
            for _ in 0..24 {
                s32_step.step(&mut l32_step).unwrap();
            }
            assert_eq!(s32_step.status(), SolverStatus::Completed);
            for i in 0..n {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    fn mean_edge_and_all_distances(graph: &CsrGraph<f64>, layout: &Layout2d<f64>) -> (f64, f64) {
        let n = layout.len();
        let mut edge_sum = 0.0;
        let mut edge_cnt = 0usize;
        for u in 0..n {
            for &v in graph.neighbors(u) {
                let vi = v as usize;
                if u < vi {
                    edge_sum +=
                        (layout.x()[u] - layout.x()[vi]).hypot(layout.y()[u] - layout.y()[vi]);
                    edge_cnt += 1;
                }
            }
        }
        let mut all_sum = 0.0;
        let mut all_cnt = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                all_sum += (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
                all_cnt += 1;
            }
        }
        (
            edge_sum / (edge_cnt.max(1) as f64),
            all_sum / (all_cnt.max(1) as f64),
        )
    }

    #[test]
    fn test_all_backends_converge_on_barbell_and_grid_without_nan() {
        let barbell = make_barbell_graph(6);
        let grid_small = make_grid_graph(4, 4);
        let grid_pure = make_grid_graph(6, 6);

        for backend in [
            TsNetBackend::Exact,
            TsNetBackend::BarnesHut,
            TsNetBackend::FftInterpolation,
            TsNetBackend::LinearFft,
        ] {
            // 1. Barbell (n=12) and 4x4 grid (n=16) with small-graph learning rate.
            let cfg_small = TsNetConfig {
                backend,
                learning_rate: Some(15.0),
                iterations_stage1: 80,
                iterations_stage2: 80,
                iterations_stage3: 100,
                lambda_r_stage3: 0.4,
                ..TsNetConfig::pure_2hop()
            };
            let mut l_barbell = seeded_layout::<f64>(barbell.num_nodes(), 0.2, 11);
            let mut solver_b =
                TsNetSolver::<f64>::from_graph_pure_2hop(&barbell, cfg_small).unwrap();
            assert_eq!(
                solver_b.run(&mut l_barbell).unwrap(),
                SolverStatus::Completed
            );
            assert!(l_barbell.is_all_finite());

            let c0_x = l_barbell.x()[..6].iter().sum::<f64>() / 6.0;
            let c0_y = l_barbell.y()[..6].iter().sum::<f64>() / 6.0;
            let c1_x = l_barbell.x()[6..].iter().sum::<f64>() / 6.0;
            let c1_y = l_barbell.y()[6..].iter().sum::<f64>() / 6.0;
            let centroid_dist = (c0_x - c1_x).hypot(c0_y - c1_y);
            let mut intra_spread = 0.0_f64;
            for i in 0..6 {
                intra_spread += (l_barbell.x()[i] - c0_x).hypot(l_barbell.y()[i] - c0_y);
                intra_spread += (l_barbell.x()[6 + i] - c1_x).hypot(l_barbell.y()[6 + i] - c1_y);
            }
            intra_spread /= 12.0;
            assert!(
                centroid_dist > intra_spread * 1.5,
                "{backend:?}: barbell failed to separate cliques (centroid_dist={centroid_dist}, intra_spread={intra_spread})"
            );

            let mut l_grid_small = seeded_layout::<f64>(grid_small.num_nodes(), 0.2, 19);
            let mut solver_gs =
                TsNetSolver::<f64>::from_graph_pure_2hop(&grid_small, cfg_small).unwrap();
            assert_eq!(
                solver_gs.run(&mut l_grid_small).unwrap(),
                SolverStatus::Completed
            );
            assert!(l_grid_small.is_all_finite());
            let (edge_s, all_s) = mean_edge_and_all_distances(&grid_small, &l_grid_small);
            assert!(
                edge_s < 0.65 * all_s,
                "{backend:?}: 4x4 grid locality failed (mean_edge={edge_s}, mean_all={all_s})"
            );

            // 2. Classic Fit-tsNET 3-stage schedule (lambda_c_stage2=1.2, lambda_r_stage3=0.6)
            // on 5x5 grid across all 4 backends.
            let grid_5x5 = make_grid_graph(5, 5);
            let cfg_classic = TsNetConfig {
                backend,
                learning_rate: Some(50.0),
                iterations_stage1: 40,
                iterations_stage2: 60,
                iterations_stage3: 40,
                lambda_c_stage2: 1.2,
                lambda_c_stage3: 0.01,
                lambda_r_stage3: 0.6,
                ..TsNetConfig::pure_2hop()
            };
            let mut l_classic = seeded_layout::<f64>(grid_5x5.num_nodes(), 1.0, 19);
            let mut solver_c =
                TsNetSolver::<f64>::from_graph_pure_2hop(&grid_5x5, cfg_classic).unwrap();
            assert_eq!(
                solver_c.run(&mut l_classic).unwrap(),
                SolverStatus::Completed
            );
            assert!(l_classic.is_all_finite());
            let (edge_c, all_c) = mean_edge_and_all_distances(&grid_5x5, &l_classic);
            assert!(
                edge_c < 0.6 * all_c,
                "{backend:?}: classic 5x5 grid locality failed (mean_edge={edge_c}, mean_all={all_c})"
            );

            // 3. Unmodified TsNetConfig::pure_2hop() preset (250/250/500, lr=200) executes cleanly
            // without NaN and remains strictly centered at the origin.
            let cfg_pure = TsNetConfig {
                backend,
                ..TsNetConfig::pure_2hop()
            };
            let mut l_grid = seeded_layout::<f64>(grid_pure.num_nodes(), 1.0, 19);
            let mut solver_g =
                TsNetSolver::<f64>::from_graph_pure_2hop(&grid_pure, cfg_pure).unwrap();
            assert_eq!(
                solver_g.run_steps(&mut l_grid, 30).unwrap(),
                SolverStatus::Running
            );
            assert!(l_grid.is_all_finite());

            let mean_x = l_grid.x().iter().sum::<f64>() / (grid_pure.num_nodes() as f64);
            let mean_y = l_grid.y().iter().sum::<f64>() / (grid_pure.num_nodes() as f64);
            assert!(mean_x.abs() < 1e-10 && mean_y.abs() < 1e-10);
        }
    }

    #[test]
    fn test_stage3_lambda_r_positive_expands_outward_regression() {
        // 1. Pure entropy repulsion stage (lambda_kl=0, lambda_c=0, lambda_r=1) must strictly
        // expand a compact point cloud across all 4 backends (verifying no sign double-inversion).
        let grid = make_grid_graph(4, 4);
        let compact_init = seeded_layout::<f64>(grid.num_nodes(), 0.25, 123);
        let init_rms = rms_radius(&compact_init);

        for backend in [
            TsNetBackend::Exact,
            TsNetBackend::BarnesHut,
            TsNetBackend::FftInterpolation,
            TsNetBackend::LinearFft,
        ] {
            let cfg = TsNetConfig {
                backend,
                learning_rate: Some(50.0),
                ..TsNetConfig::pure_2hop()
            };
            let probs = SparseProbabilities::empty(grid.num_nodes());
            let stages = vec![TsNetStage {
                iterations: 20,
                exaggeration: 1.0,
                lambda_kl: 0.0,
                lambda_c: 0.0,
                lambda_r: 1.0,
            }];
            let mut solver =
                TsNetSolver::<f64>::from_probabilities_with_stages(probs, cfg, stages).unwrap();
            let mut layout = compact_init.clone();
            solver.run(&mut layout).unwrap();
            let final_rms = rms_radius(&layout);
            assert!(
                final_rms > init_rms * 1.5,
                "{backend:?}: pure entropy failed to expand (init={init_rms}, final={final_rms})"
            );
        }

        // 2. From identical post-Stage-2 coordinates and matched Stage 3 learning rate (<= 2*N),
        // Stage 3 with lambda_r=0.6 expands outward relative to lambda_r=0.0 across all 4 backends.
        for backend in [
            TsNetBackend::Exact,
            TsNetBackend::BarnesHut,
            TsNetBackend::FftInterpolation,
            TsNetBackend::LinearFft,
        ] {
            let cfg_pre = TsNetConfig {
                backend,
                learning_rate: Some(15.0),
                iterations_stage1: 40,
                iterations_stage2: 40,
                iterations_stage3: 0,
                ..TsNetConfig::pure_2hop()
            };
            let mut post_stage2 = seeded_layout::<f64>(grid.num_nodes(), 0.5, 99);
            TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_pre)
                .unwrap()
                .run(&mut post_stage2)
                .unwrap();

            let cfg_no_rep = TsNetConfig {
                backend,
                learning_rate: Some(15.0),
                iterations_stage1: 0,
                iterations_stage2: 0,
                iterations_stage3: 40,
                lambda_c_stage3: 0.01,
                lambda_r_stage3: 0.0,
                ..TsNetConfig::pure_2hop()
            };
            let cfg_with_rep = TsNetConfig {
                lambda_r_stage3: 0.6,
                ..cfg_no_rep
            };
            let mut l_no_rep = post_stage2.clone();
            let mut l_with_rep = post_stage2;

            TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_no_rep)
                .unwrap()
                .run(&mut l_no_rep)
                .unwrap();
            TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_with_rep)
                .unwrap()
                .run(&mut l_with_rep)
                .unwrap();

            let rms_no_rep = rms_radius(&l_no_rep);
            let rms_with_rep = rms_radius(&l_with_rep);
            assert!(
                rms_with_rep > rms_no_rep,
                "{backend:?}: Stage 3 lambda_r=0.6 did not expand relative to lambda_r=0.0 ({rms_with_rep} <= {rms_no_rep})"
            );
        }
    }

    #[test]
    fn test_fit_field_fft_call_counts_and_linear_fft_forces_bilinear() {
        let grid = make_grid_graph(4, 5);
        let init = seeded_layout::<f64>(grid.num_nodes(), 1.0, 55);

        for (backend, expected_order) in [
            (TsNetBackend::FftInterpolation, InterpolationOrder::Cubic),
            (TsNetBackend::LinearFft, InterpolationOrder::Bilinear),
        ] {
            let cfg = TsNetConfig {
                backend,
                iterations_stage1: 2,
                iterations_stage2: 1,
                iterations_stage3: 2,
                lambda_r_stage3: 0.6,
                ibfft: IbFftConfig {
                    order: InterpolationOrder::Cubic,
                    ..IbFftConfig::default()
                },
                ..TsNetConfig::pure_2hop()
            };
            let mut solver = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
            assert_eq!(
                solver.fit_field.as_ref().unwrap().config().ibfft.order,
                expected_order
            );
            assert!(solver.last_fft_stats().is_none());
            assert_eq!(solver.total_fft_2d_calls(), 0);

            let mut layout = init.clone();
            // Stage 1 (lambda_r == 0): <= 5 FFTs/step (2 particle fwd + 1 kernel fwd + 2 inv).
            solver.step(&mut layout).unwrap();
            let s1 = solver.last_fft_stats().unwrap();
            assert_eq!(s1.particle_forward_ffts, 2);
            assert!(s1.total_fft_2d_calls <= 5);
            assert_eq!(s1.total_fft_2d_calls, 5);
            assert_eq!(solver.total_fft_2d_calls(), 5);

            // Finish Stage 1 & Stage 2 (still lambda_r == 0).
            solver.run_steps(&mut layout, 2).unwrap();
            assert_eq!(solver.stage_index(), 2);
            assert_eq!(solver.total_fft_2d_calls(), 15);

            // Stage 3 (lambda_r > 0): <= 8 FFTs/step (2 shared particle fwd + 2 kernel fwd + 4 inv).
            solver.step(&mut layout).unwrap();
            let s3 = solver.last_fft_stats().unwrap();
            assert_eq!(s3.particle_forward_ffts, 2);
            assert!(s3.total_fft_2d_calls <= 8);
            assert_eq!(s3.total_fft_2d_calls, 8);
            assert_eq!(solver.total_fft_2d_calls(), 23);
            assert_eq!(
                solver.fit_field.as_ref().unwrap().config().ibfft.order,
                expected_order
            );
        }

        // Exact & BarnesHut never invoke FitField2d.
        for backend in [TsNetBackend::Exact, TsNetBackend::BarnesHut] {
            let cfg = TsNetConfig {
                backend,
                iterations_stage1: 2,
                iterations_stage2: 0,
                iterations_stage3: 0,
                ..TsNetConfig::pure_2hop()
            };
            let mut solver = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
            let mut layout = init.clone();
            solver.run(&mut layout).unwrap();
            assert!(solver.last_fft_stats().is_none());
            assert_eq!(solver.total_fft_2d_calls(), 0);
        }
    }

    #[test]
    fn test_stage_lr_stability_cap_and_velocity_rescaling() {
        // 1. For small graphs (N < 100), all stages are capped at 2.0 * N even when lambda_c > 0.
        let g = make_barbell_graph(3); // n = 6
        let cfg_small = TsNetConfig {
            backend: TsNetBackend::Exact,
            iterations_stage1: 2,
            iterations_stage2: 2,
            iterations_stage3: 2,
            lambda_c_stage1: 0.1,
            lambda_c_stage2: 0.01,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.6,
            learning_rate: Some(200.0),
            momentum: 0.8,
            ..TsNetConfig::pure_2hop()
        };
        let mut solver_small = TsNetSolver::<f64>::from_graph_pure_2hop(&g, cfg_small).unwrap();
        let mut layout_small = seeded_layout::<f64>(6, 0.5, 31);
        solver_small.step(&mut layout_small).unwrap();
        assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
        solver_small.step(&mut layout_small).unwrap();
        solver_small.step(&mut layout_small).unwrap();
        assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
        solver_small.step(&mut layout_small).unwrap();
        solver_small.step(&mut layout_small).unwrap();
        assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
        assert!(layout_small.is_all_finite());

        // 2. For N = 30 (N < 100), Stage 1/2 cap is 2*30 = 60.0 -> Stage 3 (lambda_r > 0) caps at 50.0
        // and rescales velocity by 50/60.
        let g30 = make_grid_graph(5, 6); // n = 30
        let cfg_30 = TsNetConfig {
            backend: TsNetBackend::Exact,
            iterations_stage1: 2,
            iterations_stage2: 2,
            iterations_stage3: 2,
            lambda_c_stage1: 0.1,
            lambda_c_stage2: 0.01,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.6,
            learning_rate: Some(200.0),
            momentum: 0.8,
            ..TsNetConfig::pure_2hop()
        };
        let mut solver30 = TsNetSolver::<f64>::from_graph_pure_2hop(&g30, cfg_30).unwrap();
        let mut l30 = seeded_layout::<f64>(30, 0.5, 31);
        solver30.step(&mut l30).unwrap();
        assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
        solver30.step(&mut l30).unwrap();
        solver30.step(&mut l30).unwrap();
        assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
        solver30.step(&mut l30).unwrap();
        let vel_x_end_s2 = solver30.vel_x[0];
        solver30.step(&mut l30).unwrap();
        assert!((solver30.current_stage_lr() - 50.0).abs() < 1e-10);
        assert!(vel_x_end_s2.is_finite());
        assert!(l30.is_all_finite());

        // 3. For N >= 100 (N = 120), Stage 1 stability cap max_stable_lr = 16.83 * 120 = 2019.6
        // applies without the N < 100 2*N cap.
        let g120 = make_grid_graph(10, 12); // n = 120
        let cfg_120 = TsNetConfig {
            backend: TsNetBackend::BarnesHut,
            iterations_stage1: 2,
            iterations_stage2: 2,
            iterations_stage3: 2,
            lambda_c_stage1: 0.1,
            lambda_c_stage2: 0.01,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.6,
            learning_rate: Some(2500.0),
            momentum: 0.8,
            ..TsNetConfig::pure_2hop()
        };
        let mut solver120 = TsNetSolver::<f64>::from_graph_pure_2hop(&g120, cfg_120).unwrap();
        let mut l120 = seeded_layout::<f64>(120, 0.5, 31);
        solver120.step(&mut l120).unwrap();
        assert!((solver120.current_stage_lr() - 2019.6).abs() < 1e-8);
        solver120.step(&mut l120).unwrap();
        solver120.step(&mut l120).unwrap();
        assert!((solver120.current_stage_lr() - 2500.0).abs() < 1e-8);
        solver120.step(&mut l120).unwrap();
        solver120.step(&mut l120).unwrap();
        assert!((solver120.current_stage_lr() - 50.0).abs() < 1e-10);
        assert!(l120.is_all_finite());
    }

    #[test]
    fn test_constructors_from_knn_probabilities_and_partial_bfs() {
        let grid = make_grid_graph(3, 3);
        let (knn, _k2_deg, _stats) = build_pure_2hop_knn(&grid).unwrap();
        let perps = compute_nodewise_2hop_perplexities(&grid, 0.9, 5.0, 1200.0).unwrap();
        let cfg = TsNetConfig {
            iterations_stage1: 5,
            iterations_stage2: 5,
            iterations_stage3: 5,
            ..TsNetConfig::pure_2hop()
        };

        let probs = compute_knn_probabilities(&knn, &perps, &cfg.perplexity_config()).unwrap();
        let mut s_probs = TsNetSolver::<f64>::from_probabilities(probs, cfg).unwrap();
        let mut s_knn = TsNetSolver::<f64>::from_knn(&knn, Some(&perps), 1.0, cfg).unwrap();
        let mut s_graph = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();

        let mut l1 = seeded_layout::<f64>(9, 0.5, 77);
        let mut l2 = l1.clone();
        let mut l3 = l1.clone();
        s_probs.run(&mut l1).unwrap();
        s_knn.run(&mut l2).unwrap();
        s_graph.run(&mut l3).unwrap();
        for i in 0..9 {
            assert_eq!(l1.x()[i].to_bits(), l2.x()[i].to_bits());
            assert_eq!(l1.y()[i].to_bits(), l2.y()[i].to_bits());
            assert_eq!(l1.x()[i].to_bits(), l3.x()[i].to_bits());
            assert_eq!(l1.y()[i].to_bits(), l3.y()[i].to_bits());
        }

        // Uniform perplexity via from_knn(..., None, ...) and from_graph_partial_bfs.
        let mut s_uni = TsNetSolver::<f64>::from_knn(&knn, None, 4.0, cfg).unwrap();
        let mut s_bfs = TsNetSolver::<f64>::from_graph_partial_bfs(&grid, 6, 4.0, cfg).unwrap();
        let mut l_uni = seeded_layout::<f64>(9, 0.5, 77);
        let mut l_bfs = l_uni.clone();
        s_uni.run(&mut l_uni).unwrap();
        s_bfs.run(&mut l_bfs).unwrap();
        assert!(l_uni.is_all_finite());
        assert!(l_bfs.is_all_finite());
    }

    #[test]
    fn test_edge_cases_empty_singleton_two_node_coincident_and_errors() {
        for backend in [
            TsNetBackend::Exact,
            TsNetBackend::BarnesHut,
            TsNetBackend::FftInterpolation,
            TsNetBackend::LinearFft,
        ] {
            let cfg = TsNetConfig {
                backend,
                iterations_stage1: 5,
                iterations_stage2: 5,
                iterations_stage3: 5,
                lambda_r_stage3: 0.6,
                ..TsNetConfig::pure_2hop()
            };

            // 0-node graph
            let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
            let mut s0 = TsNetSolver::<f64>::from_graph_pure_2hop(&g0, cfg).unwrap();
            let mut l0 = Layout2d::<f64>::zeros(0);
            assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

            // 1-node graph
            let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
            let mut s1 = TsNetSolver::<f64>::from_graph_pure_2hop(&g1, cfg).unwrap();
            let mut l1 = Layout2d::from_xy(vec![3.5], vec![-2.0]).unwrap();
            assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
            assert_eq!(l1.x(), &[3.5]);
            assert_eq!(l1.y(), &[-2.0]);

            // 2-node graph + coincident coordinates
            let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
            let mut s2 = TsNetSolver::<f64>::from_graph_pure_2hop(&g2, cfg).unwrap();
            let mut l2 = Layout2d::from_xy(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
            assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
            assert!(l2.is_all_finite());
        }

        // Zero total iterations finishes immediately.
        let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let cfg_zero = TsNetConfig {
            iterations_stage1: 0,
            iterations_stage2: 0,
            iterations_stage3: 0,
            ..TsNetConfig::pure_2hop()
        };
        let mut s_zero = TsNetSolver::<f64>::from_graph_pure_2hop(&g2, cfg_zero).unwrap();
        assert_eq!(s_zero.status(), SolverStatus::Completed);
        assert_eq!(s_zero.total_iterations(), 0);
        let mut l2 = Layout2d::from_xy(vec![1.0, -1.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(s_zero.step(&mut l2).unwrap(), SolverStatus::Completed);

        // Non-finite layout and mismatched dimensions rejected on step().
        let mut s_err =
            TsNetSolver::<f64>::from_graph_pure_2hop(&g2, TsNetConfig::pure_2hop()).unwrap();
        let mut l_nan = Layout2d::from_xy(vec![f64::NAN, 0.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            s_err.step(&mut l_nan),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let mut l_inf = Layout2d::from_xy(vec![0.0, f64::INFINITY], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            s_err.step(&mut l_inf),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let mut l_mismatch = Layout2d::<f64>::zeros(3);
        assert!(matches!(
            s_err.step(&mut l_mismatch),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Invalid uniform_perplexity on from_knn.
        let (knn, _, _) = build_pure_2hop_knn(&g2).unwrap();
        assert!(matches!(
            TsNetSolver::<f64>::from_knn(&knn, None, 0.0, TsNetConfig::pure_2hop()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            TsNetSolver::<f64>::from_knn(&knn, None, f64::NAN, TsNetConfig::pure_2hop()),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // Invalid SparseProbabilities rejected.
        let bad_probs = SparseProbabilities {
            n_nodes: 2,
            indptr: vec![0, 1, 1],
            indices: vec![2],
            values: vec![0.5],
        };
        assert!(matches!(
            TsNetSolver::<f64>::from_probabilities(bad_probs, TsNetConfig::pure_2hop()),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        let bad_val_probs = SparseProbabilities {
            n_nodes: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            values: vec![0.5, -0.1],
        };
        assert!(matches!(
            TsNetSolver::<f64>::from_probabilities(bad_val_probs, TsNetConfig::pure_2hop()),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_auto_lr_scaling_stage3_small_n_cap_and_row_normalized_pure_2hop() {
        // 1. Small-N Stage 3 with lambda_c == 0.0 and lambda_r == 0.0 is capped at 2 * N.
        let g6 = make_barbell_graph(3); // N = 6
        let cfg_s3 = TsNetConfig {
            backend: TsNetBackend::Exact,
            iterations_stage1: 0,
            iterations_stage2: 0,
            iterations_stage3: 2,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.0,
            learning_rate: None,
            ..TsNetConfig::pure_2hop()
        };
        let mut solver6 = TsNetSolver::<f64>::from_graph_pure_2hop(&g6, cfg_s3).unwrap();
        let mut l6 = seeded_layout::<f64>(6, 0.5, 13);
        solver6.step(&mut l6).unwrap();
        assert!((solver6.current_stage_lr() - 12.0).abs() < 1e-12);
        assert!(l6.is_all_finite());

        // 2. Large-N Stage 3 (N = 3000) resolves to N/3 = 1000.0 and is never clipped by 2 * N.
        let stage3 = TsNetStage {
            iterations: 500,
            exaggeration: 1.0,
            lambda_kl: 1.0,
            lambda_c: 0.0,
            lambda_r: 0.0,
        };
        let cfg_auto = TsNetConfig::pure_2hop();
        let base_3000 = cfg_auto.resolve_learning_rate(3000);
        assert!((base_3000 - 1000.0).abs() < 1e-12);
        let stage3_lr_3000 = compute_stage_lr(&stage3, 3000, base_3000, cfg_auto.momentum);
        assert!((stage3_lr_3000 - 1000.0).abs() < 1e-12);

        // 2b. Boundary checks at n = 99 (capped at 198.0 across stages) and n = 100 (200.0).
        let stage1_c = TsNetStage {
            iterations: 10,
            exaggeration: 4.0,
            lambda_kl: 1.0,
            lambda_c: 0.1,
            lambda_r: 0.0,
        };
        let stage2_c = TsNetStage {
            iterations: 10,
            exaggeration: 1.0,
            lambda_kl: 1.0,
            lambda_c: 0.01,
            lambda_r: 0.0,
        };
        let stage3_no_r = TsNetStage {
            iterations: 10,
            exaggeration: 1.0,
            lambda_kl: 1.0,
            lambda_c: 0.0,
            lambda_r: 0.0,
        };
        let lr99 = cfg_auto.resolve_learning_rate(99);
        assert!((lr99 - 200.0).abs() < 1e-12);
        assert!((compute_stage_lr(&stage1_c, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12);
        assert!((compute_stage_lr(&stage2_c, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12);
        assert!(
            (compute_stage_lr(&stage3_no_r, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12
        );

        let lr100 = cfg_auto.resolve_learning_rate(100);
        assert!((lr100 - 200.0).abs() < 1e-12);
        assert!((compute_stage_lr(&stage1_c, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12);
        assert!((compute_stage_lr(&stage2_c, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12);
        assert!(
            (compute_stage_lr(&stage3_no_r, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12
        );
        // With explicit larger learning_rate=500.0 at n=100, Stage 1/2 use 500.0 (not capped at 200.0).
        assert!((compute_stage_lr(&stage1_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);
        assert!((compute_stage_lr(&stage2_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);

        // 3. RowNormalized vs StandardTsne in from_graph_pure_2hop on asymmetric star+tail graph.
        let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
        edges.extend([(6, 7), (7, 8), (8, 9)]);
        let star_tail = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
        let s_std = TsNetSolver::<f64>::from_graph_pure_2hop(&star_tail, cfg_auto).unwrap();
        let s_rn = TsNetSolver::<f64>::from_graph_pure_2hop(
            &star_tail,
            TsNetConfig {
                symmetrization: SymmetrizationMode::RowNormalized,
                ..cfg_auto
            },
        )
        .unwrap();
        let max_p_diff = s_std
            .p_vals
            .iter()
            .zip(s_rn.p_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_p_diff > 1e-4,
            "expected RowNormalized pure_2hop p_vals to differ from StandardTsne, diff={max_p_diff}"
        );
    }

    #[test]
    fn test_from_graph_row_normalized_uncapped_capped_and_bit_exact() {
        let grid = make_grid_graph(4, 4);
        let n = grid.num_nodes();
        let cfg = TsNetConfig {
            iterations_stage1: 8,
            iterations_stage2: 8,
            iterations_stage3: 8,
            seed: 91,
            ..TsNetConfig::pure_2hop()
        };

        for max_2hop in [None, Some(4usize)] {
            let probs = egraph_neighborhood::graph_12hop_row_normalized_probabilities(
                &grid, 0.5, max_2hop, cfg.seed,
            )
            .unwrap();
            let mut s_ref = TsNetSolver::<f64>::from_probabilities(probs, cfg).unwrap();
            let mut s_run =
                TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
            let mut s_step =
                TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();

            let mut l_ref = seeded_layout::<f64>(n, 0.5, 29);
            let mut l_run = l_ref.clone();
            let mut l_step = l_ref.clone();

            s_ref.run(&mut l_ref).unwrap();
            s_run.run(&mut l_run).unwrap();
            s_step.run_steps(&mut l_step, 10).unwrap();
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            for i in 0..n {
                assert_eq!(l_ref.x()[i].to_bits(), l_run.x()[i].to_bits());
                assert_eq!(l_ref.y()[i].to_bits(), l_run.y()[i].to_bits());
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact step == run
            let mut s32_run =
                TsNetSolver::<f32>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
            let mut s32_step =
                TsNetSolver::<f32>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
            let mut l32_run = seeded_layout::<f32>(n, 0.5, 29);
            let mut l32_step = l32_run.clone();
            s32_run.run(&mut l32_run).unwrap();
            for _ in 0..24 {
                s32_step.step(&mut l32_step).unwrap();
            }
            for i in 0..n {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }

        // Validation & edge cases
        assert!(matches!(
            TsNetSolver::<f64>::from_graph_row_normalized(&grid, -0.5, None, cfg),
            Err(CoreError::InvalidInput(_))
        ));
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s_empty =
            TsNetSolver::<f64>::from_graph_row_normalized(&empty, 0.5, None, cfg).unwrap();
        let mut l_empty = Layout2d::<f64>::zeros(0);
        assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

        // Degree-decayed row-normalized solver matches standalone probabilities
        let probs_beta = graph_12hop_row_normalized_probabilities_with_degree_decay(
            &grid,
            0.5,
            Some(4),
            0.35,
            cfg.seed,
        )
        .unwrap();
        let mut s_prob_beta = TsNetSolver::<f64>::from_probabilities(probs_beta, cfg).unwrap();
        let mut s_decay = TsNetSolver::<f64>::from_graph_row_normalized_with_degree_decay(
            &grid,
            0.5,
            Some(4),
            0.35,
            cfg,
        )
        .unwrap();
        let mut l_p = seeded_layout::<f64>(n, 0.5, 29);
        let mut l_d = l_p.clone();
        s_prob_beta.run(&mut l_p).unwrap();
        s_decay.run(&mut l_d).unwrap();
        for i in 0..n {
            assert_eq!(l_p.x()[i].to_bits(), l_d.x()[i].to_bits());
            assert_eq!(l_p.y()[i].to_bits(), l_d.y()[i].to_bits());
        }
        assert!(matches!(
            TsNetSolver::<f64>::from_graph_row_normalized_with_degree_decay(
                &grid, 0.5, None, -0.1, cfg
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_jacobi_precondition_weights_regular_isolated_and_hub() {
        // 1. Regular cycle graph (N=16, deg=2): P_{i.} == 1/N, so inv_jacobi[i] == 1.0 everywhere.
        let n_reg = 16;
        let reg_edges: Vec<(u32, u32)> = (0..n_reg as u32)
            .map(|i| (i, (i + 1) % (n_reg as u32)))
            .collect();
        let cycle = CsrGraph::<f64>::from_edges(n_reg, &reg_edges, None, false, false).unwrap();
        let cfg_on = TsNetConfig::pure_2hop();
        let s_reg_p2h = TsNetSolver::<f64>::from_graph_pure_2hop(&cycle, cfg_on).unwrap();
        let s_reg_rn =
            TsNetSolver::<f64>::from_graph_row_normalized(&cycle, 0.5, None, cfg_on).unwrap();
        assert_eq!(s_reg_p2h.inv_jacobi().len(), n_reg);
        assert_eq!(s_reg_rn.inv_jacobi().len(), n_reg);
        for &w in s_reg_p2h.inv_jacobi().iter().chain(s_reg_rn.inv_jacobi()) {
            assert!(
                (w - 1.0).abs() < 1e-12,
                "expected 1.0 on regular graph, got {w}"
            );
        }

        // 2. Disconnected graph with isolated vertex (triangle 0-1-2 + isolated vertex 3).
        let disc =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let s_disc =
            TsNetSolver::<f64>::from_graph_row_normalized(&disc, 0.0, None, cfg_on).unwrap();
        assert_eq!(s_disc.inv_jacobi().len(), 4);
        assert_eq!(s_disc.inv_jacobi()[3], 1.0);
        for &w in s_disc.inv_jacobi() {
            assert!(w.is_finite() && w > 0.0 && w <= 1.0);
        }

        // 3. Star graph (N=20, hub 0, leaves 1..19): P_{0.} = 0.5, N * P_{0.} = 10 -> inv_jacobi[0] = 0.1.
        let n_star = 20usize;
        let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
        let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
        let s_star =
            TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_on).unwrap();
        assert!((s_star.inv_jacobi()[0] - 0.1).abs() < 1e-12);
        for leaf in 1..n_star {
            assert!((s_star.inv_jacobi()[leaf] - 1.0).abs() < 1e-12);
        }

        // 4. Self-loops in raw SparseProbabilities are skipped when computing inv_jacobi.
        let self_loop_probs = SparseProbabilities {
            n_nodes: 4,
            indptr: vec![0, 2, 3, 4, 4],
            indices: vec![0, 1, 0, 2],
            values: vec![100.0, 0.5, 0.5, 0.1],
        };
        let s_sl = TsNetSolver::<f64>::from_probabilities(self_loop_probs, cfg_on).unwrap();
        assert!((s_sl.inv_jacobi()[0] - 0.5).abs() < 1e-12);
        assert!((s_sl.inv_jacobi()[1] - 0.5).abs() < 1e-12);
        assert!((s_sl.inv_jacobi()[2] - 1.0).abs() < 1e-12);
        assert!((s_sl.inv_jacobi()[3] - 1.0).abs() < 1e-12);

        // 5. jacobi_precondition = false leaves inv_jacobi empty and preserves f64/f32 step == run_steps.
        let cfg_off = TsNetConfig {
            backend: TsNetBackend::Exact,
            jacobi_precondition: false,
            iterations_stage1: 6,
            iterations_stage2: 4,
            iterations_stage3: 4,
            ..TsNetConfig::pure_2hop()
        };
        let mut s64_run =
            TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
        let mut s64_step =
            TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
        assert!(s64_run.inv_jacobi().is_empty());
        let mut l64_run = seeded_layout::<f64>(n_star, 0.5, 17);
        let mut l64_step = l64_run.clone();
        s64_run.run(&mut l64_run).unwrap();
        s64_step.run_steps(&mut l64_step, 5).unwrap();
        while !s64_step.status().is_finished() {
            s64_step.step(&mut l64_step).unwrap();
        }
        for i in 0..n_star {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        let mut s32_run =
            TsNetSolver::<f32>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
        let mut s32_step =
            TsNetSolver::<f32>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
        assert!(s32_run.inv_jacobi().is_empty());
        let mut l32_run = seeded_layout::<f32>(n_star, 0.5, 17);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        for _ in 0..14 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n_star {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_jacobi_precondition_prevents_hub_overshoot_and_512_grid_blowup() {
        // Sink-barrel graph (N=1200): hub 0 connected to 600 leaves (1..=600) plus a tail/ring (601..=1199).
        let n = 1200usize;
        let mut edges: Vec<(u32, u32)> = (1..=600u32).map(|v| (0, v)).collect();
        for u in 600..1199u32 {
            edges.push((u, u + 1));
        }
        edges.push((1199, 601));
        let barrel = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

        let make_init = || {
            let mut l = seeded_layout::<f64>(n, 0.5, 42);
            let (xs, ys) = l.xy_mut();
            xs[0] = 1.0;
            ys[0] = 0.0;
            let mx = xs.iter().sum::<f64>() / (n as f64);
            let my = ys.iter().sum::<f64>() / (n as f64);
            for i in 0..n {
                xs[i] -= mx;
                ys[i] -= my;
            }
            l
        };
        let init_f64 = make_init();
        let init_f32 = Layout2d::<f32>::from_xy(
            init_f64.x().iter().map(|&v| v as f32).collect(),
            init_f64.y().iter().map(|&v| v as f32).collect(),
        )
        .unwrap();

        // Baseline without Jacobi: Stage 1 immediately ejects hub 0 and blows up ibfft grid to 512.
        let cfg_off = TsNetConfig {
            backend: TsNetBackend::FftInterpolation,
            jacobi_precondition: false,
            iterations_stage1: 10,
            iterations_stage2: 0,
            iterations_stage3: 0,
            ..TsNetConfig::pure_2hop()
        };
        let mut s_off =
            TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_off).unwrap();
        let mut l_off = init_f64.clone();
        s_off.run_steps(&mut l_off, 2).unwrap();
        let r_hub_off = l_off.x()[0].hypot(l_off.y()[0]);
        let spec_off = s_off.fit_field.as_ref().unwrap().grid_spec().unwrap();
        assert!(
            r_hub_off > 100.0,
            "expected baseline hub ejection without Jacobi, got r_hub_off={r_hub_off}"
        );
        assert!(
            spec_off.nx == 512 || spec_off.ny == 512,
            "expected baseline 512 grid blowup without Jacobi, got {spec_off:?}"
        );

        // With Jacobi preconditioning (f64 and f32): bounded hub displacement, nx/ny < 512, step == run_steps.
        let cfg_on = TsNetConfig {
            jacobi_precondition: true,
            ..cfg_off
        };

        let mut s64_run =
            TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
        let mut s64_step =
            TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
        let mut l64_run = init_f64.clone();
        let mut l64_step = init_f64;
        assert_eq!(
            s64_run.run_steps(&mut l64_run, 10).unwrap(),
            SolverStatus::Completed
        );
        for _ in 0..10 {
            s64_step.step(&mut l64_step).unwrap();
        }
        assert!(l64_run.is_all_finite());
        let r_hub_64 = l64_run.x()[0].hypot(l64_run.y()[0]);
        let spec_64 = s64_run.fit_field.as_ref().unwrap().grid_spec().unwrap();
        assert!(
            r_hub_64 < 20.0,
            "expected bounded hub displacement (f64), got {r_hub_64}"
        );
        assert!(
            spec_64.nx < 512 && spec_64.ny < 512,
            "expected grid dims < 512 with Jacobi (f64), got {spec_64:?}"
        );
        for i in 0..n {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        let mut s32_run =
            TsNetSolver::<f32>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
        let mut s32_step =
            TsNetSolver::<f32>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
        let mut l32_run = init_f32.clone();
        let mut l32_step = init_f32;
        assert_eq!(
            s32_run.run_steps(&mut l32_run, 10).unwrap(),
            SolverStatus::Completed
        );
        for _ in 0..10 {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert!(l32_run.is_all_finite());
        let r_hub_32 = (l32_run.x()[0] as f64).hypot(l32_run.y()[0] as f64);
        let spec_32 = s32_run.fit_field.as_ref().unwrap().grid_spec().unwrap();
        assert!(
            r_hub_32 < 20.0,
            "expected bounded hub displacement (f32), got {r_hub_32}"
        );
        assert!(
            spec_32.nx < 512 && spec_32.ny < 512,
            "expected grid dims < 512 with Jacobi (f32), got {spec_32:?}"
        );
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}
