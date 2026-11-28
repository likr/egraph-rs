use egraph_core::{CoreError, CoreResult};

/// Learning rate decay schedule family for [`super::SgdSolver`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SgdScheduleKind {
    /// Exponential decay $\eta(t) = \eta_{\max} \exp(-b t)$.
    #[default]
    Exponential,
    /// Linear decay $\eta(t) = \eta_{\max} - b t$.
    Linear,
    /// Quadratic decay $\eta(t) = \eta_{\max} (1 - b t)^2$.
    Quadratic,
    /// Reciprocal decay $\eta(t) = \eta_{\max} / (1 + b t)$.
    Reciprocal,
    /// Constant learning rate $\eta(t) = \mathrm{step\_scale}$.
    Constant,
    /// Squared progress decay $\eta(t) = \max(\eta_{\max} (1 - \text{progress})^2, \eta_{\min})$.
    Squared,
}

/// Learning rate schedule parameters for [`super::SgdSolver`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SgdScheduleConfig {
    /// Total number of scheduled SGD iterations (`default: 60`).
    pub iterations: usize,
    /// Minimum learning rate ratio $\varepsilon > 0$ controlling $\eta_{\min} = \varepsilon / w_{\max}$ (`default: 0.01`).
    pub eps: f64,
    /// Decay curve family (`default: SgdScheduleKind::Exponential`).
    pub kind: SgdScheduleKind,
    /// Multiplicative step scale applied to $\eta(t)$ (`default: 1.0`).
    pub step_scale: f64,
    /// Optional upper bound on per-pair step factor $\mu = \min(\eta w, \mu_{\max})$ (`default: None`, using `1.0`).
    pub max_mu: Option<f64>,
}

impl Default for SgdScheduleConfig {
    fn default() -> Self {
        Self {
            iterations: 60,
            eps: 0.01,
            kind: SgdScheduleKind::Exponential,
            step_scale: 1.0,
            max_mu: None,
        }
    }
}

impl SgdScheduleConfig {
    /// Creates a new schedule configuration with `iterations` and default decay settings.
    #[inline]
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            ..Self::default()
        }
    }

    /// Sets the total iteration count `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the minimum learning rate ratio `eps`.
    #[inline]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the learning rate decay curve `kind`.
    #[inline]
    pub fn with_kind(mut self, kind: SgdScheduleKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the multiplicative `step_scale`.
    #[inline]
    pub fn with_step_scale(mut self, step_scale: f64) -> Self {
        self.step_scale = step_scale;
        self
    }

    /// Sets the optional upper bound `max_mu` on per-pair step factor $\mu$.
    #[inline]
    pub fn with_max_mu(mut self, max_mu: Option<f64>) -> Self {
        self.max_mu = max_mu;
        self
    }

    /// Validates all schedule hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("SgdScheduleConfig::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SgdScheduleConfig::eps must be > 0, got {}",
                self.eps
            )));
        }
        if !self.step_scale.is_finite() {
            return Err(CoreError::non_finite("SgdScheduleConfig::step_scale"));
        }
        if self.step_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SgdScheduleConfig::step_scale must be > 0, got {}",
                self.step_scale
            )));
        }
        if let Some(max_mu) = self.max_mu {
            if !max_mu.is_finite() {
                return Err(CoreError::non_finite("SgdScheduleConfig::max_mu"));
            }
            if max_mu <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "SgdScheduleConfig::max_mu must be > 0, got {max_mu}"
                )));
            }
        }
        Ok(())
    }
}

/// Symmetrization rule used for the 2-hop barrier radius $R_{\mathrm{pair}}(u, v)$.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BarrierSymmetry {
    /// Use $\max(R_2(u), R_2(v))$.
    #[default]
    SymMax,
    /// Use $\frac{1}{2}(R_2(u) + R_2(v))$.
    SymAvg,
}

/// Configuration for 3-tier Sparse Barrier SGD construction.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBarrierConfig {
    /// Hub-safe per-node cap on registered 2-hop pairs (`default: 32`).
    pub max_2hop_per_node: usize,
    /// Whether Tier 3 non-2-hop barrier pairs use one-sided hinge repulsion (`default: true`).
    pub hinge_repulsion: bool,
    /// Relative margin multiplier $\delta_{\mathrm{rel}} \ge 0$ for the 2-hop barrier (`default: 0.25`).
    pub delta_rel: f64,
    /// Symmetrization rule for $R_{\mathrm{pair}}(u, v)$ (`default: BarrierSymmetry::SymMax`).
    pub barrier_sym: BarrierSymmetry,
    /// Tier 1 (1-hop) spring weight multiplier (`default: 1.0`).
    pub omega_1: f64,
    /// Tier 2 (2-hop) spring weight multiplier (`default: 1.0`).
    pub omega_2: f64,
    /// Tier 3 (non-2-hop) weight multiplier (`default: 1.0`).
    pub gamma_non2: f64,
    /// Whether Horvitz-Thompson scaling $\sqrt{N / k_{\mathrm{neg}}}$ is applied (`default: true`).
    pub ht_repulsion: bool,
    /// Relative distance floor multiplier $\mathrm{min\_ratio} \cdot \mathrm{median}(d_1 > 0)$ (`default: 0.5`).
    pub min_ratio: f64,
    /// Absolute minimum target distance clamp $> 0$ (`default: 1e-6`).
    pub eps_dist: f64,
    /// Optional explicit per-node non-2-hop sample budget (`None` uses adaptive 2-hop perplexity in `[8, 32]`).
    pub neg_per_node: Option<usize>,
    /// Number of MaxMin shortest-path pivots used for global non-2-hop coverage (`default: 50`).
    pub num_pivots: usize,
    /// Prioritizes `degree >= 2` core vertices during MaxMin pivot selection (`default: true`).
    pub core_pivots: bool,
    /// Degree-normalization exponent $\beta \ge 0$ scaling Tier 1/2 spring weights by
    /// $(d_u d_v)^{-\beta}$ (`default: 0.0`, disabled).
    pub degree_weight_beta: f64,
    /// SGD iteration schedule (`default: SgdScheduleConfig::default()`).
    pub schedule: SgdScheduleConfig,
    /// Deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for SparseBarrierConfig {
    fn default() -> Self {
        Self {
            max_2hop_per_node: 32,
            hinge_repulsion: true,
            delta_rel: 0.25,
            barrier_sym: BarrierSymmetry::SymMax,
            omega_1: 1.0,
            omega_2: 1.0,
            gamma_non2: 1.0,
            ht_repulsion: true,
            min_ratio: 0.5,
            eps_dist: 1e-6,
            neg_per_node: None,
            num_pivots: 50,
            core_pivots: true,
            degree_weight_beta: 0.0,
            schedule: SgdScheduleConfig::default(),
            seed: 42,
        }
    }
}

impl SparseBarrierConfig {
    /// Creates a new [`SparseBarrierConfig`] with default parameters.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets `max_2hop_per_node`.
    #[inline]
    pub fn with_max_2hop_per_node(mut self, max_2hop_per_node: usize) -> Self {
        self.max_2hop_per_node = max_2hop_per_node;
        self
    }
    /// Sets `hinge_repulsion`.
    #[inline]
    pub fn with_hinge_repulsion(mut self, hinge_repulsion: bool) -> Self {
        self.hinge_repulsion = hinge_repulsion;
        self
    }
    /// Sets `delta_rel`.
    #[inline]
    pub fn with_delta_rel(mut self, delta_rel: f64) -> Self {
        self.delta_rel = delta_rel;
        self
    }
    /// Sets `barrier_sym`.
    #[inline]
    pub fn with_barrier_sym(mut self, barrier_sym: BarrierSymmetry) -> Self {
        self.barrier_sym = barrier_sym;
        self
    }
    /// Sets `omega_1`.
    #[inline]
    pub fn with_omega_1(mut self, omega_1: f64) -> Self {
        self.omega_1 = omega_1;
        self
    }
    /// Sets `omega_2`.
    #[inline]
    pub fn with_omega_2(mut self, omega_2: f64) -> Self {
        self.omega_2 = omega_2;
        self
    }
    /// Sets `gamma_non2`.
    #[inline]
    pub fn with_gamma_non2(mut self, gamma_non2: f64) -> Self {
        self.gamma_non2 = gamma_non2;
        self
    }
    /// Sets `ht_repulsion`.
    #[inline]
    pub fn with_ht_repulsion(mut self, ht_repulsion: bool) -> Self {
        self.ht_repulsion = ht_repulsion;
        self
    }
    /// Sets `min_ratio`.
    #[inline]
    pub fn with_min_ratio(mut self, min_ratio: f64) -> Self {
        self.min_ratio = min_ratio;
        self
    }
    /// Sets `eps_dist`.
    #[inline]
    pub fn with_eps_dist(mut self, eps_dist: f64) -> Self {
        self.eps_dist = eps_dist;
        self
    }
    /// Sets `neg_per_node`.
    #[inline]
    pub fn with_neg_per_node(mut self, neg_per_node: Option<usize>) -> Self {
        self.neg_per_node = neg_per_node;
        self
    }
    /// Sets `num_pivots`.
    #[inline]
    pub fn with_num_pivots(mut self, num_pivots: usize) -> Self {
        self.num_pivots = num_pivots;
        self
    }
    /// Sets `core_pivots`.
    #[inline]
    pub fn with_core_pivots(mut self, core_pivots: bool) -> Self {
        self.core_pivots = core_pivots;
        self
    }
    /// Sets `degree_weight_beta`.
    #[inline]
    pub fn with_degree_weight_beta(mut self, degree_weight_beta: f64) -> Self {
        self.degree_weight_beta = degree_weight_beta;
        self
    }
    /// Sets `schedule`.
    #[inline]
    pub fn with_schedule(mut self, schedule: SgdScheduleConfig) -> Self {
        self.schedule = schedule;
        self
    }
    /// Sets `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Summary diagnostics emitted by [`super::SgdSolver::sparse_barrier_from_landmarks`] and
/// [`super::SgdSolver::sparse_barrier_dog`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBarrierStats {
    pub n_pairs_1hop: usize,
    pub n_pairs_2hop: usize,
    pub n_pairs_non2: usize,
    pub n_pairs_hinge: usize,
    pub n_pairs_two_sided: usize,
    pub perplexity_val: f64,
    pub per_node_neg_used: usize,
    pub perplexity_source: &'static str,
    pub d_floor: f64,
    pub gamma_eff: f64,
}

/// Configuration for minimal-RAM on-the-fly streaming non-edge negative sampling SGD.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingNegativeConfig {
    /// Number of non-edge negatives sampled per vertex on each step (`default: 5`).
    pub negatives_per_node: usize,
    /// Target repulsive separation distance $> 0$ (`default: 2.0`).
    pub target_distance: f64,
    /// Negative pair weight $\ge 0$ (`default: 0.25`).
    pub weight: f64,
    /// Apply one-sided hinge repulsion when `norm < target_distance` (`default: true`).
    pub hinge_only: bool,
}

impl Default for StreamingNegativeConfig {
    fn default() -> Self {
        Self {
            negatives_per_node: 5,
            target_distance: 2.0,
            weight: 0.25,
            hinge_only: true,
        }
    }
}

impl StreamingNegativeConfig {
    /// Creates a new [`StreamingNegativeConfig`] with default parameters.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `negatives_per_node`.
    #[inline]
    pub fn with_negatives_per_node(mut self, negatives_per_node: usize) -> Self {
        self.negatives_per_node = negatives_per_node;
        self
    }

    /// Sets `target_distance`.
    #[inline]
    pub fn with_target_distance(mut self, target_distance: f64) -> Self {
        self.target_distance = target_distance;
        self
    }

    /// Sets `weight`.
    #[inline]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Sets `hinge_only`.
    #[inline]
    pub fn with_hinge_only(mut self, hinge_only: bool) -> Self {
        self.hinge_only = hinge_only;
        self
    }
}
