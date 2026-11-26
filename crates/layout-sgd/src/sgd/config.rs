use egraph_core::{CoreError, CoreResult};

/// Learning rate decay schedule family for [`super::SgdSolver`].
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

/// Symmetrization rule used for the 2-hop barrier radius $R_{\mathrm{pair}}(u, v)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BarrierSymmetry {
    /// Use $\max(R_2(u), R_2(v))$.
    #[default]
    SymMax,
    /// Use $\frac{1}{2}(R_2(u) + R_2(v))$.
    SymAvg,
}

/// Configuration for Track B 3-tier Sparse Barrier SGD construction.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Whether R3 Horvitz-Thompson scaling $\sqrt{N / k_{\mathrm{neg}}}$ is applied (`default: true`).
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
    /// $(d_u d_v)^{-\beta}$ (`default: 0.0`, disabled; recommended `0.25..=0.5` for hubs).
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

/// Summary diagnostics emitted by [`super::SgdSolver::sparse_barrier_from_landmarks`] and
/// [`super::SgdSolver::sparse_barrier_dog`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Configuration for compact-memory edge-incident negative sampling `SNAP-tFDP`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapTFdpConfig {
    /// Total optimization iterations (`default: 200`).
    pub iterations: usize,
    /// Uniform random negative samples drawn per undirected edge per iteration (`default: 3`).
    pub negatives_per_edge: usize,
    /// Initial learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Bounded $t$-FDP edge attraction scale (`default: 1.0`).
    pub alpha: f64,
    /// Student-$t$ repulsion multiplier (`default: 1.0`).
    pub repulsion_scale: f64,
    /// Initial repulsion annealing weight $\rho(0)$ (`default: 1.0`).
    pub rho_start: f64,
    /// Final repulsion annealing weight $\rho(1)$ (`default: 0.1`).
    pub rho_end: f64,
    /// Exponent $p > 0$ for $\rho(t) = \rho_{\mathrm{start}} + (\rho_{\mathrm{end}} - \rho_{\mathrm{start}}) t^p$ (`default: 1.0`).
    pub rho_power: f64,
    /// Degree-attenuation exponent $\beta \ge 0$ scaling edge attraction by $(d_u d_v)^{-\beta}$ (`default: 0.0`).
    pub degree_beta: f64,
    /// Whether edge attraction uses bounded $8/(1+d^2)$ (`true`) or legacy unbounded $1 + 8/(1+d^2)$ (`false`) (`default: true`).
    pub bounded_attraction: bool,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for SnapTFdpConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            negatives_per_edge: 3,
            learning_rate: 1.0,
            alpha: 1.0,
            repulsion_scale: 1.0,
            rho_start: 1.0,
            rho_end: 0.1,
            rho_power: 1.0,
            degree_beta: 0.0,
            bounded_attraction: true,
            seed: 42,
        }
    }
}

impl SnapTFdpConfig {
    /// Validates all `SNAP-tFDP` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("SnapTFdpConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SnapTFdpConfig::learning_rate must be > 0, got {}",
                self.learning_rate
            )));
        }
        for (ctx, val) in [
            ("SnapTFdpConfig::alpha", self.alpha),
            ("SnapTFdpConfig::repulsion_scale", self.repulsion_scale),
            ("SnapTFdpConfig::rho_start", self.rho_start),
            ("SnapTFdpConfig::rho_end", self.rho_end),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{ctx} must be >= 0, got {val}"
                )));
            }
        }
        if !self.rho_power.is_finite() {
            return Err(CoreError::non_finite("SnapTFdpConfig::rho_power"));
        }
        if self.rho_power <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SnapTFdpConfig::rho_power must be > 0, got {}",
                self.rho_power
            )));
        }
        if !self.degree_beta.is_finite() {
            return Err(CoreError::non_finite("SnapTFdpConfig::degree_beta"));
        }
        if self.degree_beta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SnapTFdpConfig::degree_beta must be >= 0, got {}",
                self.degree_beta
            )));
        }
        Ok(())
    }
}

/// Configuration for 3-tier `PaCMAP-Graph` layout optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PacmapGraphConfig {
    /// Target number of Near (`NB`) 1+2-hop neighbors per vertex (`default: 10`).
    pub n_neighbors: usize,
    /// Ratio of Mid-Near (`MN`) pairs relative to `n_neighbors` (`default: 0.5`).
    pub mn_ratio: f64,
    /// Ratio of Further (`FP`) repulsive pairs relative to `n_neighbors` (`default: 2.0`).
    pub fp_ratio: f64,
    /// Phase iteration counts `[phase1, phase2, phase3]` (`default: [100, 100, 250]`).
    ///
    /// - Phase 1: global unfolding, $w_{\mathrm{MN}}: 1000 \to 3$.
    /// - Phase 2: scaffold locking, $w = (3, 3, 1)$.
    /// - Phase 3: local refinement, $w = (1, 0, 1)$.
    pub phase_iterations: [usize; 3],
    /// Adam optimizer learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
    /// Retain all 1-hop edges even on vertices with degree > `n_neighbors` (`default: false`).
    pub keep_all_1hop: bool,
    /// Optional per-node cap on selected 1-hop neighbors (`None` = uncapped when `keep_all_1hop`,
    /// `Some(k)` = at most `k` 1-hop neighbors per vertex). Must be `>= 1` when `Some` (`default: None`).
    pub max_1hop_per_node: Option<usize>,
    /// Degree-attenuation exponent $\beta \ge 0$ scaling `NB` attraction by $(d_u d_v)^{-\beta}$
    /// (`default: 0.0`, disabled).
    pub degree_weight_beta: f64,
    /// Number of internal BFS landmarks when external landmarks are omitted (`default: 16`, must be `>= 1`).
    pub num_landmarks: usize,
}

impl Default for PacmapGraphConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            mn_ratio: 0.5,
            fp_ratio: 2.0,
            phase_iterations: [100, 100, 250],
            learning_rate: 1.0,
            seed: 42,
            keep_all_1hop: false,
            max_1hop_per_node: None,
            degree_weight_beta: 0.0,
            num_landmarks: 16,
        }
    }
}

impl PacmapGraphConfig {
    /// Validates all `PaCMAP-Graph` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.n_neighbors == 0 {
            return Err(CoreError::invalid_input(
                "PacmapGraphConfig::n_neighbors must be >= 1",
            ));
        }
        if self.num_landmarks == 0 {
            return Err(CoreError::invalid_input(
                "PacmapGraphConfig::num_landmarks must be >= 1",
            ));
        }
        if self.max_1hop_per_node == Some(0) {
            return Err(CoreError::invalid_input(
                "PacmapGraphConfig::max_1hop_per_node must be >= 1",
            ));
        }
        if !self.degree_weight_beta.is_finite() {
            return Err(CoreError::non_finite(
                "PacmapGraphConfig::degree_weight_beta",
            ));
        }
        if self.degree_weight_beta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PacmapGraphConfig::degree_weight_beta must be >= 0, got {}",
                self.degree_weight_beta
            )));
        }
        for (ctx, val) in [
            ("PacmapGraphConfig::mn_ratio", self.mn_ratio),
            ("PacmapGraphConfig::fp_ratio", self.fp_ratio),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{ctx} must be >= 0, got {val}"
                )));
            }
        }
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("PacmapGraphConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PacmapGraphConfig::learning_rate must be > 0, got {}",
                self.learning_rate
            )));
        }
        let _ = self.phase_iterations[0]
            .checked_add(self.phase_iterations[1])
            .and_then(|s| s.checked_add(self.phase_iterations[2]))
            .ok_or_else(|| {
                CoreError::invalid_input("PacmapGraphConfig total iterations overflow")
            })?;
        Ok(())
    }

    #[inline]
    pub(super) fn total_iterations(&self) -> usize {
        self.phase_iterations[0] + self.phase_iterations[1] + self.phase_iterations[2]
    }
}

/// Configuration for doubly-stochastic mini-batch diagonal-preconditioned stress SGD.
///
/// Note: This diagonal-preconditioned stress SGD preset (using [`SgdScheduleKind::Squared`])
/// is distinct from Ahmed et al. (2021) multi-criteria `(SGD)^2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd2Config {
    /// Total optimization iterations (`default: 60`).
    pub iterations: usize,
    /// Minimum learning rate ratio $\varepsilon > 0$ (`default: 0.01`).
    pub eps: f64,
    /// Multiplicative step scale (`default: 1.0`).
    pub step_scale: f64,
    /// Mini-batch pair count per preconditioned update (`default: 4096`).
    pub batch_size: usize,
    /// Diagonal curvature damping denominator floor $> 0$ (`default: 1e-4`).
    pub damping: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for Sgd2Config {
    fn default() -> Self {
        Self {
            iterations: 60,
            eps: 0.01,
            step_scale: 1.0,
            batch_size: 4096,
            damping: 1e-4,
            seed: 42,
        }
    }
}

impl Sgd2Config {
    /// Validates all diagonal-preconditioned stress SGD (`Sgd2Config`) hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("Sgd2Config::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "Sgd2Config::eps must be > 0, got {}",
                self.eps
            )));
        }
        if !self.step_scale.is_finite() {
            return Err(CoreError::non_finite("Sgd2Config::step_scale"));
        }
        if self.step_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "Sgd2Config::step_scale must be > 0, got {}",
                self.step_scale
            )));
        }
        if self.batch_size == 0 {
            return Err(CoreError::invalid_input(
                "Sgd2Config::batch_size must be >= 1",
            ));
        }
        if !self.damping.is_finite() {
            return Err(CoreError::non_finite("Sgd2Config::damping"));
        }
        if self.damping <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "Sgd2Config::damping must be > 0, got {}",
                self.damping
            )));
        }
        Ok(())
    }
}
