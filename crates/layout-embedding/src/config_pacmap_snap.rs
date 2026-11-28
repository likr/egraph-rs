use egraph_core::{CoreError, CoreResult};

/// Configuration for 3-tier `PaCMAP-Graph` layout optimization.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new [`PacmapGraphConfig`] initialized with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target number of Near (`NB`) neighbors per vertex `n_neighbors`.
    #[inline]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Sets the ratio of Mid-Near (`MN`) pairs `mn_ratio`.
    #[inline]
    pub fn with_mn_ratio(mut self, mn_ratio: f64) -> Self {
        self.mn_ratio = mn_ratio;
        self
    }

    /// Sets the ratio of Further (`FP`) repulsive pairs `fp_ratio`.
    #[inline]
    pub fn with_fp_ratio(mut self, fp_ratio: f64) -> Self {
        self.fp_ratio = fp_ratio;
        self
    }

    /// Sets the 3-phase iteration counts `[phase1, phase2, phase3]`.
    #[inline]
    pub fn with_phase_iterations(mut self, phase_iterations: [usize; 3]) -> Self {
        self.phase_iterations = phase_iterations;
        self
    }

    /// Sets the Adam learning rate `learning_rate`.
    #[inline]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the deterministic RNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets whether all 1-hop edges are retained (`keep_all_1hop`).
    #[inline]
    pub fn with_keep_all_1hop(mut self, keep_all_1hop: bool) -> Self {
        self.keep_all_1hop = keep_all_1hop;
        self
    }

    /// Sets the optional per-node cap on selected 1-hop neighbors `max_1hop_per_node`.
    #[inline]
    pub fn with_max_1hop_per_node(mut self, max_1hop_per_node: Option<usize>) -> Self {
        self.max_1hop_per_node = max_1hop_per_node;
        self
    }

    /// Sets the degree-attenuation exponent `degree_weight_beta`.
    #[inline]
    pub fn with_degree_weight_beta(mut self, degree_weight_beta: f64) -> Self {
        self.degree_weight_beta = degree_weight_beta;
        self
    }

    /// Sets the number of internal BFS landmarks `num_landmarks`.
    #[inline]
    pub fn with_num_landmarks(mut self, num_landmarks: usize) -> Self {
        self.num_landmarks = num_landmarks;
        self
    }

    /// Returns the total scheduled iterations across all three phases.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.phase_iterations[0] + self.phase_iterations[1] + self.phase_iterations[2]
    }

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
}

/// Configuration for compact-memory edge-incident negative sampling `SNAP-tFDP`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new [`SnapTFdpConfig`] initialized with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total optimization iterations `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the uniform random negative sample budget per edge `negatives_per_edge`.
    #[inline]
    pub fn with_negatives_per_edge(mut self, negatives_per_edge: usize) -> Self {
        self.negatives_per_edge = negatives_per_edge;
        self
    }

    /// Sets the initial learning rate `learning_rate`.
    #[inline]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the bounded $t$-FDP edge attraction scale `alpha`.
    #[inline]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the Student-$t$ repulsion multiplier `repulsion_scale`.
    #[inline]
    pub fn with_repulsion_scale(mut self, repulsion_scale: f64) -> Self {
        self.repulsion_scale = repulsion_scale;
        self
    }

    /// Sets the initial repulsion annealing weight `rho_start`.
    #[inline]
    pub fn with_rho_start(mut self, rho_start: f64) -> Self {
        self.rho_start = rho_start;
        self
    }

    /// Sets the final repulsion annealing weight `rho_end`.
    #[inline]
    pub fn with_rho_end(mut self, rho_end: f64) -> Self {
        self.rho_end = rho_end;
        self
    }

    /// Sets the annealing exponent `rho_power`.
    #[inline]
    pub fn with_rho_power(mut self, rho_power: f64) -> Self {
        self.rho_power = rho_power;
        self
    }

    /// Sets the degree-attenuation exponent `degree_beta`.
    #[inline]
    pub fn with_degree_beta(mut self, degree_beta: f64) -> Self {
        self.degree_beta = degree_beta;
        self
    }

    /// Sets whether bounded edge attraction (`8 / (1 + d^2)`) is used (`bounded_attraction`).
    #[inline]
    pub fn with_bounded_attraction(mut self, bounded_attraction: bool) -> Self {
        self.bounded_attraction = bounded_attraction;
        self
    }

    /// Sets the deterministic RNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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
