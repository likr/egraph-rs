use egraph_core::{CoreError, CoreResult};

/// Configuration for single-threaded deterministic `LargeVis` graph layout
/// (`Tang et al., 2016`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LargeVisConfig {
    /// Total optimization epochs (`default: 200`).
    pub iterations: usize,
    /// Number of negative samples drawn per positive edge draw (`default: 5`).
    pub negative_samples: usize,
    /// Negative repulsion weight $\gamma \ge 0$ (`default: 7.0`).
    pub gamma: f64,
    /// Initial learning rate (`default: 1.0`).
    pub initial_lr: f64,
    /// Minimum learning rate floor (`default: 1e-4`).
    pub min_lr: f64,
    /// Negative-sampling degree exponent $d_j^{\alpha}$ (`default: 0.75`).
    pub degree_exponent: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for LargeVisConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            negative_samples: 5,
            gamma: 7.0,
            initial_lr: 1.0,
            min_lr: 1e-4,
            degree_exponent: 0.75,
            seed: 42,
        }
    }
}

impl LargeVisConfig {
    /// Creates a new [`LargeVisConfig`] initialized with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total optimization epochs `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the number of negative samples per positive edge draw `negative_samples`.
    #[inline]
    pub fn with_negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }

    /// Sets the negative repulsion weight `gamma`.
    #[inline]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the initial learning rate `initial_lr`.
    #[inline]
    pub fn with_initial_lr(mut self, initial_lr: f64) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the minimum learning rate floor `min_lr`.
    #[inline]
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Sets the negative-sampling degree exponent `degree_exponent`.
    #[inline]
    pub fn with_degree_exponent(mut self, degree_exponent: f64) -> Self {
        self.degree_exponent = degree_exponent;
        self
    }

    /// Sets the deterministic RNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all `LargeVisConfig` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.gamma.is_finite() {
            return Err(CoreError::non_finite("LargeVisConfig::gamma"));
        }
        if self.gamma < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "LargeVisConfig::gamma must be >= 0, got {}",
                self.gamma
            )));
        }
        if !self.initial_lr.is_finite() {
            return Err(CoreError::non_finite("LargeVisConfig::initial_lr"));
        }
        if self.initial_lr <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "LargeVisConfig::initial_lr must be > 0, got {}",
                self.initial_lr
            )));
        }
        if !self.min_lr.is_finite() {
            return Err(CoreError::non_finite("LargeVisConfig::min_lr"));
        }
        if self.min_lr < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "LargeVisConfig::min_lr must be >= 0, got {}",
                self.min_lr
            )));
        }
        if self.min_lr > self.initial_lr {
            return Err(CoreError::invalid_input(format!(
                "LargeVisConfig::min_lr ({}) must be <= initial_lr ({})",
                self.min_lr, self.initial_lr
            )));
        }
        if !self.degree_exponent.is_finite() {
            return Err(CoreError::non_finite("LargeVisConfig::degree_exponent"));
        }
        if self.degree_exponent < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "LargeVisConfig::degree_exponent must be >= 0, got {}",
                self.degree_exponent
            )));
        }
        Ok(())
    }
}

/// Configuration for multilevel negative-sampling `DRGraph` layout (`Zhu et al., 2020`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DrGraphConfig {
    /// Shortest-path neighborhood order $k \ge 1$ for sparse similarity $NS_{ij}$ (`default: 1`).
    pub k_order: usize,
    /// Number of uniform negative samples drawn per positive edge draw (`default: 5`).
    pub negative_samples: usize,
    /// Parameterized Cauchy kernel exponent $b > 0$ in $(1 + d^{2b})^{-1}$ (`default: 2.0`).
    pub b: f64,
    /// Repulsion weight $\gamma$ on the coarsest level (`default: 0.01`).
    pub gamma_coarse: f64,
    /// Repulsion weight $\gamma$ on finer levels (`default: 0.1`).
    pub gamma_fine: f64,
    /// Multilevel coarsening stop ratio $\rho \in (0, 1]$ (`default: 0.8`).
    pub coarsen_stop_ratio: f64,
    /// Total optimization iterations partitioned across the multilevel hierarchy (`default: 400`).
    pub iterations: usize,
    /// Initial learning rate (`default: 1.0`).
    pub initial_lr: f64,
    /// Minimum learning rate floor (`default: 1e-4`).
    pub min_lr: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for DrGraphConfig {
    fn default() -> Self {
        Self {
            k_order: 1,
            negative_samples: 5,
            b: 2.0,
            gamma_coarse: 0.01,
            gamma_fine: 0.1,
            coarsen_stop_ratio: 0.8,
            iterations: 400,
            initial_lr: 1.0,
            min_lr: 1e-4,
            seed: 42,
        }
    }
}

impl DrGraphConfig {
    /// Creates a new [`DrGraphConfig`] initialized with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the shortest-path neighborhood order `k_order`.
    #[inline]
    pub fn with_k_order(mut self, k_order: usize) -> Self {
        self.k_order = k_order;
        self
    }

    /// Sets the number of uniform negative samples per positive edge draw `negative_samples`.
    #[inline]
    pub fn with_negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }

    /// Sets the Cauchy kernel exponent `b`.
    #[inline]
    pub fn with_b(mut self, b: f64) -> Self {
        self.b = b;
        self
    }

    /// Sets the coarsest-level repulsion weight `gamma_coarse`.
    #[inline]
    pub fn with_gamma_coarse(mut self, gamma_coarse: f64) -> Self {
        self.gamma_coarse = gamma_coarse;
        self
    }

    /// Sets the finer-level repulsion weight `gamma_fine`.
    #[inline]
    pub fn with_gamma_fine(mut self, gamma_fine: f64) -> Self {
        self.gamma_fine = gamma_fine;
        self
    }

    /// Sets the multilevel coarsening stop ratio `coarsen_stop_ratio`.
    #[inline]
    pub fn with_coarsen_stop_ratio(mut self, coarsen_stop_ratio: f64) -> Self {
        self.coarsen_stop_ratio = coarsen_stop_ratio;
        self
    }

    /// Sets the total optimization iterations `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the initial learning rate `initial_lr`.
    #[inline]
    pub fn with_initial_lr(mut self, initial_lr: f64) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the minimum learning rate floor `min_lr`.
    #[inline]
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Sets the deterministic RNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all `DrGraphConfig` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.k_order == 0 {
            return Err(CoreError::invalid_input(
                "DrGraphConfig::k_order must be >= 1",
            ));
        }
        if !self.b.is_finite() {
            return Err(CoreError::non_finite("DrGraphConfig::b"));
        }
        if self.b <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "DrGraphConfig::b must be > 0, got {}",
                self.b
            )));
        }
        for (ctx, val) in [
            ("DrGraphConfig::gamma_coarse", self.gamma_coarse),
            ("DrGraphConfig::gamma_fine", self.gamma_fine),
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
        if !self.coarsen_stop_ratio.is_finite() {
            return Err(CoreError::non_finite("DrGraphConfig::coarsen_stop_ratio"));
        }
        if self.coarsen_stop_ratio <= 0.0 || self.coarsen_stop_ratio > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "DrGraphConfig::coarsen_stop_ratio must be in (0, 1], got {}",
                self.coarsen_stop_ratio
            )));
        }
        if !self.initial_lr.is_finite() {
            return Err(CoreError::non_finite("DrGraphConfig::initial_lr"));
        }
        if self.initial_lr <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "DrGraphConfig::initial_lr must be > 0, got {}",
                self.initial_lr
            )));
        }
        if !self.min_lr.is_finite() {
            return Err(CoreError::non_finite("DrGraphConfig::min_lr"));
        }
        if self.min_lr < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "DrGraphConfig::min_lr must be >= 0, got {}",
                self.min_lr
            )));
        }
        if self.min_lr > self.initial_lr {
            return Err(CoreError::invalid_input(format!(
                "DrGraphConfig::min_lr ({}) must be <= initial_lr ({})",
                self.min_lr, self.initial_lr
            )));
        }
        Ok(())
    }
}

/// Contrastive loss objective family for [`super::GraphCneSolver`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphCneObjective {
    /// InfoNCE contrastive objective with Cauchy kernel (`Böhm et al., 2025`, `default`).
    #[default]
    InfoNce,
    /// Negative-sampled t-SNE objective on the attraction-repulsion spectrum (`Böhm et al., 2022`).
    NegTSne,
}

/// Configuration for `Graph-CNE` contrastive neighbor embedding (`Böhm et al., 2025`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GraphCneConfig {
    /// Contrastive loss objective (`default: GraphCneObjective::InfoNce`).
    pub objective: GraphCneObjective,
    /// Number of uniform negative samples drawn per positive edge draw (`default: 5`).
    pub negative_samples: usize,
    /// Total optimization epochs (`default: 400`).
    pub iterations: usize,
    /// Initial learning rate (`default: 1.0`).
    pub initial_lr: f64,
    /// Minimum learning rate floor (`default: 1e-4`).
    pub min_lr: f64,
    /// Repulsion scaling weight $\rho \ge 0$ (`default: 1.0`).
    pub repulsion_weight: f64,
    /// Apply row-normalized symmetrization $P = (D^{-1}A + A^T D^{-1})/(2N)$ (`default: true`).
    pub row_normalize: bool,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GraphCneConfig {
    fn default() -> Self {
        Self {
            objective: GraphCneObjective::InfoNce,
            negative_samples: 5,
            iterations: 400,
            initial_lr: 1.0,
            min_lr: 1e-4,
            repulsion_weight: 1.0,
            row_normalize: true,
            seed: 42,
        }
    }
}

impl GraphCneConfig {
    /// Creates a new [`GraphCneConfig`] initialized with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the contrastive loss `objective`.
    #[inline]
    pub fn with_objective(mut self, objective: GraphCneObjective) -> Self {
        self.objective = objective;
        self
    }

    /// Sets the number of uniform negative samples per positive edge draw `negative_samples`.
    #[inline]
    pub fn with_negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }

    /// Sets the total optimization epochs `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the initial learning rate `initial_lr`.
    #[inline]
    pub fn with_initial_lr(mut self, initial_lr: f64) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the minimum learning rate floor `min_lr`.
    #[inline]
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Sets the repulsion scaling weight `repulsion_weight`.
    #[inline]
    pub fn with_repulsion_weight(mut self, repulsion_weight: f64) -> Self {
        self.repulsion_weight = repulsion_weight;
        self
    }

    /// Sets whether row-normalized symmetrization is applied (`row_normalize`).
    #[inline]
    pub fn with_row_normalize(mut self, row_normalize: bool) -> Self {
        self.row_normalize = row_normalize;
        self
    }

    /// Sets the deterministic RNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all `GraphCneConfig` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.initial_lr.is_finite() {
            return Err(CoreError::non_finite("GraphCneConfig::initial_lr"));
        }
        if self.initial_lr <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "GraphCneConfig::initial_lr must be > 0, got {}",
                self.initial_lr
            )));
        }
        if !self.min_lr.is_finite() {
            return Err(CoreError::non_finite("GraphCneConfig::min_lr"));
        }
        if self.min_lr < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "GraphCneConfig::min_lr must be >= 0, got {}",
                self.min_lr
            )));
        }
        if self.min_lr > self.initial_lr {
            return Err(CoreError::invalid_input(format!(
                "GraphCneConfig::min_lr ({}) must be <= initial_lr ({})",
                self.min_lr, self.initial_lr
            )));
        }
        if !self.repulsion_weight.is_finite() {
            return Err(CoreError::non_finite("GraphCneConfig::repulsion_weight"));
        }
        if self.repulsion_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "GraphCneConfig::repulsion_weight must be >= 0, got {}",
                self.repulsion_weight
            )));
        }
        Ok(())
    }
}
