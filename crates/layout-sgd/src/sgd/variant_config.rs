use egraph_core::{CoreError, CoreResult};

/// Pivot selection strategy for [`super::SgdSolver::ortmann_sparse_stress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrtmannPivotStrategy {
    /// K-means shortest-path medoid refinement initialized from Core-First MaxMin (`default`).
    #[default]
    KMeansSp,
    /// Randomized MaxMin shortest-path sampling proportional to min distance to chosen pivots.
    MaxMinRandomSp,
    /// Deterministic Core-First MaxMin shortest-path pivot selection.
    MaxMinSp,
}

/// Configuration for Ortmann et al. (2016) Sparse Stress SGD (`ortmann_sparse_stress`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrtmannSparseStressConfig {
    /// Number of shortest-path pivots (`default: 100`, must be `>= 1`).
    pub num_pivots: usize,
    /// Pivot selection strategy (`default: OrtmannPivotStrategy::KMeansSp`).
    pub pivot_strategy: OrtmannPivotStrategy,
    /// Total SGD iterations (`default: 200`).
    pub iterations: usize,
    /// Optional override for initial step size $\eta_{\max}$ (`default: None`).
    pub eta_max: Option<f64>,
    /// Optional override for minimum step size $\eta_{\min}$ (`default: None`).
    pub eta_min: Option<f64>,
    /// Minimum learning rate ratio $\varepsilon > 0$ when `eta_min` is `None` (`default: 0.1`).
    pub eps: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for OrtmannSparseStressConfig {
    fn default() -> Self {
        Self {
            num_pivots: 100,
            pivot_strategy: OrtmannPivotStrategy::KMeansSp,
            iterations: 200,
            eta_max: None,
            eta_min: None,
            eps: 0.1,
            seed: 42,
        }
    }
}

impl OrtmannSparseStressConfig {
    /// Validates all `OrtmannSparseStressConfig` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.num_pivots == 0 {
            return Err(CoreError::invalid_input(
                "OrtmannSparseStressConfig::num_pivots must be >= 1",
            ));
        }
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("OrtmannSparseStressConfig::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "OrtmannSparseStressConfig::eps must be > 0, got {}",
                self.eps
            )));
        }
        if let Some(emax) = self.eta_max {
            if !emax.is_finite() {
                return Err(CoreError::non_finite("OrtmannSparseStressConfig::eta_max"));
            }
            if emax <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "OrtmannSparseStressConfig::eta_max must be > 0, got {emax}"
                )));
            }
        }
        if let Some(emin) = self.eta_min {
            if !emin.is_finite() {
                return Err(CoreError::non_finite("OrtmannSparseStressConfig::eta_min"));
            }
            if emin <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "OrtmannSparseStressConfig::eta_min must be > 0, got {emin}"
                )));
            }
        }
        if let (Some(emin), Some(emax)) = (self.eta_min, self.eta_max)
            && emin > emax
        {
            return Err(CoreError::invalid_input(format!(
                "OrtmannSparseStressConfig::eta_min ({emin}) must be <= eta_max ({emax})"
            )));
        }
        Ok(())
    }
}

/// Configuration for single-threaded deterministic `LargeVis` graph layout
/// (`Tang et al., 2016`).
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Contrastive loss objective family for [`super::SgdSolver::graph_cne`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphCneObjective {
    /// InfoNCE contrastive objective with Cauchy kernel (`Böhm et al., 2025`, `default`).
    #[default]
    InfoNce,
    /// Negative-sampled t-SNE objective on the attraction-repulsion spectrum (`Böhm et al., 2022`).
    NegTSne,
}

/// Configuration for `Graph-CNE` contrastive neighbor embedding (`Böhm et al., 2025`).
#[derive(Debug, Clone, Copy, PartialEq)]
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
