use egraph_core::{CoreError, CoreResult};

/// Pivot selection strategy for [`super::SgdSolver::ortmann_sparse_stress`].
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new [`OrtmannSparseStressConfig`] with default parameters.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `num_pivots`.
    #[inline]
    pub fn with_num_pivots(mut self, num_pivots: usize) -> Self {
        self.num_pivots = num_pivots;
        self
    }

    /// Sets `pivot_strategy`.
    #[inline]
    pub fn with_pivot_strategy(mut self, pivot_strategy: OrtmannPivotStrategy) -> Self {
        self.pivot_strategy = pivot_strategy;
        self
    }

    /// Sets `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets `eta_max`.
    #[inline]
    pub fn with_eta_max(mut self, eta_max: Option<f64>) -> Self {
        self.eta_max = eta_max;
        self
    }

    /// Sets `eta_min`.
    #[inline]
    pub fn with_eta_min(mut self, eta_min: Option<f64>) -> Self {
        self.eta_min = eta_min;
        self
    }

    /// Sets `eps`.
    #[inline]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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

/// Configuration for doubly-stochastic mini-batch preconditioned stress SGD (`Sgd2Solver`).
///
/// Note: This preconditioned stress SGD solver (using [`super::SgdScheduleKind::Squared`])
/// is distinct from Ahmed et al. (2021) multi-criteria `(SGD)^2`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd2Config {
    /// Total optimization iterations (`default: 60`).
    pub iterations: usize,
    /// Minimum learning rate ratio $\varepsilon > 0$ (`default: 0.01`).
    pub eps: f64,
    /// Multiplicative step scale (`default: 1.0`).
    pub step_scale: f64,
    /// Mini-batch pair count per preconditioned update (`default: 4096`).
    pub batch_size: usize,
    /// Curvature damping denominator floor $> 0$ (`default: 1e-4`).
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
    /// Creates a new [`Sgd2Config`] with default parameters.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets `eps`.
    #[inline]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets `step_scale`.
    #[inline]
    pub fn with_step_scale(mut self, step_scale: f64) -> Self {
        self.step_scale = step_scale;
        self
    }

    /// Sets `batch_size`.
    #[inline]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets `damping`.
    #[inline]
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Sets `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all preconditioned stress SGD (`Sgd2Config`) hyperparameters.
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
