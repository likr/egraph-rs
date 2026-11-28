use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`super::PfsPrimeOverlapRemoval`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PfsPrimeConfig {
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed for coincident-point perturbation (`default: 42`).
    pub seed: u64,
}

impl Default for PfsPrimeConfig {
    fn default() -> Self {
        Self {
            padding: 0.0,
            seed: 42,
        }
    }
}

impl PfsPrimeConfig {
    /// Creates a new [`PfsPrimeConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the extra minimum clearance added between rectangle borders.
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the explicit deterministic seed for coincident-point perturbation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all `PFS'` configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("PfsPrimeConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PfsPrimeConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}
