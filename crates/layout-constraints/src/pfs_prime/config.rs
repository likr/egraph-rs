use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`super::PfsPrimeOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
