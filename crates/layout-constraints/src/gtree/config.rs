use egraph_core::{CoreError, CoreResult};

pub(super) const MIN_EDGE_DIST: f64 = 1e-9;
pub(super) const MAX_STAGE2_FINAL_ITERS: usize = 32;

/// Hyperparameters for [`super::GTreeOverlapRemoval`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GTreeConfig {
    /// Maximum number of outer GTree iterations (`default: 20`).
    pub max_iterations: usize,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Optional compact per-step expansion cap $t_{\max} > 1.0$ (`default: None`).
    pub compact_t_max: Option<f64>,
    /// Explicit deterministic seed used for coincident-point jitter (`default: 42`).
    pub seed: u64,
}

impl Default for GTreeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            padding: 0.0,
            compact_t_max: None,
            seed: 42,
        }
    }
}

impl GTreeConfig {
    /// Creates a new [`GTreeConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of outer GTree iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the extra minimum clearance added between rectangle borders.
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the optional compact per-step expansion cap $t_{\max} > 1.0$.
    pub fn with_compact_t_max(mut self, compact_t_max: Option<f64>) -> Self {
        self.compact_t_max = compact_t_max;
        self
    }

    /// Sets the explicit deterministic seed used for coincident-point jitter.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all GTree configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("GTreeConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "GTreeConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        if let Some(t_max) = self.compact_t_max {
            if !t_max.is_finite() {
                return Err(CoreError::non_finite("GTreeConfig::compact_t_max"));
            }
            if t_max <= 1.0 {
                return Err(CoreError::invalid_input(format!(
                    "GTreeConfig::compact_t_max must be > 1.0, got {t_max}"
                )));
            }
        }
        Ok(())
    }
}
