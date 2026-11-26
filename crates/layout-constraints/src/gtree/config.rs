use egraph_core::{CoreError, CoreResult};

#[cfg(not(feature = "delaunay"))]
pub(super) const K_FALLBACK_NEIGHBORS: usize = 12;

pub(super) const MIN_EDGE_DIST: f64 = 1e-9;
pub(super) const MAX_STAGE2_FINAL_ITERS: usize = 32;

/// Hyperparameters for [`super::GTreeOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
