use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`FdebSolver`](super::FdebSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdebConfig {
    /// Number of subdivision cycles (`default: 6`).
    pub cycles: usize,
    /// Initial force step size (`default: 0.1`).
    pub s0: f64,
    /// Initial number of iterations in cycle 0 (`default: 90`).
    pub i0: usize,
    /// Multiplicative step-size decay factor per cycle (`default: 0.5`).
    pub s_step: f64,
    /// Multiplicative iteration-count decay factor per cycle (`default: 2.0 / 3.0`).
    pub i_step: f64,
    /// Edge spring constant (`default: 0.1`).
    pub spring_k: f64,
    /// Minimum edge compatibility threshold in `[0.0, 1.0]` (`default: 0.6`).
    pub minimum_edge_compatibility: f64,
}

impl Default for FdebConfig {
    fn default() -> Self {
        Self {
            cycles: 6,
            s0: 0.1,
            i0: 90,
            s_step: 0.5,
            i_step: 2.0 / 3.0,
            spring_k: 0.1,
            minimum_edge_compatibility: 0.6,
        }
    }
}

impl FdebConfig {
    /// Validates all FDEB configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        for (name, val) in [
            ("s0", self.s0),
            ("s_step", self.s_step),
            ("i_step", self.i_step),
            ("spring_k", self.spring_k),
            (
                "minimum_edge_compatibility",
                self.minimum_edge_compatibility,
            ),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite("FdebConfig"));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be >= 0.0, got {val}"
                )));
            }
        }
        if self.minimum_edge_compatibility > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "minimum_edge_compatibility must be <= 1.0, got {}",
                self.minimum_edge_compatibility
            )));
        }
        if self.cycles >= 30 {
            return Err(CoreError::invalid_input(format!(
                "cycles must be < 30 to avoid subdivision overflow, got {}",
                self.cycles
            )));
        }
        Ok(())
    }
}
