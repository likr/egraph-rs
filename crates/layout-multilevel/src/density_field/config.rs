use egraph_core::{CoreError, CoreResult};
use egraph_spatial_fft::ScreenedPoissonConfig;

/// Configuration for [`ScreenedPoissonRefiner`](super::ScreenedPoissonRefiner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedPoissonRefinerConfig {
    /// Underlying Eulerian density field configuration (`default: ScreenedPoissonConfig::default()`).
    pub field: ScreenedPoissonConfig,
    /// Maximum number of refinement iterations (`default: 20`).
    pub iterations: usize,
    /// Dimensionless gradient step multiplier (`default: 0.25`, must be `> 0.0`).
    pub step_size: f64,
    /// Heavy-ball momentum coefficient in `[0.0, 1.0)` (`default: 0.5`).
    pub momentum: f64,
    /// Per-vertex displacement clamp as a fraction of domain span (`default: 0.05`, must be `> 0.0`).
    pub max_step_ratio: f64,
    /// Optional 1-hop edge-length anchor spring weight (`default: 0.0`).
    pub anchor_weight: f64,
    /// Optional Pitfall #7 cluster centroid tether strength in `[0.0, 1.0]` (`default: 0.0`).
    pub gamma_tether: f64,
    /// Whether to recenter layout mean to the origin after each active step (`default: true`).
    pub recenter: bool,
    /// Early-convergence energy tolerance (`default: 1e-12`).
    pub energy_tol: f64,
    /// Deterministic RNG seed for breaking exact coincident degeneracy (`default: 42`).
    pub seed: u64,
}

impl Default for ScreenedPoissonRefinerConfig {
    fn default() -> Self {
        Self {
            field: ScreenedPoissonConfig::default(),
            iterations: 20,
            step_size: 0.25,
            momentum: 0.5,
            max_step_ratio: 0.05,
            anchor_weight: 0.0,
            gamma_tether: 0.0,
            recenter: true,
            energy_tol: 1e-12,
            seed: 42,
        }
    }
}

impl ScreenedPoissonRefinerConfig {
    /// Validates all refiner hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.field.validate()?;
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "step_size must be finite and > 0.0, got {}",
                self.step_size
            )));
        }
        if !self.momentum.is_finite() || !(0.0..1.0).contains(&self.momentum) {
            return Err(CoreError::invalid_input(format!(
                "momentum must be finite and in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.max_step_ratio.is_finite() || self.max_step_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "max_step_ratio must be finite and > 0.0, got {}",
                self.max_step_ratio
            )));
        }
        if !self.anchor_weight.is_finite() || self.anchor_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "anchor_weight must be finite and >= 0.0, got {}",
                self.anchor_weight
            )));
        }
        if !self.gamma_tether.is_finite() || !(0.0..=1.0).contains(&self.gamma_tether) {
            return Err(CoreError::invalid_input(format!(
                "gamma_tether must be finite and in [0.0, 1.0], got {}",
                self.gamma_tether
            )));
        }
        if !self.energy_tol.is_finite() || self.energy_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "energy_tol must be finite and >= 0.0, got {}",
                self.energy_tol
            )));
        }
        Ok(())
    }
}
