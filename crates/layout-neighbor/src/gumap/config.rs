use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::find_ab_params;

/// Hyperparameters for [`GuMapSolver`](super::GuMapSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuMapConfig {
    /// Total number of optimization epochs (`default: 200`).
    pub n_epochs: usize,
    /// Number of negative samples drawn per attractive edge (`default: 5`).
    pub negative_sample_rate: usize,
    /// Initial SGD learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Target minimum separation in the 2D embedding (`default: 0.1`).
    pub min_dist: f64,
    /// Effective scale/spread parameter in the 2D embedding (`default: 1.0`).
    pub spread: f64,
    /// Optional explicit UMAP curve parameter `a` (`None` fits from `min_dist` and `spread`).
    pub a: Option<f64>,
    /// Optional explicit UMAP curve parameter `b` (`None` fits from `min_dist` and `spread`).
    pub b: Option<f64>,
    /// Maximum binary-search iterations for $\sigma_i$ in fuzzy simplicial set construction (`default: 64`).
    pub sigma_iters: usize,
    /// Convergence tolerance for $\sigma_i$ binary search (`default: 1e-5`).
    pub sigma_tolerance: f64,
    /// Whether negative-sample repulsion symmetrically updates both endpoints (`default: true`).
    pub symmetric_negative: bool,
    /// Explicit deterministic seed for negative sampling and edge subsampling (`default: 42`).
    pub seed: u64,
}

impl Default for GuMapConfig {
    fn default() -> Self {
        Self {
            n_epochs: 200,
            negative_sample_rate: 5,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
            sigma_iters: 64,
            sigma_tolerance: 1e-5,
            symmetric_negative: true,
            seed: 42,
        }
    }
}

impl GuMapConfig {
    /// Validates all hyperparameters and resolves `(a, b)` curve parameters if omitted.
    pub fn validate(&self) -> CoreResult<()> {
        let _ = self.resolve_ab()?;
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if self.sigma_iters == 0 {
            return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
        }
        if !self.sigma_tolerance.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::sigma_tolerance"));
        }
        if self.sigma_tolerance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_tolerance must be > 0.0, got {}",
                self.sigma_tolerance
            )));
        }
        Ok(())
    }

    /// Validates `(min_dist, spread, a, b)` and returns effective `(a, b)`.
    pub fn resolve_ab(&self) -> CoreResult<(f64, f64)> {
        if !self.min_dist.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::min_dist"));
        }
        if self.min_dist < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_dist must be >= 0.0, got {}",
                self.min_dist
            )));
        }
        if !self.spread.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::spread"));
        }
        if self.spread <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "spread must be > 0.0, got {}",
                self.spread
            )));
        }
        if let Some(a_val) = self.a {
            if !a_val.is_finite() {
                return Err(CoreError::non_finite("GuMapConfig::a"));
            }
            if a_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "a must be > 0.0, got {a_val}"
                )));
            }
        }
        if let Some(b_val) = self.b {
            if !b_val.is_finite() {
                return Err(CoreError::non_finite("GuMapConfig::b"));
            }
            if b_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "b must be > 0.0, got {b_val}"
                )));
            }
        }

        let (eff_a, eff_b) = match (self.a, self.b) {
            (Some(a_val), Some(b_val)) => (a_val, b_val),
            _ => {
                let (fitted_a, fitted_b) = find_ab_params(self.min_dist, self.spread)?;
                (self.a.unwrap_or(fitted_a), self.b.unwrap_or(fitted_b))
            }
        };
        if !eff_a.is_finite() || eff_a <= 0.0 || !eff_b.is_finite() || eff_b <= 0.0 {
            return Err(CoreError::numerical_breakdown(
                "resolved UMAP (a, b) parameters must be finite and positive",
            ));
        }
        Ok((eff_a, eff_b))
    }
}
