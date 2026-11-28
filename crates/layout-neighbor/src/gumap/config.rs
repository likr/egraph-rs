use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::find_ab_params;

/// Hyperparameters for [`GuMapSolver`](super::GuMapSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GuMapConfig {
    /// Total number of optimization iterations (`default: 200`).
    pub iterations: usize,
    /// Number of negative samples drawn per attractive edge (`default: 5`).
    pub negative_sample_rate: usize,
    /// Initial SGD learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Target minimum separation in the embedding (`default: 0.1`).
    pub min_dist: f64,
    /// Effective scale/spread parameter in the embedding (`default: 1.0`).
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
            iterations: 200,
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
    /// Creates a new [`GuMapConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets total optimization iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the number of negative samples drawn per attractive edge.
    pub fn with_negative_sample_rate(mut self, negative_sample_rate: usize) -> Self {
        self.negative_sample_rate = negative_sample_rate;
        self
    }

    /// Sets the initial SGD learning rate.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the target minimum separation `min_dist`.
    pub fn with_min_dist(mut self, min_dist: f64) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Sets the effective scale/spread parameter `spread`.
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    /// Sets optional explicit UMAP curve parameter `a`.
    pub fn with_a(mut self, a: Option<f64>) -> Self {
        self.a = a;
        self
    }

    /// Sets optional explicit UMAP curve parameter `b`.
    pub fn with_b(mut self, b: Option<f64>) -> Self {
        self.b = b;
        self
    }

    /// Sets the maximum binary-search iterations `sigma_iters`.
    pub fn with_sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the binary-search convergence tolerance `sigma_tolerance`.
    pub fn with_sigma_tolerance(mut self, sigma_tolerance: f64) -> Self {
        self.sigma_tolerance = sigma_tolerance;
        self
    }

    /// Enables or disables symmetric negative-sample updates.
    pub fn with_symmetric_negative(mut self, symmetric_negative: bool) -> Self {
        self.symmetric_negative = symmetric_negative;
        self
    }

    /// Sets the deterministic RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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
