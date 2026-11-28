use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::find_ab_params;

/// Default repulsive distance-squared softening $\varepsilon$ for [`IbUmapConfig`] (`0.05`).
pub const DEFAULT_IB_UMAP_REPULSIVE_EPS: f64 = 0.05;

/// Space-independent configuration for [`IbUmapSolver`](super::IbUmapSolver) (`ibUMAP`
/// interpolation-based batch Graph UMAP).
///
/// Repulsion approximation settings (`theta`, FFT grid parameters) live on concrete backend
/// types ([`Exact`](crate::Exact), [`BarnesHut`](crate::BarnesHut), [`IbFft`](crate::IbFft),
/// [`IbUmapBackend2d`](crate::IbUmapBackend2d)).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct IbUmapConfig {
    /// Total optimization iterations (`default: 200`).
    pub iterations: usize,
    /// Initial learning rate $\alpha_0$ (`default: 1.0`).
    pub learning_rate: f64,
    /// Target minimum separation in the 2D embedding (`default: 0.1`).
    pub min_dist: f64,
    /// Effective scale/spread parameter in the 2D embedding (`default: 1.0`).
    pub spread: f64,
    /// Optional explicit UMAP curve parameter `a` (`None` fits from `min_dist` and `spread`).
    pub a: Option<f64>,
    /// Optional explicit UMAP curve parameter `b` (`None` fits from `min_dist` and `spread`).
    pub b: Option<f64>,
    /// Repulsion calibration constant $c_K$ (`default: 4.0`).
    pub c_k: f64,
    /// Repulsive distance-squared softening $\varepsilon > 0$ (`default: 0.05`).
    pub repulsive_eps: f64,
    /// Per-node step clipping multiplier $\tau_t = \tau_{\mathrm{scale}} \alpha_t$ (`default: 4.0`).
    pub tau_scale: f64,
    /// Whether to apply inverse-degree hub damping $\eta_i$ (`default: true`).
    pub hub_damping: bool,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for IbUmapConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
            c_k: 4.0,
            repulsive_eps: DEFAULT_IB_UMAP_REPULSIVE_EPS,
            tau_scale: 4.0,
            hub_damping: true,
            seed: 42,
        }
    }
}

impl IbUmapConfig {
    /// Creates a new [`IbUmapConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets total optimization iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the initial learning rate $\alpha_0$.
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

    /// Sets the repulsion calibration constant `c_k`.
    pub fn with_c_k(mut self, c_k: f64) -> Self {
        self.c_k = c_k;
        self
    }

    /// Sets the repulsive distance-squared softening `repulsive_eps`.
    pub fn with_repulsive_eps(mut self, repulsive_eps: f64) -> Self {
        self.repulsive_eps = repulsive_eps;
        self
    }

    /// Sets the per-node step clipping multiplier `tau_scale`.
    pub fn with_tau_scale(mut self, tau_scale: f64) -> Self {
        self.tau_scale = tau_scale;
        self
    }

    /// Enables or disables inverse-degree hub damping.
    pub fn with_hub_damping(mut self, hub_damping: bool) -> Self {
        self.hub_damping = hub_damping;
        self
    }

    /// Sets the deterministic RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates `(min_dist, spread, a, b)` and returns effective `(a, b)` parameters.
    pub fn resolve_ab(&self) -> CoreResult<(f64, f64)> {
        if !self.min_dist.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::min_dist"));
        }
        if self.min_dist < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_dist must be >= 0.0, got {}",
                self.min_dist
            )));
        }
        if !self.spread.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::spread"));
        }
        if self.spread <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "spread must be > 0.0, got {}",
                self.spread
            )));
        }
        if let Some(a_val) = self.a {
            if !a_val.is_finite() {
                return Err(CoreError::non_finite("IbUmapConfig::a"));
            }
            if a_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "a must be > 0.0, got {a_val}"
                )));
            }
        }
        if let Some(b_val) = self.b {
            if !b_val.is_finite() {
                return Err(CoreError::non_finite("IbUmapConfig::b"));
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

    /// Validates all `ibUMAP` hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        let _ = self.resolve_ab()?;
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if !self.c_k.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::c_k"));
        }
        if self.c_k < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "c_k must be >= 0.0, got {}",
                self.c_k
            )));
        }
        if !self.repulsive_eps.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::repulsive_eps"));
        }
        if self.repulsive_eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "repulsive_eps must be > 0.0, got {}",
                self.repulsive_eps
            )));
        }
        if !self.tau_scale.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::tau_scale"));
        }
        if self.tau_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "tau_scale must be > 0.0, got {}",
                self.tau_scale
            )));
        }
        Ok(())
    }
}
