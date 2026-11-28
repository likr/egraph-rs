use egraph_core::{CoreError, CoreResult};
use egraph_spatial_fft::ScreenedPoissonConfig;

/// Configuration for [`ScreenedPoissonRefiner`](super::ScreenedPoissonRefiner).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Optional cluster centroid tether strength in `[0.0, 1.0]` (`default: 0.0`).
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
    /// Creates a new [`ScreenedPoissonRefinerConfig`] initialized to [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the underlying [`ScreenedPoissonConfig`] `field`.
    #[inline]
    pub fn with_field(mut self, field: ScreenedPoissonConfig) -> Self {
        self.field = field;
        self
    }

    /// Sets the maximum number of refinement `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the dimensionless gradient step multiplier `step_size`.
    #[inline]
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Sets the heavy-ball `momentum` coefficient in `[0.0, 1.0)`.
    #[inline]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the per-vertex displacement clamp `max_step_ratio` as a fraction of domain span.
    #[inline]
    pub fn with_max_step_ratio(mut self, max_step_ratio: f64) -> Self {
        self.max_step_ratio = max_step_ratio;
        self
    }

    /// Sets the optional 1-hop edge-length `anchor_weight`.
    #[inline]
    pub fn with_anchor_weight(mut self, anchor_weight: f64) -> Self {
        self.anchor_weight = anchor_weight;
        self
    }

    /// Sets the optional cluster centroid tether strength `gamma_tether` in `[0.0, 1.0]`.
    #[inline]
    pub fn with_gamma_tether(mut self, gamma_tether: f64) -> Self {
        self.gamma_tether = gamma_tether;
        self
    }

    /// Sets whether to `recenter` the layout mean to the origin after each active step.
    #[inline]
    pub fn with_recenter(mut self, recenter: bool) -> Self {
        self.recenter = recenter;
        self
    }

    /// Sets the early-convergence `energy_tol`.
    #[inline]
    pub fn with_energy_tol(mut self, energy_tol: f64) -> Self {
        self.energy_tol = energy_tol;
        self
    }

    /// Sets the deterministic PRNG `seed` for breaking coincident degeneracy.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all refiner hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.field.validate()?;
        if !self.step_size.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::step_size",
            ));
        }
        if self.step_size <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "step_size must be > 0.0, got {}",
                self.step_size
            )));
        }
        if !self.momentum.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::momentum",
            ));
        }
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(CoreError::invalid_input(format!(
                "momentum must be in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.max_step_ratio.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::max_step_ratio",
            ));
        }
        if self.max_step_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "max_step_ratio must be > 0.0, got {}",
                self.max_step_ratio
            )));
        }
        if !self.anchor_weight.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::anchor_weight",
            ));
        }
        if self.anchor_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "anchor_weight must be >= 0.0, got {}",
                self.anchor_weight
            )));
        }
        if !self.gamma_tether.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::gamma_tether",
            ));
        }
        if !(0.0..=1.0).contains(&self.gamma_tether) {
            return Err(CoreError::invalid_input(format!(
                "gamma_tether must be in [0.0, 1.0], got {}",
                self.gamma_tether
            )));
        }
        if !self.energy_tol.is_finite() {
            return Err(CoreError::non_finite(
                "ScreenedPoissonRefinerConfig::energy_tol",
            ));
        }
        if self.energy_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "energy_tol must be >= 0.0, got {}",
                self.energy_tol
            )));
        }
        Ok(())
    }
}
