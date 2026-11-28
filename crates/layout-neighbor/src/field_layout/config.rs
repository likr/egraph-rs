use egraph_core::{CoreError, CoreResult};

/// Telemetry counters from the most recent `IbFft` step on [`TFdpSolver`](super::TFdpSolver) or
/// [`IbUmapSolver`](super::IbUmapSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FieldFftStats {
    /// Number of forward 2D FFTs executed on particle charge grids.
    pub particle_forward_ffts: u64,
    /// Number of forward 2D FFTs executed on kernel grids.
    pub kernel_forward_ffts: u64,
    /// Number of inverse 2D FFTs executed.
    pub inverse_ffts: u64,
    /// Total `fft_2d` calls for this step.
    pub total_fft_2d_calls: u64,
    /// Active FFT grid width.
    pub nx: usize,
    /// Active FFT grid height.
    pub ny: usize,
}

impl FieldFftStats {
    /// Creates zero-initialized [`FieldFftStats`].
    pub fn new() -> Self {
        Self::default()
    }
}

/// Power-law schedule $\rho(t)$ weighting repulsion across normalized progress $t \in [0, 1]$.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RhoSchedule {
    /// Repulsion multiplier at progress $t = 0.0$ (`default: 1.0`).
    pub start_weight: f64,
    /// Repulsion multiplier at progress $t = 1.0$ (`default: 0.1`).
    pub end_weight: f64,
    /// Exponent $p > 0$ controlling decay curvature $t^p$ (`default: 1.0`).
    pub power: f64,
}

impl Default for RhoSchedule {
    fn default() -> Self {
        Self {
            start_weight: 1.0,
            end_weight: 0.1,
            power: 1.0,
        }
    }
}

impl RhoSchedule {
    /// Creates a new [`RhoSchedule`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial repulsion multiplier `start_weight`.
    pub fn with_start_weight(mut self, start_weight: f64) -> Self {
        self.start_weight = start_weight;
        self
    }

    /// Sets the final repulsion multiplier `end_weight`.
    pub fn with_end_weight(mut self, end_weight: f64) -> Self {
        self.end_weight = end_weight;
        self
    }

    /// Sets the schedule exponent `power`.
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Validates schedule bounds and exponent.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.start_weight.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::start_weight"));
        }
        if self.start_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "start_weight must be >= 0.0, got {}",
                self.start_weight
            )));
        }
        if !self.end_weight.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::end_weight"));
        }
        if self.end_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "end_weight must be >= 0.0, got {}",
                self.end_weight
            )));
        }
        if !self.power.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::power"));
        }
        if self.power <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "power must be > 0.0, got {}",
                self.power
            )));
        }
        Ok(())
    }

    /// Evaluates $\rho(\mathrm{progress}) = \rho_{\mathrm{start}} + (\rho_{\mathrm{end}} - \rho_{\mathrm{start}}) t^p$
    /// with `progress` clamped to `[0.0, 1.0]`.
    #[inline]
    pub fn evaluate(&self, progress: f64) -> f64 {
        let t = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p = if self.power.is_finite() && self.power > 0.0 {
            self.power
        } else {
            1.0
        };
        self.start_weight + (self.end_weight - self.start_weight) * t.powf(p)
    }
}

/// Space-independent hyperparameters for [`TFdpSolver`](super::TFdpSolver) (`t-FDP` bounded
/// Student-$t$ force-directed placement).
///
/// Repulsion approximation settings (`theta`, FFT grid parameters) live on concrete backend
/// types (`Exact`, `BarnesHut`, `IbFft`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TFdpConfig {
    /// Total optimization steps (`default: 200`).
    pub iterations: usize,
    /// Base gradient descent learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Velocity momentum coefficient in `[0.0, 1.0)` (`default: 0.8`).
    pub momentum: f64,
    /// Attractive Student-$t$ scale $\alpha > 0$ (`default: 1.0`).
    pub alpha: f64,
    /// Repulsive kernel exponent $\gamma > 0$ in $(1 + d^2)^{-\gamma}$ (`default: 1.0`).
    pub gamma: f64,
    /// Dynamic repulsion schedule $\rho(t)$ across `0..iterations`.
    pub rho: RhoSchedule,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for TFdpConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            learning_rate: 1.0,
            momentum: 0.8,
            alpha: 1.0,
            gamma: 1.0,
            rho: RhoSchedule::default(),
            seed: 42,
        }
    }
}

impl TFdpConfig {
    /// Creates a new [`TFdpConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets total optimization iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the base gradient descent learning rate.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the velocity momentum coefficient.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the attractive Student-$t$ scale `alpha`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the repulsive kernel exponent `gamma`.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the dynamic repulsion schedule `rho`.
    pub fn with_rho(mut self, rho: RhoSchedule) -> Self {
        self.rho = rho;
        self
    }

    /// Sets the deterministic RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all `t-FDP` hyperparameters and nested schedules.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if !self.momentum.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::momentum"));
        }
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(CoreError::invalid_input(format!(
                "momentum must be in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.alpha.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::alpha"));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "alpha must be > 0.0, got {}",
                self.alpha
            )));
        }
        if !self.gamma.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::gamma"));
        }
        if self.gamma <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gamma must be > 0.0, got {}",
                self.gamma
            )));
        }
        self.rho.validate()
    }
}
