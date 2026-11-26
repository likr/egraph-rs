use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::find_ab_params;
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

/// Repulsion approximation backend used by [`FieldLayoutSolver`](super::FieldLayoutSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FieldRepulsionBackend {
    /// $O(N)$ interpolation-based 2D FFT convolution via [`IbFft2d`](egraph_spatial_fft::IbFft2d).
    #[default]
    IbFft,
    /// $O(N \log N)$ Barnes-Hut quadtree spatial approximation.
    BarnesHut,
    /// Exact $O(N^2)$ pairwise repulsion reference.
    Exact,
}

/// Power-law schedule $\rho(t)$ weighting repulsion across normalized progress $t \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Configuration for `t-FDP` bounded Student-$t$ force-directed placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TFdpConfig {
    /// Repulsion evaluation backend (`IbFft` by default).
    pub backend: FieldRepulsionBackend,
    /// Total optimization steps (`default: 200`).
    pub iterations: usize,
    /// Base gradient descent learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Velocity momentum coefficient in `[0.0, 1.0)` (`default: 0.8`).
    pub momentum: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 0.5`).
    pub theta: f64,
    /// Attractive Student-$t$ scale $\alpha > 0$ (`default: 1.0`).
    pub alpha: f64,
    /// Repulsive kernel exponent $\gamma > 0$ in $(1 + d^2)^{-\gamma}$ (`default: 1.0`).
    pub gamma: f64,
    /// Dynamic repulsion schedule $\rho(t)$ across `0..iterations`.
    pub rho: RhoSchedule,
    /// Interpolation-based 2D FFT grid configuration.
    pub ibfft: IbFftConfig,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for TFdpConfig {
    fn default() -> Self {
        Self {
            backend: FieldRepulsionBackend::IbFft,
            iterations: 200,
            learning_rate: 1.0,
            momentum: 0.8,
            theta: 0.5,
            alpha: 1.0,
            gamma: 1.0,
            rho: RhoSchedule::default(),
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }
}

impl TFdpConfig {
    /// Validates all `t-FDP` hyperparameters and nested schedules/configs.
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
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be >= 0.0, got {}",
                self.theta
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
        self.rho.validate()?;
        self.ibfft.validate()
    }

    /// Returns the configured FFT interpolation order (`ibfft.order`).
    #[inline]
    pub fn interpolation_order(&self) -> InterpolationOrder {
        self.ibfft.order
    }

    /// Returns the configured FFT nominal interval count (`ibfft.intervals`).
    #[inline]
    pub fn intervals(&self) -> usize {
        self.ibfft.intervals
    }

    /// Returns the configured minimum FFT grid dimension (`ibfft.min_grid_dim`).
    #[inline]
    pub fn min_grid_dim(&self) -> usize {
        self.ibfft.min_grid_dim
    }

    /// Returns the configured maximum FFT grid dimension (`ibfft.max_grid_dim`).
    #[inline]
    pub fn max_grid_dim(&self) -> usize {
        self.ibfft.max_grid_dim
    }
}

/// Configuration for `ibUMAP` (interpolation-based batch Graph UMAP).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbUmapConfig {
    /// Repulsion evaluation backend (`IbFft` by default).
    pub backend: FieldRepulsionBackend,
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
    /// Per-node step clipping multiplier $\tau_t = \tau_{\mathrm{scale}} \alpha_t$ (`default: 4.0`).
    pub tau_scale: f64,
    /// Whether to apply inverse-degree hub damping $\eta_i$ (`default: true`).
    pub hub_damping: bool,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 0.5`).
    pub theta: f64,
    /// Interpolation-based 2D FFT grid configuration.
    pub ibfft: IbFftConfig,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for IbUmapConfig {
    fn default() -> Self {
        Self {
            backend: FieldRepulsionBackend::IbFft,
            iterations: 200,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
            c_k: 4.0,
            tau_scale: 4.0,
            hub_damping: true,
            theta: 0.5,
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }
}

impl IbUmapConfig {
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

    /// Validates all `ibUMAP` hyperparameters and nested FFT configuration.
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
        if !self.tau_scale.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::tau_scale"));
        }
        if self.tau_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "tau_scale must be > 0.0, got {}",
                self.tau_scale
            )));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        self.ibfft.validate()
    }

    /// Returns the configured FFT interpolation order (`ibfft.order`).
    #[inline]
    pub fn interpolation_order(&self) -> InterpolationOrder {
        self.ibfft.order
    }

    /// Returns the configured FFT nominal interval count (`ibfft.intervals`).
    #[inline]
    pub fn intervals(&self) -> usize {
        self.ibfft.intervals
    }

    /// Returns the configured minimum FFT grid dimension (`ibfft.min_grid_dim`).
    #[inline]
    pub fn min_grid_dim(&self) -> usize {
        self.ibfft.min_grid_dim
    }

    /// Returns the configured maximum FFT grid dimension (`ibfft.max_grid_dim`).
    #[inline]
    pub fn max_grid_dim(&self) -> usize {
        self.ibfft.max_grid_dim
    }
}
