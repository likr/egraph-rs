use egraph_core::{CoreError, CoreResult, Scalar};

/// Frequency-domain Green's operator used in [`super::ScreenedPoissonField2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DensityFieldKernel {
    /// Long-range screened Poisson operator $\hat{G}(k) = (\|k\|^2 + \kappa_0^2)^{-1}$.
    #[default]
    ScreenedPoisson,
    /// Local single-Gaussian $L^2$ excess baseline $\hat{G}(k) = 1$.
    SingleGaussian,
}

/// Configuration for [`super::ScreenedPoissonField2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedPoissonConfig {
    /// Density kernel operator (`default: DensityFieldKernel::ScreenedPoisson`).
    pub kernel: DensityFieldKernel,
    /// Active interior grid resolution $G \times G$ (`default: 2048`, power of two `>= 16`).
    pub grid_size: usize,
    /// Zero-padding multiplier (`1` or `2`, `default: 2`).
    pub padding_factor: usize,
    /// Relative border margin around the particle bounding box (`default: 0.1`).
    pub margin_ratio: f64,
    /// Default Gaussian smoothing width $\sigma_0$ as a fraction of domain span (`default: 0.015`).
    pub sigma_ratio: f64,
    /// Optional explicit physical Gaussian smoothing width $\sigma_0 > 0$.
    pub sigma: Option<f64>,
    /// Default screening length $\kappa_0^{-1}$ as a fraction of domain span (`default: 0.20`).
    pub screening_length_ratio: f64,
    /// Optional explicit screening wavenumber $\kappa_0 > 0$.
    pub kappa: Option<f64>,
    /// Default upper density cap multiplier relative to uniform domain density (`default: 4.0`).
    pub rho_max_factor: f64,
    /// Optional explicit density upper bound $\rho_{\max} \ge 0$.
    pub rho_max: Option<f64>,
    /// Whether to apply the exact variational support mask $\mathbf{1}_{\{\rho_V > \rho_{\max}\}}$
    /// before the final $K_{\sigma_0}$ adjoint convolution (`default: true`).
    pub exact_adjoint: bool,
}

impl Default for ScreenedPoissonConfig {
    fn default() -> Self {
        Self {
            kernel: DensityFieldKernel::ScreenedPoisson,
            grid_size: 2048,
            padding_factor: 2,
            margin_ratio: 0.1,
            sigma_ratio: 0.015,
            sigma: None,
            screening_length_ratio: 0.20,
            kappa: None,
            rho_max_factor: 4.0,
            rho_max: None,
            exact_adjoint: true,
        }
    }
}

impl ScreenedPoissonConfig {
    /// Validates all field hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.grid_size < 16 || !self.grid_size.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "grid_size must be a power of two >= 16, got {}",
                self.grid_size
            )));
        }
        if self.padding_factor != 1 && self.padding_factor != 2 {
            return Err(CoreError::invalid_input(format!(
                "padding_factor must be 1 or 2, got {}",
                self.padding_factor
            )));
        }
        if !self.margin_ratio.is_finite() || self.margin_ratio < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin_ratio must be finite and >= 0.0, got {}",
                self.margin_ratio
            )));
        }
        if !self.sigma_ratio.is_finite() || self.sigma_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_ratio must be finite and > 0.0, got {}",
                self.sigma_ratio
            )));
        }
        if let Some(s) = self.sigma
            && (!s.is_finite() || s <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "sigma must be finite and > 0.0, got {s}"
            )));
        }
        if !self.screening_length_ratio.is_finite() || self.screening_length_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "screening_length_ratio must be finite and > 0.0, got {}",
                self.screening_length_ratio
            )));
        }
        if let Some(k) = self.kappa
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "kappa must be finite and > 0.0, got {k}"
            )));
        }
        if !self.rho_max_factor.is_finite() || self.rho_max_factor < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "rho_max_factor must be finite and >= 0.0, got {}",
                self.rho_max_factor
            )));
        }
        if let Some(r) = self.rho_max
            && (!r.is_finite() || r < 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "rho_max must be finite and >= 0.0, got {r}"
            )));
        }
        Ok(())
    }
}

/// Diagnostic telemetry produced by [`super::ScreenedPoissonField2d::evaluate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScreenedPoissonStats {
    /// Discrete Parseval variational energy $\mathcal{L}_{\mathrm{dens}}(X) \ge 0$.
    pub energy: f64,
    /// Peak smoothed density $\max_p \rho_V(p)$ on the grid.
    pub max_density: f64,
    /// Fraction of active interior grid cells where $\rho_V > \rho_{\max}$.
    pub excess_fraction: f64,
    /// Integrated excess density mass $\int \delta\rho(p)\,dp \in [0, 1]$.
    pub excess_mass: f64,
    /// Effective Gaussian smoothing width $\sigma_0$ used.
    pub sigma_used: f64,
    /// Effective screening wavenumber $\kappa_0$ used.
    pub kappa_used: f64,
    /// Effective upper density threshold $\rho_{\max}$ used.
    pub rho_max_used: f64,
    /// Active grid dimension $G$.
    pub grid_dim: usize,
    /// Padded FFT dimension $G_{\mathrm{pad}} = G \cdot \mathrm{padding\_factor}$.
    pub pad_dim: usize,
    /// Number of 2D R2C/C2R FFT calls executed in this evaluation (`0`, `2`, `4`, or `6`).
    pub fft_2d_calls: u64,
    /// `true` when zero cells exceeded $\rho_{\max}$ and the solve exited after smoothing.
    pub early_exit_no_excess: bool,
}

/// Reusable output buffers filled by [`super::ScreenedPoissonField2d::evaluate`].
#[derive(Debug, Clone)]
pub struct ScreenedPoissonOutput<S: Scalar = f64> {
    /// Descent force $x$-components $F_{i,x}^{\mathrm{dens}} = -\partial\mathcal{L}/\partial x_i$.
    pub force_x: Vec<S>,
    /// Descent force $y$-components $F_{i,y}^{\mathrm{dens}} = -\partial\mathcal{L}/\partial y_i$.
    pub force_y: Vec<S>,
    /// Variational energy $\mathcal{L}_{\mathrm{dens}}(X)$.
    pub energy: f64,
    /// Evaluation telemetry.
    pub stats: ScreenedPoissonStats,
}

impl<S: Scalar> Default for ScreenedPoissonOutput<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> ScreenedPoissonOutput<S> {
    /// Allocates an empty reusable output buffer.
    pub fn new() -> Self {
        Self {
            force_x: Vec::new(),
            force_y: Vec::new(),
            energy: 0.0,
            stats: ScreenedPoissonStats::default(),
        }
    }

    pub(super) fn resize_and_zero(&mut self, n: usize) {
        self.force_x.resize(n, S::ZERO);
        self.force_x.fill(S::ZERO);
        self.force_y.resize(n, S::ZERO);
        self.force_y.fill(S::ZERO);
        self.energy = 0.0;
        self.stats = ScreenedPoissonStats::default();
    }
}
