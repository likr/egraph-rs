mod wrappers;

use egraph_core::{CoreError, CoreResult};
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

pub use wrappers::{Auto2d, Auto3d, TFdpBackend2d, TFdpBackend3d, TsNetBackend2d, TsNetBackend3d};

/// Default vertex-count threshold at or below which [`Auto2d`] and [`Auto3d`] select [`Exact`].
pub const DEFAULT_AUTO_EXACT_THRESHOLD: usize = 256;

/// Exact $O(N^2)$ pairwise repulsion backend.
///
/// Supported across all [`RiemannianLayout`](egraph_core::RiemannianLayout) targets in
/// [`TsNetSolver`](crate::TsNetSolver) and all [`EuclideanLayout`](egraph_core::EuclideanLayout)
/// targets in [`TFdpSolver`](crate::TFdpSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Exact;

impl Exact {
    /// Creates a new [`Exact`] repulsion backend.
    pub fn new() -> Self {
        Self
    }
}

/// $O(N \log N)$ Barnes-Hut spatial tree repulsion backend
/// (`ArenaQuadtree2d` on `Layout2d`, `ArenaOctree3d` on `Layout3d`).
///
/// Supported only on [`Layout2d`](egraph_core::Layout2d) and
/// [`Layout3d`](egraph_core::Layout3d).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct BarnesHut {
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 0.5`).
    pub theta: f64,
}

impl Default for BarnesHut {
    fn default() -> Self {
        Self { theta: 0.5 }
    }
}

impl BarnesHut {
    /// Creates a new [`BarnesHut`] backend with the given opening angle threshold `theta`.
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Sets the Barnes-Hut opening angle threshold $\theta \ge 0$.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Returns the configured Barnes-Hut opening angle threshold $\theta$.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Validates Barnes-Hut parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        Ok(())
    }
}

/// Fused 2D FFT interpolation backend (`FitField2d`) for [`TsNetSolver`](crate::TsNetSolver)
/// (`Layout2d` only).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FitFft {
    /// Interpolation-based 2D FFT grid configuration (`Cubic` $p=3$ by default).
    pub ibfft: IbFftConfig,
}

impl Default for FitFft {
    fn default() -> Self {
        Self {
            ibfft: IbFftConfig::default().with_order(InterpolationOrder::Cubic),
        }
    }
}

impl FitFft {
    /// Creates a new [`FitFft`] backend with default `Cubic` ($p=3$) FFT settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the interpolation-based 2D FFT grid configuration.
    pub fn with_ibfft(mut self, ibfft: IbFftConfig) -> Self {
        self.ibfft = ibfft;
        self
    }

    /// Sets the polynomial interpolation order per grid axis.
    pub fn with_order(mut self, order: InterpolationOrder) -> Self {
        self.ibfft.order = order;
        self
    }

    /// Sets the nominal interval count used to determine base grid resolution.
    pub fn with_intervals(mut self, intervals: usize) -> Self {
        self.ibfft.intervals = intervals;
        self
    }

    /// Sets the minimum interior grid resolution per axis.
    pub fn with_min_grid_dim(mut self, min_grid_dim: usize) -> Self {
        self.ibfft.min_grid_dim = min_grid_dim;
        self
    }

    /// Sets the maximum interior grid resolution per axis.
    pub fn with_max_grid_dim(mut self, max_grid_dim: usize) -> Self {
        self.ibfft.max_grid_dim = max_grid_dim;
        self
    }

    /// Returns the configured [`IbFftConfig`].
    #[inline]
    pub fn ibfft(&self) -> &IbFftConfig {
        &self.ibfft
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

    /// Validates the FFT grid configuration.
    pub fn validate(&self) -> CoreResult<()> {
        self.ibfft.validate()
    }
}

/// Linear-time fused 2D FFT interpolation backend for [`TsNetSolver`](crate::TsNetSolver)
/// enforcing [`InterpolationOrder::Bilinear`] ($p=1$, `Layout2d` only).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFft {
    /// Interpolation-based 2D FFT grid configuration (`Bilinear` $p=1$ enforced).
    pub ibfft: IbFftConfig,
}

impl Default for LinearFft {
    fn default() -> Self {
        Self {
            ibfft: IbFftConfig::default().with_order(InterpolationOrder::Bilinear),
        }
    }
}

impl LinearFft {
    /// Creates a new [`LinearFft`] backend with default `Bilinear` ($p=1$) FFT settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the interpolation-based 2D FFT grid configuration (enforcing `Bilinear` $p=1$).
    pub fn with_ibfft(mut self, mut ibfft: IbFftConfig) -> Self {
        ibfft.order = InterpolationOrder::Bilinear;
        self.ibfft = ibfft;
        self
    }

    /// Sets the nominal interval count used to determine base grid resolution.
    pub fn with_intervals(mut self, intervals: usize) -> Self {
        self.ibfft.intervals = intervals;
        self
    }

    /// Sets the minimum interior grid resolution per axis.
    pub fn with_min_grid_dim(mut self, min_grid_dim: usize) -> Self {
        self.ibfft.min_grid_dim = min_grid_dim;
        self
    }

    /// Sets the maximum interior grid resolution per axis.
    pub fn with_max_grid_dim(mut self, max_grid_dim: usize) -> Self {
        self.ibfft.max_grid_dim = max_grid_dim;
        self
    }

    /// Returns the configured [`IbFftConfig`].
    #[inline]
    pub fn ibfft(&self) -> &IbFftConfig {
        &self.ibfft
    }

    /// Returns an [`IbFftConfig`] with [`InterpolationOrder::Bilinear`] enforced.
    #[inline]
    pub(crate) fn effective_ibfft(&self) -> IbFftConfig {
        let mut cfg = self.ibfft.clone();
        cfg.order = InterpolationOrder::Bilinear;
        cfg
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

    /// Validates the FFT grid configuration.
    pub fn validate(&self) -> CoreResult<()> {
        self.ibfft.validate()
    }
}

/// Interpolation-based 2D FFT repulsion backend (`IbFft2d`) for [`TFdpSolver`](crate::TFdpSolver)
/// (`Layout2d` only).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IbFft {
    /// Interpolation-based 2D FFT grid configuration (`Bilinear` $p=1$ by default).
    pub ibfft: IbFftConfig,
}

impl IbFft {
    /// Creates a new [`IbFft`] backend with default FFT grid settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the interpolation-based 2D FFT grid configuration.
    pub fn with_ibfft(mut self, ibfft: IbFftConfig) -> Self {
        self.ibfft = ibfft;
        self
    }

    /// Sets the polynomial interpolation order per grid axis.
    pub fn with_order(mut self, order: InterpolationOrder) -> Self {
        self.ibfft.order = order;
        self
    }

    /// Sets the nominal interval count used to determine base grid resolution.
    pub fn with_intervals(mut self, intervals: usize) -> Self {
        self.ibfft.intervals = intervals;
        self
    }

    /// Sets the minimum interior grid resolution per axis.
    pub fn with_min_grid_dim(mut self, min_grid_dim: usize) -> Self {
        self.ibfft.min_grid_dim = min_grid_dim;
        self
    }

    /// Sets the maximum interior grid resolution per axis.
    pub fn with_max_grid_dim(mut self, max_grid_dim: usize) -> Self {
        self.ibfft.max_grid_dim = max_grid_dim;
        self
    }

    /// Returns the configured [`IbFftConfig`].
    #[inline]
    pub fn ibfft(&self) -> &IbFftConfig {
        &self.ibfft
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

    /// Validates the FFT grid configuration.
    pub fn validate(&self) -> CoreResult<()> {
        self.ibfft.validate()
    }
}
