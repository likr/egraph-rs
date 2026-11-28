use egraph_core::CoreResult;
use egraph_spatial_fft::IbFftConfig;

use super::{BarnesHut, DEFAULT_AUTO_EXACT_THRESHOLD, Exact, FitFft, IbFft, LinearFft};

/// Adaptive 2D repulsion backend for [`TsNetSolver`](crate::TsNetSolver) (`Layout2d` only).
///
/// Selects [`Exact`] when `num_nodes <= exact_threshold` (default `256`) and [`LinearFft`]
/// ($p=1$) otherwise.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Auto2d {
    /// Maximum vertex count that selects [`Exact`] repulsion (`default: 256`).
    pub exact_threshold: usize,
    /// [`LinearFft`] settings used when `num_nodes > exact_threshold`.
    pub linear_fft: LinearFft,
}

impl Default for Auto2d {
    fn default() -> Self {
        Self {
            exact_threshold: DEFAULT_AUTO_EXACT_THRESHOLD,
            linear_fft: LinearFft::default(),
        }
    }
}

impl Auto2d {
    /// Creates a new [`Auto2d`] backend with default threshold `256` and [`LinearFft`] settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum vertex count that selects [`Exact`] repulsion.
    pub fn with_exact_threshold(mut self, exact_threshold: usize) -> Self {
        self.exact_threshold = exact_threshold;
        self
    }

    /// Sets the [`LinearFft`] backend used above `exact_threshold`.
    pub fn with_linear_fft(mut self, linear_fft: LinearFft) -> Self {
        self.linear_fft = linear_fft;
        self
    }

    /// Sets the [`IbFftConfig`] forwarded to [`LinearFft`] above `exact_threshold`.
    pub fn with_ibfft(mut self, ibfft: IbFftConfig) -> Self {
        self.linear_fft = self.linear_fft.clone().with_ibfft(ibfft);
        self
    }

    /// Returns the vertex-count threshold for selecting [`Exact`].
    #[inline]
    pub fn exact_threshold(&self) -> usize {
        self.exact_threshold
    }

    /// Returns the configured [`LinearFft`] fallback backend.
    #[inline]
    pub fn linear_fft(&self) -> &LinearFft {
        &self.linear_fft
    }

    /// Validates [`Auto2d`] parameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.linear_fft.validate()
    }
}

/// Adaptive 3D repulsion backend for [`TsNetSolver`](crate::TsNetSolver) (`Layout3d` only).
///
/// Selects [`Exact`] when `num_nodes <= exact_threshold` (default `256`) and [`BarnesHut`]
/// (`ArenaOctree3d`) otherwise.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Auto3d {
    /// Maximum vertex count that selects [`Exact`] repulsion (`default: 256`).
    pub exact_threshold: usize,
    /// [`BarnesHut`] settings used when `num_nodes > exact_threshold`.
    pub barnes_hut: BarnesHut,
}

impl Default for Auto3d {
    fn default() -> Self {
        Self {
            exact_threshold: DEFAULT_AUTO_EXACT_THRESHOLD,
            barnes_hut: BarnesHut::default(),
        }
    }
}

impl Auto3d {
    /// Creates a new [`Auto3d`] backend with default threshold `256` and `theta = 0.5`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum vertex count that selects [`Exact`] repulsion.
    pub fn with_exact_threshold(mut self, exact_threshold: usize) -> Self {
        self.exact_threshold = exact_threshold;
        self
    }

    /// Sets the [`BarnesHut`] backend used above `exact_threshold`.
    pub fn with_barnes_hut(mut self, barnes_hut: BarnesHut) -> Self {
        self.barnes_hut = barnes_hut;
        self
    }

    /// Sets the Barnes-Hut opening angle threshold $\theta \ge 0$ used above `exact_threshold`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.barnes_hut = self.barnes_hut.clone().with_theta(theta);
        self
    }

    /// Returns the vertex-count threshold for selecting [`Exact`].
    #[inline]
    pub fn exact_threshold(&self) -> usize {
        self.exact_threshold
    }

    /// Returns the configured [`BarnesHut`] fallback backend.
    #[inline]
    pub fn barnes_hut(&self) -> &BarnesHut {
        &self.barnes_hut
    }

    /// Returns the configured Barnes-Hut opening angle threshold $\theta$.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.barnes_hut.theta()
    }

    /// Validates [`Auto3d`] parameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.barnes_hut.validate()
    }
}

/// Dynamic wrapper over all legal [`TsNetSolver`](crate::TsNetSolver) backends on
/// [`Layout2d`](egraph_core::Layout2d) for runtime-selected bindings.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TsNetBackend2d {
    /// Exact $O(N^2)$ pairwise repulsion.
    Exact(Exact),
    /// $O(N \log N)$ 2D Barnes-Hut quadtree repulsion.
    BarnesHut(BarnesHut),
    /// Fused 2D FFT interpolation (`FitField2d`, `Cubic` $p=3$ by default).
    FitFft(FitFft),
    /// Linear-time fused 2D FFT interpolation enforcing `Bilinear` ($p=1$).
    LinearFft(LinearFft),
    /// Adaptive threshold-based switch between `Exact` and `LinearFft`.
    Auto(Auto2d),
}

impl Default for TsNetBackend2d {
    fn default() -> Self {
        Self::Auto(Auto2d::default())
    }
}

impl TsNetBackend2d {
    /// Validates the wrapped 2D backend parameters.
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::Exact(_) => Ok(()),
            Self::BarnesHut(b) => b.validate(),
            Self::FitFft(b) => b.validate(),
            Self::LinearFft(b) => b.validate(),
            Self::Auto(b) => b.validate(),
        }
    }
}

/// Dynamic wrapper over all legal [`TsNetSolver`](crate::TsNetSolver) backends on
/// [`Layout3d`](egraph_core::Layout3d) for runtime-selected bindings.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TsNetBackend3d {
    /// Exact $O(N^2)$ pairwise repulsion.
    Exact(Exact),
    /// $O(N \log N)$ 3D Barnes-Hut octree repulsion.
    BarnesHut(BarnesHut),
    /// Adaptive threshold-based switch between `Exact` and `BarnesHut`.
    Auto(Auto3d),
}

impl Default for TsNetBackend3d {
    fn default() -> Self {
        Self::Auto(Auto3d::default())
    }
}

impl TsNetBackend3d {
    /// Validates the wrapped 3D backend parameters.
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::Exact(_) => Ok(()),
            Self::BarnesHut(b) => b.validate(),
            Self::Auto(b) => b.validate(),
        }
    }
}

/// Dynamic wrapper over all legal [`TFdpSolver`](crate::TFdpSolver) backends on
/// [`Layout2d`](egraph_core::Layout2d) for runtime-selected bindings.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TFdpBackend2d {
    /// $O(N)$ interpolation-based 2D FFT repulsion.
    IbFft(IbFft),
    /// $O(N \log N)$ 2D Barnes-Hut quadtree repulsion.
    BarnesHut(BarnesHut),
    /// Exact $O(N^2)$ pairwise repulsion.
    Exact(Exact),
}

impl Default for TFdpBackend2d {
    fn default() -> Self {
        Self::IbFft(IbFft::default())
    }
}

impl TFdpBackend2d {
    /// Validates the wrapped 2D backend parameters.
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::IbFft(b) => b.validate(),
            Self::BarnesHut(b) => b.validate(),
            Self::Exact(_) => Ok(()),
        }
    }
}

/// Dynamic wrapper over all legal [`TFdpSolver`](crate::TFdpSolver) backends on
/// [`Layout3d`](egraph_core::Layout3d) for runtime-selected bindings.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TFdpBackend3d {
    /// $O(N \log N)$ 3D Barnes-Hut octree repulsion.
    BarnesHut(BarnesHut),
    /// Exact $O(N^2)$ pairwise repulsion.
    Exact(Exact),
}

impl Default for TFdpBackend3d {
    fn default() -> Self {
        Self::BarnesHut(BarnesHut::default())
    }
}

impl TFdpBackend3d {
    /// Validates the wrapped 3D backend parameters.
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::BarnesHut(b) => b.validate(),
            Self::Exact(_) => Ok(()),
        }
    }
}
