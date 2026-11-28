#[cfg(feature = "baselines")]
use egraph_core::RiemannianLayout;
use egraph_core::{CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, Scalar};
#[cfg(doc)]
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

/// Exact $O(N^2)$ all-pairs repulsion backend.
///
/// Supported on [`Layout2d`], [`Layout3d`], [`LayoutNd`], and (for
/// Fruchterman–Reingold) any [`RiemannianLayout`](egraph_core::RiemannianLayout)
/// target including [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Exact {}

impl Exact {
    /// Creates a new [`Exact`] repulsion backend.
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

/// $O(N \log N)$ Barnes-Hut spatial tree repulsion backend.
///
/// Supported on [`Layout2d`] (via [`ArenaQuadtree2d`]) and [`Layout3d`]
/// (via [`ArenaOctree3d`]); rejected at compile time on [`LayoutNd`] and
/// [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct BarnesHut {
    theta: f64,
}

impl Default for BarnesHut {
    #[inline]
    fn default() -> Self {
        Self { theta: 1.2 }
    }
}

impl BarnesHut {
    /// Creates a [`BarnesHut`] backend with opening angle threshold `theta >= 0.0` (`default: 1.2`).
    #[inline]
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Sets the Barnes-Hut opening angle threshold `theta >= 0.0`.
    #[inline]
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Returns the configured Barnes-Hut opening angle threshold $\theta$.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Validates Barnes-Hut parameters (`theta` must be finite and `>= 0.0`).
    #[inline]
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

/// Runtime-selectable repulsion backend wrapper for 2D ([`Layout2d`]) and 3D ([`Layout3d`])
/// force-directed solvers (`ForceAtlas2Solver`, `LinLogSolver`, `YifanHuSolver`).
///
/// Only implements repulsion traits for [`Layout2d`] and [`Layout3d`], so passing
/// `ForceBackend` to a [`LayoutNd`] solver fails at compile time.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ForceBackend {
    /// Exact $O(N^2)$ all-pairs repulsion.
    Exact(Exact),
    /// $O(N \log N)$ Barnes-Hut spatial tree repulsion.
    BarnesHut(BarnesHut),
}

impl Default for ForceBackend {
    #[inline]
    fn default() -> Self {
        Self::BarnesHut(BarnesHut::default())
    }
}

impl ForceBackend {
    /// Creates a [`ForceBackend::Exact`] variant.
    #[inline]
    pub fn exact() -> Self {
        Self::Exact(Exact::new())
    }

    /// Creates a [`ForceBackend::BarnesHut`] variant with opening angle `theta`.
    #[inline]
    pub fn barnes_hut(theta: f64) -> Self {
        Self::BarnesHut(BarnesHut::new(theta))
    }

    /// Validates the selected backend parameters.
    #[inline]
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::Exact(_) => Ok(()),
            Self::BarnesHut(bh) => bh.validate(),
        }
    }
}

/// Maps each supported Euclidean layout space `L` to its default repulsion backend for
/// `ForceAtlas2Solver`, `LinLogSolver`, and `YifanHuSolver`.
///
/// - [`Layout2d`] -> [`BarnesHut`]
/// - [`Layout3d`] -> [`BarnesHut`]
/// - [`LayoutNd`] -> [`Exact`]
pub trait DefaultForceBackend: Sized {
    /// Default repulsion backend type for `Self`.
    type Backend;
}

impl<S: Scalar> DefaultForceBackend for Layout2d<S> {
    type Backend = BarnesHut;
}

impl<S: Scalar> DefaultForceBackend for Layout3d<S> {
    type Backend = BarnesHut;
}

impl<S: Scalar> DefaultForceBackend for LayoutNd<S> {
    type Backend = Exact;
}

/// Fruchterman–Reingold (1991) Fig. 2 $2k$ spatial grid repulsion backend with strict
/// Euclidean cutoff $d < 2k$ ($f_r(d) = 0$ for $d \ge 2k$).
///
/// Supported on [`Layout2d`] and [`Layout3d`]; rejected at compile time on [`LayoutNd`] and
/// [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d).
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Grid {}

#[cfg(feature = "baselines")]
impl Grid {
    /// Creates a new [`Grid`] repulsion backend.
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

/// Runtime-selectable repulsion backend wrapper for 2D ([`Layout2d`]) and 3D ([`Layout3d`])
/// [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
///
/// Only implements [`FrRepulsion`](super::FrRepulsion) for [`Layout2d`] and [`Layout3d`], so
/// passing `FrBackend` to a [`LayoutNd`] or [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d)
/// solver fails at compile time.
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum FrBackend {
    /// Exact $O(N^2)$ all-pairs repulsion.
    Exact(Exact),
    /// $O(N \log N)$ Barnes-Hut spatial tree approximation.
    BarnesHut(BarnesHut),
    /// Fruchterman–Reingold (1991) Fig. 2 $2k$ spatial grid with strict $d < 2k$ cutoff.
    Grid(Grid),
}

#[cfg(feature = "baselines")]
impl Default for FrBackend {
    #[inline]
    fn default() -> Self {
        Self::Exact(Exact::default())
    }
}

#[cfg(feature = "baselines")]
impl FrBackend {
    /// Creates an [`FrBackend::Exact`] variant.
    #[inline]
    pub fn exact() -> Self {
        Self::Exact(Exact::new())
    }

    /// Creates an [`FrBackend::BarnesHut`] variant with opening angle `theta`.
    #[inline]
    pub fn barnes_hut(theta: f64) -> Self {
        Self::BarnesHut(BarnesHut::new(theta))
    }

    /// Creates an [`FrBackend::Grid`] variant.
    #[inline]
    pub fn grid() -> Self {
        Self::Grid(Grid::new())
    }

    /// Validates the selected backend parameters.
    #[inline]
    pub fn validate(&self) -> CoreResult<()> {
        match self {
            Self::Exact(_) | Self::Grid(_) => Ok(()),
            Self::BarnesHut(bh) => bh.validate(),
        }
    }
}

/// Maps each supported [`RiemannianLayout`](egraph_core::RiemannianLayout) space `L` to its
/// default Fruchterman–Reingold repulsion backend ([`Exact`]).
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub trait DefaultFrBackend {
    /// Default repulsion backend type for `Self`.
    type Backend;
}

#[cfg(feature = "baselines")]
impl<L: RiemannianLayout> DefaultFrBackend for L {
    type Backend = Exact;
}
