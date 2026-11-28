use egraph_core::{
    CoreResult, DeterministicRng, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    RiemannianLayout, Scalar, SolverStatus,
};

use super::config::GuMapConfig;
use super::manifold::{step_manifold, validate_manifold};
use super::nd::{step_nd, validate_nd};
use super::riemannian::{GuMapRiemannianScratch, step_riemannian, validate_riemannian};
use super::step2d::{step_2d, validate_basic_2d};
use super::step3d::{step_3d, validate_basic_3d};

/// Space-independent core state shared across all [`GuMapSolver`](super::GuMapSolver) targets.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuMapCore {
    pub(super) config: GuMapConfig,
    pub(super) num_nodes: usize,
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) edge_src: Vec<u32>,
    pub(super) edge_tgt: Vec<u32>,
    pub(super) edge_weight: Vec<f32>,
    pub(super) epoch: usize,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

#[derive(Debug, Clone)]
pub(super) enum GuMapSpaceKind {
    Euclidean2d,
    Euclidean3d,
    EuclideanNd { locked_dims: Option<usize> },
    Manifold2d { locked_kind: Option<ManifoldKind> },
    Riemannian(Option<GuMapRiemannianScratch>),
}

/// Opaque per-space scratch/lock workspace for [`GuMapSolver`](super::GuMapSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GuMapWorkspace(pub(super) GuMapSpaceKind);

/// Target layout capability trait for [`GuMapSolver`](super::GuMapSolver).
///
/// Implemented for [`Layout2d`], [`Layout3d`], [`LayoutNd`], and [`ManifoldLayout2d`] with
/// zero-copy fast paths. External [`RiemannianLayout`] implementations can opt into
/// `GuMapSolver` with an empty `impl GuMapTarget for MyLayout {}` block, inheriting the
/// generic Riemannian geodesic implementation automatically.
pub trait GuMapTarget: RiemannianLayout {
    #[doc(hidden)]
    fn init_gumap_workspace() -> GuMapWorkspace
    where
        Self: Sized,
    {
        GuMapWorkspace(GuMapSpaceKind::Riemannian(None))
    }

    #[doc(hidden)]
    fn validate_gumap_target(&self, core: &GuMapCore, ws: &GuMapWorkspace) -> CoreResult<()> {
        validate_riemannian(core, ws, self)
    }

    #[doc(hidden)]
    fn step_gumap_target(
        &mut self,
        core: &mut GuMapCore,
        ws: &mut GuMapWorkspace,
    ) -> CoreResult<()> {
        step_riemannian(core, ws, self)
    }
}

impl<S: Scalar> GuMapTarget for Layout2d<S> {
    #[inline]
    fn init_gumap_workspace() -> GuMapWorkspace {
        GuMapWorkspace(GuMapSpaceKind::Euclidean2d)
    }

    #[inline]
    fn validate_gumap_target(&self, core: &GuMapCore, _ws: &GuMapWorkspace) -> CoreResult<()> {
        validate_basic_2d(core, self)
    }

    #[inline]
    fn step_gumap_target(
        &mut self,
        core: &mut GuMapCore,
        _ws: &mut GuMapWorkspace,
    ) -> CoreResult<()> {
        step_2d(core, self)
    }
}

impl<S: Scalar> GuMapTarget for Layout3d<S> {
    #[inline]
    fn init_gumap_workspace() -> GuMapWorkspace {
        GuMapWorkspace(GuMapSpaceKind::Euclidean3d)
    }

    #[inline]
    fn validate_gumap_target(&self, core: &GuMapCore, _ws: &GuMapWorkspace) -> CoreResult<()> {
        validate_basic_3d(core, self)
    }

    #[inline]
    fn step_gumap_target(
        &mut self,
        core: &mut GuMapCore,
        _ws: &mut GuMapWorkspace,
    ) -> CoreResult<()> {
        step_3d(core, self)
    }
}

impl<S: Scalar> GuMapTarget for LayoutNd<S> {
    #[inline]
    fn init_gumap_workspace() -> GuMapWorkspace {
        GuMapWorkspace(GuMapSpaceKind::EuclideanNd { locked_dims: None })
    }

    #[inline]
    fn validate_gumap_target(&self, core: &GuMapCore, ws: &GuMapWorkspace) -> CoreResult<()> {
        validate_nd(core, ws, self)
    }

    #[inline]
    fn step_gumap_target(
        &mut self,
        core: &mut GuMapCore,
        ws: &mut GuMapWorkspace,
    ) -> CoreResult<()> {
        step_nd(core, ws, self)
    }
}

impl<S: Scalar> GuMapTarget for ManifoldLayout2d<S> {
    #[inline]
    fn init_gumap_workspace() -> GuMapWorkspace {
        GuMapWorkspace(GuMapSpaceKind::Manifold2d { locked_kind: None })
    }

    #[inline]
    fn validate_gumap_target(&self, core: &GuMapCore, ws: &GuMapWorkspace) -> CoreResult<()> {
        validate_manifold(core, ws, self)
    }

    #[inline]
    fn step_gumap_target(
        &mut self,
        core: &mut GuMapCore,
        ws: &mut GuMapWorkspace,
    ) -> CoreResult<()> {
        step_manifold(core, ws, self)
    }
}
