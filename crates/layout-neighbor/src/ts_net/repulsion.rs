use egraph_core::{
    CoreResult, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldLayout2d, RiemannianLayout,
    Scalar, SolverStatus,
};
use egraph_spatial_fft::FitFieldStats;

use crate::backends::{Auto2d, Auto3d, Exact};

use super::config::TsNetConfig;
use super::dim3::{TsNet3dWorkspace, step_3d, validate_3d};
use super::manifold::{TsNetManifoldWorkspace, step_manifold, validate_manifold};
use super::nd::{TsNetNdWorkspace, step_nd, validate_nd};
use super::riemannian::{TsNetRiemannianWorkspace, step_riemannian, validate_riemannian};
use super::step2d::{TsNet2dWorkspace, step_2d, validate_2d};

/// Open pluggable repulsion backend trait for [`TsNetSolver`](super::TsNetSolver).
///
/// External crates (such as custom surface or manifold layout modules) can implement
/// `TsNetRepulsion<MyLayout>` for custom accelerators (e.g., geodesic Barnes-Hut) without
/// modifying `egraph-layout-neighbor`.
pub trait TsNetRepulsion<L: LayoutStorage + ?Sized> {
    /// Per-solver mutable workspace state owned exclusively by this backend.
    type State: Clone;

    /// Validates backend parameters before solver construction.
    fn validate_config(&self) -> CoreResult<()>;

    /// Allocates backend-specific workspace state for `num_nodes` vertices.
    fn init_state(
        &self,
        num_nodes: usize,
        epsilon_r: f64,
        any_entropy: bool,
        init_entropy: bool,
    ) -> CoreResult<Self::State>;

    /// Notifies the backend when entering an optimization stage whose `compute_entropy` flag
    /// may have changed.
    #[inline]
    fn on_stage_enter(
        &self,
        _state: &mut Self::State,
        _epsilon_r: f64,
        _compute_entropy: bool,
    ) -> CoreResult<()> {
        Ok(())
    }

    /// Evaluates KL repulsion $(1/Z) \sum_{j \neq i} (x_i - x_j) / (1 + d_{ij}^2)^2$ into
    /// `rep_forces` and (when `compute_entropy` is true) unnormalized entropy repulsion
    /// $\sum_{j \neq i} (x_i - x_j) / (d_{ij} (d_{ij} + \varepsilon_r))$ into `ent_forces`.
    ///
    /// Both slices are row-major with length `layout.len() * tangent_dim`.
    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()>;

    /// Returns telemetry counters from the most recent FFT field evaluation, if applicable.
    #[inline]
    fn last_fft_stats<'a>(&self, _state: &'a Self::State) -> Option<&'a FitFieldStats> {
        None
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    fn total_fft_2d_calls(&self, _state: &Self::State) -> u64 {
        0
    }
}

/// Space-independent core state shared across all [`TsNetSolver`](super::TsNetSolver) targets.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TsNetCore {
    pub(super) config: TsNetConfig,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) p_vals: Vec<f64>,
    pub(super) inv_jacobi: Vec<f64>,
    pub(super) status: SolverStatus,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) stage_idx: usize,
    pub(super) stage_iter: usize,
    pub(super) stage_entered: bool,
    pub(super) prev_stage_lr: f64,
    pub(super) active_stage_lr: f64,
}

impl TsNetCore {
    pub(super) fn advance_stage_counters(&mut self) {
        self.iteration += 1;
        self.stage_iter += 1;
        if self.stage_iter < self.config.stages[self.stage_idx].iterations {
            self.status = SolverStatus::Running;
            return;
        }

        self.stage_iter = 0;
        self.stage_entered = false;
        if let Some(rel_next) = self.config.stages[self.stage_idx + 1..]
            .iter()
            .position(|s| s.iterations > 0)
        {
            self.stage_idx += 1 + rel_next;
            self.status = SolverStatus::Running;
        } else {
            self.status = SolverStatus::Completed;
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum TsNetSpaceKind {
    Euclidean2d(TsNet2dWorkspace),
    Euclidean3d(TsNet3dWorkspace),
    EuclideanNd(TsNetNdWorkspace),
    Manifold2d(TsNetManifoldWorkspace),
    Riemannian(TsNetRiemannianWorkspace),
}

/// Opaque per-space attraction/velocity/force scratch workspace for [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TsNetWorkspace(pub(super) TsNetSpaceKind);

impl TsNetWorkspace {
    #[cfg(test)]
    pub(super) fn vel_x_mut(&mut self) -> Option<&mut [f64]> {
        match &mut self.0 {
            TsNetSpaceKind::Euclidean2d(w) => Some(&mut w.vel_x),
            TsNetSpaceKind::Manifold2d(w) => Some(&mut w.vel_x),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(super) fn vel_x(&self) -> Option<&[f64]> {
        match &self.0 {
            TsNetSpaceKind::Euclidean2d(w) => Some(&w.vel_x),
            TsNetSpaceKind::Manifold2d(w) => Some(&w.vel_x),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(super) fn vel_y(&self) -> Option<&[f64]> {
        match &self.0 {
            TsNetSpaceKind::Euclidean2d(w) => Some(&w.vel_y),
            TsNetSpaceKind::Manifold2d(w) => Some(&w.vel_y),
            _ => None,
        }
    }
}

/// Associates each target layout space with its default [`TsNetRepulsion`] backend.
///
/// This trait is open so external [`RiemannianLayout`](egraph_core::RiemannianLayout)
/// implementations can declare a default backend (such as [`Exact`] or a custom accelerator).
pub trait DefaultTsNetBackend: LayoutStorage {
    /// Default repulsion backend used when `B` is omitted on [`TsNetSolver`](super::TsNetSolver).
    type Backend: TsNetRepulsion<Self>;

    #[doc(hidden)]
    fn init_tsnet_workspace(num_nodes: usize, any_entropy: bool) -> TsNetWorkspace
    where
        Self: Sized,
    {
        TsNetWorkspace(TsNetSpaceKind::Riemannian(TsNetRiemannianWorkspace::new(
            num_nodes,
            any_entropy,
        )))
    }

    #[doc(hidden)]
    fn validate_tsnet_target(&self, core: &TsNetCore, ws: &TsNetWorkspace) -> CoreResult<()>
    where
        Self: RiemannianLayout,
    {
        validate_riemannian(core, ws, self)
    }

    #[doc(hidden)]
    fn step_tsnet_target<B: TsNetRepulsion<Self>>(
        &mut self,
        core: &mut TsNetCore,
        ws: &mut TsNetWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()>
    where
        Self: RiemannianLayout,
    {
        step_riemannian(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTsNetBackend for Layout2d<S> {
    type Backend = Auto2d;

    #[inline]
    fn init_tsnet_workspace(num_nodes: usize, any_entropy: bool) -> TsNetWorkspace {
        TsNetWorkspace(TsNetSpaceKind::Euclidean2d(TsNet2dWorkspace::new(
            num_nodes,
            any_entropy,
        )))
    }

    #[inline]
    fn validate_tsnet_target(&self, core: &TsNetCore, _ws: &TsNetWorkspace) -> CoreResult<()> {
        validate_2d(core, self)
    }

    #[inline]
    fn step_tsnet_target<B: TsNetRepulsion<Self>>(
        &mut self,
        core: &mut TsNetCore,
        ws: &mut TsNetWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_2d(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTsNetBackend for Layout3d<S> {
    type Backend = Auto3d;

    #[inline]
    fn init_tsnet_workspace(num_nodes: usize, any_entropy: bool) -> TsNetWorkspace {
        TsNetWorkspace(TsNetSpaceKind::Euclidean3d(TsNet3dWorkspace::new(
            num_nodes,
            any_entropy,
        )))
    }

    #[inline]
    fn validate_tsnet_target(&self, core: &TsNetCore, _ws: &TsNetWorkspace) -> CoreResult<()> {
        validate_3d(core, self)
    }

    #[inline]
    fn step_tsnet_target<B: TsNetRepulsion<Self>>(
        &mut self,
        core: &mut TsNetCore,
        ws: &mut TsNetWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_3d(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTsNetBackend for LayoutNd<S> {
    type Backend = Exact;

    #[inline]
    fn init_tsnet_workspace(num_nodes: usize, any_entropy: bool) -> TsNetWorkspace {
        let _ = num_nodes;
        TsNetWorkspace(TsNetSpaceKind::EuclideanNd(TsNetNdWorkspace::new(
            any_entropy,
        )))
    }

    #[inline]
    fn validate_tsnet_target(&self, core: &TsNetCore, ws: &TsNetWorkspace) -> CoreResult<()> {
        validate_nd(core, ws, self)
    }

    #[inline]
    fn step_tsnet_target<B: TsNetRepulsion<Self>>(
        &mut self,
        core: &mut TsNetCore,
        ws: &mut TsNetWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_nd(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTsNetBackend for ManifoldLayout2d<S> {
    type Backend = Exact;

    #[inline]
    fn init_tsnet_workspace(num_nodes: usize, any_entropy: bool) -> TsNetWorkspace {
        TsNetWorkspace(TsNetSpaceKind::Manifold2d(TsNetManifoldWorkspace::new(
            num_nodes,
            any_entropy,
        )))
    }

    #[inline]
    fn validate_tsnet_target(&self, core: &TsNetCore, ws: &TsNetWorkspace) -> CoreResult<()> {
        validate_manifold(core, ws, self)
    }

    #[inline]
    fn step_tsnet_target<B: TsNetRepulsion<Self>>(
        &mut self,
        core: &mut TsNetCore,
        ws: &mut TsNetWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_manifold(self, core, ws, backend, backend_state)
    }
}
