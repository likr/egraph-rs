use egraph_core::{
    CoreResult, DeterministicRng, EuclideanLayout, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    Scalar, SolverStatus,
};

use crate::backends::{BarnesHut, Exact, IbFft};

use super::config::{FieldFftStats, TFdpConfig};
use super::t_fdp_step::{
    TFdp2dWorkspace, TFdpNdWorkspace, step_2d, step_nd, validate_2d, validate_nd,
};
use super::t_fdp_step3d::{TFdp3dWorkspace, step_3d, validate_3d};

/// Open pluggable repulsion backend trait for [`TFdpSolver`](super::TFdpSolver).
pub trait TFdpRepulsion<L: LayoutStorage + ?Sized> {
    /// Per-solver mutable workspace state owned exclusively by this backend.
    type State: Clone;

    /// Validates backend parameters before solver construction.
    fn validate_config(&self) -> CoreResult<()>;

    /// Allocates backend-specific workspace state for `num_nodes` vertices.
    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State>;

    /// Evaluates unnormalized Student-$t$ repulsion forces
    /// $\sum_{j \neq i} (x_i - x_j) / (1 + d_{ij}^2)^\gamma$ into `rep_forces`
    /// (row-major `layout.len() * dims`).
    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()>;

    /// Returns telemetry counters from the most recent `IbFft` step, if applicable.
    #[inline]
    fn last_fft_stats<'a>(&self, _state: &'a Self::State) -> Option<&'a FieldFftStats> {
        None
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    fn total_fft_2d_calls(&self, _state: &Self::State) -> u64 {
        0
    }
}

/// Space-independent core state shared across all [`TFdpSolver`](super::TFdpSolver) targets.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpCore {
    pub(super) config: TFdpConfig,
    pub(super) rep_scale: f64,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) edge_weights: Vec<f32>,
    pub(super) rng: DeterministicRng,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) status: SolverStatus,
}

impl TFdpCore {
    #[inline]
    pub(super) fn current_rho(&self) -> f64 {
        let progress = if self.total_iterations <= 1 {
            if self.iteration >= self.total_iterations {
                1.0
            } else {
                0.0
            }
        } else {
            (self.iteration as f64) / ((self.total_iterations - 1) as f64)
        };
        self.config.rho.evaluate(progress)
    }
}

#[derive(Debug, Clone)]
pub(super) enum TFdpSpaceKind {
    Euclidean2d(TFdp2dWorkspace),
    Euclidean3d(TFdp3dWorkspace),
    EuclideanNd(TFdpNdWorkspace),
}

/// Opaque per-space attraction/velocity/force scratch workspace for [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpWorkspace(pub(super) TFdpSpaceKind);

impl TFdpWorkspace {
    #[cfg(test)]
    pub(super) fn capacities(&self) -> (usize, usize, usize, usize) {
        match &self.0 {
            TFdpSpaceKind::Euclidean2d(w) => (
                w.attr_x.capacity(),
                w.vel_x.capacity(),
                w.next_x.capacity(),
                w.rep.capacity(),
            ),
            TFdpSpaceKind::Euclidean3d(w) => (
                w.attr_x.capacity(),
                w.vel_x.capacity(),
                w.next_x.capacity(),
                w.rep.capacity(),
            ),
            TFdpSpaceKind::EuclideanNd(w) => (
                w.attr.capacity(),
                w.vel.capacity(),
                w.next.capacity(),
                w.rep.capacity(),
            ),
        }
    }
}

/// Associates each [`EuclideanLayout`] target with its default [`TFdpRepulsion`] backend.
pub trait DefaultTFdpBackend: EuclideanLayout {
    /// Default repulsion backend used when `B` is omitted on [`TFdpSolver`](super::TFdpSolver).
    type Backend: TFdpRepulsion<Self>;

    #[doc(hidden)]
    fn init_tfdp_workspace(num_nodes: usize) -> TFdpWorkspace
    where
        Self: Sized;

    #[doc(hidden)]
    fn validate_tfdp_target(&self, core: &TFdpCore, ws: &TFdpWorkspace) -> CoreResult<()>;

    #[doc(hidden)]
    fn step_tfdp_target<B: TFdpRepulsion<Self>>(
        &mut self,
        core: &mut TFdpCore,
        ws: &mut TFdpWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()>;
}

impl<S: Scalar> DefaultTFdpBackend for Layout2d<S> {
    type Backend = IbFft;

    #[inline]
    fn init_tfdp_workspace(num_nodes: usize) -> TFdpWorkspace {
        TFdpWorkspace(TFdpSpaceKind::Euclidean2d(TFdp2dWorkspace::new(num_nodes)))
    }

    #[inline]
    fn validate_tfdp_target(&self, core: &TFdpCore, _ws: &TFdpWorkspace) -> CoreResult<()> {
        validate_2d(core, self)
    }

    #[inline]
    fn step_tfdp_target<B: TFdpRepulsion<Self>>(
        &mut self,
        core: &mut TFdpCore,
        ws: &mut TFdpWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_2d(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTFdpBackend for Layout3d<S> {
    type Backend = BarnesHut;

    #[inline]
    fn init_tfdp_workspace(num_nodes: usize) -> TFdpWorkspace {
        TFdpWorkspace(TFdpSpaceKind::Euclidean3d(TFdp3dWorkspace::new(num_nodes)))
    }

    #[inline]
    fn validate_tfdp_target(&self, core: &TFdpCore, _ws: &TFdpWorkspace) -> CoreResult<()> {
        validate_3d(core, self)
    }

    #[inline]
    fn step_tfdp_target<B: TFdpRepulsion<Self>>(
        &mut self,
        core: &mut TFdpCore,
        ws: &mut TFdpWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_3d(self, core, ws, backend, backend_state)
    }
}

impl<S: Scalar> DefaultTFdpBackend for LayoutNd<S> {
    type Backend = Exact;

    #[inline]
    fn init_tfdp_workspace(_num_nodes: usize) -> TFdpWorkspace {
        TFdpWorkspace(TFdpSpaceKind::EuclideanNd(TFdpNdWorkspace::new()))
    }

    #[inline]
    fn validate_tfdp_target(&self, core: &TFdpCore, ws: &TFdpWorkspace) -> CoreResult<()> {
        validate_nd(core, ws, self)
    }

    #[inline]
    fn step_tfdp_target<B: TFdpRepulsion<Self>>(
        &mut self,
        core: &mut TFdpCore,
        ws: &mut TFdpWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_nd(self, core, ws, backend, backend_state)
    }
}
