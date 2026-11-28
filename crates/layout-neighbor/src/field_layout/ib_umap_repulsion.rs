use egraph_core::{
    CoreResult, DeterministicRng, EuclideanLayout, Layout2d, LayoutStorage, Scalar, SolverStatus,
};

use crate::backends::IbFft;

use super::config::FieldFftStats;
use super::ib_umap_config::IbUmapConfig;
use super::ib_umap_step::{IbUmap2dWorkspace, step_2d, validate_2d};

/// Open pluggable repulsion backend trait for [`IbUmapSolver`](super::IbUmapSolver).
pub trait IbUmapRepulsion<L: LayoutStorage + ?Sized> {
    /// Per-solver mutable workspace state owned exclusively by this backend.
    type State: Clone;

    /// Validates backend parameters before solver construction.
    fn validate_config(&self) -> CoreResult<()>;

    /// Allocates backend-specific workspace state for `num_nodes` vertices.
    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State>;

    /// Evaluates UMAP field repulsion forces
    /// $2 b c_K \sum_{j \neq i} (x_i - x_j) / ((\varepsilon + d_{ij}^2)(1 + a d_{ij}^{2b}))$
    /// into `rep_forces` (row-major `layout.len() * dims`).
    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        a: f64,
        b: f64,
        c_k: f64,
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

/// Space-independent core state shared across all [`IbUmapSolver`](super::IbUmapSolver) targets.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IbUmapCore {
    pub(super) config: IbUmapConfig,
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) edge_weights: Vec<f32>,
    pub(super) hub_eta: Vec<f64>,
    pub(super) rng: DeterministicRng,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) status: SolverStatus,
}

#[derive(Debug, Clone)]
pub(super) enum IbUmapSpaceKind {
    Euclidean2d(IbUmap2dWorkspace),
}

/// Opaque per-space attraction/coordinate/force scratch workspace for
/// [`IbUmapSolver`](super::IbUmapSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IbUmapWorkspace(pub(super) IbUmapSpaceKind);

/// Associates each supported [`EuclideanLayout`] target with its default
/// [`IbUmapRepulsion`] backend.
pub trait DefaultIbUmapBackend: EuclideanLayout {
    /// Default repulsion backend used when `B` is omitted on [`IbUmapSolver`](super::IbUmapSolver).
    type Backend: IbUmapRepulsion<Self>;

    #[doc(hidden)]
    fn init_ib_umap_workspace(num_nodes: usize) -> IbUmapWorkspace
    where
        Self: Sized;

    #[doc(hidden)]
    fn validate_ib_umap_target(&self, core: &IbUmapCore, ws: &IbUmapWorkspace) -> CoreResult<()>;

    #[doc(hidden)]
    fn step_ib_umap_target<B: IbUmapRepulsion<Self>>(
        &mut self,
        core: &mut IbUmapCore,
        ws: &mut IbUmapWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()>;
}

impl<S: Scalar> DefaultIbUmapBackend for Layout2d<S> {
    type Backend = IbFft;

    #[inline]
    fn init_ib_umap_workspace(num_nodes: usize) -> IbUmapWorkspace {
        IbUmapWorkspace(IbUmapSpaceKind::Euclidean2d(IbUmap2dWorkspace::new(
            num_nodes,
        )))
    }

    #[inline]
    fn validate_ib_umap_target(&self, core: &IbUmapCore, _ws: &IbUmapWorkspace) -> CoreResult<()> {
        validate_2d(core, self)
    }

    #[inline]
    fn step_ib_umap_target<B: IbUmapRepulsion<Self>>(
        &mut self,
        core: &mut IbUmapCore,
        ws: &mut IbUmapWorkspace,
        backend: &B,
        backend_state: &mut B::State,
    ) -> CoreResult<()> {
        step_2d(self, core, ws, backend, backend_state)
    }
}
