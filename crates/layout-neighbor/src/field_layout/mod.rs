//! Field-based graph layout solvers (`t-FDP` and `ibUMAP`) built on interpolation FFT,
//! Barnes-Hut spatial trees, and exact $O(N^2)$ repulsion references.

mod config;
mod ib_umap;
mod ib_umap_config;
mod ib_umap_step;
mod quadtree;
mod repulsion;
mod repulsion_impls;
mod repulsion_wrappers;
mod t_fdp;
mod t_fdp_edges;
mod t_fdp_step;
mod t_fdp_step3d;
#[cfg(test)]
mod tests;

pub use config::{FieldFftStats, FieldRepulsionBackend, RhoSchedule, TFdpConfig};
pub use ib_umap::{DEFAULT_MAX_2HOP_PER_NODE, IbUmapSolver};
pub use ib_umap_config::IbUmapConfig;
pub use repulsion::{DefaultTFdpBackend, TFdpRepulsion};
#[doc(hidden)]
pub use repulsion::{TFdpCore, TFdpWorkspace};
#[doc(hidden)]
pub use repulsion_impls::{
    ExactTFdpState, TFdpBarnesHut2dState, TFdpBarnesHut3dState, TFdpIbFft2dState,
};
#[doc(hidden)]
pub use repulsion_wrappers::{TFdpBackend2dState, TFdpBackend3dState};
pub use t_fdp::TFdpSolver;
