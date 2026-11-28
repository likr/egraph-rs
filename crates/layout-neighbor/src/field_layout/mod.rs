//! Field-based graph layout solvers (`t-FDP` and `ibUMAP`) built on interpolation FFT,
//! Barnes-Hut spatial trees, and exact $O(N^2)$ repulsion references.

mod config;
mod ib_umap;
mod ib_umap_config;
mod ib_umap_step;
mod quadtree;
mod t_fdp;
mod t_fdp_dispatch;
mod t_fdp_step;
mod t_fdp_step3d;
#[cfg(test)]
mod tests;

pub use config::{FieldFftStats, FieldRepulsionBackend, RhoSchedule, TFdpConfig};
pub use ib_umap::{DEFAULT_MAX_2HOP_PER_NODE, IbUmapSolver};
pub use ib_umap_config::IbUmapConfig;
pub use t_fdp::TFdpSolver;
