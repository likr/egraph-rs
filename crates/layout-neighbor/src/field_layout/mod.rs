//! Field-based graph layout solvers (`t-FDP` and `ibUMAP`) built on interpolation FFT,
//! Barnes-Hut quadtrees, and exact $O(N^2)$ repulsion references.

mod config;
mod ib_umap;
mod quadtree;
mod solver;
mod step;
mod t_fdp;
#[cfg(test)]
mod tests;

pub use config::{FieldRepulsionBackend, IbUmapConfig, RhoSchedule, TFdpConfig};
pub use ib_umap::DEFAULT_MAX_2HOP_PER_NODE;
pub use solver::{FieldFftStats, FieldLayout, FieldLayoutSolver};
