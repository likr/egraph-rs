//! Single-threaded, pure-Rust sampling-based neighborhood graph embedding solvers for `egraph-rs`.
//!
//! Provides five independent cooperative [`IterativeSolver`](egraph_core::IterativeSolver)
//! implementations over any [`RiemannianLayout`](egraph_core::RiemannianLayout) target
//! (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, or custom surfaces):
//! - [`PacmapGraphSolver`]: 3-tier (`NB`, `MN`, `FP`) `PaCMAP-Graph` with rotation-invariant
//!   Riemannian Adam and parallel transport.
//! - [`LargeVisSolver`]: Vose-alias negative-sampling `LargeVis`.
//! - [`DrGraphSolver`]: multilevel negative-sampling `DRGraph`.
//! - [`GraphCneSolver`]: contrastive neighbor embedding (`InfoNce` / `NegTSne`).
//! - [`SnapTFdpSolver`]: compact-memory edge-incident negative-sampling `SNAP-tFDP`.

#![warn(missing_docs)]

mod config_negative;
mod config_pacmap_snap;
mod drgraph;
mod drgraph_coarsen;
mod graph_cne;
mod largevis;
mod pacmap;
mod pacmap_build;
mod pacmap_pairs;
mod sampling_utils;
mod snap_t_fdp;
mod snap_t_fdp_edges;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests_negative_sampling;
#[cfg(test)]
mod tests_pacmap_basic;
#[cfg(test)]
mod tests_pacmap_invariance;
#[cfg(test)]
mod tests_pacmap_riemannian;
#[cfg(test)]
mod tests_pacmap_topology;
#[cfg(test)]
mod tests_snap_t_fdp;

pub use config_negative::{DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig};
pub use config_pacmap_snap::{PacmapGraphConfig, SnapTFdpConfig};
pub use drgraph::DrGraphSolver;
pub use graph_cne::GraphCneSolver;
pub use largevis::LargeVisSolver;
pub use pacmap::PacmapGraphSolver;
pub use snap_t_fdp::SnapTFdpSolver;
