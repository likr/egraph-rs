//! Force-Directed Edge Bundling (`FDEB`) solver over [`egraph_core::EdgePolylines`].
//!
//! Implements Holten & Van Wijk (2009), "Force-Directed Edge Bundling for Graph Visualization",
//! with cooperative [`egraph_core::IterativeSolver`] execution across `Layout2d`, `Layout3d`, and
//! `LayoutNd` coordinate buffers.

mod compatibility;
mod config;
mod solver;
mod step;

#[cfg(test)]
mod tests;

pub use config::FdebConfig;
pub use solver::FdebSolver;

pub(super) const MIN_EDGE_LEN: f64 = 1e-12;
