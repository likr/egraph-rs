//! Force-Directed Edge Bundling (`FDEB`) solver with flat coordinate buffers and
//! cooperative `step()`/`run_steps(k)` execution.
//!
//! Implements Holten & Van Wijk (2009), "Force-Directed Edge Bundling for Graph Visualization",
//! taking `&CsrGraph` and `&Layout2d` by reference (avoiding value-drop issues), clamping
//! `acos` arguments to `[-1.0, 1.0]` to eliminate collinear `NaN`s, and guarding zero-length edges.

mod compatibility;
mod config;
mod solver;
mod step;

#[cfg(test)]
mod tests;

pub use config::FdebConfig;
pub use solver::FdebSolver;

pub(super) const MIN_EDGE_LEN: f64 = 1e-12;
