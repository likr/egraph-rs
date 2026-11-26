//! Giovannangeli et al. (2022) FORBID node overlap removal solver.
//!
//! Implements FORBID ("FORBID: Fast Overlap Removal By stochastic gradIent Descent",
//! Giovannangeli, Lalanne, Auber, Giot & Bourqui, *Computer Graphics Forum* 2022) with
//! three selectable variants:
//! - [`ForbidVariant::Forbid`]: Cumulative layout scaling across binary-search passes.
//! - [`ForbidVariant::ForbidPrime`]: Resets each binary-search pass from uniformly scaled initial layout.
//! - [`ForbidVariant::SparseForbid`]: Uses `StaticKdTree` `sparse_k`-NN pairs plus active
//!   overlapping pairs $\mathcal{O}(X')$ for sub-quadratic scaling.

mod config;
mod init;
mod sgd;
mod solver;
#[cfg(test)]
mod tests;

pub use config::{ForbidConfig, ForbidVariant};
pub use solver::ForbidOverlapRemoval;
