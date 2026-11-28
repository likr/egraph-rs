//! Pure-Rust Structure-of-Arrays (SoA) `u32` SGD stress and sparse barrier solvers,
//! streaming non-edge negative sampling, diagonal/tangent-Hessian preconditioned SGD (`Sgd2Solver`),
//! and generic [`egraph_core::RiemannianLayout`] stepping.

mod config;
mod graph_helpers;
mod ortmann;
mod ortmann_pivots;
mod pair_buffer;
mod pair_constructors;
mod pairs;
mod sgd2;
mod sgd2_kernels;
mod solver;
mod sparse_barrier;
mod stepping;
mod variant_config;
mod variant_constructors;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests_barrier_streaming;
#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_errors_and_blocking;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_manifold_modes;
#[cfg(test)]
mod tests_nd;
#[cfg(test)]
mod tests_nd_edge_cases;
#[cfg(test)]
mod tests_ortmann;
#[cfg(test)]
mod tests_sgd2_and_configs;

pub use config::{
    BarrierSymmetry, SgdScheduleConfig, SgdScheduleKind, SparseBarrierConfig, SparseBarrierStats,
    StreamingNegativeConfig,
};
pub use ortmann::{OrtmannSparseProblem, build_ortmann_sparse_problem};
pub use pair_buffer::{SgdPairBuffer, SgdPairKind, SgdPairRecord};
pub use sgd2::Sgd2Solver;
pub use solver::SgdSolver;
pub use variant_config::{OrtmannPivotStrategy, OrtmannSparseStressConfig, Sgd2Config};
