//! Pure-Rust Structure-of-Arrays (SoA) `u32` SGD solver, 3-tier sparse barrier builder,
//! streaming non-edge negative sampling, and non-Euclidean [`egraph_core::ManifoldLayout2d`]
//! updates.

mod config;
mod drgraph;
mod drgraph_coarsen;
mod graph_cne;
mod graph_sampling;
mod largevis;
mod manifold_drgraph;
mod manifold_sampling_variants;
mod manifold_variants;
mod nd;
mod nd_variants;
mod ortmann;
mod ortmann_pivots;
mod pacmap;
mod pacmap_pairs;
mod pair_constructors;
mod pairs;
mod sgd2;
mod snap_t_fdp;
mod solver;
mod sparse_barrier;
mod stepping;
mod update;
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
mod tests_negative_sampling;
#[cfg(test)]
mod tests_ortmann;
#[cfg(test)]
mod tests_pacmap_basic;
#[cfg(test)]
mod tests_pacmap_invariance;
#[cfg(test)]
mod tests_pacmap_topology;
#[cfg(test)]
mod tests_sgd2_and_configs;
#[cfg(test)]
mod tests_snap_t_fdp;

pub use config::{
    BarrierSymmetry, PacmapGraphConfig, Sgd2Config, SgdScheduleConfig, SgdScheduleKind,
    SnapTFdpConfig, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
};
#[doc(hidden)]
pub use pairs::PackedSymPair;
pub use solver::SgdSolver;
pub use stepping::SgdScheduleWindowLayout;
pub use variant_config::{
    DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig, OrtmannPivotStrategy,
    OrtmannSparseStressConfig,
};
