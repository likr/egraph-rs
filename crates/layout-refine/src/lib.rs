//! Single-threaded, pure-Rust post-layout spatial and density-field refinement solvers for `egraph-rs`.

#![warn(missing_docs)]

#[cfg(feature = "baselines")]
mod cluster_refine;
mod density_field;
mod refiner;

#[cfg(feature = "baselines")]
pub use cluster_refine::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, refine_cluster_layout,
};
pub use density_field::{ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig};
pub use refiner::{
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
