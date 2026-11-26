//! Graph coarsening and multilevel layout uncoarsening/projection primitives.

mod config;
mod contract;
mod hierarchy;
mod project;

#[cfg(test)]
mod tests;

pub use config::{
    CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
};
pub use contract::coarsen_graph;
pub use hierarchy::louvain_coarsen;
pub use project::{
    align_clustered_centroids, project_clustered_layout, project_clustered_layout_with_radii,
};
