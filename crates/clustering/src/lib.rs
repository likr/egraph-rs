//! Single-threaded, pure-Rust graph clustering and coarsening primitives for `egraph-rs`.
//!
//! `egraph-clustering` provides deterministic clustering algorithms (`KMeansPlusPlus`,
//! `SpectralClustering`, `Leiden`, `Louvain`, `LabelPropagation`, and coarsening) over flat
//! `CsrGraph` and `egraph-linalg` primitives across Native Rust, Python (PyO3), and
//! WebAssembly targets.

mod clusterer;
mod coarsen;
mod kmeans;
mod label_propagation;
mod leiden;
mod louvain;
mod spectral;

pub use clusterer::{ClusterResult, Clusterer, CommunityDetector};
pub use coarsen::{
    CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
    align_clustered_centroids, coarsen_graph, louvain_coarsen, project_clustered_layout,
    project_clustered_layout_with_radii,
};
pub use kmeans::{KMeansConfig, KMeansPlusPlus, KMeansResult, kmeans_plus_plus};
pub use label_propagation::{
    LabelPropagation, LabelPropagationConfig, LabelPropagationResult, label_propagation,
};
pub use leiden::{Leiden, LeidenConfig, LeidenObjective, LeidenResult, leiden};
pub use louvain::{Louvain, LouvainConfig, LouvainResult, louvain};
pub use spectral::{
    SpectralClustering, SpectralClusteringConfig, SpectralClusteringResult, spectral_clustering,
};
