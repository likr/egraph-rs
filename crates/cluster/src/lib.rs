//! Single-threaded, pure-Rust graph clustering and coarsening primitives for `egraph-rs`.
//!
//! `egraph-clustering` provides deterministic clustering algorithms (`KMeans++`, `Spectral`,
//! `Leiden`, `Louvain`, `LabelPropagation`, and coarsening) over flat `CsrGraph` and `egraph-linalg`
//! primitives across Native Rust, Python (PyO3), and WebAssembly targets.

pub mod coarsen;
pub mod kmeans;
pub mod label_propagation;
pub mod leiden;
pub mod louvain;
pub mod spectral;

pub use coarsen::{
    CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
    align_clustered_centroids, coarsen_graph, louvain_coarsen, project_clustered_layout,
    project_clustered_layout_with_radii,
};
pub use kmeans::{KMeansConfig, KMeansResult, kmeans_plus_plus};
pub use label_propagation::{LabelPropagationConfig, label_propagation};
pub use leiden::{LeidenConfig, LeidenObjective, LeidenResult, leiden};
pub use louvain::{LouvainConfig, LouvainResult, louvain};
pub use spectral::{SpectralClusteringConfig, spectral_clustering};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn m3_scaffold_ok() {
        assert!(!crate_version().is_empty());
    }
}
