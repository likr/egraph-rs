//! Single-threaded, pure-Rust neighborhood graph, adaptive perplexity, and fuzzy simplicial set
//! primitives for `egraph-rs`.
//!
//! `egraph-neighborhood` builds on `egraph-core`, `egraph-spatial`, and `egraph-linalg` to provide
//! deterministic neighborhood construction across Native Rust, Python (PyO3), and WebAssembly targets:
//! - [`knn`]: Validated CSR [`KnnGraph`], single-pass `VisitStamp` 1+2-hop extraction,
//!   hub-safe 2-hop pair sampling, and GuMap 1+2-hop priority CSR construction.
//! - [`perplexity`]: 2-hop adaptive perplexity sizing, `f64` $\lambda$-rescaling binary search,
//!   row-normalized symmetrization, and `VisitStamp`-backed Partial BFS.
//! - [`rerank`]: Spectral resistance/diffusion embeddings, 2-stage coarse `StaticKdTree` +
//!   full-$D$ Euclidean reranking, and hybrid adaptive $k$-NN with strict margin separation.
//! - [`fuzzy`]: UMAP / GuMap fuzzy simplicial sets and `f64` Levenberg-Marquardt `(a, b)` fitting.
//! - [`varadhan`]: R1/R2 regularized DoG Varadhan landmarks, coarse heat scaffolds, R3
//!   Horvitz-Thompson repulsion scaling, ALT lower-bound queries, and relative distance floors.

pub mod fuzzy;
pub mod knn;
pub mod perplexity;
pub mod rerank;
pub mod varadhan;

pub use fuzzy::{compute_fuzzy_simplicial_set, find_ab_params};
pub use knn::{
    KnnGraph, KnnStats, Sparse12HopPairs, build_capped_12hop_knn, build_gumap_12hop_priority_csr,
    build_pure_2hop_knn, extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs,
};
pub use perplexity::{
    DEFAULT_MAX_PERPLEXITY, DEFAULT_MEDIAN_MIN_PERPLEXITY, DEFAULT_MEDIAN_PERPLEXITY_C,
    DEFAULT_NODEWISE_MIN_PERPLEXITY, DEFAULT_NODEWISE_PERPLEXITY_C, PerplexityConfig,
    SparseProbabilities, SymmetrizationMode, compute_2hop_neighborhood_sizes,
    compute_knn_probabilities, compute_median_2hop_perplexity, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities,
    graph_12hop_row_normalized_probabilities_with_degree_decay, graph_row_normalized_probabilities,
    partial_bfs_knn, symmetrize_row_normalized,
};
pub use rerank::{
    HybridKnnConfig, build_hybrid_adaptive_knn, build_rerank_knn, compute_diffusion_embedding,
    compute_resistance_embedding,
};
pub use varadhan::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, RhghLandmarkConfig,
    RhghLandmarkStats, apply_relative_distance_floor, compute_horvitz_thompson_gamma,
    compute_pivoted_rhgh_landmarks, compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix,
};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
