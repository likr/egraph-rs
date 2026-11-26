//! 2-hop adaptive perplexity sizing, `f64` $\lambda$-rescaling binary search for sparse joint
//! probabilities, row-normalized symmetrization, and `VisitStamp`-backed Partial BFS.

mod config;
mod knn_prob;
mod row_normalized_12hop;
mod symmetrize;
mod two_hop;

#[cfg(test)]
mod tests;

pub use config::{
    DEFAULT_MAX_PERPLEXITY, DEFAULT_MEDIAN_MIN_PERPLEXITY, DEFAULT_MEDIAN_PERPLEXITY_C,
    DEFAULT_NODEWISE_MIN_PERPLEXITY, DEFAULT_NODEWISE_PERPLEXITY_C, PerplexityConfig,
    SparseProbabilities, SymmetrizationMode,
};
pub use knn_prob::{compute_knn_probabilities, partial_bfs_knn};
pub use row_normalized_12hop::{
    graph_12hop_row_normalized_probabilities,
    graph_12hop_row_normalized_probabilities_with_degree_decay,
    graph_12hop_row_normalized_probabilities_with_options,
};
pub use symmetrize::{graph_row_normalized_probabilities, symmetrize_row_normalized};
pub use two_hop::{
    compute_2hop_neighborhood_sizes, compute_median_2hop_perplexity,
    compute_nodewise_2hop_perplexities,
};
