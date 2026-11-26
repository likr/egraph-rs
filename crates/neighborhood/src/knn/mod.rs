//! Compressed Sparse Row $k$-NN storage (`KnnGraph`), single-pass `VisitStamp` 1+2-hop
//! extraction, hub-safe pair sampling, and 1+2-hop priority CSR construction.

mod capped_2hop;
mod graph;
mod priority_csr;
mod two_hop;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_random_walk;

#[cfg(test)]
pub(crate) use capped_2hop::extract_sparse_1hop_2hop_pairs_inner;
pub use capped_2hop::{
    TwoHopSamplingMode, build_capped_12hop_knn, extract_sparse_1hop_2hop_pairs,
    extract_sparse_1hop_2hop_pairs_with_mode,
};
pub use graph::{KnnGraph, KnnStats, Sparse12HopPairs};
pub use priority_csr::build_gumap_12hop_priority_csr;
#[allow(unused_imports)]
pub(crate) use priority_csr::cmp_dist_then_id;
pub(crate) use two_hop::canonical_unweighted_1hop;
pub use two_hop::{build_pure_2hop_knn, extract_1hop_2hop_csr};
