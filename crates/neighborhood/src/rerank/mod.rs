//! Spectral resistance/diffusion embeddings, 2-stage coarse `StaticKdTree` + full-$D$ Euclidean
//! reranking $k$-NN, and hybrid 1+2-hop + adaptive residual $k$-NN construction.

mod embeddings;
mod hybrid;
mod kdtree_rerank;

#[cfg(test)]
mod tests;

pub use embeddings::{compute_diffusion_embedding, compute_resistance_embedding};
pub use hybrid::{HybridKnnConfig, build_hybrid_adaptive_knn};
pub use kdtree_rerank::build_rerank_knn;
