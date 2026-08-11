//! Omega implementation for creating SGD instances from spectral embeddings.

use ndarray::Array2;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingValue;
use petgraph_layout_sgd::{RandomPairSparseSgd, Sgd};
use petgraph_linalg_kernel::EmbeddingKernel;
use petgraph_linalg_kernel::KernelDistance;
use rand::Rng;

/// Omega builder for creating SGD instances from spectral embeddings.
#[derive(Debug, Clone)]
pub struct Omega<S> {
    /// Number of random pairs per node  
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
}

impl<S> Omega<S>
where
    S: DrawingValue,
{
    /// Creates a new Omega with default values.
    pub fn new() -> Self {
        Self {
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
        }
    }

    /// Sets the number of random pairs per node.
    pub fn k(&mut self, k: usize) -> &mut Self {
        self.k = k;
        self
    }

    /// Sets the minimum distance between node pairs.
    pub fn min_dist(&mut self, min_dist: S) -> &mut Self {
        self.min_dist = min_dist;
        self
    }

    /// Builds an SGD instance from precomputed embedding.
    pub fn build<G, R>(&self, graph: G, embedding: &Array2<S>, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: petgraph_drawing::DrawingIndex + std::hash::Hash + Eq,
        R: Rng,
    {
        let kernel = EmbeddingKernel::new(graph, embedding.clone());
        let distance_matrix = KernelDistance::new(kernel).min_dist(self.min_dist);
        let mut random_pair_sgd = RandomPairSparseSgd::new();
        random_pair_sgd.k(self.k);
        random_pair_sgd.build(graph, &distance_matrix, rng)
    }
}

impl<S> Default for Omega<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}
