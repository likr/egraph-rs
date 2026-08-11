//! Kernel matrix representation based on coordinate embeddings.

use ndarray::{Array2, Zip};
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::Kernel;
use petgraph_drawing::DrawingValue;
use std::collections::HashMap;
use std::hash::Hash;

/// A kernel matrix computed from Euclidean/spectral embedding coordinates.
///
/// K(i, j) = dot_product(embedding[i], embedding[j])
#[derive(Debug, Clone)]
pub struct EmbeddingKernel<N, S> {
    pub embedding: Array2<S>,
    pub node_indices: HashMap<N, usize>,
}

impl<N, S> EmbeddingKernel<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue,
{
    /// Creates a new EmbeddingKernel from coordinates and node mapping.
    pub fn new<G>(graph: G, embedding: Array2<S>) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();
        Self {
            embedding,
            node_indices,
        }
    }
}

impl<N, S> Kernel<N, S> for EmbeddingKernel<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let row_i = self.embedding.row(i);
        let row_j = self.embedding.row(j);

        let mut sum = S::zero();
        Zip::from(row_i).and(row_j).for_each(|&a, &b| {
            sum += a * b;
        });

        sum
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.embedding.nrows();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}
