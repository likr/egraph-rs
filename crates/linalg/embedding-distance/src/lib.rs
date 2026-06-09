//! Distance matrix representation based on coordinate embeddings.

use ndarray::{Array2, Zip};
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::Distance;
use petgraph_drawing::DrawingValue;
use std::collections::HashMap;
use std::hash::Hash;

/// A distance matrix computed from Euclidean/spectral embedding coordinates.
///
/// d(i, j) = max(euclidean_distance(embedding[i], embedding[j]), min_dist)
#[derive(Debug, Clone)]
pub struct EmbeddingDistanceMatrix<N, S> {
    embedding: Array2<S>,
    node_indices: HashMap<N, usize>,
    min_dist: S,
}

impl<N, S> EmbeddingDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue,
{
    /// Creates a new EmbeddingDistanceMatrix from coordinates and node mapping.
    pub fn new<G>(graph: G, embedding: Array2<S>, min_dist: S) -> Self
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
            min_dist,
        }
    }
}

impl<N, S> Distance<N, S> for EmbeddingDistanceMatrix<N, S>
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
        if i == j {
            return S::zero();
        }
        let row_i = self.embedding.row(i);
        let row_j = self.embedding.row(j);

        let mut sum = S::zero();
        Zip::from(row_i).and(row_j).for_each(|&a, &b| {
            let diff = a - b;
            sum += diff * diff;
        });

        sum.sqrt().max(self.min_dist)
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
