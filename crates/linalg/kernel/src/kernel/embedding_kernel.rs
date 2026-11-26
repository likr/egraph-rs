use crate::*;
use ndarray::Zip;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::collections::HashMap;
use std::hash::Hash;

/// A kernel matrix computed from Euclidean/spectral embedding coordinates.
///
/// `K(i, j) = dot_product(embedding[i], embedding[j])`
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
    ///
    /// Panics if `embedding.nrows()` does not match the number of nodes in `graph`.
    pub fn new<G>(graph: G, embedding: Array2<S>) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        Self::try_new(graph, embedding).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Creates a new EmbeddingKernel, returning an error if `embedding.nrows()` does not match
    /// the graph node count.
    pub fn try_new<G>(graph: G, embedding: Array2<S>) -> Result<Self, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();
        if embedding.nrows() != node_indices.len() {
            return Err(format!(
                "Embedding row count ({}) must match graph node count ({})",
                embedding.nrows(),
                node_indices.len()
            ));
        }
        Ok(Self {
            embedding,
            node_indices,
        })
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
