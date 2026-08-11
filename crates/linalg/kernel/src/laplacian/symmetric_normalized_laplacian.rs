use crate::*;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::hash::Hash;

/// Symmetric Normalized Graph Laplacian: L_sym = D^{-1/2} L D^{-1/2}
#[derive(Debug, Clone, Copy, Default)]
pub struct SymmetricNormalizedLaplacian;

impl<G, S> Laplacian<G, S> for SymmetricNormalizedLaplacian
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    S: DrawingValue + Default,
{
    fn build(
        &self,
        graph: G,
        length: &mut impl FnMut(G::EdgeRef) -> S,
    ) -> SparseSymmetricMatrix<S> {
        SparseSymmetricMatrix::symmetric_normalized_laplacian(graph, length)
    }
}
