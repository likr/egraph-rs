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

/// Standard Graph Laplacian: L = D - A
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardLaplacian;

impl<G, S> Laplacian<G, S> for StandardLaplacian
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
        SparseSymmetricMatrix::standard_laplacian(graph, length)
    }
}
