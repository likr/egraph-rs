use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix};
use ndarray::NdFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers};
use std::hash::Hash;

pub fn warshall_floyd<G, F, S>(graph: G, length: F) -> FullDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
{
    let mut distance = FullDistanceMatrix::new(graph);
    let mut length = length;
    let n = distance.shape().0;

    for u in graph.node_identifiers() {
        for e in graph.edges(u) {
            distance.set(e.source(), e.target(), length(e));
        }
    }
    for i in 0..n {
        distance.set_by_index(i, i, S::zero());
    }

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d = distance.get_by_index(i, k) + distance.get_by_index(k, j);
                if d < distance.get_by_index(i, j) {
                    distance.set_by_index(i, j, d);
                }
            }
        }
    }

    distance
}
