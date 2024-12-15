use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers};
use std::{cmp::Reverse, collections::BinaryHeap, hash::Hash};

use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};

pub fn dijkstra_with_distance_matrix<G, S, F, D>(
    graph: G,
    length: F,
    s: G::NodeId,
    distance_matrix: &mut D,
) where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
    D: DistanceMatrix<G::NodeId, S>,
{
    let mut length = length;
    let k = distance_matrix.row_index(s).unwrap();
    let j = distance_matrix.col_index(s).unwrap();
    let mut queue = BinaryHeap::new();
    queue.push((Reverse(OrderedFloat(S::zero())), s));
    distance_matrix.set_by_index(k, j, S::zero());
    while let Some((Reverse(OrderedFloat(d)), u)) = queue.pop() {
        for edge in graph.edges(u) {
            let v = edge.target();
            let j = distance_matrix.col_index(v).unwrap();
            let e = d + length(edge);
            if e < distance_matrix.get_by_index(k, j) {
                queue.push((Reverse(OrderedFloat(e)), v));
                distance_matrix.set_by_index(k, j, e);
            }
        }
    }
}

pub fn multi_source_dijkstra<G, S, F>(
    graph: G,
    length: F,
    sources: &[G::NodeId],
) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
{
    let mut length = length;
    let mut distance_matrix = SubDistanceMatrix::new(graph, sources);
    for &u in sources.iter() {
        dijkstra_with_distance_matrix(graph, &mut length, u, &mut distance_matrix);
    }
    distance_matrix
}

pub fn all_sources_dijkstra<G, S, F>(graph: G, length: F) -> FullDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
{
    let mut length = length;
    let mut distance_matrix = FullDistanceMatrix::new(graph);
    for u in graph.node_identifiers() {
        dijkstra_with_distance_matrix(graph, &mut length, u, &mut distance_matrix);
    }
    distance_matrix
}

pub fn dijkstra<G, S, F>(graph: G, length: F, s: G::NodeId) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
{
    multi_source_dijkstra(graph, length, &[s])
}
