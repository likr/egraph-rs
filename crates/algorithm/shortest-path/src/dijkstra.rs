use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::{
    visit::{EdgeRef, IntoEdgesDirected, IntoNodeIdentifiers},
    Incoming, Outgoing,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    f32::INFINITY,
    hash::Hash,
};

pub fn dijkstra_with_distance_matrix<G, F>(
    graph: G,
    indices: &HashMap<G::NodeId, usize>,
    length: &mut F,
    s: G::NodeId,
    distance_matrix: &mut Array2<f32>,
    k: usize,
) where
    G: IntoEdgesDirected + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> f32,
{
    let mut queue = BinaryHeap::new();
    queue.push((Reverse(OrderedFloat(0.)), s));
    distance_matrix[[indices[&s], k]] = 0.;
    while let Some((Reverse(OrderedFloat(d)), u)) = queue.pop() {
        for edge in graph.edges_directed(u, Outgoing) {
            let v = edge.target();
            let e = d + length(edge);
            if e < distance_matrix[[indices[&v], k]] {
                queue.push((Reverse(OrderedFloat(e)), v));
                distance_matrix[[indices[&v], k]] = e;
            }
        }
        for edge in graph.edges_directed(u, Incoming) {
            let v = edge.source();
            let e = d + length(edge);
            if e < distance_matrix[[indices[&v], k]] {
                queue.push((Reverse(OrderedFloat(e)), v));
                distance_matrix[[indices[&v], k]] = e;
            }
        }
    }
}

pub fn multi_source_dijkstra<G, F>(graph: G, length: &mut F, sources: &[G::NodeId]) -> Array2<f32>
where
    G: IntoEdgesDirected + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> f32,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let n = indices.len();
    let k = sources.len();
    let mut distance_matrix = Array::from_elem((n, k), INFINITY);
    for c in 0..k {
        dijkstra_with_distance_matrix(graph, &indices, length, sources[c], &mut distance_matrix, c);
    }
    distance_matrix
}

pub fn all_sources_dijkstra<G, F>(graph: G, length: &mut F) -> Array2<f32>
where
    G: IntoEdgesDirected + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> f32,
{
    let sources = graph.node_identifiers().collect::<Vec<_>>();
    multi_source_dijkstra(graph, length, &sources)
}

pub fn dijkstra<G, F>(graph: G, length: &mut F, s: G::NodeId) -> Array1<f32>
where
    G: IntoEdgesDirected + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> f32,
{
    let distance_matrix = multi_source_dijkstra(graph, length, &[s]);
    let n = distance_matrix.shape()[0];
    distance_matrix.into_shape(n).unwrap()
}
