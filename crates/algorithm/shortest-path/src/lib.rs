use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::{
    visit::{EdgeRef, IntoEdgeReferences, IntoEdgesDirected, IntoNodeIdentifiers},
    Incoming, Outgoing,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    f32::INFINITY,
    hash::Hash,
};

pub fn warshall_floyd<G, F>(graph: G, length: &mut F) -> Array2<f32>
where
    G: IntoEdgeReferences + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> f32,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let n = indices.len();
    let mut distance = Array::from_elem((n, n), INFINITY);

    for e in graph.edge_references() {
        let i = indices[&e.source()];
        let j = indices[&e.target()];
        let d = length(e);
        distance[[i, j]] = d;
        distance[[j, i]] = d;
    }
    for i in 0..n {
        distance[[i, i]] = 0.;
    }

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d = distance[[i, k]] + distance[[k, j]];
                if d < distance[[i, j]] {
                    distance[[i, j]] = d;
                }
            }
        }
    }

    distance
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
        let s = sources[c];
        let mut queue = BinaryHeap::new();
        queue.push((Reverse(OrderedFloat(0.)), s));
        distance_matrix[[indices[&s], c]] = 0.;
        while let Some((Reverse(OrderedFloat(d)), u)) = queue.pop() {
            for edge in graph.edges_directed(u, Outgoing) {
                let v = edge.target();
                let e = d + length(edge);
                if e < distance_matrix[[indices[&v], c]] {
                    queue.push((Reverse(OrderedFloat(e)), v));
                    distance_matrix[[indices[&v], c]] = e;
                }
            }
            for edge in graph.edges_directed(u, Incoming) {
                let v = edge.source();
                let e = d + length(edge);
                if e < distance_matrix[[indices[&v], c]] {
                    queue.push((Reverse(OrderedFloat(e)), v));
                    distance_matrix[[indices[&v], c]] = e;
                }
            }
        }
    }
    distance_matrix
}
