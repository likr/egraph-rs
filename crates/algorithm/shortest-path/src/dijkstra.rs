use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers};
use std::{cmp::Reverse, collections::BinaryHeap, hash::Hash};

use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};

/// Computes the shortest path distances from a single source node `s` using Dijkstra's algorithm
/// and populates the corresponding row in the provided `distance_matrix`.
///
/// This function uses a binary heap for efficiency and handles weighted edges.
/// It modifies the `distance_matrix` in place.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoEdges` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
/// * `F`: The type of the function/closure used to get edge lengths.
/// * `D`: The distance matrix type, implementing `DistanceMatrix<G::NodeId, S>`.
///
/// # Arguments
///
/// * `graph`: The graph to perform Dijkstra's algorithm on.
/// * `length`: A function or closure that takes an `EdgeRef` and returns its length (`S`).
/// * `s`: The starting node ID for the algorithm.
/// * `distance_matrix`: A mutable reference to the distance matrix to be populated.
///   The distances from `s` will be written into the row corresponding to `s`.
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

/// Computes the shortest path distances from a set of source nodes using Dijkstra's algorithm.
///
/// This function runs Dijkstra's algorithm starting from each node in the `sources` slice.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoEdges` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
/// * `F`: The type of the function/closure used to get edge lengths.
///
/// # Arguments
///
/// * `graph`: The graph to perform Dijkstra's algorithm on.
/// * `length`: A function or closure that takes an `EdgeRef` and returns its length (`S`).
///   Note: This function might be called multiple times for the same edge if there are multiple sources.
/// * `sources`: A slice of node IDs representing the starting points.
///
/// # Returns
///
/// A `SubDistanceMatrix` containing the shortest path distances from each source node.
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

/// Computes the shortest path distances between all pairs of nodes using Dijkstra's algorithm.
///
/// This function runs Dijkstra's algorithm starting from every node in the graph.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoEdges` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
/// * `F`: The type of the function/closure used to get edge lengths.
///
/// # Arguments
///
/// * `graph`: The graph to perform Dijkstra's algorithm on.
/// * `length`: A function or closure that takes an `EdgeRef` and returns its length (`S`).
///   Note: This function will be called multiple times for each edge.
///
/// # Returns
///
/// A `FullDistanceMatrix` containing the shortest path distances between all pairs of nodes.
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

/// Computes the shortest path distances from a single source node `s` using Dijkstra's algorithm.
///
/// This is a convenience wrapper around `multi_source_dijkstra` for a single source node.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoEdges` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
/// * `F`: The type of the function/closure used to get edge lengths.
///
/// # Arguments
///
/// * `graph`: The graph to perform Dijkstra's algorithm on.
/// * `length`: A function or closure that takes an `EdgeRef` and returns its length (`S`).
/// * `s`: The starting node ID for the algorithm.
///
/// # Returns
///
/// A `SubDistanceMatrix` containing the shortest path distances from the source node `s`.
pub fn dijkstra<G, S, F>(graph: G, length: F, s: G::NodeId) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
    F: FnMut(G::EdgeRef) -> S,
    S: NdFloat,
{
    multi_source_dijkstra(graph, length, &[s])
}
