use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix};
use ndarray::NdFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers};
use std::hash::Hash;

/// Computes the shortest path distances between all pairs of nodes using the Floyd-Warshall algorithm.
///
/// This algorithm is capable of handling negative edge weights, but not negative cycles.
/// If a negative cycle is detected, the distance for nodes involved in or reachable from the cycle
/// might not be correctly represented (this implementation doesn't explicitly handle negative infinity).
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoEdges` and `IntoNodeIdentifiers`.
/// * `F`: The type of the function/closure used to get edge lengths.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
///
/// # Arguments
///
/// * `graph`: The graph to compute the all-pairs shortest paths for.
/// * `length`: A function or closure that takes an `EdgeRef` and returns its length (`S`).
///
/// # Returns
///
/// A `FullDistanceMatrix` containing the shortest path distances between all pairs of nodes.
/// Distances for unreachable pairs remain infinity. Distances on negative cycles might be incorrect.
pub fn warshall_floyd<G, S, F>(graph: G, length: F) -> FullDistanceMatrix<G::NodeId, S>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Copy, // Added Copy trait bound
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
