use ndarray::Array1;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeIndexable};
use petgraph_algorithm_shortest_path::{
    dijkstra_with_distance_matrix, DistanceMatrix, PivotedDistanceMatrix,
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::prelude::*;
use std::collections::HashMap;

/// Implementation of the maximal-minimal distance pivot selection strategy.
///
/// This function selects pivot nodes that are maximally distant from each other,
/// which helps ensure good coverage of the graph for sparse layouts.
///
/// The strategy works as follows:
/// 1. Select a random node as the first pivot
/// 2. Compute shortest paths from this pivot to all other nodes
/// 3. Select the next pivot as the node that maximizes its minimum distance to all existing pivots
/// 4. Repeat steps 2-3 until the desired number of pivots is selected
///
/// # Parameters
/// * `graph` - The input graph to select pivots from
/// * `length` - A function that maps edges to their lengths/weights
/// * `h` - The number of pivot nodes to select
/// * `rng` - The random number generator to use
///
/// # Returns
/// A tuple containing:
/// - A vector of selected pivot node IDs
/// - A PivotedDistanceMatrix containing distances from pivots to all nodes
pub(super) fn max_min_random_sp<G, F, R, S>(
    graph: G,
    length: F,
    h: usize,
    rng: &mut R,
) -> (Vec<G::NodeId>, PivotedDistanceMatrix<G::NodeId, S>)
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex + Ord,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
    S: DrawingValue,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let nodes = graph.node_identifiers().collect::<Vec<_>>();
    let mut length = length;
    let n = indices.len();
    if n == 0 || h == 0 {
        return (vec![], PivotedDistanceMatrix::empty(graph));
    }
    let mut pivot = vec![];
    pivot.push(nodes[rng.gen_range(0..n)]);
    let mut distance_matrix = PivotedDistanceMatrix::empty(graph);
    distance_matrix.push(pivot[0]);
    dijkstra_with_distance_matrix(graph, &mut length, pivot[0], &mut distance_matrix);
    let mut min_d = Array1::from_elem(n, S::infinity());
    for k in 1..h {
        for j in 0..n {
            min_d[j] = min_d[j].min(distance_matrix.get_by_index(k - 1, j));
        }
        pivot.push(nodes[proportional_sampling(&min_d, rng)]);
        distance_matrix.push(pivot[k]);
        dijkstra_with_distance_matrix(graph, &mut length, pivot[k], &mut distance_matrix);
    }
    (pivot, distance_matrix)
}

/// Selects a node index with probability proportional to its distance value.
pub(crate) fn proportional_sampling<R, S>(values: &Array1<S>, rng: &mut R) -> usize
where
    R: Rng,
    S: DrawingValue,
{
    let n = values.len();
    assert!(n > 0, "proportional_sampling requires non-empty values");

    // On disconnected graphs, unreachable nodes have +inf distance; sample uniformly among
    // them so pivots naturally cover separate connected components without range overflow.
    let inf_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (v.is_infinite() && *v > S::zero()).then_some(i))
        .collect();
    if !inf_indices.is_empty() {
        return inf_indices[rng.gen_range(0..inf_indices.len())];
    }

    let mut s = 0.0_f32;
    let mut pos_count = 0usize;
    let mut last_pos = n - 1;
    for i in 0..n {
        if let Some(v) = values[i].to_f32() {
            if v.is_finite() && v > 0.0 {
                s += v;
                pos_count += 1;
                last_pos = i;
            }
        }
    }
    if !s.is_finite() {
        let mut pick = rng.gen_range(0..pos_count);
        for i in 0..n {
            if let Some(v) = values[i].to_f32() {
                if v.is_finite() && v > 0.0 {
                    if pick == 0 {
                        return i;
                    }
                    pick -= 1;
                }
            }
        }
        return last_pos;
    }
    if s <= 0.0 {
        return rng.gen_range(0..n);
    }
    let x = rng.gen_range(0.0..s);
    s = 0.0;
    for i in 0..n {
        if let Some(v) = values[i].to_f32() {
            if v.is_finite() && v > 0.0 {
                s += v;
                if x < s {
                    return i;
                }
            }
        }
    }
    last_pos
}
