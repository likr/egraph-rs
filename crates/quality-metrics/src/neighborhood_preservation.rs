use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNeighbors, NodeIndexable};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex};
use std::collections::HashSet;

/// Calculates the neighborhood preservation metric for a graph layout.
///
/// This metric assesses how well the layout preserves local neighborhoods from
/// the original graph structure. It calculates the ratio of nodes that are both
/// graph-theoretical neighbors and spatial neighbors in the layout.
///
/// The implementation works by:
/// 1. Identifying all edges in the graph
/// 2. For each node, finding its k nearest neighbors in the layout (where k is its degree in the graph)
/// 3. Calculating the ratio of neighbors that are preserved (i.e., are both connected in the
///    graph and close in the layout)
///
/// A higher value indicates better preservation of the graph's neighborhood structure.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `f32` value in the range [0, 1] representing the neighborhood preservation metric.
/// A value of 1 indicates perfect preservation of neighborhoods, while 0 indicates
/// no preservation.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn neighborhood_preservation<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences + IntoNeighbors + NodeIndexable,
    G::NodeId: DrawingIndex,
{
    let mut graph_edges = HashSet::new();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        graph_edges.insert((graph.to_index(u), graph.to_index(v)));
        graph_edges.insert((graph.to_index(v), graph.to_index(u)));
    }

    let n = drawing.len();
    let mut points = Array2::zeros((n, 2));
    for i in 0..n {
        points[[i, 0]] = drawing.raw_entry(i).0;
        points[[i, 1]] = drawing.raw_entry(i).1;
    }
    let nn = BallTree::new().from_batch(&points, L2Dist).unwrap();

    let mut cap = 0;
    let mut cup = graph_edges.len();
    for i in 0..n {
        let u = *drawing.node_id(i);
        let x = drawing.raw_entry(i).0;
        let y = drawing.raw_entry(i).1;
        let d = graph.neighbors(u).count();
        let query = arr1(&[x, y]);
        let neighbors = nn.k_nearest(query.view(), d + 1).unwrap();
        for &(_, j) in neighbors.iter() {
            if i == j {
                continue;
            }
            let v = *drawing.node_id(j);
            if graph_edges.contains(&(graph.to_index(u), graph.to_index(v))) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    }

    cap as f32 / cup as f32
}
