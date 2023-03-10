use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNeighbors, NodeIndexable};
use petgraph_drawing::Drawing;
use std::{collections::HashSet, hash::Hash};

pub fn neighborhood_preservation<G>(graph: G, drawing: &Drawing<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences + IntoNeighbors + NodeIndexable,
    G::NodeId: Eq + Hash,
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
        points[[i, 0]] = drawing.coordinates[[i, 0]];
        points[[i, 1]] = drawing.coordinates[[i, 1]];
    }
    let nn = BallTree::new().from_batch(&points, L2Dist).unwrap();

    let mut cap = 0;
    let mut cup = graph_edges.len();
    for i in 0..n {
        let u = drawing.indices[i];
        let x = drawing.coordinates[[i, 0]];
        let y = drawing.coordinates[[i, 1]];
        let d = graph.neighbors(u).count();
        let query = arr1(&[x, y]);
        let neighbors = nn.k_nearest(query.view(), d + 1).unwrap();
        for &(_, j) in neighbors.iter() {
            if i == j {
                continue;
            }
            let v = drawing.indices[j];
            if graph_edges.contains(&(graph.to_index(u), graph.to_index(v))) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    }

    cap as f32 / cup as f32
}
