use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::prelude::*;
use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use std::collections::HashSet;

pub fn neighborhood_preservation<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let mut graph_edges = HashSet::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        graph_edges.insert((u.index(), v.index()));
        graph_edges.insert((v.index(), u.index()));
    }

    let n = coordinates.len();
    let mut points = Array2::zeros((n, 2));
    for i in 0..n {
        points[[i, 0]] = coordinates.points[i].x;
        points[[i, 1]] = coordinates.points[i].y;
    }
    let nn = BallTree::new().from_batch(&points, L2Dist).unwrap();

    let mut cap = 0;
    let mut cup = graph_edges.len();
    for i in 0..n {
        let u = coordinates.indices[i];
        let p = coordinates.points[i];
        let x = p.x;
        let y = p.y;
        let d = graph.neighbors_undirected(u).count();
        let query = arr1(&[x, y]);
        let neighbors = nn.k_nearest(query.view(), d + 1).unwrap();
        for &(_, j) in neighbors.iter() {
            if i == j {
                continue;
            }
            let v = coordinates.indices[j];
            if graph_edges.contains(&(u.index(), v.index())) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    }

    cap as f32 / cup as f32
}
