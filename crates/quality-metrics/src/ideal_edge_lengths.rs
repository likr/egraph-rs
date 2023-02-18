use ndarray::prelude::*;
use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use std::collections::HashMap;

pub fn ideal_edge_lengths<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
    d: &Array2<f32>,
) -> f32 {
    let node_indices = graph
        .node_indices()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let mut s = 0.;
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let (x1, y1) = coordinates.position(u).unwrap();
        let (x2, y2) = coordinates.position(v).unwrap();
        let l = d[[node_indices[&u], node_indices[&v]]];
        s += (((x1 - x2).hypot(y1 - y2) - l) / l).powi(2);
    }
    s
}
