use crate::edge_angle::edge_angle;
use itertools::Itertools;
use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use std::f32::consts::PI;

pub fn angular_resolution<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let mut min_angle = std::f32::MAX;
    for u in graph.node_indices() {
        let (x0, y0) = coordinates.position(u).unwrap();
        for pair in graph.neighbors_undirected(u).combinations(2) {
            let v = pair[0];
            let w = pair[1];
            let (x1, y1) = coordinates.position(v).unwrap();
            let (x2, y2) = coordinates.position(w).unwrap();
            if let Some(angle) = edge_angle(x1 - x0, y1 - y0, x2 - x0, y2 - y0) {
                min_angle = min_angle.min(angle)
            }
        }
    }

    let max_degree = graph
        .node_indices()
        .map(|u| graph.neighbors_undirected(u).count())
        .max()
        .unwrap_or(0);

    min_angle * max_degree as f32 / (2. * PI)
}
