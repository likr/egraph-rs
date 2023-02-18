use crate::edge_angle::edge_angle;
use itertools::Itertools;
use petgraph::graph::{EdgeIndex, Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use std::f32::consts::PI;

pub fn crossing_edges<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> Vec<(EdgeIndex<Ix>, EdgeIndex<Ix>)> {
    let mut crossing_edges = vec![];
    for item in graph.edge_indices().combinations(2) {
        let e1 = item[0];
        let e2 = item[1];
        let (source1, target1) = graph.edge_endpoints(e1).unwrap();
        let (source2, target2) = graph.edge_endpoints(e2).unwrap();
        if source1 == source2 || target1 == target2 {
            continue;
        }
        let (x11, y11) = coordinates.position(source1).unwrap();
        let (x12, y12) = coordinates.position(target1).unwrap();
        let (x21, y21) = coordinates.position(source2).unwrap();
        let (x22, y22) = coordinates.position(target2).unwrap();
        let s = (x12 - x11) * (y21 - y11) - (y11 - y12) * (x21 - x11);
        let t = (x12 - x11) * (y22 - y11) - (y11 - y12) * (x22 - x11);
        if s * t > 0. {
            continue;
        }
        let s = (x21 - x22) * (y11 - y21) - (y21 - y22) * (x11 - x21);
        let t = (x21 - x22) * (y12 - y21) - (y21 - y22) * (x12 - x21);
        if s * t > 0. {
            continue;
        }
        crossing_edges.push((e1, e2));
    }
    crossing_edges
}

pub fn crossing_number<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let crossing_edges = crossing_edges(graph, coordinates);
    crossing_angle_with_crossing_edges(graph, coordinates, &crossing_edges)
}
pub fn crossing_number_with_crossing_edges<Ix: IndexType>(
    crossing_edges: &[(EdgeIndex<Ix>, EdgeIndex<Ix>)],
) -> f32 {
    crossing_edges.len() as f32
}

pub fn crossing_angle<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let crossing_edges = crossing_edges(graph, coordinates);
    crossing_angle_with_crossing_edges(graph, coordinates, &crossing_edges)
}

pub fn crossing_angle_with_crossing_edges<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
    crossing_edges: &[(EdgeIndex<Ix>, EdgeIndex<Ix>)],
) -> f32 {
    let mut s = 0.;
    for &(e1, e2) in crossing_edges {
        let (source1, target1) = graph.edge_endpoints(e1).unwrap();
        let (source2, target2) = graph.edge_endpoints(e2).unwrap();
        let (x11, y11) = coordinates.position(source1).unwrap();
        let (x12, y12) = coordinates.position(target1).unwrap();
        let (x21, y21) = coordinates.position(source2).unwrap();
        let (x22, y22) = coordinates.position(target2).unwrap();
        if let Some(t) = edge_angle(x11 - x12, y11 - y12, x21 - x22, y21 - y22) {
            let t = t.min(PI - t);
            s += t.cos().powi(2);
        }
    }
    s
}
