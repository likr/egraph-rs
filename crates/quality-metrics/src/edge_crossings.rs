use crate::edge_angle::edge_angle;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{Drawing2D, DrawingIndex, DrawingTorus, Tuple2D};
use std::f32::consts::PI;

fn cross(x11: f32, y11: f32, x12: f32, y12: f32, x21: f32, y21: f32, x22: f32, y22: f32) -> bool {
    let s = (x12 - x11) * (y21 - y11) - (y11 - y12) * (x21 - x11);
    let t = (x12 - x11) * (y22 - y11) - (y11 - y12) * (x22 - x11);
    if s * t > 0. {
        return false;
    }
    let s = (x21 - x22) * (y11 - y21) - (y21 - y22) * (x11 - x21);
    let t = (x21 - x22) * (y12 - y21) - (y21 - y22) * (x12 - x21);
    if s * t > 0. {
        return false;
    }
    true
}

pub type CrossingEdges = Vec<(f32, f32, f32, f32, f32, f32, f32, f32)>;

pub fn crossing_edges<G>(graph: G, drawing: &Drawing2D<G::NodeId, f32>) -> CrossingEdges
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        for &(p, q) in drawing.edge_segments(u, v).unwrap().iter() {
            let Tuple2D(x1, y1) = p;
            let Tuple2D(x2, y2) = q;
            edges.push((u, v, x1, y1, x2, y2));
        }
    }
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for j in 0..i {
            let (source2, target2, x21, y21, x22, y22) = edges[j];
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            if cross(x11, y11, x12, y12, x21, y21, x22, y22) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    crossing_edges
}

pub fn crossing_edges_torus<G>(graph: G, drawing: &DrawingTorus<G::NodeId, f32>) -> CrossingEdges
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        for &(p, q) in drawing.edge_segments(u, v).unwrap().iter() {
            edges.push((u, v, p.0 .0, p.1 .0, q.0 .0, q.1 .0));
        }
    }
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for j in 0..i {
            let (source2, target2, x21, y21, x22, y22) = edges[j];
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            if cross(x11, y11, x12, y12, x21, y21, x22, y22) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    crossing_edges
}

pub fn crossing_number<G>(graph: G, drawing: &Drawing2D<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_number_with_crossing_edges(&crossing_edges)
}

pub fn crossing_number_with_crossing_edges(crossing_edges: &CrossingEdges) -> f32 {
    crossing_edges.len() as f32
}

pub fn crossing_angle<G>(graph: G, drawing: &Drawing2D<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_angle_with_crossing_edges(&crossing_edges)
}

pub fn crossing_angle_with_crossing_edges(crossing_edges: &CrossingEdges) -> f32 {
    let mut s = 0.;
    for (x11, y11, x12, y12, x21, y21, x22, y22) in crossing_edges.iter() {
        if let Some(t) = edge_angle(x11 - x12, y11 - y12, x21 - x22, y21 - y22) {
            let t = t.min(PI - t);
            s += t.cos().powi(2);
        }
    }
    s
}
