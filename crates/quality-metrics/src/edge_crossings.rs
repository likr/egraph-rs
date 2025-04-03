use crate::edge_angle::edge_angle;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{DrawingEuclidean2d, DrawingIndex, DrawingTorus2d, MetricEuclidean2d};
use std::f32::consts::PI;

#[derive(Clone, Copy)]
struct Line {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn cross(line1: &Line, line2: &Line) -> bool {
    let dx1 = line1.x2 - line1.x1;
    let dy1 = line1.y2 - line1.y1;
    let dx2 = line2.x2 - line2.x1;
    let dy2 = line2.y2 - line2.y1;

    let s1 = dx1 * (line2.y1 - line1.y1) - dy1 * (line2.x1 - line1.x1);
    let t1 = dx1 * (line2.y2 - line1.y1) - dy1 * (line2.x2 - line1.x1);

    if s1 * t1 > 0. {
        return false;
    }

    let s2 = dx2 * (line1.y1 - line2.y1) - dy2 * (line1.x1 - line2.x1);
    let t2 = dx2 * (line1.y2 - line2.y1) - dy2 * (line1.x2 - line2.x1);

    if s2 * t2 > 0. {
        return false;
    }

    true
}

pub type CrossingEdges = Vec<(f32, f32, f32, f32, f32, f32, f32, f32)>;

pub fn crossing_edges<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> CrossingEdges
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        for &(p, q) in drawing.edge_segments(u, v).unwrap().iter() {
            let MetricEuclidean2d(x1, y1) = p;
            let MetricEuclidean2d(x2, y2) = q;
            edges.push((u, v, x1, y1, x2, y2));
        }
    }
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for &(source2, target2, x21, y21, x22, y22) in edges.iter().take(i) {
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            let line1 = Line {
                x1: x11,
                y1: y11,
                x2: x12,
                y2: y12,
            };
            let line2 = Line {
                x1: x21,
                y1: y21,
                x2: x22,
                y2: y22,
            };
            if cross(&line1, &line2) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    crossing_edges
}

pub fn crossing_edges_torus<G>(graph: G, drawing: &DrawingTorus2d<G::NodeId, f32>) -> CrossingEdges
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
        for &(source2, target2, x21, y21, x22, y22) in edges.iter().take(i) {
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            let line1 = Line {
                x1: x11,
                y1: y11,
                x2: x12,
                y2: y12,
            };
            let line2 = Line {
                x1: x21,
                y1: y21,
                x2: x22,
                y2: y22,
            };
            if cross(&line1, &line2) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    crossing_edges
}

pub fn crossing_number<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
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

pub fn crossing_angle<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_angle_with_crossing_edges(&crossing_edges)
}

pub fn crossing_angle_with_crossing_edges(crossing_edges: &CrossingEdges) -> f32 {
    let mut s = 0.;
    for &(x11, y11, x12, y12, x21, y21, x22, y22) in crossing_edges.iter() {
        if let Some(t) = edge_angle(x11 - x12, y11 - y12, x21 - x22, y21 - y22) {
            let t = t.min(PI - t);
            s += t.cos().powi(2);
        }
    }
    s
}
