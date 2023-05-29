use crate::edge_angle::edge_angle;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{Drawing, DrawingIndex};
use std::f32::consts::PI;

pub fn crossing_edges<G>(
    graph: G,
    drawing: &Drawing<G::NodeId, f32>,
) -> Vec<((G::NodeId, G::NodeId), (G::NodeId, G::NodeId))>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let edges = graph
        .edge_references()
        .map(|e| {
            let u = e.source();
            let v = e.target();
            let (x1, y1) = drawing.position(u).unwrap();
            let (x2, y2) = drawing.position(v).unwrap();
            (u, v, x1, y1, x2, y2)
        })
        .collect::<Vec<_>>();
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for j in 0..i {
            let (source2, target2, x21, y21, x22, y22) = edges[j];
            if source1 == source2 || target1 == target2 {
                continue;
            }
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
            crossing_edges.push(((source1, target1), (source2, target2)));
        }
    }
    crossing_edges
}

pub fn crossing_number<G>(graph: G, drawing: &Drawing<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_number_with_crossing_edges(&crossing_edges)
}

pub fn crossing_number_with_crossing_edges<E>(crossing_edges: &[(E, E)]) -> f32 {
    crossing_edges.len() as f32
}

pub fn crossing_angle<G>(graph: G, drawing: &Drawing<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_angle_with_crossing_edges(drawing, &crossing_edges)
}

pub fn crossing_angle_with_crossing_edges<N>(
    drawing: &Drawing<N, f32>,
    crossing_edges: &[((N, N), (N, N))],
) -> f32
where
    N: Copy + DrawingIndex,
{
    let mut s = 0.;
    for &((source1, target1), (source2, target2)) in crossing_edges.iter() {
        let (x11, y11) = drawing.position(source1).unwrap();
        let (x12, y12) = drawing.position(target1).unwrap();
        let (x21, y21) = drawing.position(source2).unwrap();
        let (x22, y22) = drawing.position(target2).unwrap();
        if let Some(t) = edge_angle(x11 - x12, y11 - y12, x21 - x22, y21 - y22) {
            let t = t.min(PI - t);
            s += t.cos().powi(2);
        }
    }
    s
}
