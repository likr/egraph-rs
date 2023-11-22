use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{Drawing, DrawingIndex};

pub fn gabriel_graph_property<G>(graph: G, drawing: &Drawing<G::NodeId, (f32, f32)>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let n = drawing.len();
    let mut s = 0.;
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let (x1, y1) = drawing.position(u).unwrap();
        let (x2, y2) = drawing.position(v).unwrap();
        let cx = (x1 + x2) / 2.;
        let cy = (y1 + y2) / 2.;
        let r = (x1 - x2).hypot(y1 - y2) / 2.;
        for i in 0..n {
            s += (r - (drawing.coordinates[i].0 - cx).hypot(drawing.coordinates[i].1 - cy))
                .max(0.)
                .powi(2);
        }
    }
    s
}
