use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, MetricEuclidean2d};

pub fn gabriel_graph_property<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
{
    let n = drawing.len();
    let mut s = 0.;
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let MetricEuclidean2d(x1, y1) = drawing.position(u).unwrap();
        let MetricEuclidean2d(x2, y2) = drawing.position(v).unwrap();
        let cx = (x1 + x2) / 2.;
        let cy = (y1 + y2) / 2.;
        let r = (x1 - x2).hypot(y1 - y2) / 2.;
        for i in 0..n {
            s += (r - (drawing.raw_entry(i).0 - cx).hypot(drawing.raw_entry(i).1 - cy))
                .max(0.)
                .powi(2);
        }
    }
    s
}
