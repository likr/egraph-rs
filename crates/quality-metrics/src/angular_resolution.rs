use crate::edge_angle::edge_angle;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use petgraph_drawing::{DrawingEuclidean2d, DrawingIndex, MetricEuclidean2d};

pub fn angular_resolution<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
where
    G: IntoNodeIdentifiers + IntoNeighbors,
    G::NodeId: DrawingIndex,
{
    let mut s = 0.;
    for u in graph.node_identifiers() {
        let MetricEuclidean2d(x0, y0) = drawing.position(u).unwrap();
        let neighbors = graph.neighbors(u).collect::<Vec<_>>();
        let n = neighbors.len();
        for i in 1..n {
            let v = neighbors[i];
            let MetricEuclidean2d(x1, y1) = drawing.position(v).unwrap();
            for j in 0..i {
                let w = neighbors[j];
                let MetricEuclidean2d(x2, y2) = drawing.position(w).unwrap();
                if let Some(angle) = edge_angle(x1 - x0, y1 - y0, x2 - x0, y2 - y0) {
                    s += (-angle).exp()
                }
            }
        }
    }
    s
}
