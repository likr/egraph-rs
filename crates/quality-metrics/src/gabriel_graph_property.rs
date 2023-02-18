use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;

pub fn gabriel_graph_property<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let n = graph.node_count();
    let mut s = 0.;
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let (x1, y1) = coordinates.position(u).unwrap();
        let (x2, y2) = coordinates.position(v).unwrap();
        let cx = (x1 + x2) / 2.;
        let cy = (y1 + y2) / 2.;
        let r = (x1 - x2).hypot(y1 - y2) / 2.;
        for i in 0..n {
            s += (r - (coordinates.points[i].x - cx).hypot(coordinates.points[i].y - cy))
                .max(0.)
                .powi(2);
        }
    }
    s
}
