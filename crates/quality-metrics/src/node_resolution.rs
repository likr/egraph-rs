use itertools::Itertools;
use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;

pub fn node_resolution<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let r = 1. / (graph.node_count() as f32).sqrt();

    let mut d_max = 0f32;
    for item in graph.node_indices().combinations(2) {
        let u = item[0];
        let v = item[1];
        let (x1, y1) = coordinates.position(u).unwrap();
        let (x2, y2) = coordinates.position(v).unwrap();
        d_max = d_max.max((x1 - x2).hypot(y1 - y2));
    }

    let mut s = 0.;
    for item in graph.node_indices().combinations(2) {
        let u = item[0];
        let v = item[1];
        let (x1, y1) = coordinates.position(u).unwrap();
        let (x2, y2) = coordinates.position(v).unwrap();
        s += (1. - (x1 - x2).hypot(y1 - y2) / (r * d_max)).powi(2);
    }
    s
}
