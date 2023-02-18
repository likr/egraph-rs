use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;

pub fn node_resolution<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let n = graph.node_count();
    let r = 1. / (n as f32).sqrt();

    let mut d_max = 0f32;
    for i in 1..n {
        for j in 0..i {
            let dx = coordinates.points[i].x - coordinates.points[j].x;
            let dy = coordinates.points[i].y - coordinates.points[j].y;
            d_max = d_max.max((dx).hypot(dy));
        }
    }

    let mut s = 0.;
    for i in 1..n {
        for j in 0..i {
            let dx = coordinates.points[i].x - coordinates.points[j].x;
            let dy = coordinates.points[i].y - coordinates.points[j].y;
            s += (1. - (dx).hypot(dy) / (r * d_max)).powi(2);
        }
    }
    s
}
