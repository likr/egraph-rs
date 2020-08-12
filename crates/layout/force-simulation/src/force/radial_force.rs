use crate::{Force, Point, MIN_DISTANCE};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;

pub struct RadialForce {
    params: Vec<Option<(f32, f32, f32, f32)>>,
}

impl RadialForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<(f32, f32, f32, f32)>,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut accessor: F,
    ) -> RadialForce {
        let params = graph.node_indices().map(|u| accessor(graph, u)).collect();
        RadialForce { params }
    }
}

impl Force for RadialForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for i in 0..points.len() {
            if let Some((si, ri, xi, yi)) = self.params[i] {
                let point = points.get_mut(i).unwrap();
                let dx = if (point.x - xi).abs() < MIN_DISTANCE {
                    MIN_DISTANCE
                } else {
                    point.x - xi
                };
                let dy = if (point.y - yi).abs() < MIN_DISTANCE {
                    MIN_DISTANCE
                } else {
                    point.y - yi
                };
                let d = (dx * dx + dy * dy).sqrt();
                let k = (ri - d) * si * alpha / d;
                point.vx += dx * k;
                point.vy += dy * k;
            }
        }
    }
}
