use crate::MIN_DISTANCE;
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::{Force, ForceToNode, Point};

#[derive(Force)]
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

impl ForceToNode for RadialForce {
    fn apply_to_node(&self, i: usize, points: &mut [Point], alpha: f32) {
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
