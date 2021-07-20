use super::group_indices;
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::{Force, Point, MIN_DISTANCE};
use std::collections::HashMap;

pub struct GroupManyBodyForce {
    groups: HashMap<usize, Vec<usize>>,
    strength: Vec<f32>,
}

impl GroupManyBodyForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
        F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut strength_accessor: F1,
        mut group_accessor: F2,
    ) -> GroupManyBodyForce {
        let strength = graph
            .node_indices()
            .map(|u| strength_accessor(graph, u))
            .collect::<Vec<_>>();
        let groups = {
            let groups = graph
                .node_indices()
                .map(|u| group_accessor(graph, u))
                .collect::<Vec<_>>();
            group_indices(&groups)
        };
        GroupManyBodyForce { groups, strength }
    }
}

impl Force for GroupManyBodyForce {
    fn apply(&self, points: &mut [Point], alpha: f32) {
        for indices in self.groups.values() {
            let n = indices.len();
            for i in 0..n {
                let a = indices[i];
                let x0 = points[a].x;
                let y0 = points[a].y;
                let mut dvx = 0.;
                let mut dvy = 0.;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let b = indices[j];
                    let strength = self.strength[b];
                    let dx = points[b].x - x0;
                    let dy = points[b].y - y0;
                    let l = (dx * dx + dy * dy).max(MIN_DISTANCE);
                    dvx += dx * strength * alpha / l;
                    dvy += dy * strength * alpha / l;
                }
                points[a].vx += dvx;
                points[a].vy += dvy
            }
        }
    }
}
