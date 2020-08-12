use crate::{Force, Point};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::force::PositionForce;
use std::collections::HashMap;

pub struct GroupPositionForce {
    position_force: PositionForce,
}

impl GroupPositionForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
        F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
        F3: FnMut(usize) -> f32,
        F4: FnMut(usize) -> f32,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        strength_accessor: F1,
        mut group_accessor: F2,
        mut group_x_accessor: F3,
        mut group_y_accessor: F4,
    ) -> GroupPositionForce {
        let groups = graph
            .node_indices()
            .map(|u| (u, group_accessor(graph, u)))
            .collect::<HashMap<_, _>>();
        GroupPositionForce {
            position_force: PositionForce::new(
                graph,
                strength_accessor,
                |_, u| Some(group_x_accessor(groups[&u])),
                |_, u| Some(group_y_accessor(groups[&u])),
            ),
        }
    }
}

impl Force for GroupPositionForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        self.position_force.apply(points, alpha);
    }
}
