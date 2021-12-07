use crate::{Force, Point};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force::position_force;
use std::collections::HashMap;

#[derive(Copy, Clone)]
pub struct NodeArgument {
    pub group: usize,
    pub strength: f32,
}

#[derive(Copy, Clone)]
pub struct GroupArgument {
    pub x: f32,
    pub y: f32,
}

pub struct GroupPositionForce {
    position_force: position_force::PositionForce,
}

impl GroupPositionForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> NodeArgument,
        F2: FnMut(&Graph<N, E, Ty, Ix>, usize) -> GroupArgument,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut node_accessor: F1,
        mut group_accessor: F2,
    ) -> GroupPositionForce {
        let mut group_position = HashMap::new();
        GroupPositionForce {
            position_force: position_force::PositionForce::new(graph, |graph, u| {
                let arg = node_accessor(graph, u);
                let group = arg.group;
                let strength = Some(arg.strength);
                if !group_position.contains_key(&group) {
                    let group_arg = group_accessor(graph, group);
                    group_position.insert(group, (group_arg.x, group_arg.y));
                }
                let (x, y) = group_position[&group];
                let x = Some(x);
                let y = Some(y);
                position_force::NodeArgument { strength, x, y }
            }),
        }
    }
}

impl Force for GroupPositionForce {
    fn apply(&self, points: &mut [Point], alpha: f32) {
        self.position_force.apply(points, alpha);
    }
}
