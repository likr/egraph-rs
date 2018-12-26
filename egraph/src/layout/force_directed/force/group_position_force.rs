use super::force::{Force, ForceContext};
use super::position_force::PositionForceContext;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;

pub struct GroupPositionForce<N, E, Ty: EdgeType, Ix: IndexType> {
    pub strength: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub group_x: Box<Fn(usize) -> f32>,
    pub group_y: Box<Fn(usize) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> GroupPositionForce<N, E, Ty, Ix> {
    pub fn new() -> GroupPositionForce<N, E, Ty, Ix> {
        GroupPositionForce {
            strength: Box::new(|_, _| 0.1),
            group: Box::new(|_, _| 0),
            group_x: Box::new(|_| 0.),
            group_y: Box::new(|_| 0.),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for GroupPositionForce<N, E, Ty, Ix> {
    fn build(&self, graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph
            .node_indices()
            .map(|index| strength_accessor(graph, index))
            .collect();

        let group_accessor = &self.group;

        let group_x_accessor = &self.group_x;
        let x = graph
            .node_indices()
            .map(|index| Some(group_x_accessor(group_accessor(graph, index))))
            .collect();

        let group_y_accessor = &self.group_y;
        let y = graph
            .node_indices()
            .map(|index| Some(group_y_accessor(group_accessor(graph, index))))
            .collect();

        Box::new(PositionForceContext::new(strength, x, y))
    }
}
