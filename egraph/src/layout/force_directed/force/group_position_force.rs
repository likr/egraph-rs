use super::position_force::PositionForceContext;
use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext};

pub struct GroupPositionForce<G> {
    pub strength: Box<Fn(&Graph<G>, NodeIndex) -> f32>,
    pub group: Box<Fn(&Graph<G>, NodeIndex) -> usize>,
    pub group_x: Box<Fn(usize) -> f32>,
    pub group_y: Box<Fn(usize) -> f32>,
}

impl<G> GroupPositionForce<G> {
    pub fn new() -> GroupPositionForce<G> {
        GroupPositionForce {
            strength: Box::new(|_, _| 0.1),
            group: Box::new(|_, _| 0),
            group_x: Box::new(|_| 0.),
            group_y: Box::new(|_| 0.),
        }
    }
}

impl<G> Force<G> for GroupPositionForce<G> {
    fn build(&self, graph: &Graph<G>) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph.nodes().map(|u| strength_accessor(graph, u)).collect();

        let group_accessor = &self.group;

        let group_x_accessor = &self.group_x;
        let x = graph
            .nodes()
            .map(|u| Some(group_x_accessor(group_accessor(graph, u))))
            .collect();

        let group_y_accessor = &self.group_y;
        let y = graph
            .nodes()
            .map(|u| Some(group_y_accessor(group_accessor(graph, u))))
            .collect();

        Box::new(PositionForceContext::new(strength, x, y))
    }
}
