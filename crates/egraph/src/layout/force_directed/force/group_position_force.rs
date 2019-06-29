use super::position_force::PositionForceContext;
use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext};
use std::marker::PhantomData;

pub struct GroupPositionForce<D, G: Graph<D>> {
    pub strength: Box<dyn Fn(&G, NodeIndex) -> f32>,
    pub group: Box<dyn Fn(&G, NodeIndex) -> usize>,
    pub group_x: Box<dyn Fn(usize) -> f32>,
    pub group_y: Box<dyn Fn(usize) -> f32>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> GroupPositionForce<D, G> {
    pub fn new() -> GroupPositionForce<D, G> {
        GroupPositionForce {
            strength: Box::new(|_, _| 0.1),
            group: Box::new(|_, _| 0),
            group_x: Box::new(|_| 0.),
            group_y: Box::new(|_| 0.),
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for GroupPositionForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
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
