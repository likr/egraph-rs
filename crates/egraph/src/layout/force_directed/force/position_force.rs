use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};
use std::marker::PhantomData;

pub struct PositionForceContext {
    strength: Vec<f32>,
    x: Vec<Option<f32>>,
    y: Vec<Option<f32>>,
}

impl PositionForceContext {
    pub fn new(
        strength: Vec<f32>,
        x: Vec<Option<f32>>,
        y: Vec<Option<f32>>,
    ) -> PositionForceContext {
        PositionForceContext { strength, x, y }
    }
}

impl ForceContext for PositionForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for i in 0..points.len() {
            let strength = self.strength[i];
            let point = points.get_mut(i).unwrap();
            if let Some(xi) = self.x[i] {
                point.vx += (xi - point.x) * alpha * strength;
            }
            if let Some(yi) = self.y[i] {
                point.vy += (yi - point.y) * alpha * strength;
            }
        }
    }
}

pub struct PositionForce<D, G: Graph<D>> {
    pub strength: Box<dyn Fn(&G, NodeIndex) -> f32>,
    pub x: Box<dyn Fn(&G, NodeIndex) -> Option<f32>>,
    pub y: Box<dyn Fn(&G, NodeIndex) -> Option<f32>>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> PositionForce<D, G> {
    pub fn new() -> PositionForce<D, G> {
        PositionForce {
            strength: Box::new(|_, _| 0.1),
            x: Box::new(|_, _| None),
            y: Box::new(|_, _| None),
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for PositionForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph.nodes().map(|u| strength_accessor(graph, u)).collect();

        let x_accessor = &self.x;
        let x = graph.nodes().map(|u| x_accessor(graph, u)).collect();

        let y_accessor = &self.y;
        let y = graph.nodes().map(|u| y_accessor(graph, u)).collect();

        Box::new(PositionForceContext::new(strength, x, y))
    }
}
