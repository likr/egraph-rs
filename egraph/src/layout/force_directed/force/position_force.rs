use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};

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

pub struct PositionForce<G> {
    pub strength: Box<Fn(&Graph<G>, NodeIndex) -> f32>,
    pub x: Box<Fn(&Graph<G>, NodeIndex) -> Option<f32>>,
    pub y: Box<Fn(&Graph<G>, NodeIndex) -> Option<f32>>,
}

impl<G> PositionForce<G> {
    pub fn new() -> PositionForce<G> {
        PositionForce {
            strength: Box::new(|_, _| 0.1),
            x: Box::new(|_, _| None),
            y: Box::new(|_, _| None),
        }
    }
}

impl<G> Force<G> for PositionForce<G> {
    fn build(&self, graph: &Graph<G>) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph.nodes().map(|u| strength_accessor(graph, u)).collect();

        let x_accessor = &self.x;
        let x = graph.nodes().map(|u| x_accessor(graph, u)).collect();

        let y_accessor = &self.y;
        let y = graph.nodes().map(|u| y_accessor(graph, u)).collect();

        Box::new(PositionForceContext::new(strength, x, y))
    }
}
