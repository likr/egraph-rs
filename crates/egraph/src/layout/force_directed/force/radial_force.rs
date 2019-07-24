use super::MIN_DISTANCE;
use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};
use std::marker::PhantomData;

pub struct RadialForceContext {
    strength: Vec<f32>,
    radius: Vec<Option<f32>>,
    x: f32,
    y: f32,
}

impl RadialForceContext {
    pub fn new(strength: Vec<f32>, radius: Vec<Option<f32>>, x: f32, y: f32) -> RadialForceContext {
        RadialForceContext {
            strength,
            radius,
            x,
            y,
        }
    }
}

impl ForceContext for RadialForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for i in 0..points.len() {
            if let Some(ri) = self.radius[i] {
                let strength = self.strength[i];
                let point = points.get_mut(i).unwrap();
                let dx = (point.x - self.x);
                let dy = (point.y - self.y);
                let d = (dx * dx + dy * dy).sqrt();
                let k = (ri - d) * strength * alpha / d;
                point.vx += dx * k;
                point.vy += dy * k;
            }
        }
    }
}

pub struct RadialForce<D, G: Graph<D>> {
    pub strength: Box<dyn Fn(&G, NodeIndex) -> f32>,
    pub radius: Box<dyn Fn(&G, NodeIndex) -> Option<f32>>,
    pub x: f32,
    pub y: f32,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> RadialForce<D, G> {
    pub fn new() -> RadialForce<D, G> {
        RadialForce {
            strength: Box::new(|_, _| 0.1),
            radius: Box::new(|_, _| None),
            x: 0.,
            y: 0.,
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for RadialForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph.nodes().map(|u| strength_accessor(graph, u)).collect();

        let radius_accessor = &self.radius;
        let radius = graph.nodes().map(|u| radius_accessor(graph, u)).collect();

        Box::new(RadialForceContext::new(strength, radius, self.x, self.y))
    }
}
