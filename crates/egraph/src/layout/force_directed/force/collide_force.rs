use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};
use std::marker::PhantomData;

pub struct CollideForceContext {
    radius: Vec<f32>,
    strength: f32,
}

impl CollideForceContext {
    pub fn new(radius: Vec<f32>, strength: f32) -> CollideForceContext {
        CollideForceContext { radius, strength }
    }
}

impl ForceContext for CollideForceContext {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        let n = points.len();
        for i in 0..n {
            let xi = points[i].x + points[i].vx;
            let yi = points[i].y + points[i].vy;
            let ri = self.radius[i];
            for j in (i + 1)..n {
                let xj = points[j].x + points[j].vx;
                let yj = points[j].y + points[j].vy;
                let rj = self.radius[j];
                let dx = xi - xj;
                let dy = yi - yj;
                let r = ri + rj;
                let l2 = (dx * dx + dy * dy).max(1e-6);
                if l2 < r * r {
                    let l = l2.sqrt();
                    let d = (r - l) / l * self.strength;
                    let rr = (rj * rj) / (ri * ri + rj * rj);
                    points[i].vx += (dx * d) * rr;
                    points[i].vy += (dy * d) * rr;
                    points[j].vx -= (dx * d) * (1. - rr);
                    points[j].vy -= (dy * d) * (1. - rr);
                }
            }
        }
    }
}

pub struct CollideForce<D, G: Graph<D>> {
    pub radius: Box<dyn Fn(&G, NodeIndex) -> f32>,
    pub strength: f32,
    pub iterations: usize,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> CollideForce<D, G> {
    pub fn new() -> CollideForce<D, G> {
        CollideForce {
            radius: Box::new(|_, _| 1.),
            strength: 0.7,
            iterations: 1,
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for CollideForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
        let radius_accessor = &self.radius;
        let radius = graph.nodes().map(|u| radius_accessor(graph, u)).collect();
        Box::new(CollideForceContext::new(radius, self.strength))
    }
}
