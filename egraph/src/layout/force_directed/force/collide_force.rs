use super::force::{Force, ForceContext, Point};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;

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

pub struct CollideForce<N, E, Ty: EdgeType, Ix: IndexType> {
    pub radius: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    pub strength: f32,
    pub iterations: usize,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> CollideForce<N, E, Ty, Ix> {
    pub fn new() -> CollideForce<N, E, Ty, Ix> {
        CollideForce {
            radius: Box::new(|_, _| 1.),
            strength: 0.7,
            iterations: 1,
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for CollideForce<N, E, Ty, Ix> {
    fn build(&self, graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        let radius_accessor = &self.radius;
        let radius = graph
            .node_indices()
            .map(|a| radius_accessor(graph, a))
            .collect();
        Box::new(CollideForceContext::new(radius, self.strength))
    }
}
