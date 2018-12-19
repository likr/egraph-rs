use super::force::{Force, ForceContext, Point};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;

pub struct CenterForceContext {}

impl CenterForceContext {
    fn new() -> CenterForceContext {
        CenterForceContext {}
    }
}

impl ForceContext for CenterForceContext {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        let cx = points.iter().map(|p| p.x).sum::<f32>() / points.len() as f32;
        let cy = points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32;
        for point in points.iter_mut() {
            point.x -= cx;
            point.y -= cy;
        }
    }
}

pub struct CenterForce {}

impl CenterForce {
    pub fn new() -> CenterForce {
        CenterForce {}
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for CenterForce {
    fn build(&self, _graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        Box::new(CenterForceContext::new())
    }
}
