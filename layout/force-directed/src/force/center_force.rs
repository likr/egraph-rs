use super::force::{Force, ForceContext, Point};
use egraph_interface::Graph;

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

impl Force for CenterForce {
    fn build(&self, _graph: &Graph) -> Box<ForceContext> {
        Box::new(CenterForceContext::new())
    }
}
