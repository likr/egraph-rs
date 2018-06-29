use force::{Point, Force};

pub struct CenterForce {}

impl CenterForce {
    pub fn new() -> CenterForce {
        CenterForce {}
    }
}

impl Force for CenterForce {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        let cx = points.iter().map(|p| p.x).sum::<f32>() / points.len() as f32;
        let cy = points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32;
        for point in points.iter_mut() {
            point.x -= cx;
            point.y -= cy;
        }
    }
}
