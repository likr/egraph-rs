use crate::{Force, Point};

pub struct CenterForce {}

impl CenterForce {
    pub fn new() -> CenterForce {
        CenterForce {}
    }

    pub fn as_force(self) -> Box<dyn Force> {
        Box::new(self)
    }
}

impl Force for CenterForce {
    fn apply(&self, points: &mut [Point], _alpha: f32) {
        let cx = points.iter().map(|p| p.x).sum::<f32>() / points.len() as f32;
        let cy = points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32;
        for point in points.iter_mut() {
            point.x -= cx;
            point.y -= cy;
        }
    }
}

impl AsRef<dyn Force> for CenterForce {
    fn as_ref(&self) -> &(dyn Force + 'static) {
        self
    }
}
