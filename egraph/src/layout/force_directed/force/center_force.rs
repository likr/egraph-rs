use super::force::{Force, Point};

pub struct CenterForce {
    strength: f32,
}

impl CenterForce {
    pub fn new() -> CenterForce {
        CenterForce { strength: 1.0 }
    }
}

impl Force for CenterForce {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        let cx = points.iter().map(|p| p.x).sum::<f32>() / points.len() as f32;
        let cy = points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32;
        for point in points.iter_mut() {
            point.x -= cx * self.strength;
            point.y -= cy * self.strength;
        }
    }

    fn get_strength(&self) -> f32 {
        self.strength
    }

    fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }
}
