use super::force::{Force, Point};

pub struct PositionForce {
    strength: f32,
    x: f32,
    y: f32,
}

impl PositionForce {
    pub fn new(x: f32, y: f32) -> PositionForce {
        PositionForce {
            strength: 1.0,
            x,
            y,
        }
    }
}

impl Force for PositionForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for point in points.iter_mut() {
            point.vx -= (point.x + point.vx - self.x) * alpha * self.strength;
            point.vy -= (point.y + point.vy - self.y) * alpha * self.strength;
        }
    }

    fn get_strength(&self) -> f32 {
        self.strength
    }

    fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }
}
