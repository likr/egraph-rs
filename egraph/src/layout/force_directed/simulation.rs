use super::force::{Force, Point};

pub struct Simulation {
    pub forces: Vec<Box<Force>>,
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            forces: Vec::new(),
            alpha: 1.,
            alpha_min: 0.001,
            alpha_target: 0.,
            velocity_decay: 0.6,
        }
    }

    pub fn start(&mut self, points: &mut Vec<Point>) {
        let alpha_decay = 1. - (self.alpha_min as f32).powf(1. / 300.);
        loop {
            self.alpha += (self.alpha_target - self.alpha) * alpha_decay;
            self.step(points);
            if self.alpha < self.alpha_min {
                break;
            }
        }
    }

    pub fn step(&mut self, points: &mut Vec<Point>) {
        for force in self.forces.iter() {
            force.apply(points, self.alpha);
        }
        for point in points.iter_mut() {
            point.vx *= self.velocity_decay;
            point.x += point.vx;
            point.vy *= self.velocity_decay;
            point.y += point.vy;
        }
    }
}
