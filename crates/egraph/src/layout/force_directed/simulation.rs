use super::force::{Force, ForceContext, Point};
use crate::Graph;
use std::cell::RefCell;
use std::rc::Rc;

pub struct SimulationContext {
    forces: Vec<Box<ForceContext>>,
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl SimulationContext {
    fn new(
        forces: Vec<Box<ForceContext>>,
        alpha: f32,
        alpha_min: f32,
        alpha_target: f32,
        velocity_decay: f32,
        iterations: usize,
    ) -> SimulationContext {
        SimulationContext {
            forces,
            alpha,
            alpha_min,
            alpha_target,
            velocity_decay,
            iterations,
        }
    }

    pub fn start(&mut self, points: &mut Vec<Point>) {
        let alpha_decay = 1. - self.alpha_min.powf(1. / self.iterations as f32);
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

pub struct Simulation<G> {
    builders: Vec<Rc<RefCell<Force<G>>>>,
    pub alpha_start: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl<G> Simulation<G> {
    pub fn new() -> Simulation<G> {
        Simulation {
            builders: Vec::new(),
            alpha_start: 1.,
            alpha_min: 0.001,
            alpha_target: 0.,
            velocity_decay: 0.6,
            iterations: 300,
        }
    }

    pub fn build(&self, graph: &Graph<G>) -> SimulationContext {
        let forces = self
            .builders
            .iter()
            .map(|builder| builder.borrow().build(graph))
            .collect();
        SimulationContext::new(
            forces,
            self.alpha_start,
            self.alpha_min,
            self.alpha_target,
            self.velocity_decay,
            self.iterations,
        )
    }

    pub fn add(&mut self, force: Rc<RefCell<Force<G>>>) {
        self.builders.push(force);
    }

    pub fn get(&self, index: usize) -> Option<Rc<RefCell<Force<G>>>> {
        self.builders.get(index).map(|f| f.clone())
    }
}
