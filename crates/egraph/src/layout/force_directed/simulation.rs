use super::force::{Force, ForceContext, Point};
use crate::Graph;
use std::cell::RefCell;
use std::rc::Rc;

pub struct SimulationContext {
    forces: Vec<Box<dyn ForceContext>>,
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl SimulationContext {
    fn new(
        forces: Vec<Box<dyn ForceContext>>,
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
        loop {
            self.step(points);
            if self.is_finished() {
                break;
            }
        }
    }

    pub fn step(&mut self, points: &mut Vec<Point>) {
        let alpha_decay = 1. - self.alpha_min.powf(1. / self.iterations as f32);
        self.alpha += (self.alpha_target - self.alpha) * alpha_decay;
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

    pub fn is_finished(&self) -> bool {
        self.alpha < self.alpha_min
    }
}

pub struct Simulation<D, G: Graph<D>> {
    builders: Vec<Rc<RefCell<dyn Force<D, G>>>>,
    pub alpha_start: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl<D, G: Graph<D>> Simulation<D, G> {
    pub fn new() -> Simulation<D, G> {
        Simulation {
            builders: Vec::new(),
            alpha_start: 1.,
            alpha_min: 0.001,
            alpha_target: 0.,
            velocity_decay: 0.6,
            iterations: 300,
        }
    }

    pub fn build(&self, graph: &G) -> SimulationContext {
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

    pub fn add(&mut self, force: Rc<RefCell<dyn Force<D, G>>>) {
        self.builders.push(force);
    }

    pub fn get(&self, index: usize) -> Option<Rc<RefCell<dyn Force<D, G>>>> {
        self.builders.get(index).map(|f| f.clone())
    }
}
