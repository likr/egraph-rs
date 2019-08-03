use super::force::{
    CenterForce, Force, ForceContext, LinkForce, ManyBodyForce, Point, PositionForce,
};
use super::initial_placement;
use crate::Graph;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Simulation {
    forces: Vec<Box<dyn ForceContext>>,
    pub indices: HashMap<usize, usize>,
    pub points: Vec<Point>,
    pub alpha_decay: f32,
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl Simulation {
    fn new(
        forces: Vec<Box<dyn ForceContext>>,
        indices: HashMap<usize, usize>,
        points: Vec<Point>,
        alpha: f32,
        alpha_min: f32,
        alpha_target: f32,
        velocity_decay: f32,
        iterations: usize,
    ) -> Simulation {
        Simulation {
            forces,
            indices,
            points,
            alpha,
            alpha_min,
            alpha_target,
            alpha_decay: 1. - alpha_min.powf(1. / iterations as f32),
            velocity_decay,
            iterations,
        }
    }

    pub fn run(&mut self) {
        while !self.is_finished() {
            self.step();
        }
    }

    pub fn step(&mut self) {
        self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay;
        for force in self.forces.iter() {
            force.apply(&mut self.points, self.alpha);
        }
        for point in self.points.iter_mut() {
            point.vx *= self.velocity_decay;
            point.x += point.vx;
            point.vy *= self.velocity_decay;
            point.y += point.vy;
        }
    }

    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn is_finished(&self) -> bool {
        self.alpha < self.alpha_min
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.alpha = alpha_start;
    }

    pub fn x(&self, u: usize) -> f32 {
        self.points[self.indices[&u]].x
    }

    pub fn y(&self, u: usize) -> f32 {
        self.points[self.indices[&u]].y
    }
}

pub struct SimulationBuilder<D, G: Graph<D>> {
    builders: HashMap<usize, Rc<RefCell<dyn Force<D, G>>>>,
    pub alpha_start: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl<D, G: Graph<D>> SimulationBuilder<D, G> {
    pub fn new() -> SimulationBuilder<D, G> {
        SimulationBuilder {
            builders: HashMap::new(),
            alpha_start: 1.,
            alpha_min: 0.001,
            alpha_target: 0.,
            velocity_decay: 0.6,
            iterations: 300,
        }
    }

    pub fn build(&self, graph: &G) -> Simulation {
        let points = initial_placement(graph.node_count());
        let indices = graph
            .nodes()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let forces = self
            .builders
            .values()
            .map(|builder| builder.borrow().build(graph))
            .collect();
        Simulation::new(
            forces,
            indices,
            points,
            self.alpha_start,
            self.alpha_min,
            self.alpha_target,
            self.velocity_decay,
            self.iterations,
        )
    }

    pub fn start(&self, graph: &G) -> Simulation {
        let mut simulation = self.build(&graph);
        simulation.run();
        simulation
    }

    pub fn add(&mut self, force: Rc<RefCell<dyn Force<D, G>>>) -> usize {
        let index = self.builders.len();
        self.builders.insert(index, force);
        index
    }

    pub fn get(&self, index: usize) -> Option<Rc<RefCell<dyn Force<D, G>>>> {
        self.builders.get(&index).map(|f| f.clone())
    }

    pub fn remove(&mut self, index: usize) -> Option<Rc<RefCell<dyn Force<D, G>>>> {
        self.builders.remove(&index)
    }
}

impl<D: 'static, G: Graph<D> + 'static> Default for SimulationBuilder<D, G> {
    fn default() -> SimulationBuilder<D, G> {
        let mut builder = SimulationBuilder::new();
        let many_body_force = ManyBodyForce::new();
        builder.add(Rc::new(RefCell::new(many_body_force)));
        let link_force = LinkForce::new();
        builder.add(Rc::new(RefCell::new(link_force)));
        let center_force = CenterForce::new();
        builder.add(Rc::new(RefCell::new(center_force)));
        builder
    }
}

impl<D: 'static, G: Graph<D> + 'static> SimulationBuilder<D, G> {
    pub fn default_connected() -> SimulationBuilder<D, G> {
        SimulationBuilder::default()
    }

    pub fn default_non_connected() -> SimulationBuilder<D, G> {
        let mut builder = SimulationBuilder::new();
        let many_body_force = ManyBodyForce::new();
        builder.add(Rc::new(RefCell::new(many_body_force)));
        let link_force = LinkForce::new();
        builder.add(Rc::new(RefCell::new(link_force)));
        let mut position_force = PositionForce::new();
        position_force.x = Box::new(|_, _| Some(0.));
        position_force.y = Box::new(|_, _| Some(0.));
        builder.add(Rc::new(RefCell::new(position_force)));
        builder
    }
}
