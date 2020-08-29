use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Point {
        Point {
            x: x,
            y: y,
            vx: 0.,
            vy: 0.,
        }
    }
}

pub trait Force {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32);
}

pub struct Simulation<Ix: IndexType> {
    indices: Vec<NodeIndex<Ix>>,
    points: Vec<Point>,
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub velocity_decay: f32,
    pub iterations: usize,
}

impl<Ix: IndexType> Simulation<Ix> {
    pub fn new<N, E, Ty: EdgeType, F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> (f32, f32)>(
        graph: &Graph<N, E, Ty, Ix>,
        mut initial_placement: F,
    ) -> Simulation<Ix> {
        let indices = graph.node_indices().collect::<Vec<_>>();
        let points = indices
            .iter()
            .map(|&u| {
                let (x, y) = initial_placement(graph, u);
                Point::new(x, y)
            })
            .collect::<Vec<_>>();
        let alpha = 1.;
        let alpha_min = 0.001;
        let alpha_target = 0.;
        let velocity_decay = 0.6;
        let iterations = 300;
        Simulation {
            indices,
            points,
            alpha,
            alpha_min,
            alpha_target,
            velocity_decay,
            iterations,
        }
    }

    pub fn run<T: AsRef<dyn Force>>(&mut self, forces: &[T]) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
        while !self.is_finished() {
            self.step(forces);
        }
        self.coordinates()
    }

    pub fn run_step<T: AsRef<dyn Force>>(
        &mut self,
        n: usize,
        forces: &[T],
    ) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
        for _ in 0..n {
            self.step(forces);
        }
        self.coordinates()
    }

    pub fn step<T: AsRef<dyn Force>>(&mut self, forces: &[T]) {
        let alpha_decay = 1. - self.alpha_min.powf(1. / self.iterations as f32);
        self.alpha += (self.alpha_target - self.alpha) * alpha_decay;
        self.apply_forces(forces, self.alpha);
    }

    pub fn apply_forces<T: AsRef<dyn Force>>(&mut self, forces: &[T], alpha: f32) {
        for force in forces {
            force.as_ref().apply(&mut self.points, alpha);
        }
        for point in self.points.iter_mut() {
            point.vx *= self.velocity_decay;
            point.x += point.vx;
            point.vy *= self.velocity_decay;
            point.y += point.vy;
        }
    }

    pub fn is_finished(&self) -> bool {
        self.alpha < self.alpha_min
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.alpha = alpha_start;
    }

    pub fn coordinates(&self) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
        self.indices
            .iter()
            .zip(self.points.iter())
            .map(|(&u, p)| (u, (p.x, p.y)))
            .collect::<HashMap<_, _>>()
    }
}
