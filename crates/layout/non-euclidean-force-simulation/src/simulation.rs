use crate::map::Map;
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::{ForceToNode, Point};
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct Simulation<Ix: IndexType, M: Map> {
  indices: Vec<NodeIndex<Ix>>,
  points: Vec<Point>,
  buffer: Vec<Point>,
  pub alpha: f32,
  pub alpha_min: f32,
  pub alpha_target: f32,
  pub velocity_decay: f32,
  pub iterations: usize,
  map: PhantomData<M>,
}

impl<Ix: IndexType, M: Map> Simulation<Ix, M> {
  pub fn new<N, E, Ty: EdgeType, F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> (f32, f32)>(
    graph: &Graph<N, E, Ty, Ix>,
    mut initial_placement: F,
  ) -> Simulation<Ix, M> {
    let indices = graph.node_indices().collect::<Vec<_>>();
    let points = indices
      .iter()
      .map(|&u| {
        let (x, y) = initial_placement(graph, u);
        Point::new(x, y)
      })
      .collect::<Vec<_>>();
    let buffer = vec![Point::new(0., 0.); points.len()];
    let alpha = 1.;
    let alpha_min = 0.001;
    let alpha_target = 0.;
    let velocity_decay = 0.6;
    let iterations = 300;
    let map = PhantomData;
    Simulation {
      indices,
      points,
      buffer,
      alpha,
      alpha_min,
      alpha_target,
      velocity_decay,
      iterations,
      map,
    }
  }

  pub fn run<T: AsRef<dyn ForceToNode>>(
    &mut self,
    forces: &[T],
  ) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    while !self.is_finished() {
      self.step(forces);
    }
    self.coordinates()
  }

  pub fn run_step<T: AsRef<dyn ForceToNode>>(
    &mut self,
    n: usize,
    forces: &[T],
  ) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    for _ in 0..n {
      self.step(forces);
    }
    self.coordinates()
  }

  pub fn step<T: AsRef<dyn ForceToNode>>(&mut self, forces: &[T]) {
    let alpha_decay = 1. - self.alpha_min.powf(1. / self.iterations as f32);
    self.alpha += (self.alpha_target - self.alpha) * alpha_decay;
    self.apply_forces(forces, self.alpha);
  }

  pub fn apply_forces<T: AsRef<dyn ForceToNode>>(&mut self, forces: &[T], alpha: f32) {
    let n = self.points.len();
    for force in forces {
      for u in 0..n {
        let Point { x: cx, y: cy, .. } = self.points[u];
        for v in 0..n {
          let Point { x, y, .. } = self.points[v];
          let (zx, zy) = M::to_tangent_space((cx, cy), (x, y));
          self.buffer[v] = Point::new(zx, zy);
        }
        force.as_ref().apply_to_node(u, &mut self.points, alpha);
        self.points[u].vx = self.buffer[u].vx;
        self.points[u].vy = self.buffer[u].vy;
      }
    }
    for point in self.points.iter_mut() {
      point.vx *= self.velocity_decay;
      point.vy *= self.velocity_decay;
      let (x, y) = M::from_tangent_space((point.x, point.y), (point.vx, point.vy));
      point.x = x;
      point.y = y;
    }
  }

  pub fn is_finished(&self) -> bool {
    self.alpha < self.alpha_min
  }

  pub fn reset(&mut self, alpha_start: f32) {
    self.alpha = alpha_start;
  }

  pub fn coordinates(&self) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    self
      .indices
      .iter()
      .zip(self.points.iter())
      .map(|(&u, p)| (u, (p.x, p.y)))
      .collect::<HashMap<_, _>>()
  }
}
