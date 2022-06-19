use crate::{Force, ForceToNode};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;
use std::f32::consts::PI;

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
            x,
            y,
            vx: 0.,
            vy: 0.,
        }
    }
}

pub struct Coordinates<Ix: IndexType> {
    pub indices: Vec<NodeIndex<Ix>>,
    pub points: Vec<Point>,
    index_map: HashMap<NodeIndex<Ix>, usize>,
}

impl<Ix: IndexType> Coordinates<Ix> {
    pub fn new<N, E, Ty: EdgeType>(graph: &Graph<N, E, Ty, Ix>) -> Coordinates<Ix> {
        let indices = graph.node_indices().collect::<Vec<_>>();
        let index_map = indices
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let points = vec![Point::new(0., 0.); indices.len()];
        Coordinates {
            indices,
            points,
            index_map,
        }
    }

    pub fn iter(&self) -> CoordinatesIterator<Ix> {
        CoordinatesIterator {
            coordinates: self,
            index: 0,
        }
    }

    pub fn x(&self, u: NodeIndex<Ix>) -> Option<f32> {
        self.index_map.get(&u).map(|&i| self.points[i].x)
    }

    pub fn y(&self, u: NodeIndex<Ix>) -> Option<f32> {
        self.index_map.get(&u).map(|&i| self.points[i].y)
    }

    pub fn position(&self, u: NodeIndex<Ix>) -> Option<(f32, f32)> {
        self.index_map
            .get(&u)
            .map(|&i| (self.points[i].x, self.points[i].y))
    }

    pub fn set_x(&mut self, u: NodeIndex<Ix>, x: f32) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.points[i].x = x;
            Some(())
        } else {
            None
        }
    }

    pub fn set_y(&mut self, u: NodeIndex<Ix>, y: f32) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.points[i].y = y;
            Some(())
        } else {
            None
        }
    }

    pub fn set_position(&mut self, u: NodeIndex<Ix>, (x, y): (f32, f32)) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.points[i].x = x;
            self.points[i].y = y;
            Some(())
        } else {
            None
        }
    }
    pub fn vx(&self, u: NodeIndex<Ix>) -> Option<f32> {
        self.index_map.get(&u).map(|&i| self.points[i].vx)
    }

    pub fn vy(&self, u: NodeIndex<Ix>) -> Option<f32> {
        self.index_map.get(&u).map(|&i| self.points[i].vy)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn centralize(&mut self) {
        let cx = self.points.iter().map(|p| p.x).sum::<f32>() / self.points.len() as f32;
        let cy = self.points.iter().map(|p| p.y).sum::<f32>() / self.points.len() as f32;
        for point in self.points.iter_mut() {
            point.x -= cx;
            point.y -= cy;
        }
    }

    pub fn update_position(&mut self, velocity_decay: f32) {
        for point in self.points.iter_mut() {
            point.vx *= velocity_decay;
            point.vy *= velocity_decay;
            point.x += point.vx;
            point.y += point.vy;
        }
    }

    pub fn update_with<F: FnMut(&mut [Point], f32)>(
        &mut self,
        alpha: f32,
        velocity_decay: f32,
        f: &mut F,
    ) {
        f(&mut self.points, alpha);
        self.update_position(velocity_decay);
    }

    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        for i in 0..self.points.len() {
            self.points[i].x = self.points[i].x.clamp(x0, x1);
            self.points[i].y = self.points[i].y.clamp(y0, y1);
        }
    }

    pub fn apply_forces<T: AsRef<dyn Force>>(
        &mut self,
        forces: &[T],
        alpha: f32,
        velocity_decay: f32,
    ) {
        self.update_with(alpha, velocity_decay, &mut |points, alpha| {
            for force in forces {
                force.as_ref().apply(points, alpha);
            }
        });
    }

    pub fn apply_forces_to_node<T: AsRef<dyn ForceToNode>>(
        &mut self,
        u: usize,
        forces: &[T],
        alpha: f32,
        velocity_decay: f32,
    ) {
        self.update_with(alpha, velocity_decay, &mut |points, alpha| {
            for force in forces {
                force.as_ref().apply_to_node(u, points, alpha);
            }
        });
    }

    pub fn initial_placement<N, E, Ty: EdgeType>(graph: &Graph<N, E, Ty, Ix>) -> Coordinates<Ix> {
        let mut coordinates = Coordinates::new(graph);
        for (i, u) in graph.node_indices().enumerate() {
            let r = 10. * (i as usize as f32).sqrt();
            let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            coordinates.set_position(u, (x, y));
        }
        coordinates
    }
}

pub struct CoordinatesIterator<'a, Ix: IndexType> {
    coordinates: &'a Coordinates<Ix>,
    index: usize,
}

impl<'a, Ix: IndexType> Iterator for CoordinatesIterator<'a, Ix> {
    type Item = (NodeIndex<Ix>, Point);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        self.index += 1;
        if index < self.coordinates.points.len() {
            Some((
                self.coordinates.indices[index],
                self.coordinates.points[index],
            ))
        } else {
            None
        }
    }
}
