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
            x: x,
            y: y,
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

impl<Ix: IndexType> Coordinates<Ix> {}

impl<Ix: IndexType> Coordinates<Ix> {
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

pub fn initial_placement<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Coordinates<Ix> {
    let indices = graph.node_indices().collect::<Vec<_>>();
    let mut index_map = HashMap::new();
    let n = indices.len();
    let mut points = Vec::with_capacity(n);
    for (i, &u) in indices.iter().enumerate() {
        index_map.insert(u, i);
        let r = 10. * (i as usize as f32).sqrt();
        let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push(Point::new(x, y));
    }
    Coordinates {
        indices,
        points,
        index_map,
    }
}
