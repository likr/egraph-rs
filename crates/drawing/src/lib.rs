pub use ndarray::prelude::*;
pub use num_traits::{cast::FromPrimitive, clamp, float::FloatConst};
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
};

pub trait DrawingIndex: Eq + Hash {}
impl<T> DrawingIndex for T where T: Eq + Hash {}
pub trait DrawingValue: NdFloat {}
impl<T> DrawingValue for T where T: NdFloat {}

pub trait Metric {
    type V: DrawingValue;
    fn distance(&self, other: &Self) -> Self::V;
}

impl<S> Metric for (S, S)
where
    S: DrawingValue,
{
    type V = S;
    fn distance(&self, other: &Self) -> Self::V {
        (self.0 - other.0).hypot(self.1 - other.1)
    }
}

pub struct Drawing<N, M> {
    pub indices: Vec<N>,
    pub coordinates: Vec<M>,
    index_map: HashMap<N, usize>,
}

impl<N, M> Drawing<N, M>
where
    N: DrawingIndex,
    M: Metric + Copy + Clone + Default,
{
    pub fn new<G>(graph: G) -> Drawing<N, M>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
    {
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<N>>();
        let index_map = indices
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let coordinates = vec![M::default(); indices.len()];
        Drawing {
            indices,
            coordinates,
            index_map,
        }
    }

    pub fn iter(&self) -> DrawingIterator<N, M>
    where
        N: Copy,
    {
        DrawingIterator {
            drawing: self,
            index: 0,
        }
    }

    pub fn position(&self, u: N) -> Option<M> {
        self.index_map.get(&u).map(|&i| self.coordinates[i])
    }

    pub fn set_position(&mut self, u: N, pos: M) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[i] = pos;
            Some(())
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct DrawingIterator<'a, N, M> {
    drawing: &'a Drawing<N, M>,
    index: usize,
}

impl<'a, N, M> Iterator for DrawingIterator<'a, N, M>
where
    N: Copy + DrawingIndex,
    M: Metric + Default + Copy,
{
    type Item = (N, M);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        self.index += 1;
        if index < self.drawing.coordinates.len() {
            Some((self.drawing.indices[index], self.drawing.coordinates[index]))
        } else {
            None
        }
    }
}

impl<N, S> Drawing<N, (S, S)>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn x(&self, u: N) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[i].0)
    }

    pub fn y(&self, u: N) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[i].1)
    }

    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[i].0 = value;
            Some(())
        } else {
            None
        }
    }

    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[i].1 = value;
            Some(())
        } else {
            None
        }
    }

    pub fn centralize(&mut self)
    where
        S: FromPrimitive + Default,
    {
        let mut l = S::infinity();
        let mut r = S::neg_infinity();
        let mut t = S::infinity();
        let mut b = S::neg_infinity();
        for i in 0..self.len() {
            l = l.min(self.coordinates[i].0);
            r = r.max(self.coordinates[i].0);
            t = t.min(self.coordinates[i].1);
            b = b.max(self.coordinates[i].1);
        }
        let w = r - l;
        let h = b - t;
        for i in 0..self.len() {
            self.coordinates[i].0 -= l + w / S::from(2.).unwrap();
            self.coordinates[i].1 -= t + h / S::from(2.).unwrap();
        }
    }

    pub fn clamp_region(&mut self, x0: S, y0: S, x1: S, y1: S)
    where
        S: Default,
    {
        for i in 0..self.len() {
            self.coordinates[i].0 = clamp(self.coordinates[i].0, x0, x1);
            self.coordinates[i].1 = clamp(self.coordinates[i].1, y0, y1);
        }
    }

    pub fn initial_placement<G>(graph: G) -> Drawing<N, (S, S)>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        Drawing::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn initial_placement_with_node_order<G>(graph: G, nodes: &[G::NodeId]) -> Drawing<N, (S, S)>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut drawing = Drawing::new(graph);
        for (i, &u) in nodes.iter().enumerate() {
            let r = S::from_usize(10).unwrap() * S::from_usize(i).unwrap().sqrt();
            let theta = S::PI()
                * (S::from_usize(3).unwrap() - S::from_usize(5).unwrap().sqrt())
                * (S::from_usize(i).unwrap());
            let x = r * theta.cos();
            let y = r * theta.sin();
            drawing.set_position(u.into(), (x, y));
        }
        drawing
    }

    pub fn initial_placement_with_bfs_order<G>(graph: G, s: G::NodeId) -> Drawing<N, (S, S)>
    where
        G: IntoNeighbors + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut queue = VecDeque::new();
        queue.push_back(s);
        let mut order = HashMap::new();
        order.insert(s, 0);
        let mut index = 1usize;
        while let Some(u) = queue.pop_front() {
            for v in graph.neighbors(u) {
                if !order.contains_key(&v) {
                    queue.push_back(v);
                    order.insert(v, index);
                    index += 1;
                }
            }
        }
        let mut nodes = graph.node_identifiers().collect::<Vec<_>>();
        nodes.sort_by_key(|&u| order.get(&u).or(Some(&std::usize::MAX)));
        Drawing::initial_placement_with_node_order(graph, &nodes)
    }
}
