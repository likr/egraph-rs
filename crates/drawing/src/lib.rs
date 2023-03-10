use ndarray::prelude::*;
use num_traits::{cast::FromPrimitive, clamp, float::FloatConst};
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
};

pub struct Drawing<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    pub indices: Vec<N>,
    pub coordinates: Array2<S>,
    index_map: HashMap<N, usize>,
}

impl<N, S> Drawing<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    pub fn new<G>(graph: G) -> Drawing<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Eq + Hash + Into<N>,
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
        let coordinates = Array2::zeros((indices.len(), 2));
        Drawing {
            indices,
            coordinates,
            index_map,
        }
    }

    pub fn iter(&self) -> DrawingIterator<N, S>
    where
        N: Copy,
    {
        DrawingIterator {
            drawing: self,
            index: 0,
        }
    }

    pub fn x(&self, u: N) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[[i, 0]])
    }

    pub fn y(&self, u: N) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[[i, 1]])
    }

    pub fn position(&self, u: N) -> Option<(S, S)> {
        self.index_map
            .get(&u)
            .map(|&i| (self.coordinates[[i, 0]], self.coordinates[[i, 1]]))
    }

    pub fn set_x(&mut self, u: N, x: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 0]] = x;
            Some(())
        } else {
            None
        }
    }

    pub fn set_y(&mut self, u: N, y: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 1]] = y;
            Some(())
        } else {
            None
        }
    }

    pub fn set_position(&mut self, u: N, (x, y): (S, S)) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 0]] = x;
            self.coordinates[[i, 1]] = y;
            Some(())
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn centralize(&mut self)
    where
        S: FromPrimitive,
    {
        if let Some(c) = self.coordinates.mean_axis(Axis(0)) {
            self.coordinates -= &c;
        }
    }

    pub fn clamp_region(&mut self, x0: S, y0: S, x1: S, y1: S) {
        for i in 0..self.len() {
            self.coordinates[[i, 0]] = clamp(self.coordinates[[i, 0]], x0, x1);
            self.coordinates[[i, 1]] = clamp(self.coordinates[[i, 1]], y0, y1);
        }
    }

    pub fn initial_placement<G>(graph: G) -> Drawing<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Eq + Hash + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        Drawing::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn initial_placement_with_node_order<G>(graph: G, nodes: &[G::NodeId]) -> Drawing<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Eq + Hash + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive,
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

    pub fn initial_placement_with_bfs_order<G>(graph: G, s: G::NodeId) -> Drawing<N, S>
    where
        G: IntoNeighbors + IntoNodeIdentifiers,
        G::NodeId: Eq + Hash + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive,
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

pub struct DrawingIterator<'a, N, S>
where
    N: Copy + Eq + Hash,
    S: NdFloat,
{
    drawing: &'a Drawing<N, S>,
    index: usize,
}

impl<'a, N, S> Iterator for DrawingIterator<'a, N, S>
where
    N: Copy + Eq + Hash,
    S: NdFloat,
{
    type Item = (N, (S, S));
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        self.index += 1;
        if index < self.drawing.coordinates.len() {
            Some((
                self.drawing.indices[index],
                (
                    self.drawing.coordinates[[index, 0]],
                    self.drawing.coordinates[[index, 1]],
                ),
            ))
        } else {
            None
        }
    }
}
