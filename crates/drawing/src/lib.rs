use ndarray::prelude::*;
use num_traits::{cast::FromPrimitive, clamp, float::FloatConst};
use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

pub struct Drawing<G, S>
where
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat + FromPrimitive,
{
    pub indices: Vec<G::NodeId>,
    pub coordinates: Array2<S>,
    index_map: HashMap<G::NodeId, usize>,
}

impl<G, S> Drawing<G, S>
where
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat + FromPrimitive + FloatConst,
{
    pub fn new(graph: G) -> Drawing<G, S> {
        let indices = graph.node_identifiers().collect::<Vec<_>>();
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

    pub fn iter(&self) -> DrawingIterator<G, S> {
        DrawingIterator {
            drawing: self,
            index: 0,
        }
    }

    pub fn x(&self, u: G::NodeId) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[[i, 0]])
    }

    pub fn y(&self, u: G::NodeId) -> Option<S> {
        self.index_map.get(&u).map(|&i| self.coordinates[[i, 1]])
    }

    pub fn position(&self, u: G::NodeId) -> Option<(S, S)> {
        self.index_map
            .get(&u)
            .map(|&i| (self.coordinates[[i, 0]], self.coordinates[[i, 1]]))
    }

    pub fn set_x(&mut self, u: G::NodeId, x: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 0]] = x;
            Some(())
        } else {
            None
        }
    }

    pub fn set_y(&mut self, u: G::NodeId, y: S) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 1]] = y;
            Some(())
        } else {
            None
        }
    }

    pub fn set_position(&mut self, u: G::NodeId, (x, y): (S, S)) -> Option<()> {
        if let Some(&i) = self.index_map.get(&u) {
            self.coordinates[[i, 0]] = x;
            self.coordinates[[i, 1]] = y;
            Some(())
        } else {
            None
        }
    }
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn centralize(&mut self) {
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

    pub fn initial_placement(graph: G) -> Drawing<G, S> {
        let mut drawing = Drawing::new(graph);
        for (i, u) in graph.node_identifiers().enumerate() {
            let r = S::from_usize(10).unwrap() * S::from_usize(i).unwrap().sqrt();
            let theta = S::PI()
                * (S::from_usize(3).unwrap() - S::from_usize(5).unwrap().sqrt())
                * (S::from_usize(i).unwrap());
            let x = r * theta.cos();
            let y = r * theta.sin();
            drawing.set_position(u, (x, y));
        }
        drawing
    }
}

pub struct DrawingIterator<'a, G, S>
where
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat + FromPrimitive,
{
    drawing: &'a Drawing<G, S>,
    index: usize,
}

impl<'a, G, S> Iterator for DrawingIterator<'a, G, S>
where
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat + FromPrimitive,
{
    type Item = (G::NodeId, (S, S));
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
