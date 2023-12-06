use crate::{DrawingIndex, Metric};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

pub struct Drawing<N, M> {
    pub indices: Vec<N>,
    pub coordinates: Vec<M>,
    index_map: HashMap<N, usize>,
}

impl<N, M> Drawing<N, M>
where
    N: DrawingIndex,
    M: Metric,
{
    pub fn new<G>(graph: G) -> Drawing<N, M>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        M: Clone + Default,
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

    pub fn position(&self, u: N) -> Option<&M> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    pub fn position_mut(&mut self, u: N) -> Option<&mut M> {
        self.index_map.get(&u).map(|&i| &mut self.coordinates[i])
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
