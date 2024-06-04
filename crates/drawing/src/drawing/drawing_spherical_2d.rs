use crate::{DeltaSpherical2d, Drawing, DrawingIndex, DrawingValue, MetricSpherical2d};
use num_traits::{FloatConst, FromPrimitive};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

pub struct DrawingSpherical2d<N, S> {
    indices: Vec<N>,
    coordinates: Vec<MetricSpherical2d<S>>,
    index_map: HashMap<N, usize>,
}

impl<N, S> DrawingSpherical2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: Default,
    {
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<N>>();
        Self::from_node_indices(&indices)
    }

    pub fn from_node_indices(indices: &[N]) -> Self
    where
        N: Copy,
        S: Default,
    {
        let indices = indices.to_vec();
        let index_map = indices
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let coordinates = vec![MetricSpherical2d::default(); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
        }
    }

    pub fn lon(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0)
    }

    pub fn lat(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1)
    }

    pub fn set_lon(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = value)
    }

    pub fn set_lat(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.1 = value)
    }

    pub fn initial_placement<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut drawing = Self::new(graph);
        let n = drawing.len();
        let d = S::PI() * S::from_usize(2).unwrap() / S::from_usize(n).unwrap();
        for i in 0..n {
            drawing.coordinates[i].0 = d * S::from_usize(i).unwrap();
            drawing.coordinates[i].1 = S::PI() / S::from_usize(4).unwrap();
        }
        drawing
    }
}

impl<N, S> Drawing for DrawingSpherical2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricSpherical2d<S>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dimension(&self) -> usize {
        2
    }

    fn position(&self, u: Self::Index) -> Option<&Self::Item> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    fn position_mut(&mut self, u: Self::Index) -> Option<&mut Self::Item> {
        self.index_map.get(&u).map(|&i| &mut self.coordinates[i])
    }

    fn node_id(&self, i: usize) -> &Self::Index {
        &self.indices[i]
    }

    fn index(&self, u: Self::Index) -> usize {
        self.index_map[&u]
    }

    fn raw_entry(&self, i: usize) -> &Self::Item {
        &self.coordinates[i]
    }

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item {
        &mut self.coordinates[i]
    }

    fn delta(&self, i: usize, j: usize) -> DeltaSpherical2d<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
