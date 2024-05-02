use crate::{DeltaHyperbolic2d, Drawing, DrawingIndex, DrawingValue, MetricHyperbolic2d};
use num_traits::{FloatConst, FromPrimitive};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

pub struct DrawingHyperbolic2d<N, S> {
    indices: Vec<N>,
    coordinates: Vec<MetricHyperbolic2d<S>>,
    index_map: HashMap<N, usize>,
}

impl<N, S> DrawingHyperbolic2d<N, S>
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
        let coordinates = vec![MetricHyperbolic2d::default(); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
        }
    }

    pub fn x(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0)
    }

    pub fn y(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1)
    }

    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = value)
    }

    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
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
            drawing.coordinates[i].0 =
                S::from_f32(0.5).unwrap() * (S::from_usize(i).unwrap() * d).cos();
            drawing.coordinates[i].1 =
                S::from_f32(0.5).unwrap() * (S::from_usize(i).unwrap() * d).sin();
        }
        drawing
    }
}

impl<N, S> Drawing for DrawingHyperbolic2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricHyperbolic2d<S>;

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

    fn index(&self, i: usize) -> &Self::Index {
        &self.indices[i]
    }

    fn raw_entry(&self, i: usize) -> &Self::Item {
        &self.coordinates[i]
    }

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item {
        &mut self.coordinates[i]
    }

    fn delta(&self, i: usize, j: usize) -> DeltaHyperbolic2d<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
