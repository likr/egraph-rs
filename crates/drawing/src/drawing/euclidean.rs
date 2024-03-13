use crate::{drawing::Drawing, metric::euclidean::MetricEuclidean, DrawingIndex, DrawingValue};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

pub struct DrawingEuclidean<N, S> {
    indices: Vec<N>,
    coordinates: Vec<MetricEuclidean<S>>,
    index_map: HashMap<N, usize>,
    dimension: usize,
}

impl<N, S> DrawingEuclidean<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn new<G>(graph: G, dimension: usize) -> Self
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
        Self::from_node_indices(&indices, dimension)
    }

    pub fn from_node_indices(indices: &[N], dimension: usize) -> Self
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
        let coordinates = vec![MetricEuclidean::new(dimension); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
            dimension,
        }
    }
}

// pub fn iter(&self) -> DrawingIterator<N, M>
// where
//     N: Copy,
// {
//     DrawingIterator {
//         drawing: self,
//         index: 0,
//     }
// }

impl<N, S> Drawing for DrawingEuclidean<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricEuclidean<S>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn position(&self, u: N) -> Option<&Self::Item> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    fn position_mut(&mut self, u: N) -> Option<&mut Self::Item> {
        self.index_map.get(&u).map(|&i| &mut self.coordinates[i])
    }

    fn raw_entry(&self, i: usize) -> &Self::Item {
        &self.coordinates[i]
    }

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item {
        &mut self.coordinates[i]
    }
}

// pub struct DrawingIterator<'a, N, M> {
//     drawing: &'a DrawingEuclidean<N, M>,
//     index: usize,
// }
//
// impl<'a, N, M> Iterator for DrawingIterator<'a, N, M>
// where
//     N: Copy + DrawingIndex,
//     M: Metric + Default + Copy,
// {
//     type Item = (N, M);
//     fn next(&mut self) -> Option<Self::Item> {
//         let index = self.index;
//         self.index += 1;
//         if index < self.drawing.coordinates.len() {
//             Some((self.drawing.indices[index], self.drawing.coordinates[index]))
//         } else {
//             None
//         }
//     }
// }
//
