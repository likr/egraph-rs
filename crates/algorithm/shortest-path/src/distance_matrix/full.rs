use super::{collect_node_index_map, DistanceMatrix, IndexIterator};
use ndarray::prelude::*;
use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

/// A distance matrix where the rows and columns represent the same set of nodes,
/// typically all nodes in the graph.
///
/// This implementation uses an `ndarray::Array2` internally.
/// Node identifiers (`N`) are mapped to `usize` indices for array access.
#[derive(Clone)]
pub struct FullDistanceMatrix<N, S> {
    /// Vector of node identifiers, mapping index to node identifier.
    indices: Vec<N>,
    /// Hash map from node identifier to index.
    index_map: HashMap<N, usize>,
    d: Array2<S>,
}

impl<N, S> DistanceMatrix<N, S> for FullDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]])
    }

    fn set(&mut self, u: N, v: N, d: S) -> Option<()> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]] = d)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        self.d[[i, j]]
    }

    fn set_by_index(&mut self, i: usize, j: usize, d: S) {
        self.d[[i, j]] = d;
    }

    fn shape(&self) -> (usize, usize) {
        (self.indices.len(), self.indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }

    fn row_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator::new(&self.indices)
    }

    fn col_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator::new(&self.indices)
    }
}

impl<N, S> petgraph_linalg_kernel::Distance<N, S> for FullDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: NdFloat,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]])
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        self.d[[i, j]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.indices.len(), self.indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }
}

impl<N, S> FullDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let (indices, index_map) = collect_node_index_map(graph);
        let n = indices.len();
        Self {
            indices,
            index_map,
            d: Array::from_elem((n, n), S::infinity()),
        }
    }

    /// Creates a `FullDistanceMatrix` backed by a pre-computed 2D array of shape `(n, n)`.
    ///
    /// # Panics
    ///
    /// Panics if `array.dim() != (n, n)`, where `n` is the number of nodes in `graph`.
    pub fn from_array<G>(graph: G, array: Array2<S>) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let (indices, index_map) = collect_node_index_map(graph);
        let n = indices.len();
        assert_eq!(
            array.dim(),
            (n, n),
            "FullDistanceMatrix::from_array: expected shape ({}, {}), got {:?}",
            n,
            n,
            array.dim()
        );
        Self {
            indices,
            index_map,
            d: array,
        }
    }

    /// Returns a reference to the underlying `(n, n)` distance array.
    pub fn as_array(&self) -> &Array2<S> {
        &self.d
    }

    fn index(&self, u: N, v: N) -> Option<(usize, usize)> {
        self.index_map
            .get(&u)
            .zip(self.index_map.get(&v))
            .map(|(&i, &j)| (i, j))
    }
}
