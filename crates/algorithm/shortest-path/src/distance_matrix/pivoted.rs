use super::{collect_node_index_map, DistanceMatrix, IndexIterator};
use ndarray::prelude::*;
use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

/// A distance matrix where the rows and columns can represent different sets of nodes.
///
/// This implementation uses an `ndarray::Array2` internally.
/// Node identifiers (`N`) are mapped to `usize` indices for both rows and columns.
/// A distance matrix storing shortest paths from a specific set of pivot nodes to all nodes in a graph.
#[derive(Debug, Clone)]
pub struct PivotedDistanceMatrix<N, S> {
    /// Vector of node identifiers that make up the rows (pivot nodes).
    row_indices: Vec<N>,
    /// Hash map from node identifier to row index.
    row_index_map: HashMap<N, usize>,
    /// Vector of node identifiers that make up the columns (typically all nodes).
    col_indices: Vec<N>,
    /// Hash map from node identifier to column index.
    col_index_map: HashMap<N, usize>,
    /// The underlying 2D array storing distances. `d[[row, col]]`
    d: Array2<S>,
}

pub type SubDistanceMatrix<N, S> = PivotedDistanceMatrix<N, S>;

impl<N, S> DistanceMatrix<N, S> for PivotedDistanceMatrix<N, S>
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
        (self.row_indices.len(), self.col_indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.row_index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.col_index_map.get(&u).copied()
    }

    fn row_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator::new(&self.row_indices)
    }

    fn col_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator::new(&self.col_indices)
    }
}

impl<N, S> petgraph_linalg_kernel::Distance<N, S> for PivotedDistanceMatrix<N, S>
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
        (self.row_indices.len(), self.col_indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.row_index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.col_index_map.get(&u).copied()
    }
}

impl<N, S> PivotedDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    /// Returns the pivot nodes.
    pub fn pivots(&self) -> &[N] {
        &self.row_indices
    }

    /// Creates an empty `PivotedDistanceMatrix` for the given graph context.
    ///
    /// The resulting matrix will have zero rows and columns corresponding to all nodes in the graph.
    /// This is useful as a starting point if rows (source nodes) are added dynamically later using `push`.
    pub fn empty<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        Self::new(graph, &[])
    }

    /// Creates a new `PivotedDistanceMatrix` for the given graph and a specific set of source nodes.
    ///
    /// The rows of the matrix correspond to the `sources` provided, and the columns correspond
    /// to all nodes in the `graph`. All distances are initialized to infinity.
    pub fn new<G>(graph: G, sources: &[G::NodeId]) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let row_indices = sources.iter().map(|&u| u.into()).collect::<Vec<_>>();
        let mut row_index_map = HashMap::with_capacity(row_indices.len());
        for (i, &u) in row_indices.iter().enumerate() {
            row_index_map.insert(u, i);
        }
        let (col_indices, col_index_map) = collect_node_index_map(graph);
        let d = Array::from_elem((row_indices.len(), col_indices.len()), S::infinity());
        Self {
            row_indices,
            row_index_map,
            col_indices,
            col_index_map,
            d,
        }
    }

    /// Creates a `PivotedDistanceMatrix` backed by a pre-computed 2D array of shape `(pivots.len(), n)`.
    ///
    /// # Panics
    ///
    /// Panics if `array.dim() != (pivots.len(), n)`, where `n` is the number of nodes in `graph`,
    /// or if any pivot node is not present in `graph`.
    pub fn from_array<G>(graph: G, pivots: Vec<G::NodeId>, array: Array2<S>) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let (col_indices, col_index_map) = collect_node_index_map(graph);
        let n = col_indices.len();
        let row_indices = pivots.into_iter().map(|u| u.into()).collect::<Vec<_>>();
        let mut row_index_map = HashMap::with_capacity(row_indices.len());
        for (i, &u) in row_indices.iter().enumerate() {
            assert!(
                col_index_map.contains_key(&u),
                "PivotedDistanceMatrix::from_array: pivot node not found in graph"
            );
            row_index_map.insert(u, i);
        }
        assert_eq!(
            array.dim(),
            (row_indices.len(), n),
            "PivotedDistanceMatrix::from_array: expected shape ({}, {}), got {:?}",
            row_indices.len(),
            n,
            array.dim()
        );
        Self {
            row_indices,
            row_index_map,
            col_indices,
            col_index_map,
            d: array,
        }
    }

    /// Returns a reference to the underlying `(pivots.len(), n)` distance array.
    pub fn as_array(&self) -> &Array2<S> {
        &self.d
    }

    /// Adds a new row to the `PivotedDistanceMatrix` corresponding to the node identifier `u`.
    ///
    /// The new row is initialized with infinity values.
    /// This allows adding source nodes dynamically after initial creation.
    pub fn push(&mut self, u: N)
    where
        N: Copy,
    {
        self.row_index_map.insert(u, self.row_indices.len());
        self.row_indices.push(u);
        self.d
            .push(
                Axis(0),
                Array::from_elem(self.col_indices.len(), S::infinity()).view(),
            )
            .ok();
    }

    fn index(&self, u: N, v: N) -> Option<(usize, usize)> {
        self.row_index_map
            .get(&u)
            .zip(self.col_index_map.get(&v))
            .map(|(&i, &j)| (i, j))
    }
}
