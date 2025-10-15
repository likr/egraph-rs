use ndarray::prelude::*;
use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

/// A trait representing a distance matrix, used for graph algorithms.
///
/// This trait provides methods to access and modify distances between nodes,
/// represented by both their identifier (`N`) and their internal index (`usize`).
/// The nodes in the rows and columns can be different sets.
///
/// `N` is the type of the node identifier (e.g., `NodeIndex`).
/// `S` is the type of the distance value, typically a floating-point type implementing `NdFloat`.
pub trait DistanceMatrix<N, S> {
    /// Returns the distance between nodes `u` (row) and `v` (column).
    ///
    /// Returns `Option::None` if either `u` is not in the row domain or `v` is not in the column domain.
    fn get(&self, u: N, v: N) -> Option<S>;

    /// Sets the distance between nodes `u` (row) and `v` (column) to `d`.
    ///
    /// Returns `Option::None` if either `u` is not in the row domain or `v` is not in the column domain.
    /// Returns `Some(())` on success.
    fn set(&mut self, u: N, v: N, d: S) -> Option<()>;

    /// Returns the distance between the node at row index `i` and the node at column index `j`.
    ///
    /// This method bypasses the hash lookup for node identifiers and is thus potentially faster than `get`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of bounds for the matrix dimensions.
    fn get_by_index(&self, i: usize, j: usize) -> S;

    /// Sets the distance between the node at row index `i` and the node at column index `j` to `d`.
    ///
    /// This method bypasses the hash lookup for node identifiers and is thus potentially faster than `set`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of bounds for the matrix dimensions.
    fn set_by_index(&mut self, i: usize, j: usize, d: S);

    /// Returns the dimensions (number of rows, number of columns) of the distance matrix.
    fn shape(&self) -> (usize, usize);

    /// Returns the row index associated with node identifier `u`.
    ///
    /// Returns `Option::None` if `u` is not found in the row indices.
    fn row_index(&self, u: N) -> Option<usize>;

    /// Returns the column index associated with node identifier `u`.
    ///
    /// Returns `Option::None` if `u` is not found in the column indices.
    fn col_index(&self, u: N) -> Option<usize>;

    /// Returns an iterator over the node identifiers corresponding to the rows.
    fn row_indices(&'_ self) -> IndexIterator<'_, N>;

    /// Returns an iterator over the node identifiers corresponding to the columns.
    fn col_indices(&'_ self) -> IndexIterator<'_, N>;
}

/// An iterator over the node identifiers in a distance matrix dimension (row or column).
pub struct IndexIterator<'a, N> {
    indices: &'a Vec<N>,
    index: usize,
}

impl<N> Iterator for IndexIterator<'_, N>
where
    N: Copy,
{
    type Item = N;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        self.index += 1;
        if index < self.indices.len() {
            Some(self.indices[index])
        } else {
            None
        }
    }
}

/// A distance matrix where the rows and columns represent the same set of nodes,
/// typically all nodes in the graph.
///
/// This implementation uses an `ndarray::Array2` internally.
/// Node identifiers (`N`) are mapped to `usize` indices for array access.
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
        IndexIterator {
            indices: &self.indices,
            index: 0,
        }
    }

    fn col_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator {
            indices: &self.indices,
            index: 0,
        }
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
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<_>>();
        let mut index_map = HashMap::new();
        for (i, &u) in indices.iter().enumerate() {
            index_map.insert(u, i);
        }
        let n = indices.len();
        Self {
            indices,
            index_map,
            d: Array::from_elem((n, n), S::infinity()),
        }
    }

    fn index(&self, u: N, v: N) -> Option<(usize, usize)> {
        self.index_map
            .get(&u)
            .zip(self.index_map.get(&v))
            .map(|(&i, &j)| (i, j))
    }
}

/// A distance matrix where the rows and columns can represent different sets of nodes.
///
/// This implementation uses an `ndarray::Array2` internally.
/// Node identifiers (`N`) are mapped to `usize` indices for both rows and columns.
/// This is useful for algorithms where distances are calculated from a subset of source nodes
/// to all other nodes (e.g., single-source shortest path).
pub struct SubDistanceMatrix<N, S> {
    /// Vector of node identifiers that make up the rows (typically source nodes).
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

impl<N, S> DistanceMatrix<N, S> for SubDistanceMatrix<N, S>
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
        IndexIterator {
            indices: &self.row_indices,
            index: 0,
        }
    }

    fn col_indices(&'_ self) -> IndexIterator<'_, N> {
        IndexIterator {
            indices: &self.col_indices,
            index: 0,
        }
    }
}

impl<N, S> SubDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    /// Creates an empty `SubDistanceMatrix` for the given graph context.
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

    /// Creates a new `SubDistanceMatrix` for the given graph and a specific set of source nodes.
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
        let mut row_index_map = HashMap::new();
        for (i, &u) in row_indices.iter().enumerate() {
            row_index_map.insert(u, i);
        }
        let col_indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<_>>();
        let mut col_index_map = HashMap::new();
        for (i, &u) in col_indices.iter().enumerate() {
            col_index_map.insert(u, i);
        }
        let d = Array::from_elem((row_indices.len(), col_indices.len()), S::infinity());
        Self {
            row_indices,
            row_index_map,
            col_indices,
            col_index_map,
            d,
        }
    }

    /// Adds a new row to the `SubDistanceMatrix` corresponding to the node identifier `u`.
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
