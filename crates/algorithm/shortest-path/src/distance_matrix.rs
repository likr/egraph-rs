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

fn collect_node_index_map<G, N>(graph: G) -> (Vec<N>, HashMap<N, usize>)
where
    G: IntoNodeIdentifiers,
    G::NodeId: Into<N>,
    N: Eq + Hash + Copy,
{
    let indices = graph
        .node_identifiers()
        .map(|u| u.into())
        .collect::<Vec<_>>();
    let mut index_map = HashMap::with_capacity(indices.len());
    for (i, &u) in indices.iter().enumerate() {
        index_map.insert(u, i);
    }
    (indices, index_map)
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

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{NodeIndex, UnGraph};

    #[test]
    fn test_full_distance_matrix_from_and_as_array() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let _n1 = g.add_node(());
        let n2 = g.add_node(());

        let arr = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let dm: FullDistanceMatrix<NodeIndex, f64> =
            FullDistanceMatrix::from_array(&g, arr.clone());
        assert_eq!(dm.shape(), (3, 3));
        assert_eq!(dm.as_array(), &arr);
        assert_eq!(dm.get(n0, n2), Some(2.0));
        assert_eq!(dm.get_by_index(1, 2), 1.5);
    }

    #[test]
    #[should_panic(expected = "FullDistanceMatrix::from_array")]
    fn test_full_distance_matrix_from_array_bad_shape() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        g.add_node(());
        g.add_node(());
        let arr = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5]]);
        let _: FullDistanceMatrix<NodeIndex, f64> = FullDistanceMatrix::from_array(&g, arr);
    }

    #[test]
    fn test_pivoted_distance_matrix_from_and_as_array() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());

        let pivots = vec![n2, n0];
        let arr = arr2(&[[2.0, 1.5, 0.0], [0.0, 1.0, 2.0]]);
        let dm: PivotedDistanceMatrix<NodeIndex, f64> =
            PivotedDistanceMatrix::from_array(&g, pivots.clone(), arr.clone());
        assert_eq!(dm.shape(), (2, 3));
        assert_eq!(dm.pivots(), &pivots);
        assert_eq!(dm.as_array(), &arr);
        assert_eq!(dm.get(n2, n1), Some(1.5));
        assert_eq!(dm.get(n0, n2), Some(2.0));
        assert_eq!(dm.get(n1, n0), None);
        assert_eq!(dm.get_by_index(0, 1), 1.5);
    }

    #[test]
    #[should_panic(expected = "PivotedDistanceMatrix::from_array")]
    fn test_pivoted_distance_matrix_from_array_bad_shape() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let _n2 = g.add_node(());
        let arr = arr2(&[[0.0, 1.0], [1.0, 0.0], [2.0, 1.5]]);
        let _: PivotedDistanceMatrix<NodeIndex, f64> =
            PivotedDistanceMatrix::from_array(&g, vec![n0, n1], arr);
    }
}
