mod full;
mod pivoted;

use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

pub use full::FullDistanceMatrix;
pub use pivoted::{PivotedDistanceMatrix, SubDistanceMatrix};

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

impl<'a, N> IndexIterator<'a, N> {
    pub(super) fn new(indices: &'a Vec<N>) -> Self {
        Self { indices, index: 0 }
    }
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

pub(super) fn collect_node_index_map<G, N>(graph: G) -> (Vec<N>, HashMap<N, usize>)
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

#[cfg(test)]
mod tests;
