use ndarray::ScalarOperand;
pub mod distance;
pub mod kernel;
pub mod laplacian;
pub mod solver;

pub use distance::*;
pub use kernel::*;
pub use laplacian::*;
pub use solver::*;

use ndarray::{Array1, Array2};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::hash::Hash;

/// A trait representing a read-only distance matrix.
pub trait Distance<N, S> {
    /// Returns the distance between nodes `u` (row) and `v` (column).
    fn get(&self, u: N, v: N) -> Option<S>;

    /// Returns the distance between the node at row index `i` and the node at column index `j`.
    fn get_by_index(&self, i: usize, j: usize) -> S;

    /// Returns the dimensions (number of rows, number of columns) of the distance matrix.
    fn shape(&self) -> (usize, usize);

    /// Returns the row index associated with node identifier `u`.
    fn row_index(&self, u: N) -> Option<usize>;

    /// Returns the column index associated with node identifier `u`.
    fn col_index(&self, u: N) -> Option<usize>;
}

/// A trait for constructing a graph Laplacian matrix.
pub trait Laplacian<G, S>
where
    G: petgraph::visit::IntoEdgeReferences,
{
    /// Builds the Laplacian matrix from the graph and the edge length/weight function.
    fn build(&self, graph: G, length: &mut impl FnMut(G::EdgeRef) -> S)
        -> SparseSymmetricMatrix<S>;
}

/// A kernel matrix interface.
pub trait Kernel<N, S> {
    /// Returns the kernel value for nodes `u` (row) and `v` (column).
    fn get(&self, u: N, v: N) -> Option<S>;

    /// Returns the kernel value for identical points K(i, j).
    fn get_by_index(&self, i: usize, j: usize) -> S;

    /// Returns the dimensions (number of rows, number of columns) of the kernel matrix.
    fn shape(&self) -> (usize, usize);

    /// Returns the row index associated with node identifier `u`.
    fn row_index(&self, u: N) -> Option<usize>;

    /// Returns the column index associated with node identifier `u`.
    fn col_index(&self, u: N) -> Option<usize>;
}

impl<N, S, T> Distance<N, S> for &T
where
    T: Distance<N, S> + ?Sized,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        (**self).get(u, v)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        (**self).get_by_index(i, j)
    }

    fn shape(&self) -> (usize, usize) {
        (**self).shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        (**self).row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        (**self).col_index(u)
    }
}

impl<N, S, T> Kernel<N, S> for &T
where
    T: Kernel<N, S> + ?Sized,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        (**self).get(u, v)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        (**self).get_by_index(i, j)
    }

    fn shape(&self) -> (usize, usize) {
        (**self).shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        (**self).row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        (**self).col_index(u)
    }
}

impl<N, S, K> Kernel<N, S> for Box<K>
where
    K: Kernel<N, S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        (**self).get(u, v)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        (**self).get_by_index(i, j)
    }

    fn shape(&self) -> (usize, usize) {
        (**self).shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        (**self).row_index(u)
    }

    fn col_index(&self, v: N) -> Option<usize> {
        (**self).col_index(v)
    }
}

/// A kernel matrix that computes elements from specific pivot nodes.
pub trait PivotedKernel<S> {
    /// Returns the indices of the pivot nodes.
    fn pivots(&self) -> &[usize];

    /// Queries the (pivot_idx, j) element of the kernel matrix, where `pivot_idx` is the index into the `pivots()` array.
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S;
}
