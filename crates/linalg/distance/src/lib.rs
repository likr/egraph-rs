mod sparse_symmetric_matrix;

pub use sparse_symmetric_matrix::SparseSymmetricMatrix;

use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};

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

/// Standard Graph Laplacian: L = D - A
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardLaplacian;

impl<G, S> Laplacian<G, S> for StandardLaplacian
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    S: DrawingValue + Default,
{
    fn build(
        &self,
        graph: G,
        length: &mut impl FnMut(G::EdgeRef) -> S,
    ) -> SparseSymmetricMatrix<S> {
        SparseSymmetricMatrix::standard_laplacian(graph, length)
    }
}

/// Symmetric Normalized Graph Laplacian: L_sym = D^{-1/2} L D^{-1/2}
#[derive(Debug, Clone, Copy, Default)]
pub struct SymmetricNormalizedLaplacian;

impl<G, S> Laplacian<G, S> for SymmetricNormalizedLaplacian
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    S: DrawingValue + Default,
{
    fn build(
        &self,
        graph: G,
        length: &mut impl FnMut(G::EdgeRef) -> S,
    ) -> SparseSymmetricMatrix<S> {
        SparseSymmetricMatrix::symmetric_normalized_laplacian(graph, length)
    }
}

/// A kernel matrix interface.
pub trait Kernel<S> {
    /// Returns the kernel value for identical points K(i, j).
    fn get(&self, i: usize, j: usize) -> S;

    /// Returns the number of nodes in the graph.
    fn n(&self) -> usize;
}

use std::marker::PhantomData;

/// Gaussian Kernel applied to a distance matrix: K(x, y) = exp(-gamma * d(x, y)^2)
#[derive(Debug, Clone, Copy)]
pub struct GaussianKernel<N, D, S> {
    pub distance: D,
    pub gamma: S,
    _marker: PhantomData<N>,
}

impl<N, D, S> GaussianKernel<N, D, S> {
    pub fn new(distance: D, gamma: S) -> Self {
        Self {
            distance,
            gamma,
            _marker: PhantomData,
        }
    }
}

impl<N, D, S> Kernel<S> for GaussianKernel<N, D, S>
where
    D: Distance<N, S>,
    S: DrawingValue,
{
    fn get(&self, i: usize, j: usize) -> S {
        let d = self.distance.get_by_index(i, j);
        (-self.gamma * d * d).exp()
    }

    fn n(&self) -> usize {
        self.distance.shape().0
    }
}

/// A distance matrix wrapping a kernel, representing distance in the kernel space.
/// d_K(i, j) = sqrt(K(i, i) + K(j, j) - 2 * K(i, j))
#[derive(Debug, Clone)]
pub struct KernelDistance<K> {
    pub kernel: K,
}

impl<K> KernelDistance<K> {
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }
}

impl<N, S, K> Distance<N, S> for KernelDistance<K>
where
    K: Kernel<S>,
    S: DrawingValue,
{
    fn get(&self, _u: N, _v: N) -> Option<S> {
        None
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ii = self.kernel.get(i, i);
        let k_jj = self.kernel.get(j, j);
        let k_ij = self.kernel.get(i, j);
        let diff = k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij;
        diff.max(S::zero()).sqrt()
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.kernel.n();
        (n, n)
    }

    fn row_index(&self, _u: N) -> Option<usize> {
        None
    }

    fn col_index(&self, _u: N) -> Option<usize> {
        None
    }
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
