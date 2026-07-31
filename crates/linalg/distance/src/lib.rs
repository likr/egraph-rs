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

/// A kernel function applied to distances.
pub trait Kernel<S> {
    /// Applies the kernel function to a distance value.
    fn apply(&self, distance: S) -> S;

    /// Returns the kernel value for identical points K(x, x).
    fn self_kernel(&self) -> S;
}

/// Gaussian Kernel: K(x, y) = exp(-gamma * d(x, y)^2)
#[derive(Debug, Clone, Copy)]
pub struct GaussianKernel<S> {
    pub gamma: S,
}

impl<S> GaussianKernel<S> {
    pub fn new(gamma: S) -> Self {
        Self { gamma }
    }
}

impl<S> Kernel<S> for GaussianKernel<S>
where
    S: DrawingValue,
{
    fn apply(&self, distance: S) -> S {
        (-self.gamma * distance * distance).exp()
    }

    fn self_kernel(&self) -> S {
        S::one()
    }
}

/// A distance matrix wrapping another distance matrix with a kernel, representing distance in the kernel space.
/// d_K(u, v) = sqrt(K(u, u) + K(v, v) - 2 * K(u, v))
#[derive(Debug, Clone)]
pub struct KernelDistance<D, K> {
    pub distance: D,
    pub kernel: K,
}

impl<D, K> KernelDistance<D, K> {
    pub fn new(distance: D, kernel: K) -> Self {
        Self { distance, kernel }
    }
}

impl<N, S, D, K> Distance<N, S> for KernelDistance<D, K>
where
    D: Distance<N, S>,
    K: Kernel<S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        self.distance.get(u, v).map(|d| {
            let k_uu = self.kernel.self_kernel();
            let k_vv = self.kernel.self_kernel();
            let k_uv = self.kernel.apply(d);
            let diff = k_uu + k_vv - S::from_f32(2.0).unwrap() * k_uv;
            diff.max(S::zero()).sqrt()
        })
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let d = self.distance.get_by_index(i, j);
        let k_ii = self.kernel.self_kernel();
        let k_jj = self.kernel.self_kernel();
        let k_ij = self.kernel.apply(d);
        let diff = k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij;
        diff.max(S::zero()).sqrt()
    }

    fn shape(&self) -> (usize, usize) {
        self.distance.shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.distance.row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.distance.col_index(u)
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
