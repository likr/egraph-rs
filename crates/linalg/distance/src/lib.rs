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

use std::hash::Hash;

/// A distance matrix wrapping a kernel, representing distance in the kernel space.
/// d_K(i, j) = sqrt(K(i, i) + K(j, j) - 2 * K(i, j))
#[derive(Debug, Clone, Copy)]
pub struct KernelDistance<K, S> {
    pub kernel: K,
    pub min_dist: S,
}

impl<K, S> KernelDistance<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            min_dist: S::zero(),
        }
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }
}

impl<N, S, K> Distance<N, S> for KernelDistance<K, S>
where
    N: Eq + Hash + Copy,
    K: Kernel<N, S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ii = self.kernel.get_by_index(i, i);
        let k_jj = self.kernel.get_by_index(j, j);
        let k_ij = self.kernel.get_by_index(i, j);
        let diff = k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij;
        diff.max(S::zero()).sqrt().max(self.min_dist)
    }

    fn shape(&self) -> (usize, usize) {
        self.kernel.shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.kernel.row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.kernel.col_index(u)
    }
}

/// Gaussian Kernel applied to a base Kernel: K(x, y) = exp(-gamma * d_K(x, y)^2)
#[derive(Debug, Clone, Copy)]
pub struct GaussianKernel<K, S> {
    pub distance: KernelDistance<K, S>,
    pub gamma: S,
}

impl<K, S> GaussianKernel<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K, gamma: S) -> Self {
        Self {
            distance: KernelDistance::new(kernel),
            gamma,
        }
    }
}

impl<N, S, K> Kernel<N, S> for GaussianKernel<K, S>
where
    N: Eq + Hash + Copy,
    K: Kernel<N, S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let d = self.distance.get_by_index(i, j);
        (-self.gamma * d * d).exp()
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

/// Exponential Kernel applied to a base Kernel: K(x, y) = exp(-gamma * d_K(x, y))
#[derive(Debug, Clone, Copy)]
pub struct ExponentialKernel<K, S> {
    pub distance: KernelDistance<K, S>,
    pub gamma: S,
}

impl<K, S> ExponentialKernel<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K, gamma: S) -> Self {
        Self {
            distance: KernelDistance::new(kernel),
            gamma,
        }
    }
}

impl<N, S, K> Kernel<N, S> for ExponentialKernel<K, S>
where
    N: Eq + Hash + Copy,
    K: Kernel<N, S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let d = self.distance.get_by_index(i, j);
        (-self.gamma * d).exp()
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

/// T-Kernel applied to a base Kernel: K(x, y) = 1 / (1 + d_K(x, y)^2 / alpha)^((alpha + 1) / 2)
#[derive(Debug, Clone, Copy)]
pub struct TKernel<K, S> {
    pub distance: KernelDistance<K, S>,
    pub alpha: S,
}

impl<K, S> TKernel<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K, alpha: S) -> Self {
        Self {
            distance: KernelDistance::new(kernel),
            alpha,
        }
    }
}

impl<N, S, K> Kernel<N, S> for TKernel<K, S>
where
    N: Eq + Hash + Copy,
    K: Kernel<N, S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let d = self.distance.get_by_index(i, j);
        let one = S::one();
        let val = one + d * d / self.alpha;
        let power = (self.alpha + one) / S::from_f32(2.0).unwrap();
        one / val.powf(power)
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
