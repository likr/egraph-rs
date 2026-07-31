//! Sparse symmetric matrix representation and operations.

use ndarray::Array1;
use num_traits::Zero;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// A sparse symmetric matrix represented in edge list format.
///
/// This structure stores only the lower triangular part of the matrix
/// (where i < j) to avoid redundancy. For symmetric matrices, A[i,j] = A[j,i],
/// so we only need to store one of these values.
///
/// The matrix is represented as:
/// - A list of edges (i, j, value) where i < j (off-diagonal elements)
/// - A vector of diagonal elements
///
/// This representation is efficient for graph Laplacians and other sparse
/// symmetric matrices commonly used in spectral graph theory.
#[derive(Debug, Clone)]
pub struct SparseSymmetricMatrix<T> {
    /// Number of rows/columns in the matrix
    n: usize,
    /// Off-diagonal entries: (row, col, value) where row < col
    edges: Vec<(usize, usize, T)>,
    /// Diagonal entries: diagonal[i] is the (i,i) element
    diagonal: Vec<T>,
}

impl<T> SparseSymmetricMatrix<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign + Default + Zero,
{
    /// Creates a new sparse symmetric matrix with specified dimension.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            diagonal: vec![T::default(); n],
        }
    }

    /// Creates a sparse symmetric matrix from edges and diagonal elements.
    pub fn from_parts(n: usize, edges: Vec<(usize, usize, T)>, diagonal: Vec<T>) -> Self {
        for &(i, j, _) in &edges {
            assert!(i < j, "Edges must have i < j for lower triangular storage");
            assert!(i < n && j < n, "Edge indices out of bounds");
        }
        assert_eq!(diagonal.len(), n, "Diagonal length must match dimension");

        Self { n, edges, diagonal }
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Sets a diagonal element.
    pub fn set_diagonal(&mut self, i: usize, value: T) {
        assert!(i < self.n, "Index out of bounds");
        self.diagonal[i] = value;
    }

    /// Adds to a diagonal element.
    pub fn add_to_diagonal(&mut self, i: usize, value: T) {
        assert!(i < self.n, "Index out of bounds");
        self.diagonal[i] += value;
    }

    /// Adds an off-diagonal edge.
    pub fn add_edge(&mut self, i: usize, j: usize, value: T) {
        assert!(i < j, "i must be < j for lower triangular storage");
        assert!(i < self.n && j < self.n, "Indices out of bounds");
        self.edges.push((i, j, value));
    }

    /// Computes the matrix-vector product: y = A * x
    pub fn multiply(&self, x: &Array1<T>) -> Array1<T> {
        assert_eq!(x.len(), self.n, "Vector dimension mismatch");

        let mut y = Array1::zeros(self.n);

        for i in 0..self.n {
            y[i] = self.diagonal[i] * x[i];
        }

        for &(i, j, value) in &self.edges {
            y[i] += value * x[j];
            y[j] += value * x[i];
        }

        y
    }

    /// Computes the matrix-vector product in-place: y = A * x
    pub fn multiply_into(&self, x: &Array1<T>, y: &mut Array1<T>) {
        assert_eq!(x.len(), self.n, "Input vector dimension mismatch");
        assert_eq!(y.len(), self.n, "Output vector dimension mismatch");

        for i in 0..self.n {
            y[i] = self.diagonal[i] * x[i];
        }

        for &(i, j, value) in &self.edges {
            y[i] += value * x[j];
            y[j] += value * x[i];
        }
    }

    /// Returns a reference to the edges.
    pub fn edges(&self) -> &[(usize, usize, T)] {
        &self.edges
    }

    /// Returns a reference to the diagonal.
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }
}

impl<T> SparseSymmetricMatrix<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Default
        + PartialOrd,
{
    /// Creates a scaled and shifted matrix: (scale * A) - shift * I
    pub fn scale_and_shift(&self, scale: T, shift: T) -> Self {
        let mut edges = Vec::with_capacity(self.edges.len());
        for &(i, j, value) in &self.edges {
            edges.push((i, j, scale * value));
        }

        let mut diagonal = Vec::with_capacity(self.n);
        for &d in &self.diagonal {
            diagonal.push(scale * d - shift);
        }

        Self {
            n: self.n,
            edges,
            diagonal,
        }
    }
}

impl<T> SparseSymmetricMatrix<T>
where
    T: DrawingValue + Default,
{
    /// Builds standard graph Laplacian matrix: L = D - A
    pub fn standard_laplacian<G, F>(graph: G, mut length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> T,
    {
        let n = graph.node_count();
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        let mut matrix = SparseSymmetricMatrix::new(n);
        let mut degrees = vec![T::zero(); n];

        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];
            let weight = length(edge);

            if i != j {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                matrix.add_edge(min_idx, max_idx, -weight);
                degrees[i] += weight;
                degrees[j] += weight;
            }
        }

        for (i, &deg) in degrees.iter().enumerate().take(n) {
            matrix.set_diagonal(i, deg);
        }

        matrix
    }

    /// Builds symmetric normalized graph Laplacian matrix: L_sym = D^{-1/2} L D^{-1/2}
    pub fn symmetric_normalized_laplacian<G, F>(graph: G, mut length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> T,
    {
        let n = graph.node_count();
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        let mut degrees = vec![T::zero(); n];
        let mut raw_edges = Vec::new();

        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];
            let weight = length(edge);

            if i != j {
                degrees[i] += weight;
                degrees[j] += weight;
                raw_edges.push((i, j, weight));
            }
        }

        let mut matrix = SparseSymmetricMatrix::new(n);

        for (i, &deg) in degrees.iter().enumerate() {
            if deg > T::zero() {
                matrix.set_diagonal(i, T::one());
            } else {
                matrix.set_diagonal(i, T::zero());
            }
        }

        for (i, j, w) in raw_edges {
            let deg_i = degrees[i];
            let deg_j = degrees[j];
            if deg_i > T::zero() && deg_j > T::zero() {
                let norm_w = -w / (deg_i * deg_j).sqrt();
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                matrix.add_edge(min_idx, max_idx, norm_w);
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        assert_eq!(matrix.dim(), 3);
        assert_eq!(matrix.edges().len(), 0);
        assert_eq!(matrix.diagonal().len(), 3);
    }

    #[test]
    fn test_multiply_identity() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 1.0);
        matrix.set_diagonal(1, 1.0);
        matrix.set_diagonal(2, 1.0);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = matrix.multiply(&x);

        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 3.0);
    }
}
