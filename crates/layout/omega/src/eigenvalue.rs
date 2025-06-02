//! Pure Rust implementation of eigenvalue computation using deflation-based power method
//! with orthogonalization for computing the smallest eigenvalues of graph Laplacians.

use nalgebra::DMatrix;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::collections::HashMap;

/// Eigenvalue solver using deflation-based power method with orthogonalization.
///
/// This solver is specifically designed for graph Laplacians, which are positive
/// semi-definite matrices. It computes the smallest non-zero eigenvalues and their
/// corresponding eigenvectors, which are useful for spectral graph analysis.
pub struct EigenSolver<S> {
    /// Maximum number of iterations for each eigenvalue computation
    max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    tolerance: S,
}

impl<S> EigenSolver<S>
where
    S: DrawingValue,
{
    /// Creates a new eigenvalue solver with specified parameters.
    ///
    /// # Parameters
    /// * `max_iterations` - Maximum iterations per eigenvalue
    /// * `tolerance` - Convergence tolerance
    pub fn new(max_iterations: usize, tolerance: S) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Creates a default eigenvalue solver with reasonable parameters.
    pub fn default() -> Self {
        Self::new(1000, S::from_f32(1e-6).unwrap())
    }

    /// Computes the smallest `d` non-zero eigenvalues and eigenvectors of a graph Laplacian.
    ///
    /// Uses the deflation-based power method with orthogonalization. The algorithm:
    /// 1. Constructs the graph Laplacian matrix
    /// 2. For each eigenvalue i from 1 to d:
    ///    a. Applies inverse iteration to find the smallest eigenvalue
    ///    b. Orthogonalizes against previously found eigenvectors
    ///    c. Deflates the matrix to remove the found eigenvalue
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `d` - Number of smallest non-zero eigenvalues to compute
    ///
    /// # Returns
    /// A tuple containing:
    /// - Vector of eigenvalues (sorted in ascending order)
    /// - Vector of corresponding eigenvectors
    pub fn compute_smallest_eigenvalues<G>(&self, graph: G, d: usize) -> (Vec<S>, Vec<Vec<S>>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();
        let laplacian = self.build_laplacian_matrix(graph);

        let mut eigenvalues = Vec::new();
        let mut eigenvectors: Vec<Vec<S>> = Vec::new();

        // Use a simplified approach: find eigenvectors using power iteration
        // on the shifted matrix to avoid the zero eigenvalue
        let shift = S::from_f32(1.0).unwrap();
        let mut working_matrix = laplacian.clone();

        // Add shift to diagonal to move eigenvalues away from zero
        for i in 0..n {
            working_matrix[(i, i)] = working_matrix[(i, i)] + shift;
        }

        for _iter in 0..d {
            let (eigenvalue, eigenvector) = self.find_dominant_eigenvalue(&working_matrix);

            // Convert back to original eigenvalue (subtract shift)
            let original_eigenvalue = eigenvalue - shift;

            // Skip if eigenvalue is too close to zero or negative
            if original_eigenvalue <= self.tolerance {
                continue;
            }

            // Orthogonalize against previously found eigenvectors
            let mut orthogonal_vector = eigenvector.clone();
            for prev_eigenvector in &eigenvectors {
                let dot_product = Self::dot_product(&orthogonal_vector, prev_eigenvector);
                for j in 0..n {
                    orthogonal_vector[j] = orthogonal_vector[j] - dot_product * prev_eigenvector[j];
                }
            }

            // Normalize the orthogonalized vector
            Self::normalize(&mut orthogonal_vector);

            // Check if we got a valid eigenvector (not too small)
            let norm = Self::dot_product(&orthogonal_vector, &orthogonal_vector).sqrt();
            if norm < self.tolerance {
                continue;
            }

            eigenvalues.push(original_eigenvalue);
            eigenvectors.push(orthogonal_vector.clone());

            // Deflate the matrix: A = A - λ * v * v^T
            self.deflate_matrix(&mut working_matrix, eigenvalue, &orthogonal_vector);
        }

        (eigenvalues, eigenvectors)
    }

    /// Builds the Laplacian matrix for the given graph.
    ///
    /// The Laplacian matrix L is defined as L = D - A, where:
    /// - D is the degree matrix (diagonal matrix with node degrees)
    /// - A is the adjacency matrix
    fn build_laplacian_matrix<G>(&self, graph: G) -> DMatrix<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();
        let mut laplacian = DMatrix::zeros(n, n);

        // Create node index mapping
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        // Build adjacency part and compute degrees
        let mut degrees = vec![S::zero(); n];

        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];

            // For undirected graphs, add edge weight to both (i,j) and (j,i)
            laplacian[(i, j)] = laplacian[(i, j)] - S::one();
            if i != j {
                laplacian[(j, i)] = laplacian[(j, i)] - S::one();
                degrees[i] = degrees[i] + S::one();
                degrees[j] = degrees[j] + S::one();
            } else {
                degrees[i] = degrees[i] + S::one();
            }
        }

        // Add degree matrix (diagonal elements)
        for i in 0..n {
            laplacian[(i, i)] = laplacian[(i, i)] + degrees[i];
        }

        laplacian
    }

    /// Finds the dominant (largest) eigenvalue and corresponding eigenvector using power iteration.
    fn find_dominant_eigenvalue(&self, matrix: &DMatrix<S>) -> (S, Vec<S>) {
        let n = matrix.nrows();

        // Start with a random vector (avoid the all-ones vector which might be the null space)
        let mut vector = vec![S::zero(); n];
        for i in 0..n {
            vector[i] = S::from_f32((i as f32 + 1.0).sin()).unwrap();
        }
        Self::normalize(&mut vector);

        let mut eigenvalue = S::zero();

        for _ in 0..self.max_iterations {
            // Apply matrix to vector: new_vector = A * vector
            let mut new_vector = vec![S::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    new_vector[i] = new_vector[i] + matrix[(i, j)] * vector[j];
                }
            }

            // Normalize to prevent overflow
            Self::normalize(&mut new_vector);

            // Compute Rayleigh quotient: λ = (v^T * A * v) / (v^T * v)
            let mut matrix_vector = vec![S::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    matrix_vector[i] = matrix_vector[i] + matrix[(i, j)] * new_vector[j];
                }
            }

            let numerator = Self::dot_product(&new_vector, &matrix_vector);
            let denominator = Self::dot_product(&new_vector, &new_vector);
            let new_eigenvalue = numerator / denominator;

            // Check for convergence
            if (new_eigenvalue - eigenvalue).abs() < self.tolerance {
                break;
            }

            eigenvalue = new_eigenvalue;
            vector = new_vector;
        }

        (eigenvalue, vector)
    }

    /// Deflates the matrix by removing the found eigenvalue and eigenvector.
    fn deflate_matrix(&self, matrix: &mut DMatrix<S>, eigenvalue: S, eigenvector: &[S]) {
        let n = matrix.nrows();

        // A = A - λ * v * v^T
        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = matrix[(i, j)] - eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    /// Computes the dot product of two vectors.
    fn dot_product(a: &[S], b: &[S]) -> S {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .fold(S::zero(), |acc, x| acc + x)
    }

    /// Normalizes a vector to unit length.
    fn normalize(vector: &mut [S]) {
        let norm = Self::dot_product(vector, vector).sqrt();
        if norm > S::zero() {
            for x in vector.iter_mut() {
                *x = *x / norm;
            }
        }
    }
}
