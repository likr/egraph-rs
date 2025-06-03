//! Pure Rust implementation of eigenvalue computation using inverse power method
//! with Gram-Schmidt orthogonalization and Conjugate Gradient solver for computing
//! the smallest non-zero eigenvalues of graph Laplacians.

use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::collections::HashMap;

/// Eigenvalue solver using inverse power method with Gram-Schmidt orthogonalization.
///
/// This solver implements the algorithm specified for computing non-zero eigenvalues
/// of graph Laplacians using:
/// 1. Inverse power method for each eigenvalue
/// 2. Conjugate Gradient method for solving linear systems Ly = x
/// 3. Gram-Schmidt orthogonalization against previously found eigenvectors
/// 4. Sequential computation of λ2, λ3, ..., λ(N+1)
pub struct EigenSolver<S> {
    /// Maximum number of iterations for inverse power method
    max_iterations: usize,
    /// Maximum number of iterations for CG method
    cg_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    tolerance: S,
    /// Convergence tolerance for CG method
    cg_tolerance: S,
    /// Convergence tolerance for eigenvector changes
    vector_tolerance: S,
}

impl<S> EigenSolver<S>
where
    S: DrawingValue,
{
    /// Creates a new eigenvalue solver with specified parameters.
    ///
    /// # Parameters
    /// * `max_iterations` - Maximum iterations for inverse power method
    /// * `cg_max_iterations` - Maximum iterations for CG method
    /// * `tolerance` - Convergence tolerance for eigenvalues
    /// * `cg_tolerance` - Convergence tolerance for CG method
    /// * `vector_tolerance` - Convergence tolerance for eigenvectors
    pub fn new(
        max_iterations: usize,
        cg_max_iterations: usize,
        tolerance: S,
        cg_tolerance: S,
        vector_tolerance: S,
    ) -> Self {
        Self {
            max_iterations,
            cg_max_iterations,
            tolerance,
            cg_tolerance,
            vector_tolerance,
        }
    }

    /// Creates a default eigenvalue solver with reasonable parameters.
    pub fn default() -> Self {
        Self::new(
            1000,                        // max_iterations
            1000,                        // cg_max_iterations
            S::from_f32(1e-10).unwrap(), // tolerance
            S::from_f32(1e-10).unwrap(), // cg_tolerance
            S::from_f32(1e-10).unwrap(), // vector_tolerance
        )
    }

    /// Computes the smallest `n_target` non-zero eigenvalues and eigenvectors of a graph Laplacian.
    ///
    /// Implements the algorithm specified:
    /// 1. Sequential computation of λ2, λ3, ..., λ(n_target+1)
    /// 2. Inverse power method with CG solver for each eigenvalue
    /// 3. Gram-Schmidt orthogonalization against previously found eigenvectors
    /// 4. Rayleigh quotient for eigenvalue estimation
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `n_target` - Number of smallest non-zero eigenvalues to compute
    ///
    /// # Returns
    /// A tuple containing:
    /// - Vector of eigenvalues (λ2, λ3, ..., λ(n_target+1))
    /// - Vector of corresponding eigenvectors
    pub fn compute_smallest_eigenvalues<G>(
        &self,
        graph: G,
        n_target: usize,
    ) -> (Vec<S>, Vec<Vec<S>>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();

        // Initialize found eigenvectors with v1 = (1,1,...,1)^T / sqrt(n)
        let mut found_eigenvectors: Vec<Vec<S>> = Vec::new();
        let mut v1 = vec![S::one() / (n as f32).sqrt().into(); n];
        Self::normalize(&mut v1);
        found_eigenvectors.push(v1);

        let mut found_eigenvalues: Vec<S> = Vec::new();

        // For k = 1, ..., n_target: find the (k+1)-th eigenvalue λ(k+1) and eigenvector v(k+1)
        for _ in 1..=n_target {
            // Step 1: Initialize random vector and orthogonalize against found eigenvectors
            let mut x_iter = self.generate_random_vector(n);
            self.gram_schmidt_orthogonalize(&mut x_iter, &found_eigenvectors);
            Self::normalize(&mut x_iter);

            let mut lambda_prev_est = S::zero();
            let mut converged = false;

            // Step 2: Inverse power method iteration
            for _iter in 0..self.max_iterations {
                // Step 2a: Solve Ly = x_iter using CG method
                let y_solved = self.solve_with_conjugate_gradient(graph, &x_iter);

                // Step 2b: Orthogonalize y against found eigenvectors (for numerical stability)
                let mut y_orth = y_solved;
                self.gram_schmidt_orthogonalize(&mut y_orth, &found_eigenvectors);

                // Step 2c: Normalize
                Self::normalize(&mut y_orth);
                let x_next_iter = y_orth;

                // Step 2d: Compute eigenvalue estimate using Rayleigh quotient
                let lx = self.laplacian_multiply(graph, &x_next_iter);
                let numerator = Self::dot_product(&x_next_iter, &lx);
                let denominator = Self::dot_product(&x_next_iter, &x_next_iter);
                let lambda_est = numerator / denominator;

                // Step 2e: Check convergence
                let eigenvalue_converged = (lambda_est - lambda_prev_est).abs() < self.tolerance;
                let vector_converged = {
                    let diff: Vec<S> = x_next_iter
                        .iter()
                        .zip(x_iter.iter())
                        .map(|(&a, &b)| a - b)
                        .collect();
                    let diff_norm = Self::dot_product(&diff, &diff).sqrt();
                    diff_norm < self.vector_tolerance
                };

                if eigenvalue_converged || vector_converged {
                    found_eigenvalues.push(lambda_est);
                    found_eigenvectors.push(x_next_iter);
                    converged = true;
                    break;
                }

                // Step 2f: Update for next iteration
                x_iter = x_next_iter;
                lambda_prev_est = lambda_est;
            }

            if !converged {
                // If didn't converge, still store the best estimate
                let lx = self.laplacian_multiply(graph, &x_iter);
                let numerator = Self::dot_product(&x_iter, &lx);
                let denominator = Self::dot_product(&x_iter, &x_iter);
                let lambda_est = numerator / denominator;

                found_eigenvalues.push(lambda_est);
                found_eigenvectors.push(x_iter);
            }
        }

        // Return only the non-zero eigenvalues and eigenvectors (skip the first one which is v1)
        let result_eigenvectors = found_eigenvectors.into_iter().skip(1).collect();

        (found_eigenvalues, result_eigenvectors)
    }
    /// Generates a random vector of specified size.
    fn generate_random_vector(&self, n: usize) -> Vec<S> {
        let mut vector = Vec::with_capacity(n);
        for i in 0..n {
            // Use deterministic values for reproducibility
            let value = S::from_f32(((i as f32 + 1.0) * 0.123456).sin()).unwrap();
            vector.push(value);
        }
        vector
    }

    /// Performs Gram-Schmidt orthogonalization of a vector against known vectors.
    ///
    /// Implements: x_orth = x - Σ(dot(x, v_i) * v_i) for all known vectors v_i
    fn gram_schmidt_orthogonalize(&self, vector: &mut Vec<S>, known_vectors: &[Vec<S>]) {
        for known_vector in known_vectors {
            let dot_product = Self::dot_product(vector, known_vector);
            for j in 0..vector.len() {
                vector[j] = vector[j] - dot_product * known_vector[j];
            }
        }
    }

    /// Solves the linear system Ly = b using the Conjugate Gradient method.
    ///
    /// This implements CG for the semi-positive definite Laplacian matrix L.
    /// Since L has a zero eigenvalue, we solve for the component orthogonal to
    /// the null space (the constant vector).
    fn solve_with_conjugate_gradient<G>(&self, graph: G, b: &[S]) -> Vec<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();
        let mut x = vec![S::zero(); n]; // Initial guess: zero vector
        let mut r = b.to_vec(); // Initial residual r = b - Lx = b (since x = 0)
        let mut p = r.clone(); // Initial search direction

        let mut rsold = Self::dot_product(&r, &r);

        for _iter in 0..self.cg_max_iterations {
            // Compute Ap = L * p
            let ap = self.laplacian_multiply(graph, &p);

            // Compute alpha = (r^T * r) / (p^T * Ap)
            let pap = Self::dot_product(&p, &ap);
            if pap.abs() < self.cg_tolerance {
                break; // Avoid division by zero
            }
            let alpha = rsold / pap;

            // Update solution: x = x + alpha * p
            for i in 0..n {
                x[i] = x[i] + alpha * p[i];
            }

            // Update residual: r = r - alpha * Ap
            for i in 0..n {
                r[i] = r[i] - alpha * ap[i];
            }

            let rsnew = Self::dot_product(&r, &r);

            // Check for convergence
            if rsnew.sqrt() < self.cg_tolerance {
                break;
            }

            // Compute beta = (r_new^T * r_new) / (r_old^T * r_old)
            let beta = rsnew / rsold;

            // Update search direction: p = r + beta * p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
        }

        x
    }

    /// Computes the Laplacian matrix-vector product Lv efficiently without building the full matrix.
    ///
    /// For each vertex i: (Lv)_i = degree(i) * v_i - Σ(v_j for j neighbor of i)
    fn laplacian_multiply<G>(&self, graph: G, vector: &[S]) -> Vec<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();
        let mut result = vec![S::zero(); n];

        // Create node index mapping
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        // Compute degrees
        let mut degrees = vec![S::zero(); n];
        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];

            if i != j {
                degrees[i] = degrees[i] + S::one();
                degrees[j] = degrees[j] + S::one();
            } else {
                degrees[i] = degrees[i] + S::one();
            }
        }

        // Initialize result with degree * vector
        for i in 0..n {
            result[i] = degrees[i] * vector[i];
        }

        // Subtract adjacency contribution
        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];

            result[i] = result[i] - vector[j];
            if i != j {
                result[j] = result[j] - vector[i];
            }
        }

        result
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
