//! Pure Rust implementation of eigenvalue computation using inverse power method
//! with Gram-Schmidt orthogonalization and Conjugate Gradient solver for computing
//! the smallest non-zero eigenvalues of graph Laplacians.

use ndarray::{s, Array1, Array2, ArrayView2};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;
use std::collections::HashMap;

/// Precomputed Laplacian structure for efficient matrix operations.
///
/// This structure caches the graph topology and edge weights to avoid
/// repeated computations during eigenvalue iteration.
#[derive(Debug, Clone)]
pub struct LaplacianStructure<S> {
    /// Number of nodes in the graph
    n: usize,
    /// List of edges with their weights: (source_index, target_index, weight)
    edges: Vec<(usize, usize, S)>,
    /// Degree of each node (sum of incident edge weights)
    degrees: Vec<S>,
}

impl<S> LaplacianStructure<S>
where
    S: DrawingValue,
{
    /// Creates a new LaplacianStructure from a graph with edge weights.
    pub fn new<G, F>(graph: G, mut edge_weight: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
    {
        let n = graph.node_count();

        // Create node index mapping
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        let mut edges = Vec::new();
        let mut degrees = vec![S::zero(); n];

        // Process edges and compute degrees
        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];
            let weight = edge_weight(edge);

            edges.push((i, j, weight));

            if i != j {
                degrees[i] += weight;
                degrees[j] += weight;
            } else {
                degrees[i] += weight;
            }
        }

        LaplacianStructure { n, edges, degrees }
    }

    /// Computes the Laplacian matrix-vector product Lv efficiently.
    ///
    /// For each vertex i: (Lv)_i = degree(i) * v_i - Σ(weight_ij * v_j for j neighbor of i)
    pub fn multiply(&self, vector: &[S]) -> Vec<S> {
        let mut result = vec![S::zero(); self.n];

        // Initialize result with degree * vector
        for i in 0..self.n {
            result[i] = self.degrees[i] * vector[i];
        }

        // Subtract adjacency contribution
        for &(i, j, weight) in &self.edges {
            result[i] -= weight * vector[j];
            if i != j {
                result[j] -= weight * vector[i];
            }
        }

        result
    }

    /// Computes the Laplacian quadratic form x^T L x efficiently in O(|E|) time.
    ///
    /// Uses the fact that x^T L x = Σ_{(i,j) ∈ E} weight_ij * (x_i - x_j)^2
    pub fn quadratic_form(&self, vector: &[S]) -> S {
        let mut result = S::zero();

        for &(i, j, weight) in &self.edges {
            let diff = vector[i] - vector[j];
            result += weight * diff * diff;
        }

        result
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.n
    }
}

/// Generates a random vector of specified size using the provided RNG.
pub fn generate_random_vector<S, R>(n: usize, rng: &mut R) -> Array1<S>
where
    S: DrawingValue,
    R: Rng,
{
    Array1::from_shape_fn(n, |_| S::from_f32(rng.gen_range(-1.0..1.0)).unwrap())
}

/// Performs Gram-Schmidt orthogonalization of a vector against known vectors.
///
/// Implements: x_orth = x - Σ(dot(x, v_i) * v_i) for all known vectors v_i
pub fn gram_schmidt_orthogonalize<S>(vector: &mut Array1<S>, known_vectors: &ArrayView2<S>)
where
    S: DrawingValue,
{
    for col in known_vectors.columns() {
        let dot_product_value = vector.dot(&col);
        *vector -= &(col.to_owned() * dot_product_value);
    }
}

/// Normalizes an Array1 vector to unit length.
pub fn normalize<S>(vector: &mut Array1<S>)
where
    S: DrawingValue,
{
    let norm = vector.dot(vector).sqrt();
    if norm > S::zero() {
        *vector /= norm;
    }
}

/// Solves the linear system Ly = b using the Conjugate Gradient method with Array1 input/output.
///
/// This implements CG for the semi-positive definite Laplacian matrix L.
/// Since L has a zero eigenvalue, we solve for the component orthogonal to
/// the null space (the constant vector).
///
/// # Parameters
/// * `laplacian` - The Laplacian structure
/// * `b` - Right-hand side vector as Array1
/// * `cg_max_iterations` - Maximum iterations for CG method
/// * `cg_tolerance` - Convergence tolerance for CG method
pub fn solve_with_conjugate_gradient<S>(
    laplacian: &LaplacianStructure<S>,
    b: &Array1<S>,
    cg_max_iterations: usize,
    cg_tolerance: S,
) -> Array1<S>
where
    S: DrawingValue,
{
    let n = laplacian.node_count();
    let mut x = Array1::zeros(n); // Initial guess: zero vector
    let mut r = b.clone(); // Initial residual r = b - Lx = b (since x = 0)
    let mut p = r.clone(); // Initial search direction

    let mut rsold = r.dot(&r);

    for _iter in 0..cg_max_iterations {
        // Compute Ap = L * p
        let ap_vec = laplacian.multiply(p.as_slice().unwrap());
        let ap = Array1::from_vec(ap_vec);

        // Compute alpha = (r^T * r) / (p^T * Ap)
        let pap = p.dot(&ap);
        if pap.abs() < cg_tolerance {
            break; // Avoid division by zero
        }
        let alpha = rsold / pap;

        // Update solution: x = x + alpha * p
        x += &(&p * alpha);

        // Update residual: r = r - alpha * Ap
        r -= &(&ap * alpha);

        let rsnew = r.dot(&r);

        // Check for convergence
        if rsnew.sqrt() < cg_tolerance {
            break;
        }

        // Compute beta = (r_new^T * r_new) / (r_old^T * r_old)
        let beta = rsnew / rsold;

        // Update search direction: p = r + beta * p
        p = &r + &(&p * beta);

        rsold = rsnew;
    }

    x
}

/// Computes the smallest `n_target` non-zero eigenvalues and eigenvectors using a precomputed LaplacianStructure.
///
/// Implements the algorithm specified for computing non-zero eigenvalues of graph Laplacians using:
/// 1. Sequential computation of λ1, λ2, ..., λn_target (smallest non-zero eigenvalues)
/// 2. Inverse power method with CG solver for each eigenvalue
/// 3. Gram-Schmidt orthogonalization against previously found eigenvectors
/// 4. Optimized Rayleigh quotient using quadratic form
///
/// # Parameters
/// * `laplacian` - Precomputed Laplacian structure
/// * `n_target` - Number of smallest non-zero eigenvalues to compute
/// * `max_iterations` - Maximum iterations for inverse power method
/// * `cg_max_iterations` - Maximum iterations for CG method
/// * `tolerance` - Convergence tolerance for eigenvalues
/// * `cg_tolerance` - Convergence tolerance for CG method
/// * `vector_tolerance` - Convergence tolerance for eigenvectors
/// * `rng` - Random number generator for initial vectors
///
/// # Returns
/// A tuple containing:
/// - Array1 of eigenvalues (\lambda_0, \lambda_1, \cdots, \lambda_{n_target})
/// - Array2 of corresponding eigenvectors (each column is an eigenvector)
#[allow(clippy::too_many_arguments)]
pub fn compute_smallest_eigenvalues_with_laplacian<S, R>(
    laplacian: &LaplacianStructure<S>,
    n_target: usize,
    max_iterations: usize,
    cg_max_iterations: usize,
    tolerance: S,
    cg_tolerance: S,
    vector_tolerance: S,
    rng: &mut R,
) -> (Array1<S>, Array2<S>)
where
    S: DrawingValue,
    R: Rng,
{
    let n = laplacian.node_count();

    // Initialize storage for eigenvalues and eigenvectors using ndarray
    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    eigenvectors
        .column_mut(0)
        .fill(S::one() / S::from_usize(n).unwrap().sqrt());

    // For k = 1, ..., n_target: find the k-th smallest non-zero eigenvalue and eigenvector
    for k in 1..=n_target {
        // Step 1: Initialize random vector and orthogonalize against found eigenvectors
        let mut x_iter = generate_random_vector(n, rng);

        // Orthogonalize against previously found eigenvectors
        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize(&mut x_iter, &found_vecs);
        normalize(&mut x_iter);

        let mut lambda_prev_est = S::zero();
        let mut converged = false;

        // Step 2: Inverse power method iteration
        for _iter in 0..max_iterations {
            // Step 2a: Solve Ly = x_iter using CG method
            let y_solved =
                solve_with_conjugate_gradient(laplacian, &x_iter, cg_max_iterations, cg_tolerance);

            // Step 2b: Orthogonalize y against found eigenvectors (for numerical stability)
            let mut y_orth = y_solved;
            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize(&mut y_orth, &found_vecs);

            // Step 2c: Normalize
            normalize(&mut y_orth);
            let x_next_iter = y_orth;

            // Step 2d: Compute eigenvalue estimate using optimized Rayleigh quotient
            let numerator = laplacian.quadratic_form(x_next_iter.as_slice().unwrap());
            let denominator = x_next_iter.dot(&x_next_iter);
            let lambda_est = numerator / denominator;

            // Step 2e: Check convergence
            let eigenvalue_converged = (lambda_est - lambda_prev_est).abs() < tolerance;
            let vector_converged = {
                let diff = &x_next_iter - &x_iter;
                let diff_norm = diff.dot(&diff).sqrt();
                diff_norm < vector_tolerance
            };

            if eigenvalue_converged || vector_converged {
                eigenvalues[k] = lambda_est;
                eigenvectors.column_mut(k).assign(&x_next_iter);
                converged = true;
                break;
            }

            // Step 2f: Update for next iteration
            x_iter = x_next_iter;
            lambda_prev_est = lambda_est;
        }

        if !converged {
            // If didn't converge, still store the best estimate
            let numerator = laplacian.quadratic_form(x_iter.as_slice().unwrap());
            let denominator = x_iter.dot(&x_iter);
            let lambda_est = numerator / denominator;

            eigenvalues[k] = lambda_est;
            eigenvectors.column_mut(k).assign(&x_iter);
        }
    }

    (eigenvalues, eigenvectors)
}

/// Computes the smallest `n_target` non-zero eigenvalues and eigenvectors of a graph Laplacian.
///
/// This is a convenience function that creates a LaplacianStructure and calls the optimized version
/// with default parameters.
///
/// # Parameters
/// * `graph` - The input graph
/// * `n_target` - Number of smallest non-zero eigenvalues to compute
///
/// # Returns
/// A tuple containing:
/// - Array1 of eigenvalues (λ1, λ2, ..., λn_target)
/// - Array2 of corresponding eigenvectors (each column is an eigenvector)
pub fn compute_smallest_eigenvalues<G, S>(graph: G, n_target: usize) -> (Array1<S>, Array2<S>)
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let laplacian = LaplacianStructure::new(graph, |_| S::one());
    let mut rng = rand::thread_rng();
    compute_smallest_eigenvalues_with_laplacian(
        &laplacian,
        n_target,
        1000,                       // max_iterations
        100,                        // cg_max_iterations
        S::from_f32(1e-4).unwrap(), // tolerance
        S::from_f32(1e-4).unwrap(), // cg_tolerance
        S::from_f32(1e-4).unwrap(), // vector_tolerance
        &mut rng,
    )
}
