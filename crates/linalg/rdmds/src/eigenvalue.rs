//! Pure Rust implementation of eigenvalue computation using inverse power method
//! with Gram-Schmidt orthogonalization and Conjugate Gradient solver for computing
//! the smallest non-zero eigenvalues of graph Laplacians.

use ndarray::{Array1, Array2, ArrayView2, s};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;
use std::collections::HashMap;

/// IC(0) Incomplete Cholesky preconditioner for sparse symmetric positive definite matrices.
///
/// This structure stores the lower triangular factor L such that LL^T ≈ A,
/// where the sparsity pattern of L matches the lower triangle of A (no fill-in).
/// The preconditioner solves M^{-1}r by solving Ly = r and L^T z = y.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner<S> {
    /// Number of nodes in the matrix
    n: usize,
    /// Diagonal entries of the L factor
    diagonal: Vec<S>,
    /// Row-wise storage: row_entries[i] contains (column_index, value) pairs for row i
    /// Only stores off-diagonal entries where column_index < i
    row_entries: Vec<Vec<(usize, S)>>,
    /// Column-wise storage: col_entries[j] contains (row_index, value) pairs for column j
    /// Only stores off-diagonal entries where row_index > j
    col_entries: Vec<Vec<(usize, S)>>,
}

impl<S> IncompleteCholeskyPreconditioner<S>
where
    S: DrawingValue,
{
    /// Creates an IC(0) preconditioner from a LaplacianStructure.
    ///
    /// Performs incomplete Cholesky factorization with zero fill-in, maintaining
    /// the sparsity pattern of the lower triangle of the Laplacian matrix.
    /// This achieves O(|E|) complexity by leveraging the graph structure.
    pub fn from_laplacian(laplacian: &LaplacianStructure<S>) -> Self {
        let n = laplacian.n;

        // Build adjacency lists with HashMap for O(1) lookup
        let mut adjacency: Vec<HashMap<usize, S>> = vec![HashMap::new(); n];
        for &(i, j, weight) in &laplacian.edges {
            adjacency[i].insert(j, weight);
            adjacency[j].insert(i, weight);
        }

        // Initialize storage
        let mut diagonal = vec![S::zero(); n];
        let mut row_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
        let mut col_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];

        // Build initial sparsity pattern: collect lower triangular entries
        for i in 0..n {
            for (&j, &weight) in &adjacency[i] {
                if j < i {
                    // Add to row-wise storage: row i has entry (column j, value)
                    row_entries[i].push((j, -weight));
                    // Add to column-wise storage: column j has entry (row i, value)
                    col_entries[j].push((i, -weight));
                }
            }
        }

        // Sort entries by column index for efficient lookup
        for i in 0..n {
            row_entries[i].sort_by_key(|&(col, _)| col);
            col_entries[i].sort_by_key(|&(row, _)| row);
        }

        // Perform IC(0) factorization: L L^T = A
        for i in 0..n {
            // Compute diagonal element L_ii
            let mut sum = S::zero();

            // sum += L_ik^2 for k < i where L_ik != 0
            for &(_, l_ik) in &row_entries[i] {
                sum += l_ik * l_ik;
            }

            // L_ii = sqrt(A_ii - sum)
            let aii = laplacian.degrees[i]; // Diagonal entry of Laplacian
            diagonal[i] = (aii - sum).sqrt();

            if diagonal[i] <= S::zero() {
                panic!("non positive definite");
            }

            // Update off-diagonal elements L_ji for j > i (only process actual edges)
            for (&j, &weight) in &adjacency[i] {
                if j > i {
                    // Find entry (j, i) in row j's storage
                    if let Some(entry_pos) = row_entries[j].iter().position(|&(col, _)| col == i) {
                        // Compute L_ji = (A_ji - sum) / L_ii
                        let mut sum = S::zero();

                        // Find common indices k < i where both L_jk and L_ik are non-zero
                        // Use two-pointer technique on sorted lists
                        let mut i_ptr = 0;
                        let mut j_ptr = 0;

                        while i_ptr < row_entries[i].len() && j_ptr < row_entries[j].len() {
                            let (i_col, i_val) = row_entries[i][i_ptr];
                            let (j_col, j_val) = row_entries[j][j_ptr];

                            if i_col == j_col && i_col < i {
                                sum += i_val * j_val;
                                i_ptr += 1;
                                j_ptr += 1;
                            } else if i_col < j_col {
                                i_ptr += 1;
                            } else {
                                j_ptr += 1;
                            }
                        }

                        // A_ji = -weight (negative of edge weight)
                        let a_ji = -weight;
                        let new_value = (a_ji - sum) / diagonal[i];

                        // Update both storage structures
                        row_entries[j][entry_pos].1 = new_value;
                        // Find and update corresponding entry in col_entries[i]
                        if let Some(col_pos) = col_entries[i].iter().position(|&(row, _)| row == j)
                        {
                            col_entries[i][col_pos].1 = new_value;
                        }
                    }
                }
            }
        }

        IncompleteCholeskyPreconditioner {
            n,
            diagonal,
            row_entries,
            col_entries,
        }
    }

    /// Applies the IC(0) preconditioner: solves M^{-1} * r = z
    ///
    /// This performs two triangular solves:
    /// 1. Forward solve: L * y = r
    /// 2. Backward solve: L^T * z = y
    ///
    /// Complexity: O(|E|) leveraging sparsity of L.
    pub fn apply(&self, r: &Array1<S>, z: &mut Array1<S>) {
        let mut y = Array1::zeros(self.n);

        // Forward solve: L * y = r
        for i in 0..self.n {
            let mut sum = S::zero();

            // sum += L_ij * y_j for j < i
            for &(j, l_ij) in &self.row_entries[i] {
                sum += l_ij * y[j];
            }

            y[i] = (r[i] - sum) / self.diagonal[i];
        }

        // Backward solve: L^T * z = y
        z.fill(S::zero());

        for i in (0..self.n).rev() {
            let mut sum = S::zero();

            // sum += L_ji * z_j for j > i (where L_ji is transpose of L_ij)
            // Use column-wise storage for efficient access
            for &(j, l_ji) in &self.col_entries[i] {
                sum += l_ji * z[j];
            }

            z[i] = (y[i] - sum) / self.diagonal[i];
        }
    }

    /// Returns the number of nodes in the preconditioner matrix.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Returns the number of diagonal entries.
    pub fn diagonal_count(&self) -> usize {
        self.diagonal.len()
    }
}

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
    pub fn new<G, F>(graph: G, edge_weight: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
    {
        Self::new_with_shift(graph, edge_weight, S::zero())
    }

    /// Creates a new LaplacianStructure with shift parameter, precomputing L + cI matrix.
    ///
    /// This constructor creates a positive definite matrix L + cI by adding the shift value c
    /// to the diagonal elements. This improves convergence of the conjugate gradient method
    /// since the matrix becomes positive definite instead of semi-positive definite.
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `edge_weight` - Function to extract edge weights
    /// * `shift` - Shift parameter c to add to diagonal (makes matrix positive definite)
    pub fn new_with_shift<G, F>(graph: G, mut edge_weight: F, shift: S) -> Self
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

            if i == j {
                panic!("self loop");
            }
            edges.push((i, j, weight));
            degrees[i] += weight;
            degrees[j] += weight;
        }

        // Add shift to diagonal elements to create L + cI matrix
        for degree in &mut degrees {
            *degree += shift;
        }

        LaplacianStructure { n, edges, degrees }
    }

    /// Computes the Laplacian matrix-vector product Lv efficiently.
    ///
    /// For each vertex i: (Lv)_i = degree(i) * v_i - Σ(weight_ij * v_j for j neighbor of i)
    pub fn multiply(&self, vector: &Array1<S>, result: &mut Array1<S>) {
        for i in 0..self.n {
            result[i] = self.degrees[i] * vector[i];
        }
        for &(i, j, weight) in &self.edges {
            result[i] -= weight * vector[j];
            result[j] -= weight * vector[i];
        }
    }

    /// Computes the Laplacian quadratic form x^T L x efficiently in O(|E|) time.
    ///
    /// Uses the fact that x^T L x = Σ_{(i,j) ∈ E} weight_ij * (x_i - x_j)^2
    pub fn quadratic_form(&self, vector: &Array1<S>) -> S {
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

/// Solves the linear system Ly = b using the Conjugate Gradient method with IC(0) preconditioning.
///
/// This implements preconditioned CG for the semi-positive definite Laplacian matrix L.
/// Uses IC(0) incomplete Cholesky preconditioner for improved convergence compared to
/// Jacobi preconditioning.
///
/// # Parameters
/// * `laplacian` - The Laplacian structure
/// * `preconditioner` - The IC(0) preconditioner
/// * `b` - Right-hand side vector as Array1
/// * `x` - Initial guess and solution vector
/// * `cg_max_iterations` - Maximum iterations for CG method
/// * `cg_tolerance` - Convergence tolerance for CG method
pub fn solve_with_conjugate_gradient<S>(
    laplacian: &LaplacianStructure<S>,
    preconditioner: &IncompleteCholeskyPreconditioner<S>,
    b: &Array1<S>,
    x: &mut Array1<S>,
    cg_max_iterations: usize,
    cg_tolerance: S,
) where
    S: DrawingValue,
{
    let n = laplacian.node_count();
    let mut r = Array1::zeros(n);
    let mut z = Array1::zeros(n);
    let mut q = Array1::zeros(n);

    // r = b - A*x
    laplacian.multiply(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }

    // z = M^{-1} * r (apply IC(0) preconditioner)
    preconditioner.apply(&r, &mut z);
    let mut p = z.clone();

    let mut rsold = r.dot(&z);

    for _iter in 0..cg_max_iterations {
        // q = A * p
        laplacian.multiply(&p, &mut q);
        let alpha = rsold / p.dot(&q);

        // Update x and r
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        // Apply preconditioner: z = M^{-1} * r
        preconditioner.apply(&r, &mut z);

        let rsnew = r.dot(&z);
        if rsnew < cg_tolerance * cg_tolerance {
            break;
        }

        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = beta * p[i] + z[i];
        }

        rsold = rsnew;
    }
}

/// Computes d-dimensional spectral coordinates and eigenvalues using edge weights and custom parameters.
///
/// This function combines the spectral coordinate computation with eigenvalue computation,
/// returning both the final embedding coordinates and the computed eigenvalues.
/// It excludes the zero eigenvalue and returns only d non-zero eigenvalues and their eigenvectors.
///
/// # Parameters
/// * `graph` - The input graph
/// * `length` - Function to extract edge weights/lengths
/// * `shift` - Shift parameter for creating positive definite matrix L + cI
/// * `eigenvalue_max_iterations` - Maximum iterations for eigenvalue computation
/// * `cg_max_iterations` - Maximum iterations for CG method
/// * `eigenvalue_tolerance` - Convergence tolerance for eigenvalue computation
/// * `cg_tolerance` - Convergence tolerance for CG method
/// * `d` - Number of spectral dimensions
/// * `rng` - Random number generator for eigenvalue computation
///
/// # Returns
/// A tuple containing:
/// - Array2 where coordinates.row(i) contains the d-dimensional coordinate for node i
/// - Array1 of d non-zero eigenvalues (λ_1, λ_2, ..., λ_d)
pub fn eigendecomposition<S, G, F, R>(
    graph: G,
    length: F,
    shift: S,
    eigenvalue_max_iterations: usize,
    cg_max_iterations: usize,
    eigenvalue_tolerance: S,
    cg_tolerance: S,
    d: usize,
    rng: &mut R,
) -> (Array2<S>, Array1<S>)
where
    S: DrawingValue,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
{
    let n = graph.node_count();

    // Create weighted Laplacian structure with shift for positive definite matrix L + cI
    let laplacian = LaplacianStructure::new_with_shift(graph, length, shift);

    // Step 1: Compute smallest d non-zero eigenvalues and eigenvectors (+ zero eigenvalue)
    let (all_eigenvalues, all_eigenvectors) = compute_smallest_eigenvalues(
        &laplacian,
        d,
        eigenvalue_max_iterations,
        cg_max_iterations,
        eigenvalue_tolerance,
        cg_tolerance,
        rng,
    );

    // Step 2: Extract only the non-zero eigenvalues and eigenvectors (skip index 0)
    let mut eigenvalues = Array1::zeros(d);
    let mut eigenvectors = Array2::zeros((n, d));

    for i in 0..d {
        eigenvalues[i] = all_eigenvalues[i + 1] - shift; // Subtract shift and skip zero eigenvalue
    }

    for i in 0..d {
        eigenvectors
            .column_mut(i)
            .assign(&all_eigenvectors.column(i + 1)); // Skip zero eigenvector
    }

    // Step 3: Create coordinates by dividing eigenvectors by sqrt of eigenvalues
    for dim in 0..d {
        let mut eigenvector = eigenvectors.column_mut(dim);
        eigenvector /= eigenvalues[dim].sqrt();
    }

    (eigenvectors, eigenvalues)
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
/// * `rng` - Random number generator for initial vectors
///
/// # Returns
/// A tuple containing:
/// - Array1 of eigenvalues (\lambda_0, \lambda_1, \cdots, \lambda_{n_target})
/// - Array2 of corresponding eigenvectors (each column is an eigenvector)
pub fn compute_smallest_eigenvalues<S, R>(
    laplacian: &LaplacianStructure<S>,
    n_target: usize,
    max_iterations: usize,
    cg_max_iterations: usize,
    tolerance: S,
    cg_tolerance: S,
    rng: &mut R,
) -> (Array1<S>, Array2<S>)
where
    S: DrawingValue,
    R: Rng,
{
    let n = laplacian.node_count();

    // Create IC(0) preconditioner from the Laplacian (computed once and reused)
    let preconditioner = IncompleteCholeskyPreconditioner::from_laplacian(laplacian);

    // Initialize storage for eigenvalues and eigenvectors using ndarray
    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    eigenvectors
        .column_mut(0)
        .fill(S::one() / S::from_usize(n).unwrap().sqrt());
    let mut y = Array1::zeros(n);

    // For k = 1, ..., n_target: find the k-th smallest non-zero eigenvalue and eigenvector
    for k in 1..=n_target {
        // Step 1: Initialize random vector and orthogonalize against found eigenvectors
        let mut x_iter = generate_random_vector(n, rng);

        // Orthogonalize against previously found eigenvectors
        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize(&mut x_iter, &found_vecs);
        normalize(&mut x_iter);

        let mut lambda_prev_est = S::zero();

        // Step 2: Inverse power method iteration
        for _iter in 0..max_iterations {
            // Step 2a: Solve Ly = x_iter using CG method with IC(0) preconditioning
            solve_with_conjugate_gradient(
                laplacian,
                &preconditioner,
                &x_iter,
                &mut y,
                cg_max_iterations,
                cg_tolerance,
            );
            let mut x_next_iter = y.clone();

            // Step 2b: Orthogonalize y against found eigenvectors (for numerical stability)
            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize(&mut x_next_iter, &found_vecs);

            // Step 2c: Normalize
            normalize(&mut x_next_iter);

            // Step 2d: Compute eigenvalue estimate using optimized Rayleigh quotient
            let numerator = laplacian.quadratic_form(&x_next_iter);
            let denominator = x_next_iter.dot(&x_next_iter);
            let lambda_est = numerator / denominator;

            // Step 2e: Check convergence
            let converged = (lambda_est - lambda_prev_est).abs() < tolerance;

            // Step 2f: Update for next iteration
            x_iter = x_next_iter;
            lambda_prev_est = lambda_est;

            if converged {
                break;
            }
        }

        eigenvalues[k] = lambda_prev_est;
        eigenvectors.column_mut(k).assign(&x_iter);
    }

    (eigenvalues, eigenvectors)
}
