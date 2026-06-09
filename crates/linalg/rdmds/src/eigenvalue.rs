//! Pure Rust implementation of eigenvalue computation using inverse power method
//! with Gram-Schmidt orthogonalization and Conjugate Gradient solver.

use ndarray::{Array1, Array2, ArrayView2, s};
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_distance::Laplacian;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_spmv::SparseSymmetricMatrix;
use rand::Rng;
use std::collections::HashMap;

/// IC(0) Incomplete Cholesky preconditioner for sparse symmetric positive definite matrices.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner<S> {
    n: usize,
    diagonal: Vec<S>,
    row_entries: Vec<Vec<(usize, S)>>,
    col_entries: Vec<Vec<(usize, S)>>,
}

impl<S> IncompleteCholeskyPreconditioner<S>
where
    S: DrawingValue + Default,
{
    /// Creates an IC(0) preconditioner from a SparseSymmetricMatrix.
    pub fn from_matrix(matrix: &SparseSymmetricMatrix<S>) -> Self {
        let n = matrix.dim();

        // Build adjacency lists with HashMap
        let mut adjacency: Vec<HashMap<usize, S>> = vec![HashMap::new(); n];
        for &(i, j, val) in matrix.edges() {
            adjacency[i].insert(j, val);
            adjacency[j].insert(i, val);
        }

        // Initialize storage
        let mut diagonal = vec![S::zero(); n];
        let mut row_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
        let mut col_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];

        // Build initial sparsity pattern: collect lower triangular entries
        for i in 0..n {
            for (&j, &val) in &adjacency[i] {
                if j < i {
                    row_entries[i].push((j, val));
                    col_entries[j].push((i, val));
                }
            }
        }

        // Sort entries by column index
        for i in 0..n {
            row_entries[i].sort_by_key(|&(col, _)| col);
            col_entries[i].sort_by_key(|&(row, _)| row);
        }

        // Perform IC(0) factorization: L L^T = A
        for i in 0..n {
            let mut sum = S::zero();
            for &(_, l_ik) in &row_entries[i] {
                sum += l_ik * l_ik;
            }

            let aii = matrix.diagonal()[i];
            diagonal[i] = (aii - sum).max(S::zero()).sqrt();

            if diagonal[i] <= S::zero() {
                diagonal[i] = S::from_f32(1e-6).unwrap();
            }

            for (&j, &val) in &adjacency[i] {
                let entry_pos = if j > i {
                    row_entries[j].iter().position(|&(col, _)| col == i)
                } else {
                    None
                };
                if let Some(entry_pos) = entry_pos {
                    let mut sum = S::zero();
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

                    let a_ji = val;
                    let new_value = (a_ji - sum) / diagonal[i];

                    row_entries[j][entry_pos].1 = new_value;
                    if let Some(col_pos) = col_entries[i].iter().position(|&(row, _)| row == j) {
                        col_entries[i][col_pos].1 = new_value;
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
    pub fn apply(&self, r: &Array1<S>, z: &mut Array1<S>) {
        let mut y = Array1::zeros(self.n);

        for i in 0..self.n {
            let mut sum = S::zero();
            for &(j, l_ij) in &self.row_entries[i] {
                sum += l_ij * y[j];
            }
            y[i] = (r[i] - sum) / self.diagonal[i];
        }

        z.fill(S::zero());
        for i in (0..self.n).rev() {
            let mut sum = S::zero();
            for &(j, l_ji) in &self.col_entries[i] {
                sum += l_ji * z[j];
            }
            z[i] = (y[i] - sum) / self.diagonal[i];
        }
    }
}

/// Generates a random vector of specified size.
pub fn generate_random_vector<S, R>(n: usize, rng: &mut R) -> Array1<S>
where
    S: DrawingValue,
    R: Rng,
{
    Array1::from_shape_fn(n, |_| S::from_f32(rng.gen_range(-1.0..1.0)).unwrap())
}

/// Performs Gram-Schmidt orthogonalization.
pub fn gram_schmidt_orthogonalize<S>(vector: &mut Array1<S>, known_vectors: &ArrayView2<S>)
where
    S: DrawingValue,
{
    for col in known_vectors.columns() {
        let dot_product_value = vector.dot(&col);
        *vector -= &(col.to_owned() * dot_product_value);
    }
}

/// Normalizes a vector.
pub fn normalize<S>(vector: &mut Array1<S>)
where
    S: DrawingValue,
{
    let norm = vector.dot(vector).sqrt();
    if norm > S::zero() {
        *vector /= norm;
    }
}

/// Solves linear system Ly = b using Conjugate Gradient with IC(0) preconditioning.
pub fn solve_with_conjugate_gradient<S>(
    matrix: &SparseSymmetricMatrix<S>,
    preconditioner: &IncompleteCholeskyPreconditioner<S>,
    b: &Array1<S>,
    x: &mut Array1<S>,
    cg_max_iterations: usize,
    cg_tolerance: S,
) where
    S: DrawingValue + Default,
{
    let n = matrix.dim();
    let mut r = Array1::zeros(n);
    let mut z = Array1::zeros(n);
    let mut q = Array1::zeros(n);

    matrix.multiply_into(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }

    preconditioner.apply(&r, &mut z);
    let mut p = z.clone();

    let mut rsold = r.dot(&z);

    for _iter in 0..cg_max_iterations {
        matrix.multiply_into(&p, &mut q);
        let alpha = rsold / p.dot(&q);

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

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

/// Computes d-dimensional spectral coordinates and eigenvalues.
#[allow(clippy::too_many_arguments)]
pub fn eigendecomposition<S, G, F, R, L>(
    graph: G,
    length: F,
    shift: S,
    eigenvalue_max_iterations: usize,
    cg_max_iterations: usize,
    eigenvalue_tolerance: S,
    cg_tolerance: S,
    d: usize,
    laplacian_builder: L,
    rng: &mut R,
) -> (Array2<S>, Array1<S>)
where
    S: DrawingValue + Default,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
    L: Laplacian<G, S>,
{
    let n = graph.node_count();

    let laplacian = laplacian_builder
        .build(graph, &mut { length })
        .scale_and_shift(S::one(), -shift);

    let (all_eigenvalues, all_eigenvectors) = compute_smallest_eigenvalues(
        &laplacian,
        d,
        eigenvalue_max_iterations,
        cg_max_iterations,
        eigenvalue_tolerance,
        cg_tolerance,
        rng,
    );

    let mut eigenvalues = Array1::zeros(d);
    let mut eigenvectors = Array2::zeros((n, d));

    for i in 0..d {
        eigenvalues[i] = all_eigenvalues[i + 1] - shift;
    }

    for i in 0..d {
        eigenvectors
            .column_mut(i)
            .assign(&all_eigenvectors.column(i + 1));
    }

    for dim in 0..d {
        let mut eigenvector = eigenvectors.column_mut(dim);
        eigenvector /= eigenvalues[dim].max(S::zero()).sqrt();
    }

    (eigenvectors, eigenvalues)
}

/// Computes smallest eigenvalues and eigenvectors.
pub fn compute_smallest_eigenvalues<S, R>(
    matrix: &SparseSymmetricMatrix<S>,
    n_target: usize,
    max_iterations: usize,
    cg_max_iterations: usize,
    tolerance: S,
    cg_tolerance: S,
    rng: &mut R,
) -> (Array1<S>, Array2<S>)
where
    S: DrawingValue + Default,
    R: Rng,
{
    let n = matrix.dim();

    let preconditioner = IncompleteCholeskyPreconditioner::from_matrix(matrix);

    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    eigenvectors
        .column_mut(0)
        .fill(S::one() / S::from_usize(n).unwrap().sqrt());
    let mut y = Array1::zeros(n);

    for k in 1..=n_target {
        let mut x_iter = generate_random_vector(n, rng);

        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize(&mut x_iter, &found_vecs);
        normalize(&mut x_iter);

        let mut lambda_prev_est = S::zero();

        for _iter in 0..max_iterations {
            solve_with_conjugate_gradient(
                matrix,
                &preconditioner,
                &x_iter,
                &mut y,
                cg_max_iterations,
                cg_tolerance,
            );
            let mut x_next_iter = y.clone();

            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize(&mut x_next_iter, &found_vecs);

            normalize(&mut x_next_iter);

            // Rayleigh quotient: x^T A x / x^T x
            let numerator = x_next_iter.dot(&matrix.multiply(&x_next_iter));
            let denominator = x_next_iter.dot(&x_next_iter);
            let lambda_est = numerator / denominator;

            let converged = (lambda_est - lambda_prev_est).abs() < tolerance;

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
