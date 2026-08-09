use crate::solvers::LinearSolver;
use ndarray::{Array1, Array2, ArrayView2, s};
use petgraph_distance::SparseSymmetricMatrix;
use petgraph_drawing::DrawingValue;
use rand::Rng;

pub struct EigendecompositionResult<S> {
    pub eigenvectors: Array2<S>,
    pub eigenvalues: Array1<S>,
    pub cg_iterations: Vec<usize>,
    pub power_iterations: Vec<usize>,
}

/// Generates a random vector of specified size.
pub fn generate_random_vector<S, R>(n: usize, rng: &mut R) -> Array1<S>
where
    S: DrawingValue,
    R: Rng,
{
    Array1::from_shape_fn(n, |_| S::from_f32(rng.gen_range(-1.0..1.0)).unwrap())
}

/// Performs standard Gram-Schmidt orthogonalization.
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

/// Computes smallest eigenvalues for Standard Laplacian.
pub fn compute_smallest_eigenvalues<S, R, Solver>(
    n_target: usize,
    max_iterations: usize,
    tolerance: S,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let matrix = solver.matrix();
    let n = matrix.dim();

    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    let mut cg_iterations = Vec::new();
    let mut power_iterations = Vec::new();

    // 0-eigenvector for Standard Laplacian L * 1 = 0
    eigenvectors
        .column_mut(0)
        .fill(S::one() / S::from_usize(n).unwrap().sqrt());

    cg_iterations.push(0);
    power_iterations.push(0);

    let mut y = Array1::zeros(n);

    for k in 1..=n_target {
        let mut x_iter = generate_random_vector(n, rng);

        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize(&mut x_iter, &found_vecs);
        normalize(&mut x_iter);

        let mut lambda_prev_est = S::zero();
        let mut total_cg_iters = 0;
        let mut power_iter = 0;

        for iter in 0..max_iterations {
            power_iter = iter + 1;
            let cg_iters = solver.solve(&x_iter, &mut y);
            total_cg_iters += cg_iters;
            let mut x_next_iter = y.clone();

            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize(&mut x_next_iter, &found_vecs);
            normalize(&mut x_next_iter);

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
        cg_iterations.push(total_cg_iters);
        power_iterations.push(power_iter);
    }

    EigendecompositionResult {
        eigenvectors,
        eigenvalues,
        cg_iterations,
        power_iterations,
    }
}

/// Computes smallest eigenvalues for Symmetric Normalized Laplacian.
pub fn compute_smallest_eigenvalues_symmetric_normalized<S, R, Solver>(
    n_target: usize,
    max_iterations: usize,
    tolerance: S,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let matrix = solver.matrix();
    let n = matrix.dim();

    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    let mut cg_iterations = Vec::new();
    let mut power_iterations = Vec::new();

    // 0-eigenvector for Symmetric Normalized Laplacian L_sym * D^{1/2} 1 = 0
    let mut zero_vec = matrix.stationary_vector();
    normalize(&mut zero_vec);
    eigenvectors.column_mut(0).assign(&zero_vec);

    cg_iterations.push(0);
    power_iterations.push(0);

    let mut y = Array1::zeros(n);

    for k in 1..=n_target {
        let mut x_iter = generate_random_vector(n, rng);

        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize(&mut x_iter, &found_vecs);
        normalize(&mut x_iter);

        let mut lambda_prev_est = S::zero();
        let mut total_cg_iters = 0;
        let mut power_iter = 0;

        for iter in 0..max_iterations {
            power_iter = iter + 1;
            let cg_iters = solver.solve(&x_iter, &mut y);
            total_cg_iters += cg_iters;
            let mut x_next_iter = y.clone();

            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize(&mut x_next_iter, &found_vecs);
            normalize(&mut x_next_iter);

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
        cg_iterations.push(total_cg_iters);
        power_iterations.push(power_iter);
    }

    EigendecompositionResult {
        eigenvectors,
        eigenvalues,
        cg_iterations,
        power_iterations,
    }
}

/// Computes smallest eigenvalues for Random Walk Normalized Laplacian.
pub fn compute_smallest_eigenvalues_random_walk_normalized<S, R, Solver>(
    degrees: &Array1<S>,
    n_target: usize,
    max_iterations: usize,
    tolerance: S,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let matrix = solver.matrix();
    let n = matrix.dim();

    let mut eigenvalues = Array1::zeros(n_target + 1);
    let mut eigenvectors = Array2::zeros((n, n_target + 1));
    let mut cg_iterations = Vec::new();
    let mut power_iterations = Vec::new();

    // 0-eigenvector for Random Walk Normalized Laplacian L_rw * 1 = 0
    eigenvectors
        .column_mut(0)
        .fill(S::one() / S::from_usize(n).unwrap().sqrt());

    cg_iterations.push(0);
    power_iterations.push(0);

    let mut y = Array1::zeros(n);

    for k in 1..=n_target {
        let mut x_iter = generate_random_vector(n, rng);

        let found_vecs = eigenvectors.slice(s![.., ..k]);
        gram_schmidt_orthogonalize_weighted(&mut x_iter, &found_vecs, degrees);
        normalize_weighted(&mut x_iter, degrees);

        let mut lambda_prev_est = S::zero();
        let mut total_cg_iters = 0;
        let mut power_iter = 0;

        for iter in 0..max_iterations {
            power_iter = iter + 1;
            let mut d_x = x_iter.clone();
            for i in 0..n {
                d_x[i] *= degrees[i];
            }
            let cg_iters = solver.solve(&d_x, &mut y);
            total_cg_iters += cg_iters;
            let mut x_next_iter = y.clone();

            let found_vecs = eigenvectors.slice(s![.., ..k]);
            gram_schmidt_orthogonalize_weighted(&mut x_next_iter, &found_vecs, degrees);
            normalize_weighted(&mut x_next_iter, degrees);

            let mut matrix_x = Array1::zeros(n);
            matrix.multiply_into(&x_next_iter, &mut matrix_x);
            let numerator = x_next_iter.dot(&matrix_x);
            let mut denominator = S::zero();
            for i in 0..n {
                denominator += x_next_iter[i] * x_next_iter[i] * degrees[i];
            }

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
        cg_iterations.push(total_cg_iters);
        power_iterations.push(power_iter);
    }

    EigendecompositionResult {
        eigenvectors,
        eigenvalues,
        cg_iterations,
        power_iterations,
    }
}

/// Computes d-dimensional spectral coordinates and eigenvalues.
pub fn eigendecomposition<S, R, Solver>(
    shift: S,
    eigenvalue_max_iterations: usize,
    eigenvalue_tolerance: S,
    d: usize,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let n = solver.matrix().dim();

    let mut result = compute_smallest_eigenvalues(
        d,
        eigenvalue_max_iterations,
        eigenvalue_tolerance,
        solver,
        rng,
    );

    for i in 0..=d {
        result.eigenvalues[i] -= shift;
    }

    let mut final_eigenvalues = Array1::zeros(d);
    let mut final_eigenvectors = Array2::zeros((n, d));

    for i in 0..d {
        final_eigenvalues[i] = result.eigenvalues[i + 1];
        final_eigenvectors
            .column_mut(i)
            .assign(&result.eigenvectors.column(i + 1));
    }

    result.eigenvectors = final_eigenvectors;
    result.eigenvalues = final_eigenvalues;
    result.cg_iterations.remove(0);
    result.power_iterations.remove(0);

    result
}

/// Computes d-dimensional spectral coordinates and eigenvalues for Symmetric Normalized Laplacian.
pub fn eigendecomposition_symmetric_normalized<S, R, Solver>(
    shift: S,
    eigenvalue_max_iterations: usize,
    eigenvalue_tolerance: S,
    d: usize,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let n = solver.matrix().dim();

    let mut result = compute_smallest_eigenvalues_symmetric_normalized(
        d,
        eigenvalue_max_iterations,
        eigenvalue_tolerance,
        solver,
        rng,
    );

    for i in 0..=d {
        result.eigenvalues[i] -= shift;
    }

    let mut final_eigenvalues = Array1::zeros(d);
    let mut final_eigenvectors = Array2::zeros((n, d));

    for i in 0..d {
        final_eigenvalues[i] = result.eigenvalues[i + 1];
        final_eigenvectors
            .column_mut(i)
            .assign(&result.eigenvectors.column(i + 1));
    }

    result.eigenvectors = final_eigenvectors;
    result.eigenvalues = final_eigenvalues;
    result.cg_iterations.remove(0);
    result.power_iterations.remove(0);

    result
}

/// Computes d-dimensional spectral coordinates and eigenvalues for Random Walk Normalized Laplacian.
pub fn eigendecomposition_random_walk_normalized<S, R, Solver>(
    shift: S,
    eigenvalue_max_iterations: usize,
    eigenvalue_tolerance: S,
    d: usize,
    solver: &Solver,
    rng: &mut R,
) -> EigendecompositionResult<S>
where
    S: DrawingValue + Default,
    R: Rng,
    Solver: LinearSolver<S>,
{
    let matrix = solver.matrix();
    let n = matrix.dim();

    let mut degrees = Array1::zeros(n);
    for i in 0..n {
        degrees[i] = matrix.diagonal()[i] + shift;
    }

    let mut result = compute_smallest_eigenvalues_random_walk_normalized(
        &degrees,
        d,
        eigenvalue_max_iterations,
        eigenvalue_tolerance,
        solver,
        rng,
    );

    for i in 0..=d {
        result.eigenvalues[i] -= shift;
    }

    let mut final_eigenvalues = Array1::zeros(d);
    let mut final_eigenvectors = Array2::zeros((n, d));

    for i in 0..d {
        final_eigenvalues[i] = result.eigenvalues[i + 1];
        final_eigenvectors
            .column_mut(i)
            .assign(&result.eigenvectors.column(i + 1));
    }

    result.eigenvectors = final_eigenvectors;
    result.eigenvalues = final_eigenvalues;
    result.cg_iterations.remove(0);
    result.power_iterations.remove(0);

    result
}

fn gram_schmidt_orthogonalize_weighted<S>(
    x: &mut Array1<S>,
    basis: &ArrayView2<S>,
    weights: &Array1<S>,
) where
    S: DrawingValue + Default,
{
    let n = x.len();
    let d = basis.ncols();
    for i in 0..d {
        let v = basis.column(i);
        let mut proj_coeff_num = S::zero();
        let mut proj_coeff_den = S::zero();
        for j in 0..n {
            proj_coeff_num += x[j] * v[j] * weights[j];
            proj_coeff_den += v[j] * v[j] * weights[j];
        }
        let proj_coeff = proj_coeff_num / proj_coeff_den;
        for j in 0..n {
            x[j] -= proj_coeff * v[j];
        }
    }
}

fn normalize_weighted<S>(x: &mut Array1<S>, weights: &Array1<S>)
where
    S: DrawingValue + Default,
{
    let n = x.len();
    let mut norm_sq = S::zero();
    for i in 0..n {
        norm_sq += x[i] * x[i] * weights[i];
    }
    let norm = norm_sq.sqrt();
    if norm > S::zero() {
        for i in 0..n {
            x[i] /= norm;
        }
    }
}
