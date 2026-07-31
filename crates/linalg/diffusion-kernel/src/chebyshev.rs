//! Chebyshev polynomial approximation for matrix functions.
//!
//! This module implements Chebyshev polynomial approximation of exp(-tL) where L
//! is a graph Laplacian matrix, using Clenshaw's recurrence algorithm for stability.

use ndarray::{Array1, Array2};
use num_traits::Float;
use petgraph_distance::SparseSymmetricMatrix;
use std::f64::consts::PI;

/// Approximates exp(-tL) @ vectors using Chebyshev polynomial expansion.
///
/// This function computes K @ vectors where K = exp(-tL), using sparse
/// matrix operations for O(d * (|V| + |E|) * k) complexity.
///
/// # Parameters
/// * `laplacian` - Laplacian matrix (sparse symmetric)
/// * `t` - Diffusion time parameter
/// * `degree` - Degree of polynomial approximation
/// * `lambda_max` - Maximum eigenvalue of L
/// * `vectors` - Random vectors matrix of shape (n, num_vectors)
///
/// # Returns
/// KV - Result of K @ vectors where K = exp(-tL), shape (n, num_vectors)
pub(crate) fn chebyshev_approximation<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    t: T,
    degree: usize,
    lambda_max: T,
    vectors: &Array2<T>,
) -> Array2<T>
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    // Scale L to [-1, 1]: L' = (2L/lambda_max) - I
    let two = T::from(2.0).unwrap();
    let scale = two / lambda_max;
    let l_scaled = laplacian.scale_and_shift(scale, T::one());

    // Compute Chebyshev coefficients for exp(-t * lambda_max * (x + 1) / 2)
    let coeffs = compute_chebyshev_coefficients(t, lambda_max, degree);

    // Evaluate Chebyshev polynomial at L_scaled applied to vectors
    evaluate_chebyshev_polynomial(&l_scaled, &coeffs, vectors)
}

/// Approximates exp(-tL) @ vector using Chebyshev polynomial expansion.
pub(crate) fn chebyshev_approximation_vec<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    t: T,
    degree: usize,
    lambda_max: T,
    vector: &Array1<T>,
) -> Array1<T>
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let two = T::from(2.0).unwrap();
    let scale = two / lambda_max;
    let l_scaled = laplacian.scale_and_shift(scale, T::one());

    let coeffs = compute_chebyshev_coefficients(t, lambda_max, degree);

    evaluate_chebyshev_polynomial_vec(&l_scaled, &coeffs, vector)
}

/// Computes Chebyshev coefficients for exp(-t * lambda_max * (x + 1) / 2).
///
/// # Parameters
/// * `t` - Diffusion time parameter
/// * `lambda_max` - Maximum eigenvalue
/// * `degree` - Degree of approximation
///
/// # Returns
/// Chebyshev coefficients
fn compute_chebyshev_coefficients<T>(t: T, lambda_max: T, degree: usize) -> Vec<T>
where
    T: Float,
{
    // Number of points for numerical integration
    let n_points = (1000_usize).max(10 * degree);

    let mut coeffs = vec![T::zero(); degree + 1];

    // Compute coefficients using Chebyshev-Gauss quadrature
    for (j, coeff) in coeffs.iter_mut().enumerate() {
        let mut sum = T::zero();

        for k in 0..n_points {
            // Chebyshev nodes in [-1, 1]
            let theta = T::from(PI * (k as f64 + 0.5) / n_points as f64).unwrap();
            let x = theta.cos();

            // Function value: exp(-t * lambda_max * (x + 1) / 2)
            let exponent = -t * lambda_max * (x + T::one()) / T::from(2.0).unwrap();
            let f = exponent.exp();

            // Chebyshev polynomial T_j(x) = cos(j * arccos(x))
            let t_j = (T::from(j).unwrap() * theta).cos();

            sum = sum + f * t_j;
        }

        *coeff = T::from(2.0).unwrap() * sum / T::from(n_points).unwrap();
    }

    // First coefficient has weight 1 instead of 2
    coeffs[0] = coeffs[0] / T::from(2.0).unwrap();

    coeffs
}

/// Evaluates Chebyshev polynomial at matrix L_scaled applied to vectors.
///
/// Uses Clenshaw's recurrence algorithm for stability, computing
/// polynomial(L_scaled) @ vectors efficiently using sparse matrix-vector
/// products. Complexity: O(d * (|V| + |E|) * k) where d is degree,
/// |V| is number of vertices, |E| is number of edges, k is number of vectors.
///
/// # Parameters
/// * `l_scaled` - Scaled Laplacian matrix in [-1, 1] (sparse)
/// * `coeffs` - Chebyshev coefficients
/// * `vectors` - Matrix of shape (n, num_vectors)
///
/// # Returns
/// Result of polynomial(L_scaled) @ vectors, shape (n, num_vectors)
fn evaluate_chebyshev_polynomial<T>(
    l_scaled: &SparseSymmetricMatrix<T>,
    coeffs: &[T],
    vectors: &Array2<T>,
) -> Array2<T>
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let n = l_scaled.dim();
    let num_vectors = vectors.ncols();
    let degree = coeffs.len() - 1;

    let mut result = Array2::zeros((n, num_vectors));

    // Apply Clenshaw's algorithm to each vector
    for i in 0..num_vectors {
        let v = vectors.column(i).to_owned();

        // Clenshaw's algorithm for this vector
        let mut b_k_plus_2 = Array1::zeros(n);
        let mut b_k_plus_1 = Array1::zeros(n);

        for k in (1..=degree).rev() {
            // b_k = coeffs[k] * v + 2 * L_scaled @ b_k_plus_1 - b_k_plus_2
            let l_times_b = l_scaled.multiply(&b_k_plus_1);
            let b_k = &v * coeffs[k] + &(l_times_b * T::from(2.0).unwrap()) - &b_k_plus_2;

            b_k_plus_2 = b_k_plus_1;
            b_k_plus_1 = b_k;
        }

        // Final step
        let l_times_b = l_scaled.multiply(&b_k_plus_1);
        let final_result = &v * coeffs[0] + &l_times_b - &b_k_plus_2;

        for j in 0..n {
            result[[j, i]] = final_result[j];
        }
    }

    result
}

/// Evaluates Chebyshev polynomial at matrix L_scaled applied to a single vector.
fn evaluate_chebyshev_polynomial_vec<T>(
    l_scaled: &SparseSymmetricMatrix<T>,
    coeffs: &[T],
    v: &Array1<T>,
) -> Array1<T>
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let n = l_scaled.dim();
    let degree = coeffs.len() - 1;

    let mut b_k_plus_2 = Array1::zeros(n);
    let mut b_k_plus_1 = Array1::zeros(n);

    for k in (1..=degree).rev() {
        let l_times_b = l_scaled.multiply(&b_k_plus_1);
        let b_k = v * coeffs[k] + &(l_times_b * T::from(2.0).unwrap()) - &b_k_plus_2;

        b_k_plus_2 = b_k_plus_1;
        b_k_plus_1 = b_k;
    }

    let l_times_b = l_scaled.multiply(&b_k_plus_1);
    v * coeffs[0] + &l_times_b - &b_k_plus_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_chebyshev_approximation_identity() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 1.0);
        matrix.set_diagonal(1, 1.0);
        matrix.set_diagonal(2, 1.0);

        let vectors = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = chebyshev_approximation(&matrix, 0.0, 10, 1.0, &vectors);

        for i in 0..3 {
            for j in 0..2 {
                assert!((result[[i, j]] - vectors[[i, j]]).abs() < 1e-6);
            }
        }
    }
}
