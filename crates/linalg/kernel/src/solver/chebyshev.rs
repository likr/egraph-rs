//! Chebyshev polynomial approximation for matrix functions.
//!
//! This module implements Chebyshev polynomial approximation of exp(-tL) where L
//! is a graph Laplacian matrix, using Clenshaw's recurrence algorithm for stability.
use ndarray::ScalarOperand;

use crate::SparseSymmetricMatrix;
use ndarray::{Array1, Array2};
use num_traits::Float;
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
pub fn chebyshev_approximation<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    t: T,
    degree: usize,
    lambda_max: T,
    vectors: &Array2<T>,
) -> Array2<T>
where
    T: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    let u1 = laplacian.stationary_vector();
    let (kv, _, _) =
        chebyshev_approximation_with_baseline(laplacian, t, degree, lambda_max, vectors, &u1);
    kv
}

/// Approximates exp(-tL) @ vectors with baseline subtraction for variance reduction.
///
/// Returns (kv, w, residual_diag_sum) where kv is K @ vectors, w is residual matrix
/// exp(-tL) @ v_sub, and residual_diag_sum accumulates (v_r o exp(-tL) v_{sub, r}) across all samples.
pub fn chebyshev_approximation_with_baseline<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    t: T,
    degree: usize,
    lambda_max: T,
    vectors: &Array2<T>,
    u1: &Array1<T>,
) -> (Array2<T>, Array2<T>, Vec<T>)
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let coeffs = compute_chebyshev_coefficients(t, lambda_max, degree);
    chebyshev_eval_with_baseline(laplacian, &coeffs, T::one(), lambda_max, vectors, u1)
}

/// Approximates a band-pass kernel g_hat(L) = sum_m c_m exp(-t_m L) applied to vectors.
pub fn chebyshev_bandpass_approximation<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    terms: &[(T, T)],
    degree: usize,
    lambda_max: T,
    vectors: &Array2<T>,
) -> Array2<T>
where
    T: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    let u1 = laplacian.stationary_vector();
    let (kv, _, _) = chebyshev_bandpass_approximation_with_baseline(
        laplacian, terms, degree, lambda_max, vectors, &u1,
    );
    kv
}

/// Approximates a band-pass kernel g_hat(L) = sum_m c_m exp(-t_m L) applied to vectors
/// with baseline subtraction on the stationary eigenvector u1.
///
/// Projects out u1 (whose exact response is g_hat(0) = sum_m c_m), evaluates the polynomial
/// on the residual, and reconstructs K @ v = w + g_hat(0) * (u1^T v) u1.
pub fn chebyshev_bandpass_approximation_with_baseline<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    terms: &[(T, T)],
    degree: usize,
    lambda_max: T,
    vectors: &Array2<T>,
    u1: &Array1<T>,
) -> (Array2<T>, Array2<T>, Vec<T>)
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let coeffs = compute_chebyshev_bandpass_coefficients(terms, lambda_max, degree);
    let g_hat_0 = terms
        .iter()
        .map(|&(c, _)| c)
        .fold(T::zero(), |acc, c| acc + c);
    chebyshev_eval_with_baseline(laplacian, &coeffs, g_hat_0, lambda_max, vectors, u1)
}

fn safe_scale_and_shift<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    lambda_max: T,
) -> SparseSymmetricMatrix<T>
where
    T: Float + std::ops::AddAssign + Default,
{
    let two = T::from(2.0).unwrap();
    // Guard lambda_max == 0 (e.g. edgeless graph) so scale stays finite; L = 0 maps to -I.
    let safe_lmax = if lambda_max > T::zero() {
        lambda_max
    } else {
        T::one()
    };
    laplacian.scale_and_shift(two / safe_lmax, T::one())
}

fn chebyshev_eval_with_baseline<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    coeffs: &[T],
    g_hat_0: T,
    lambda_max: T,
    vectors: &Array2<T>,
    u1: &Array1<T>,
) -> (Array2<T>, Array2<T>, Vec<T>)
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let l_scaled = safe_scale_and_shift(laplacian, lambda_max);

    let n = l_scaled.dim();
    let num_vectors = vectors.ncols();
    let mut kv = Array2::zeros((n, num_vectors));
    let mut w_mat = Array2::zeros((n, num_vectors));
    let mut residual_diag_sum = vec![T::zero(); n];

    for i in 0..num_vectors {
        let v = vectors.column(i).to_owned();

        // Baseline Subtraction: project out stationary eigenvector u1
        // v_sub = v - (u1^T v) u1
        let dot_u1_v: T = u1.iter().zip(v.iter()).map(|(&u, &x)| u * x).sum();
        let mut v_sub = Array1::zeros(n);
        for j in 0..n {
            v_sub[j] = v[j] - dot_u1_v * u1[j];
        }

        // Compute w = g_hat(L) @ v_sub using Clenshaw's algorithm
        let w = evaluate_chebyshev_polynomial_vec(&l_scaled, coeffs, &v_sub);

        // Reconstruct full K @ v = w + g_hat(0) * (u1^T v) u1
        for j in 0..n {
            let full_kv_j = w[j] + g_hat_0 * dot_u1_v * u1[j];
            kv[[j, i]] = full_kv_j;
            w_mat[[j, i]] = w[j];
            residual_diag_sum[j] += v[j] * w[j];
        }
    }

    (kv, w_mat, residual_diag_sum)
}

/// Computes the baseline-subtracted Hutchinson diagonal estimate from residual sums.
pub(crate) fn compute_hutchinson_diagonal_from_residual<T>(
    residual_diag_sum: &[T],
    u1: &Array1<T>,
    g_hat_0: T,
    num_vectors: usize,
) -> Array1<T>
where
    T: Float,
{
    let n = residual_diag_sum.len();
    let m_t = T::from(num_vectors).unwrap();
    let mut diag = Array1::zeros(n);
    for j in 0..n {
        diag[j] = (residual_diag_sum[j] / m_t) + g_hat_0 * u1[j] * u1[j];
    }
    diag
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
    let l_scaled = safe_scale_and_shift(laplacian, lambda_max);

    let coeffs = compute_chebyshev_coefficients(t, lambda_max, degree);

    evaluate_chebyshev_polynomial_vec(&l_scaled, &coeffs, vector)
}

/// Approximates g_hat(L) @ vector = (sum_m c_m exp(-t_m L)) @ vector using Chebyshev expansion.
pub fn chebyshev_bandpass_approximation_vec<T>(
    laplacian: &SparseSymmetricMatrix<T>,
    terms: &[(T, T)],
    degree: usize,
    lambda_max: T,
    vector: &Array1<T>,
) -> Array1<T>
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
{
    let l_scaled = safe_scale_and_shift(laplacian, lambda_max);

    let coeffs = compute_chebyshev_bandpass_coefficients(terms, lambda_max, degree);

    evaluate_chebyshev_polynomial_vec(&l_scaled, &coeffs, vector)
}

/// Computes Chebyshev coefficients for g_hat(lambda) = sum_m c_m * exp(-t_m * lambda).
///
/// Handles `t_m == 0` exactly without quadrature since exp(0) = 1 has zeroth coefficient 1
/// and all higher Chebyshev coefficients equal to 0.
pub fn compute_chebyshev_bandpass_coefficients<S>(
    terms: &[(S, S)],
    lambda_max: S,
    order: usize,
) -> Vec<S>
where
    S: Float,
{
    let mut coeffs = vec![S::zero(); order + 1];
    for &(c_m, t_m) in terms {
        if c_m == S::zero() {
            continue;
        }
        if t_m == S::zero() {
            coeffs[0] = coeffs[0] + c_m;
        } else {
            let term_coeffs = compute_chebyshev_coefficients(t_m, lambda_max, order);
            for (dst, &src) in coeffs.iter_mut().zip(term_coeffs.iter()) {
                *dst = *dst + c_m * src;
            }
        }
    }
    coeffs
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
#[allow(dead_code)]
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
    use crate::estimate_lambda_max;
    use ndarray::Array2;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn sparse_to_dense(matrix: &SparseSymmetricMatrix<f64>) -> Array2<f64> {
        let n = matrix.dim();
        let mut dense = Array2::zeros((n, n));
        for i in 0..n {
            dense[[i, i]] = matrix.diagonal()[i];
        }
        for &(i, j, w) in matrix.edges() {
            dense[[i, j]] += w;
            dense[[j, i]] += w;
        }
        dense
    }

    fn dense_expm_neg(l_dense: &Array2<f64>, t: f64) -> Array2<f64> {
        let n = l_dense.nrows();
        let mut a = l_dense * (-t);
        let max_row_norm = (0..n)
            .map(|i| (0..n).map(|j| a[[i, j]].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let s = if max_row_norm > 0.5 {
            max_row_norm.log2().ceil().max(0.0) as u32 + 1
        } else {
            0
        };
        let scale = 2.0_f64.powi(-(s as i32));
        a *= scale;

        let mut res = Array2::eye(n);
        let mut term = Array2::eye(n);
        for k in 1..=30 {
            term = term.dot(&a) / (k as f64);
            res = &res + &term;
        }
        for _ in 0..s {
            res = res.dot(&res);
        }
        res
    }

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

    #[test]
    fn test_bandpass_coefficients_zero_time_exact() {
        let coeffs = compute_chebyshev_bandpass_coefficients(&[(0.25, 0.0), (-0.1, 0.0)], 2.0, 10);
        assert!((coeffs[0] - 0.15).abs() < 1e-12);
        for &c in &coeffs[1..] {
            assert_eq!(c, 0.0);
        }
    }

    #[test]
    fn test_chebyshev_bandpass_vs_dense_expm() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
        for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)] {
            g.add_edge(nodes[u], nodes[v], ());
        }

        let lap_std = SparseSymmetricMatrix::standard_laplacian(&g, |_| 1.0);
        let lap_sym = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0);

        let terms = vec![(1.5, 0.4), (-0.75, 1.8), (0.25, 0.0)];
        let mut rng = StdRng::seed_from_u64(42);

        for lap in [&lap_std, &lap_sym] {
            let lmax = estimate_lambda_max(lap, &mut rng, 100, 1e-6) * 1.05;
            let l_dense = sparse_to_dense(lap);
            let mut k_dense = Array2::<f64>::zeros((5, 5));
            for &(c, t) in &terms {
                k_dense = k_dense + dense_expm_neg(&l_dense, t) * c;
            }

            let eye = Array2::eye(5);
            let k_cheb = chebyshev_bandpass_approximation(lap, &terms, 25, lmax, &eye);
            for i in 0..5 {
                for j in 0..5 {
                    assert!(
                        (k_cheb[[i, j]] - k_dense[[i, j]]).abs() < 1e-7,
                        "mismatch at ({}, {}): {} vs {}",
                        i,
                        j,
                        k_cheb[[i, j]],
                        k_dense[[i, j]]
                    );
                }
            }

            let v = Array1::from_vec(vec![1.0, -0.5, 2.0, 0.3, -1.2]);
            let vec_cheb = chebyshev_bandpass_approximation_vec(lap, &terms, 25, lmax, &v);
            let vec_dense = k_dense.dot(&v);
            for i in 0..5 {
                assert!(
                    (vec_cheb[i] - vec_dense[i]).abs() < 1e-7,
                    "vec mismatch at {}: {} vs {}",
                    i,
                    vec_cheb[i],
                    vec_dense[i]
                );
            }

            // Verify baseline stationary response g_hat(0) = sum c_m = 1.0
            let u1 = lap.stationary_vector();
            let u1_col = u1.clone().insert_axis(ndarray::Axis(1));
            let (kv, w_mat, _) =
                chebyshev_bandpass_approximation_with_baseline(lap, &terms, 25, lmax, &u1_col, &u1);
            let g_hat_0: f64 = terms.iter().map(|&(c, _)| c).sum();
            for i in 0..5 {
                assert!(w_mat[[i, 0]].abs() < 1e-10);
                assert!((kv[[i, 0]] - g_hat_0 * u1[i]).abs() < 1e-10);
            }
        }
    }
}
