//! Power method for estimating the maximum eigenvalue of a symmetric matrix.

use ndarray::Array1;
use num_traits::Float;
use petgraph_linalg_spmv::SparseSymmetricMatrix;
use rand::{Rng, RngExt};

/// Estimates the maximum eigenvalue of a symmetric matrix using the power method.
///
/// The power method iteratively computes v_{k+1} = A*v_k / ||A*v_k|| and estimates
/// λ_max ≈ v_k^T * A * v_k / (v_k^T * v_k).
///
/// # Parameters
/// * `matrix` - The symmetric matrix to analyze
/// * `rng` - Random number generator for initialization
/// * `max_iterations` - Maximum number of iterations (default: 100)
/// * `tolerance` - Convergence tolerance (default: 1e-6)
///
/// # Returns
/// Estimated maximum eigenvalue
pub fn estimate_lambda_max<T, R>(
    matrix: &SparseSymmetricMatrix<T>,
    rng: &mut R,
    max_iterations: usize,
    tolerance: T,
) -> T
where
    T: Float + std::iter::Sum + std::ops::AddAssign + Default + ndarray::ScalarOperand,
    R: Rng,
{
    let n = matrix.dim();

    // Initialize with random vector
    let mut v = Array1::from_shape_fn(n, |_| T::from(rng.random_range(-1.0..1.0)).unwrap());

    // Normalize
    let norm = v.iter().map(|&x| x * x).sum::<T>().sqrt();
    v = v / norm;

    let mut lambda_prev = T::zero();

    for _ in 0..max_iterations {
        // Compute Av
        let av = matrix.multiply(&v);

        // Estimate eigenvalue: λ ≈ v^T * A * v
        let lambda = v
            .iter()
            .zip(av.iter())
            .map(|(&vi, &avi)| vi * avi)
            .sum::<T>();

        // Check convergence
        if (lambda - lambda_prev).abs() < tolerance {
            return lambda;
        }

        // Normalize Av for next iteration
        let av_norm = av.iter().map(|&x| x * x).sum::<T>().sqrt();
        if av_norm > T::zero() {
            v = av / av_norm;
        }

        lambda_prev = lambda;
    }

    lambda_prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_estimate_lambda_max_identity() {
        // Identity matrix has λ_max = 1
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 1.0);
        matrix.set_diagonal(1, 1.0);
        matrix.set_diagonal(2, 1.0);

        let mut rng = StdRng::seed_from_u64(42);
        let lambda_max = estimate_lambda_max(&matrix, &mut rng, 100, 1e-6);

        assert!((lambda_max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_estimate_lambda_max_scaled_identity() {
        // 5*I has λ_max = 5
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 5.0);
        matrix.set_diagonal(1, 5.0);
        matrix.set_diagonal(2, 5.0);

        let mut rng = StdRng::seed_from_u64(42);
        let lambda_max = estimate_lambda_max(&matrix, &mut rng, 100, 1e-6);

        assert!((lambda_max - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_estimate_lambda_max_tridiagonal() {
        // Tridiagonal matrix:
        // [2  1  0]
        // [1  2  1]
        // [0  1  2]
        // λ_max ≈ 3.414 (2 + sqrt(2))
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 2.0);
        matrix.set_diagonal(1, 2.0);
        matrix.set_diagonal(2, 2.0);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);

        let mut rng = StdRng::seed_from_u64(42);
        let lambda_max = estimate_lambda_max(&matrix, &mut rng, 1000, 1e-6);

        let expected = 2.0 + 2.0_f64.sqrt();
        assert!((lambda_max - expected).abs() < 1e-3);
    }
}
