//! Power method for estimating the maximum eigenvalue of a symmetric matrix.

use ndarray::Array1;
use num_traits::Float;
use petgraph_distance::SparseSymmetricMatrix;
use rand::Rng;

/// Estimates the maximum eigenvalue of a symmetric matrix using the power method.
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

    let mut v = Array1::from_shape_fn(n, |_| T::from(rng.gen_range(-1.0..1.0)).unwrap());

    let norm = v.iter().map(|&x| x * x).sum::<T>().sqrt();
    v = v / norm;

    let mut lambda_prev = T::zero();

    for _ in 0..max_iterations {
        let av = matrix.multiply(&v);

        let lambda = v
            .iter()
            .zip(av.iter())
            .map(|(&vi, &avi)| vi * avi)
            .sum::<T>();

        if (lambda - lambda_prev).abs() < tolerance {
            return lambda;
        }

        let av_norm = av.iter().map(|&x| x * x).sum::<T>().sqrt();
        if av_norm > T::zero() {
            v = av / av_norm;
        }

        lambda_prev = lambda;
    }

    lambda_prev
}
