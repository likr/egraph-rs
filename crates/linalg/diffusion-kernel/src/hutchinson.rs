//! Hutchinson trace estimator for approximating kernel matrix elements.

use ndarray::Array2;
use num_traits::Float;
use rand::Rng;

/// Hutchinson estimator for querying kernel matrix elements.
#[derive(Debug, Clone)]
pub struct HutchinsonEstimator<T> {
    v: Array2<T>,
    kv: Array2<T>,
    num_vectors: usize,
}

impl<T> HutchinsonEstimator<T> {
    pub fn new(v: Array2<T>, kv: Array2<T>) -> Self {
        assert_eq!(v.shape(), kv.shape(), "V and KV must have the same shape");
        let num_vectors = v.ncols();
        Self { v, kv, num_vectors }
    }

    pub fn n(&self) -> usize {
        self.v.nrows()
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

impl<T> HutchinsonEstimator<T>
where
    T: Float + std::iter::Sum,
{
    pub fn query(&self, i: usize, j: usize) -> T {
        if i == j {
            self.query_diagonal(i)
        } else {
            self.query_symmetric(i, j)
        }
    }

    fn query_diagonal(&self, i: usize) -> T {
        let mut sum = T::zero();
        for l in 0..self.num_vectors {
            sum = sum + self.v[[i, l]] * self.kv[[i, l]];
        }
        sum / T::from(self.num_vectors).unwrap()
    }

    fn query_symmetric(&self, i: usize, j: usize) -> T {
        let mut sum1 = T::zero();
        let mut sum2 = T::zero();

        for l in 0..self.num_vectors {
            sum1 = sum1 + self.v[[i, l]] * self.kv[[j, l]];
            sum2 = sum2 + self.v[[j, l]] * self.kv[[i, l]];
        }

        (sum1 + sum2) / T::from(2 * self.num_vectors).unwrap()
    }
}

pub fn generate_rademacher_vectors<T, R>(n: usize, num_vectors: usize, rng: &mut R) -> Array2<T>
where
    T: Float,
    R: Rng,
{
    Array2::from_shape_fn((n, num_vectors), |_| {
        if rng.gen::<bool>() {
            T::one()
        } else {
            -T::one()
        }
    })
}
