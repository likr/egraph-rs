//! Hutchinson trace estimator for approximating kernel matrix elements.

use ndarray::Array2;
use num_traits::Float;
use rand::Rng;

/// Hutchinson estimator for querying kernel matrix elements and estimating distances.
#[derive(Debug, Clone)]
pub struct HutchinsonEstimator<T> {
    v: Array2<T>,
    kv: Array2<T>,
    num_vectors: usize,
    diag: Vec<T>,
}

impl<T> HutchinsonEstimator<T>
where
    T: Float + std::iter::Sum,
{
    pub fn new(v: Array2<T>, kv: Array2<T>) -> Self {
        assert_eq!(v.shape(), kv.shape(), "V and KV must have the same shape");
        let num_vectors = v.ncols();
        let n = v.nrows();

        let v_slice = v.as_slice().expect("Array2 V must be contiguous");
        let kv_slice = kv.as_slice().expect("Array2 KV must be contiguous");

        let num_vectors_t = T::from(num_vectors).unwrap();
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            let v_row = &v_slice[i * num_vectors..(i + 1) * num_vectors];
            let kv_row = &kv_slice[i * num_vectors..(i + 1) * num_vectors];
            let sum: T = v_row.iter().zip(kv_row.iter()).map(|(&x, &y)| x * y).sum();
            diag.push(sum / num_vectors_t);
        }

        Self {
            v,
            kv,
            num_vectors,
            diag,
        }
    }

    pub fn n(&self) -> usize {
        self.v.nrows()
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn diag(&self) -> &[T] {
        &self.diag
    }

    pub fn v(&self) -> &Array2<T> {
        &self.v
    }

    pub fn kv(&self) -> &Array2<T> {
        &self.kv
    }

    pub fn query(&self, i: usize, j: usize) -> T {
        if i == j {
            self.query_diagonal(i)
        } else {
            self.query_symmetric(i, j)
        }
    }

    pub fn query_diagonal(&self, i: usize) -> T {
        self.diag[i]
    }

    pub fn query_symmetric(&self, i: usize, j: usize) -> T {
        let m = self.num_vectors;
        let v_slice = self.v.as_slice().expect("Array2 V must be contiguous");
        let kv_slice = self.kv.as_slice().expect("Array2 KV must be contiguous");

        let v_i = &v_slice[i * m..(i + 1) * m];
        let kv_i = &kv_slice[i * m..(i + 1) * m];
        let v_j = &v_slice[j * m..(j + 1) * m];
        let kv_j = &kv_slice[j * m..(j + 1) * m];

        let sum1: T = v_i.iter().zip(kv_j.iter()).map(|(&x, &y)| x * y).sum();
        let sum2: T = v_j.iter().zip(kv_i.iter()).map(|(&x, &y)| x * y).sum();

        let two_m = T::from(2 * self.num_vectors).unwrap();
        (sum1 + sum2) / two_m
    }

    pub fn distance_kernel(&self, i: usize, j: usize) -> T {
        if i == j {
            return T::zero();
        }
        let k_ii = self.query_diagonal(i);
        let k_jj = self.query_diagonal(j);
        let k_ij = self.query_symmetric(i, j);
        let two = T::from(2.0).unwrap();
        let diff = k_ii + k_jj - two * k_ij;
        diff.max(T::zero()).sqrt()
    }

    pub fn distance_l2(&self, i: usize, j: usize) -> T {
        if i == j {
            return T::zero();
        }
        let m = self.num_vectors;
        let kv_slice = self.kv.as_slice().expect("Array2 KV must be contiguous");

        let kv_i = &kv_slice[i * m..(i + 1) * m];
        let kv_j = &kv_slice[j * m..(j + 1) * m];

        let sum_sq: T = kv_i
            .iter()
            .zip(kv_j.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        let m_t = T::from(m).unwrap();
        (sum_sq / m_t).sqrt()
    }

    pub fn distance(&self, i: usize, j: usize) -> T {
        self.distance_kernel(i, j)
    }
}

pub(crate) fn generate_rademacher_vectors<T, R>(
    n: usize,
    num_vectors: usize,
    rng: &mut R,
) -> Array2<T>
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
