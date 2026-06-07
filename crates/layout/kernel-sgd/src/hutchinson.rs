//! Hutchinson trace estimator for approximating kernel matrix elements.
//!
//! This module implements the Hutchinson trace estimator with symmetry optimization
//! to efficiently query individual elements of the diffusion kernel matrix K = exp(-tL).

use ndarray::Array2;
use num_traits::Float;
use rand::{Rng, RngExt};

/// Hutchinson estimator for querying kernel matrix elements.
///
/// This structure stores random vectors V and the result of applying the kernel KV,
/// allowing efficient queries of individual kernel matrix elements using the formula:
/// K[i,j] ≈ (1/k) Σ_l V[i,l] * (KV)[j,l]
///
/// For symmetric matrices, we optimize by using K[i,j] = K[j,i] to effectively
/// double the number of samples:
/// K[i,j] ≈ (1/2k) [Σ_l V[i,l]*KV[j,l] + Σ_l V[j,l]*KV[i,l]]
#[derive(Debug)]
pub struct HutchinsonEstimator<T> {
    /// Random vectors matrix (n, num_vectors)
    v: Array2<T>,
    /// Product K @ V (n, num_vectors)
    kv: Array2<T>,
    /// Number of random vectors
    num_vectors: usize,
}

impl<T> HutchinsonEstimator<T> {
    /// Creates a new Hutchinson estimator from precomputed K @ V.
    ///
    /// # Parameters
    /// * `v` - Random vectors matrix (n, num_vectors)
    /// * `kv` - Product K @ V (n, num_vectors)
    ///
    /// # Returns
    /// A new Hutchinson estimator
    pub fn new(v: Array2<T>, kv: Array2<T>) -> Self {
        assert_eq!(v.shape(), kv.shape(), "V and KV must have the same shape");
        let num_vectors = v.ncols();

        Self { v, kv, num_vectors }
    }

    /// Returns the number of nodes in the kernel matrix.
    pub fn n(&self) -> usize {
        self.v.nrows()
    }

    /// Returns the number of random vectors used.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

impl<T> HutchinsonEstimator<T>
where
    T: Float + std::iter::Sum,
{
    /// Queries the (i, j) element of the approximated kernel matrix with symmetry optimization.
    ///
    /// For symmetric matrices, uses K[i,j] = K[j,i] to effectively double the sample count:
    /// K[i,j] ≈ (1/2k) [Σ_l V[i,l]*KV[j,l] + Σ_l V[j,l]*KV[i,l]]
    ///
    /// This gives us 2k samples worth of accuracy with the same k random vectors.
    ///
    /// # Parameters
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    /// Approximated kernel value at (i, j)
    pub fn query(&self, i: usize, j: usize) -> T {
        if i == j {
            // Diagonal element: no symmetry benefit
            self.query_diagonal(i)
        } else {
            // Off-diagonal: use symmetry to double samples
            self.query_symmetric(i, j)
        }
    }

    /// Queries a diagonal element K[i,i].
    fn query_diagonal(&self, i: usize) -> T {
        // K[i,i] ≈ (1/k) Σ_l V[i,l] * KV[i,l]
        let mut sum = T::zero();
        for l in 0..self.num_vectors {
            sum = sum + self.v[[i, l]] * self.kv[[i, l]];
        }
        sum / T::from(self.num_vectors).unwrap()
    }

    /// Queries an off-diagonal element K[i,j] with symmetry optimization.
    ///
    /// Uses both K[i,j] and K[j,i] to effectively get 2k samples:
    /// K[i,j] ≈ (1/2k) [Σ_l V[i,l]*KV[j,l] + Σ_l V[j,l]*KV[i,l]]
    fn query_symmetric(&self, i: usize, j: usize) -> T {
        let mut sum1 = T::zero();
        let mut sum2 = T::zero();

        for l in 0..self.num_vectors {
            // First direction: V[i,l] * KV[j,l]
            sum1 = sum1 + self.v[[i, l]] * self.kv[[j, l]];
            // Second direction: V[j,l] * KV[i,l] (using symmetry)
            sum2 = sum2 + self.v[[j, l]] * self.kv[[i, l]];
        }

        // Average both estimates for 2x effective samples
        (sum1 + sum2) / T::from(2 * self.num_vectors).unwrap()
    }
}

/// Generates Rademacher random vectors (entries are ±1 with equal probability).
///
/// # Parameters
/// * `n` - Dimension of vectors
/// * `num_vectors` - Number of random vectors to generate
/// * `rng` - Random number generator
///
/// # Returns
/// Matrix of shape (n, num_vectors) where each column is a Rademacher vector
pub fn generate_rademacher_vectors<T, R>(n: usize, num_vectors: usize, rng: &mut R) -> Array2<T>
where
    T: Float,
    R: Rng,
{
    Array2::from_shape_fn((n, num_vectors), |_| {
        if rng.random::<bool>() {
            T::one()
        } else {
            -T::one()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_hutchinson_estimator_identity() {
        // For identity matrix K=I, we have KV = V
        // So K[i,j] should be ≈ 0 for i≠j and ≈ 1 for i=j
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10;
        let num_vectors = 100;

        let v = generate_rademacher_vectors(n, num_vectors, &mut rng);
        let kv = v.clone(); // K = I, so KV = V

        let estimator = HutchinsonEstimator::new(v, kv);

        // Diagonal elements should be close to 1
        for i in 0..n {
            let val: f64 = estimator.query(i, i);
            assert!((val - 1.0).abs() < 0.2, "Diagonal K[{},{}] = {}", i, i, val);
        }

        // Off-diagonal elements should be close to 0
        for i in 0..3 {
            for j in (i + 1)..3 {
                let val: f64 = estimator.query(i, j);
                assert!(val.abs() < 0.3, "Off-diagonal K[{},{}] = {}", i, j, val);
            }
        }
    }

    #[test]
    fn test_hutchinson_estimator_symmetry() {
        // Test that symmetry optimization works: K[i,j] should equal K[j,i]
        let mut rng = StdRng::seed_from_u64(42);
        let n = 5;
        let num_vectors = 50;

        let v = generate_rademacher_vectors(n, num_vectors, &mut rng);
        // Create some arbitrary symmetric kernel result
        let mut kv = Array2::zeros((n, num_vectors));
        for i in 0..n {
            for j in 0..num_vectors {
                kv[[i, j]] = v[[i, j]] * 0.8 + v[[(i + 1) % n, j]] * 0.2;
            }
        }

        let estimator = HutchinsonEstimator::new(v, kv);

        // Check symmetry for a few pairs
        for i in 0..3 {
            for j in (i + 1)..4 {
                let k_ij: f64 = estimator.query(i, j);
                let k_ji: f64 = estimator.query(j, i);
                // Should be exactly equal due to our symmetric implementation
                assert!(
                    (k_ij - k_ji).abs() < 1e-10,
                    "K[{},{}]={} != K[{},{}]={}",
                    i,
                    j,
                    k_ij,
                    j,
                    i,
                    k_ji
                );
            }
        }
    }

    #[test]
    fn test_generate_rademacher_vectors() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10;
        let num_vectors = 5;

        let v: Array2<f64> = generate_rademacher_vectors(n, num_vectors, &mut rng);

        assert_eq!(v.shape(), &[n, num_vectors]);

        // All entries should be ±1
        for i in 0..n {
            for j in 0..num_vectors {
                let val = v[[i, j]];
                assert!(
                    val == 1.0 || val == -1.0,
                    "Invalid Rademacher value: {}",
                    val
                );
            }
        }
    }
}
