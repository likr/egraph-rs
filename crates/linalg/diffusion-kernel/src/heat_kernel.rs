//! Shared trait interface for heat diffusion kernels.

use ndarray::ScalarOperand;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Common trait for heat diffusion kernel implementations.
pub trait HeatKernel<S> {
    /// Queries the (i, j) element of the heat kernel matrix.
    fn get(&self, i: usize, j: usize) -> S;

    /// Returns the number of nodes in the graph.
    fn n(&self) -> usize;

    /// Returns the diffusion time parameter t.
    fn t(&self) -> S;

    /// Computes the heat diffusion distance between node i and node j.
    fn distance(&self, i: usize, j: usize) -> S
    where
        S: Float + ScalarOperand + DrawingValue,
    {
        if i == j {
            return S::zero();
        }
        let k_ii = self.get(i, i);
        let k_jj = self.get(j, j);
        let k_ij = self.get(i, j);

        let eps = S::from_f64(1e-15).unwrap();
        let k_ii_clamped = k_ii.max(eps);
        let k_jj_clamped = k_jj.max(eps);
        let sqrt_prod = (k_ii_clamped * k_jj_clamped).sqrt();

        let ratio = (k_ij / sqrt_prod).clamp(eps, S::one());
        let four_t = S::from_f64(4.0).unwrap() * self.t();
        (-four_t * ratio.ln()).max(S::zero()).sqrt()
    }

    /// Computes the single-source heat diffusion distance vector from a pivot node.
    fn pivot_distance_vector(&self, pivot: usize) -> Vec<S>
    where
        S: Float + ScalarOperand + DrawingValue,
    {
        let n = self.n();
        assert!(pivot < n, "Pivot index out of bounds");

        let eps = S::from_f64(1e-15).unwrap();
        let mut distances = vec![S::zero(); n];
        let four_t = S::from_f64(4.0).unwrap() * self.t();

        // Precompute diagonal elements K_jj for all j
        let mut diag = vec![S::zero(); n];
        for (j, diag_j) in diag.iter_mut().enumerate() {
            *diag_j = self.get(j, j).max(eps);
        }

        let sqrt_k_pp = diag[pivot].sqrt();

        // Compute row K_{pivot, j} for all j
        for j in 0..n {
            if j == pivot {
                distances[j] = S::zero();
            } else {
                let k_pj = self.get(pivot, j);
                let sqrt_k_jj = diag[j].sqrt();
                let ratio = (k_pj / (sqrt_k_pp * sqrt_k_jj)).clamp(eps, S::one());
                distances[j] = (-four_t * ratio.ln()).max(S::zero()).sqrt();
            }
        }

        distances
    }
}
