//! Exact heat diffusion kernel matrix computed via high-order Chebyshev polynomial expansion.

use crate::{
    chebyshev::chebyshev_approximation, heat_kernel::HeatKernel, power_method::estimate_lambda_max,
};
use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use petgraph_distance::SparseSymmetricMatrix;
use petgraph_drawing::DrawingValue;

/// Exact heat kernel exp(-tL) computed via Chebyshev polynomial expansion for all matrix columns.
#[derive(Debug, Clone)]
pub struct ExactDiffusionKernel<S> {
    n: usize,
    t: S,
    matrix: Array2<S>,
}

impl<S> ExactDiffusionKernel<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    /// Creates a new ExactDiffusionKernel with automatic lambda_max estimation.
    pub fn new(laplacian: &SparseSymmetricMatrix<S>, t: S, degree: usize) -> Self {
        let mut rng = rand::thread_rng();
        let lambda_max = estimate_lambda_max(laplacian, &mut rng, 100, S::from_f64(1e-4).unwrap());
        Self::new_with_lambda_max(laplacian, t, degree, lambda_max)
    }

    /// Creates an ExactDiffusionKernel with externally provided lambda_max.
    pub fn new_with_lambda_max(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        lambda_max: S,
    ) -> Self {
        let n = laplacian.dim();

        // Construct identity matrix I of size (n, n)
        let mut eye = Array2::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = S::one();
        }

        // Compute exp(-tL) @ I = exp(-tL)
        let matrix = chebyshev_approximation(laplacian, t, degree, lambda_max, &eye);

        Self { n, t, matrix }
    }

    /// Queries the (i, j) element of the exact heat kernel matrix.
    pub fn get(&self, i: usize, j: usize) -> S {
        assert!(i < self.n && j < self.n, "Index out of bounds");
        self.matrix[[i, j]]
    }

    /// Computes the heat diffusion distance between node i and node j.
    pub fn distance(&self, i: usize, j: usize) -> S {
        HeatKernel::distance(self, i, j)
    }

    /// Computes the single-source heat diffusion distance vector from a pivot node.
    pub fn pivot_distance_vector(&self, pivot: usize) -> Vec<S> {
        HeatKernel::pivot_distance_vector(self, pivot)
    }

    /// Returns the number of nodes in the graph.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the diffusion time parameter t.
    pub fn t(&self) -> S {
        self.t
    }

    /// Returns a reference to the full kernel matrix.
    pub fn matrix(&self) -> &Array2<S> {
        &self.matrix
    }
}

impl<S> HeatKernel<S> for ExactDiffusionKernel<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    fn get(&self, i: usize, j: usize) -> S {
        self.get(i, j)
    }

    fn n(&self) -> usize {
        self.n()
    }

    fn t(&self) -> S {
        self.t()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph_distance::{Laplacian, StandardLaplacian};

    #[test]
    fn test_exact_diffusion_kernel_basic() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());

        let laplacian = StandardLaplacian.build(&g, &mut |_| 1.0);
        let kernel = ExactDiffusionKernel::new(&laplacian, 1.0, 30);

        assert_eq!(kernel.n(), 3);
        assert_eq!(kernel.t(), 1.0);

        for i in 0..3 {
            assert!(kernel.get(i, i) > 0.0);
        }

        assert!((kernel.get(0, 1) - kernel.get(1, 0)).abs() < 1e-10);

        assert_eq!(kernel.distance(0, 0), 0.0);
        assert!(kernel.distance(0, 1) > 0.0);

        let dist_vec = kernel.pivot_distance_vector(0);
        assert_eq!(dist_vec.len(), 3);
        assert_eq!(dist_vec[0], 0.0);
        assert!(dist_vec[1] > 0.0);
    }
}
