use crate::*;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::hash::Hash;

pub(crate) fn compute_degrees<S: Float + std::ops::AddAssign + Default>(
    matrix: &SparseSymmetricMatrix<S>,
) -> Vec<f64> {
    let mut degrees = vec![0.0f64; matrix.dim()];
    for &(i, j, w) in matrix.edges() {
        let abs_w = w.to_f64().unwrap().abs();
        degrees[i] += abs_w;
        degrees[j] += abs_w;
    }
    for (deg, &diag) in degrees.iter_mut().zip(matrix.diagonal()) {
        if *deg == 0.0f64 && diag.to_f64().unwrap() > 0.0f64 {
            *deg = diag.to_f64().unwrap();
        }
    }
    degrees
}

pub(crate) fn to_f64_matrix<S: Float + std::ops::AddAssign + Default>(
    matrix: &SparseSymmetricMatrix<S>,
) -> SparseSymmetricMatrix<f64> {
    let mut matrix_f64 = SparseSymmetricMatrix::new(matrix.dim());
    for &(i, j, w) in matrix.edges() {
        matrix_f64.add_edge(i, j, w.to_f64().unwrap());
    }
    for (i, &w) in matrix.diagonal().iter().enumerate() {
        matrix_f64.set_diagonal(i, w.to_f64().unwrap());
    }
    matrix_f64
}

/// Builder for MultiscaleDiffusionKernel
pub struct MultiscaleDiffusionKernelBuilder<'a, S> {
    pub matrix: &'a SparseSymmetricMatrix<S>,
    pub alpha: S,
    pub tol: f64,
    pub max_iter: usize,
}

impl<'a, S: Float + num_traits::FromPrimitive + std::ops::AddAssign + Default>
    MultiscaleDiffusionKernelBuilder<'a, S>
{
    pub fn new(matrix: &'a SparseSymmetricMatrix<S>, alpha: S) -> Self {
        Self {
            matrix,
            alpha,
            tol: 1e-7,
            max_iter: 100,
        }
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn build(self) -> Result<MultiscaleDiffusionKernel<S>, String> {
        if !self.alpha.is_finite() || self.alpha < S::zero() || self.alpha >= S::one() {
            return Err("alpha must be finite and in [0, 1)".to_string());
        }
        let n = self.matrix.dim();
        let degrees = compute_degrees(self.matrix);
        let matrix_f64 = to_f64_matrix(self.matrix);

        let mut b = vec![0.0f64; n * n];
        for i in 0..n {
            b[i * n + i] = 1.0;
        }

        let mut y = vec![0.0f64; n * n];
        let mut buffers = BicgstabSolverBuffers::new(n, n);

        solve_batched_bicgstab(
            &matrix_f64,
            &degrees,
            self.alpha.to_f64().unwrap(),
            &b,
            &mut y,
            n,
            self.tol,
            self.max_iter,
            &mut buffers,
        );

        let mut array = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                array[[i, j]] = S::from_f64(y[i * n + j]).unwrap();
            }
        }

        Ok(MultiscaleDiffusionKernel { n, matrix: array })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_multiscale_diffusion_kernel_builder_rejects_invalid_alpha() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        for bad_alpha in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            let err = MultiscaleDiffusionKernelBuilder::new(&lap, bad_alpha)
                .build()
                .unwrap_err();
            assert!(err.contains("alpha"));
        }
        assert!(MultiscaleDiffusionKernelBuilder::new(&lap, 0.0)
            .build()
            .is_ok());
    }
}
