use crate::kernel::multiscale_diffusion_kernel_builder::{compute_degrees, to_f64_matrix};
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

/// Builder for PivotedMultiscaleDiffusionKernel
pub struct PivotedMultiscaleDiffusionKernelBuilder<'a, S> {
    pub matrix: &'a SparseSymmetricMatrix<S>,
    pub alpha: S,
    pub pivots: Vec<usize>,
    pub tol: f64,
    pub max_iter: usize,
}

impl<'a, S: Float + num_traits::FromPrimitive + std::ops::AddAssign + Default>
    PivotedMultiscaleDiffusionKernelBuilder<'a, S>
{
    pub fn new(matrix: &'a SparseSymmetricMatrix<S>, alpha: S, pivots: Vec<usize>) -> Self {
        Self {
            matrix,
            alpha,
            pivots,
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

    pub fn build(self) -> Result<PivotedMultiscaleDiffusionKernel<S>, String> {
        let n = self.matrix.dim();
        let m = self.pivots.len();
        let degrees = compute_degrees(self.matrix);
        let matrix_f64 = to_f64_matrix(self.matrix);

        let mut b = vec![0.0f64; n * m];
        for (k, &p) in self.pivots.iter().enumerate() {
            if p >= n {
                return Err("Pivot index out of bounds".to_string());
            }
            b[p * m + k] = 1.0;
        }

        let mut y = vec![0.0f64; n * m];
        let mut buffers = BicgstabSolverBuffers::new(n, m);

        solve_batched_bicgstab(
            &matrix_f64,
            &degrees,
            self.alpha.to_f64().unwrap(),
            &b,
            &mut y,
            m,
            self.tol,
            self.max_iter,
            &mut buffers,
        );

        let mut pivot_vectors = Vec::with_capacity(m);
        for k in 0..m {
            let mut vec_k = Array1::zeros(n);
            for i in 0..n {
                vec_k[i] = S::from_f64(y[i * m + k]).unwrap();
            }
            pivot_vectors.push(vec_k);
        }

        Ok(PivotedMultiscaleDiffusionKernel {
            pivots: self.pivots,
            vectors: pivot_vectors,
        })
    }
}
