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
