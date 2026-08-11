//! Multiscale diffusion kernels computed via Batched BiCGSTAB.

use crate::bicgstab::{solve_batched_bicgstab, BicgstabSolverBuffers};
use crate::traits::PivotedKernel;
use ndarray::{Array1, Array2};
use num_traits::Float;
use petgraph_distance::{Kernel, SparseSymmetricMatrix};

fn compute_degrees<S: Float + std::ops::AddAssign + Default>(
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

fn to_f64_matrix<S: Float + std::ops::AddAssign + Default>(
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
    matrix: &'a SparseSymmetricMatrix<S>,
    alpha: S,
    tol: f64,
    max_iter: usize,
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

/// Exact multiscale diffusion kernel (I - alpha P)^{-1}.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionKernel<S> {
    n: usize,
    matrix: Array2<S>,
}

impl<S: Float> Kernel<usize, S> for MultiscaleDiffusionKernel<S> {
    fn get(&self, u: usize, v: usize) -> Option<S> {
        if u < self.n && v < self.n {
            Some(self.get_by_index(u, v))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        assert!(i < self.n && j < self.n, "Index out of bounds");
        self.matrix[[i, j]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.n)
    }

    fn col_index(&self, v: usize) -> Option<usize> {
        Some(v).filter(|&j| j < self.n)
    }
}

/// Builder for PivotedMultiscaleDiffusionKernel
pub struct PivotedMultiscaleDiffusionKernelBuilder<'a, S> {
    matrix: &'a SparseSymmetricMatrix<S>,
    alpha: S,
    pivots: Vec<usize>,
    tol: f64,
    max_iter: usize,
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

/// Pivoted multiscale diffusion kernel.
#[derive(Debug, Clone)]
pub struct PivotedMultiscaleDiffusionKernel<S> {
    pivots: Vec<usize>,
    vectors: Vec<Array1<S>>,
}

impl<S: Float> PivotedKernel<S> for PivotedMultiscaleDiffusionKernel<S> {
    fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S {
        self.vectors[pivot_idx][j]
    }
}
