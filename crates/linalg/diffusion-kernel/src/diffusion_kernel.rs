//! Diffusion kernel matrix computation via high-order Chebyshev polynomial expansion.

use crate::chebyshev::{chebyshev_approximation, chebyshev_approximation_vec};
use crate::power_method::estimate_lambda_max;
use crate::traits::PivotedKernel;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use petgraph_distance::{Kernel, SparseSymmetricMatrix};
use rand::Rng;

/// Builder for DiffusionKernel
pub struct DiffusionKernelBuilder<'a, S> {
    laplacian: &'a SparseSymmetricMatrix<S>,
    t: S,
    degree: usize,
    lambda_max: Option<S>,
}

impl<'a, S> DiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(laplacian: &'a SparseSymmetricMatrix<S>, t: S, degree: usize) -> Self {
        Self {
            laplacian,
            t,
            degree,
            lambda_max: None,
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<DiffusionKernel<S>, String> {
        let lambda_max = match self.lambda_max {
            Some(l) => l,
            None => estimate_lambda_max(self.laplacian, rng, 100, S::from_f64(1e-4).unwrap()),
        };

        let n = self.laplacian.dim();
        let mut eye = Array2::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = S::one();
        }

        let matrix = chebyshev_approximation(self.laplacian, self.t, self.degree, lambda_max, &eye);

        Ok(DiffusionKernel { n, matrix })
    }
}

/// Exact heat kernel exp(-tL) computed via Chebyshev polynomial expansion for all matrix columns.
#[derive(Debug, Clone)]
pub struct DiffusionKernel<S> {
    n: usize,
    matrix: Array2<S>,
}

impl<S: Float + ScalarOperand> Kernel<usize, S> for DiffusionKernel<S> {
    fn get(&self, u: usize, v: usize) -> Option<S> {
        if u < self.n && v < self.n {
            Some(self.matrix[[u, v]])
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

    fn col_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.n)
    }
}

/// Builder for PivotedDiffusionKernel
pub struct PivotedDiffusionKernelBuilder<'a, S> {
    laplacian: &'a SparseSymmetricMatrix<S>,
    t: S,
    degree: usize,
    pivots: Vec<usize>,
    lambda_max: Option<S>,
}

impl<'a, S> PivotedDiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        laplacian: &'a SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        pivots: Vec<usize>,
    ) -> Self {
        Self {
            laplacian,
            t,
            degree,
            pivots,
            lambda_max: None,
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<PivotedDiffusionKernel<S>, String> {
        let lambda_max = match self.lambda_max {
            Some(l) => l,
            None => estimate_lambda_max(self.laplacian, rng, 100, S::from_f64(1e-4).unwrap()),
        };

        let n = self.laplacian.dim();
        let mut pivot_vectors = Vec::with_capacity(self.pivots.len());

        for &p in &self.pivots {
            let mut e = Array1::zeros(n);
            if p < n {
                e[p] = S::one();
            } else {
                return Err("Pivot index out of bounds".to_string());
            }
            let heat_vec =
                chebyshev_approximation_vec(self.laplacian, self.t, self.degree, lambda_max, &e);
            pivot_vectors.push(heat_vec);
        }

        Ok(PivotedDiffusionKernel {
            pivots: self.pivots,
            vectors: pivot_vectors,
        })
    }
}

/// Pivoted diffusion kernel computing distance from specific sources.
#[derive(Debug, Clone)]
pub struct PivotedDiffusionKernel<S> {
    pivots: Vec<usize>,
    vectors: Vec<Array1<S>>,
}

impl<S: Float + ScalarOperand> PivotedKernel<S> for PivotedDiffusionKernel<S> {
    fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S {
        self.vectors[pivot_idx][j]
    }
}
