use crate::*;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;
use std::hash::Hash;

/// Builder for DiffusionKernel
pub struct DiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub t: S,
    pub degree: usize,
    pub lambda_max: Option<S>,
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
