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

/// Builder for PivotedDiffusionKernel
pub struct PivotedDiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub t: S,
    pub degree: usize,
    pub pivots: Vec<usize>,
    pub lambda_max: Option<S>,
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
