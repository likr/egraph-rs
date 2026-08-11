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
use std::hash::Hash;

/// Low-rank spectral approximation of the heat kernel exp(-tL).
#[derive(Debug, Clone)]
pub struct LowRankDiffusionKernel<S> {
    pub eigenvectors: Array2<S>,
    pub coefficients: Array1<S>,
}

impl<S: Float> LowRankDiffusionKernel<S> {
    pub fn new(t: S, eigenvalues: Array1<S>, eigenvectors: Array2<S>) -> Self {
        let rank_plus_1 = eigenvalues.len();
        let mut coefficients = Array1::zeros(rank_plus_1);
        for k in 0..rank_plus_1 {
            coefficients[k] = (-t * eigenvalues[k]).exp();
        }
        Self {
            eigenvectors,
            coefficients,
        }
    }
}

impl<S: Float + ScalarOperand + num_traits::FromPrimitive> Kernel<usize, S>
    for LowRankDiffusionKernel<S>
{
    fn get(&self, u: usize, v: usize) -> Option<S> {
        let n = self.eigenvectors.nrows();
        if u < n && v < n {
            Some(self.get_by_index(u, v))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let n = self.eigenvectors.nrows();
        assert!(i < n && j < n, "Index out of bounds");
        let mut sum = S::one() / S::from_usize(n).unwrap();
        for k in 0..self.coefficients.len() {
            sum =
                sum + self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.eigenvectors.nrows();
        (n, n)
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.eigenvectors.nrows())
    }

    fn col_index(&self, v: usize) -> Option<usize> {
        Some(v).filter(|&j| j < self.eigenvectors.nrows())
    }
}
