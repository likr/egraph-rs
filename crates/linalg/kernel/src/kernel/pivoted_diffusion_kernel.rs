use crate::*;
use ndarray::{Array1, ScalarOperand};
use num_traits::Float;

/// Pivoted diffusion kernel computing distance from specific sources.
#[derive(Debug, Clone)]
pub struct PivotedDiffusionKernel<S> {
    pub pivots: Vec<usize>,
    pub vectors: Vec<Array1<S>>,
    pub diagonal: Option<Array1<S>>,
}

impl<S: Float + ScalarOperand> PivotedKernel<S> for PivotedDiffusionKernel<S> {
    fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S {
        self.vectors[pivot_idx][j]
    }

    fn diagonal(&self, j: usize) -> Option<S> {
        self.diagonal.as_ref().and_then(|d| d.get(j).copied())
    }

    fn node_count(&self) -> Option<usize> {
        self.vectors
            .first()
            .map(|v| v.len())
            .or_else(|| self.diagonal.as_ref().map(|d| d.len()))
    }
}
