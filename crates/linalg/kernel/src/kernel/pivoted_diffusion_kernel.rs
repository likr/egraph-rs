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

/// Pivoted diffusion kernel computing distance from specific sources.
#[derive(Debug, Clone)]
pub struct PivotedDiffusionKernel<S> {
    pub pivots: Vec<usize>,
    pub vectors: Vec<Array1<S>>,
}

impl<S: Float + ScalarOperand> PivotedKernel<S> for PivotedDiffusionKernel<S> {
    fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S {
        self.vectors[pivot_idx][j]
    }
}
