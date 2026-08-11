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

/// Exact multiscale diffusion kernel (I - alpha P)^{-1}.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionKernel<S> {
    pub n: usize,
    pub matrix: Array2<S>,
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
