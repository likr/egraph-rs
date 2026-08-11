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

/// A distance matrix wrapping a kernel, representing distance in the kernel space.
/// d_K(i, j) = sqrt(K(i, i) + K(j, j) - 2 * K(i, j))
#[derive(Debug, Clone, Copy)]
pub struct KernelDistance<K, S> {
    pub kernel: K,
    pub min_dist: S,
}

impl<K, S> KernelDistance<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            min_dist: S::zero(),
        }
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }
}

impl<N, S, K> Distance<N, S> for KernelDistance<K, S>
where
    N: Eq + Hash + Copy,
    K: Kernel<N, S>,
    S: DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ii = self.kernel.get_by_index(i, i);
        let k_jj = self.kernel.get_by_index(j, j);
        let k_ij = self.kernel.get_by_index(i, j);
        let diff = k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij;
        diff.max(S::zero()).sqrt().max(self.min_dist)
    }

    fn shape(&self) -> (usize, usize) {
        self.kernel.shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.kernel.row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.kernel.col_index(u)
    }
}
