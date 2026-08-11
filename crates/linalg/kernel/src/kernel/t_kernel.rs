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

/// T-Kernel applied to a base Kernel: K(x, y) = 1 / (1 + d_K(x, y)^2 / alpha)^((alpha + 1) / 2)
#[derive(Debug, Clone, Copy)]
pub struct TKernel<K, S> {
    pub distance: KernelDistance<K, S>,
    pub alpha: S,
}

impl<K, S> TKernel<K, S>
where
    S: DrawingValue,
{
    pub fn new(kernel: K, alpha: S) -> Self {
        Self {
            distance: KernelDistance::new(kernel),
            alpha,
        }
    }
}

impl<N, S, K> Kernel<N, S> for TKernel<K, S>
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
        let d = self.distance.get_by_index(i, j);
        let one = S::one();
        let val = one + d * d / self.alpha;
        let power = (self.alpha + one) / S::from_f32(2.0).unwrap();
        one / val.powf(power)
    }

    fn shape(&self) -> (usize, usize) {
        self.distance.shape()
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.distance.row_index(u)
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.distance.col_index(u)
    }
}
