use crate::*;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use std::collections::HashMap;
use std::hash::Hash;

/// A distance matrix for pivoted kernels computing `(-alpha * log(K_pj + beta))^p`
#[derive(Debug, Clone)]
pub struct PivotedNegLogDistance<N, S, K> {
    pub kernel: K,
    pub alpha: S,
    pub beta: S,
    pub p: S,
    pub min_dist: S,
    pub node_indices: HashMap<N, usize>,
}

impl<N, S, K> PivotedNegLogDistance<N, S, K>
where
    K: PivotedKernel<S>,
{
    pub fn pivots(&self) -> &[usize] {
        self.kernel.pivots()
    }
}

impl<N, S, K> Distance<N, S> for PivotedNegLogDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float,
    K: PivotedKernel<S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let pivots = self.kernel.pivots();

        let pivot_idx = if let Some(idx) = pivots.iter().position(|&p| p == i) {
            idx
        } else if let Some(idx) = pivots.iter().position(|&p| p == j) {
            idx
        } else {
            return S::infinity();
        };

        let target_j = if pivots[pivot_idx] == i { j } else { i };

        let k_pj = self.kernel.get_from_pivot(pivot_idx, target_j);
        if k_pj + self.beta <= S::zero() {
            return S::infinity();
        }
        let val = (-self.alpha * (k_pj + self.beta).ln()).max(S::zero());
        val.powf(self.p).max(self.min_dist)
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.node_indices.len();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}
