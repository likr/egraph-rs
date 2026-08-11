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

/// A distance matrix for full kernels computing `(-alpha * log(K_ij + beta))^p`
#[derive(Debug, Clone)]
pub struct NegLogDistance<N, S, K> {
    pub kernel: K,
    pub alpha: S,
    pub beta: S,
    pub p: S,
    pub min_dist: S,
    pub node_indices: HashMap<N, usize>,
}

impl<N, S, K> Distance<N, S> for NegLogDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float,
    K: Kernel<usize, S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ij = self.kernel.get_by_index(i, j);
        if k_ij + self.beta <= S::zero() {
            return S::infinity();
        }
        let val = (-self.alpha * (k_ij + self.beta).ln()).max(S::zero());
        val.powf(self.p).max(self.min_dist)
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.kernel.shape().0;
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}
