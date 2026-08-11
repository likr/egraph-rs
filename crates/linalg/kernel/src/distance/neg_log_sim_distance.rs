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

/// A distance matrix computing `alpha * (log(K_ii + beta) - 2 * log(K_ij + beta) + log(K_jj + beta))^p`
#[derive(Debug, Clone)]
pub struct NegLogSimDistance<N, S, K> {
    pub kernel: K,
    pub alpha: S,
    pub beta: S,
    pub p: S,
    pub min_dist: S,
    pub node_indices: HashMap<N, usize>,
}

impl<N, S, K> Distance<N, S> for NegLogSimDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float + num_traits::FromPrimitive,
    K: Kernel<usize, S>,
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

        if k_ii + self.beta <= S::zero()
            || k_jj + self.beta <= S::zero()
            || k_ij + self.beta <= S::zero()
        {
            return S::infinity();
        }
        let log_ii = (k_ii + self.beta).ln();
        let log_jj = (k_jj + self.beta).ln();
        let log_ij = (k_ij + self.beta).ln();

        let two = S::from_f64(2.0).unwrap();
        let val = (self.alpha * (log_ii - two * log_ij + log_jj)).max(S::zero());
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
