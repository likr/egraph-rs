//! Distance matrices derived from kernels.

use crate::traits::PivotedKernel;
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::{Distance, Kernel};
use std::collections::HashMap;
use std::hash::Hash;

/// Builder for NegLogSimDistance
pub struct NegLogSimDistanceBuilder<K, S> {
    kernel: K,
    alpha: S,
    beta: S,
}

impl<K, S: Float> NegLogSimDistanceBuilder<K, S> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            alpha: S::one(),
            beta: S::zero(),
        }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: S) -> Self {
        self.beta = beta;
        self
    }

    pub fn build<G, N>(self, graph: G) -> Result<NegLogSimDistance<N, S, K>, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        Ok(NegLogSimDistance {
            kernel: self.kernel,
            alpha: self.alpha,
            beta: self.beta,
            node_indices,
        })
    }
}

/// A distance matrix computing `alpha * (log(K_ii + beta) - 2 * log(K_ij + beta) + log(K_jj + beta))`
#[derive(Debug, Clone)]
pub struct NegLogSimDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    node_indices: HashMap<N, usize>,
}

impl<N, S, K> Distance<N, S> for NegLogSimDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float + num_traits::FromPrimitive,
    K: Kernel<S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ii = self.kernel.get(i, i);
        let k_jj = self.kernel.get(j, j);
        let k_ij = self.kernel.get(i, j);

        let log_ii = (k_ii + self.beta).ln();
        let log_jj = (k_jj + self.beta).ln();
        let log_ij = (k_ij + self.beta).ln();

        let two = S::from_f64(2.0).unwrap();
        self.alpha * (log_ii - two * log_ij + log_jj)
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.kernel.n();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}

/// Builder for NegLogDistance
pub struct NegLogDistanceBuilder<K, S> {
    kernel: K,
    alpha: S,
    beta: S,
}

impl<K, S: Float> NegLogDistanceBuilder<K, S> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            alpha: S::one(),
            beta: S::zero(),
        }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: S) -> Self {
        self.beta = beta;
        self
    }

    pub fn build<G, N>(self, graph: G) -> Result<NegLogDistance<N, S, K>, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        Ok(NegLogDistance {
            kernel: self.kernel,
            alpha: self.alpha,
            beta: self.beta,
            node_indices,
        })
    }
}

/// A distance matrix for full kernels computing `-alpha * log(K_ij + beta)`
#[derive(Debug, Clone)]
pub struct NegLogDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    node_indices: HashMap<N, usize>,
}

impl<N, S, K> Distance<N, S> for NegLogDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float,
    K: Kernel<S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let k_ij = self.kernel.get(i, j);
        -self.alpha * (k_ij + self.beta).ln()
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.kernel.n();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}

/// Builder for PivotedNegLogDistance
pub struct PivotedNegLogDistanceBuilder<K, S> {
    kernel: K,
    alpha: S,
    beta: S,
}

impl<K, S: Float> PivotedNegLogDistanceBuilder<K, S> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            alpha: S::one(),
            beta: S::zero(),
        }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: S) -> Self {
        self.beta = beta;
        self
    }

    pub fn build<G, N>(self, graph: G) -> Result<PivotedNegLogDistance<N, S, K>, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        Ok(PivotedNegLogDistance {
            kernel: self.kernel,
            alpha: self.alpha,
            beta: self.beta,
            node_indices,
        })
    }
}

/// A distance matrix for pivoted kernels computing `-alpha * log(K_pj + beta)`
#[derive(Debug, Clone)]
pub struct PivotedNegLogDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    node_indices: HashMap<N, usize>,
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
        -self.alpha * (k_pj + self.beta).ln()
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
