//! Distance matrices derived from kernels.

use crate::traits::PivotedKernel;
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::{Distance, Kernel};
use std::collections::HashMap;
use std::hash::Hash;

/// Builder for NegLogSimDistance
#[derive(Clone)]
pub struct NegLogSimDistanceBuilder<S> {
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
}

impl<S: Float + num_traits::FromPrimitive> NegLogSimDistanceBuilder<S> {
    pub fn new() -> Self {
        Self {
            alpha: S::one(),
            beta: S::zero(),
            p: S::from_f64(0.5).unwrap(),
            min_dist: S::zero(),
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

    pub fn p(mut self, p: S) -> Self {
        self.p = p;
        self
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    pub fn build<G, N, K>(self, graph: G, kernel: K) -> Result<NegLogSimDistance<N, S, K>, String>
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
            kernel,
            alpha: self.alpha,
            beta: self.beta,
            p: self.p,
            min_dist: self.min_dist,
            node_indices,
        })
    }
}

impl<S: Float + num_traits::FromPrimitive> Default for NegLogSimDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A distance matrix computing `alpha * (log(K_ii + beta) - 2 * log(K_ij + beta) + log(K_jj + beta))^p`
#[derive(Debug, Clone)]
pub struct NegLogSimDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
    node_indices: HashMap<N, usize>,
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

        if k_ii + self.beta <= S::zero() || k_jj + self.beta <= S::zero() || k_ij + self.beta <= S::zero() {
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

/// Builder for NegLogDistance
#[derive(Clone)]
pub struct NegLogDistanceBuilder<S> {
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
}

impl<S: Float + num_traits::FromPrimitive> NegLogDistanceBuilder<S> {
    pub fn new() -> Self {
        Self {
            alpha: S::one(),
            beta: S::zero(),
            p: S::from_f64(0.5).unwrap(),
            min_dist: S::zero(),
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

    pub fn p(mut self, p: S) -> Self {
        self.p = p;
        self
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    pub fn build<G, N, K>(self, graph: G, kernel: K) -> Result<NegLogDistance<N, S, K>, String>
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
            kernel,
            alpha: self.alpha,
            beta: self.beta,
            p: self.p,
            min_dist: self.min_dist,
            node_indices,
        })
    }
}

impl<S: Float + num_traits::FromPrimitive> Default for NegLogDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A distance matrix for full kernels computing `(-alpha * log(K_ij + beta))^p`
#[derive(Debug, Clone)]
pub struct NegLogDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
    node_indices: HashMap<N, usize>,
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

/// Builder for PivotedNegLogDistance
#[derive(Clone)]
pub struct PivotedNegLogDistanceBuilder<S> {
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
}

impl<S: Float + num_traits::FromPrimitive> PivotedNegLogDistanceBuilder<S> {
    pub fn new() -> Self {
        Self {
            alpha: S::one(),
            beta: S::zero(),
            p: S::from_f64(0.5).unwrap(),
            min_dist: S::zero(),
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

    pub fn p(mut self, p: S) -> Self {
        self.p = p;
        self
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    pub fn build<G, N, K>(
        self,
        graph: G,
        kernel: K,
    ) -> Result<PivotedNegLogDistance<N, S, K>, String>
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
            kernel,
            alpha: self.alpha,
            beta: self.beta,
            p: self.p,
            min_dist: self.min_dist,
            node_indices,
        })
    }
}

impl<S: Float + num_traits::FromPrimitive> Default for PivotedNegLogDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A distance matrix for pivoted kernels computing `(-alpha * log(K_pj + beta))^p`
#[derive(Debug, Clone)]
pub struct PivotedNegLogDistance<N, S, K> {
    kernel: K,
    alpha: S,
    beta: S,
    p: S,
    min_dist: S,
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
