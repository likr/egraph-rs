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

/// Builder for PivotedNegLogDistance
#[derive(Clone)]
pub struct PivotedNegLogDistanceBuilder<S> {
    pub alpha: S,
    pub beta: S,
    pub p: S,
    pub min_dist: S,
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
