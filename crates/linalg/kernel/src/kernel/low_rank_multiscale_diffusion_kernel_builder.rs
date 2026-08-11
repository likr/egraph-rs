use crate::*;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;
use std::hash::Hash;

/// Builder for LowRankMultiscaleDiffusionKernel
#[derive(Clone)]
pub struct LowRankMultiscaleDiffusionKernelBuilder<S> {
    pub alpha: S,
    pub rank: usize,
    pub shift: S,
    pub eigenvalue_max_iterations: usize,
    pub cg_max_iterations: usize,
    pub eigenvalue_tolerance: S,
    pub cg_tolerance: S,
}

impl<S> LowRankMultiscaleDiffusionKernelBuilder<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            alpha: S::from_f64(0.85).unwrap(),
            rank: 10,
            shift: S::from_f64(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f64(1e-2).unwrap(),
            cg_tolerance: S::from_f64(1e-4).unwrap(),
        }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    pub fn shift(mut self, shift: S) -> Self {
        self.shift = shift;
        self
    }

    pub fn eigenvalue_max_iterations(mut self, max_iterations: usize) -> Self {
        self.eigenvalue_max_iterations = max_iterations;
        self
    }

    pub fn cg_max_iterations(mut self, cg_max_iterations: usize) -> Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    pub fn eigenvalue_tolerance(mut self, tolerance: S) -> Self {
        self.eigenvalue_tolerance = tolerance;
        self
    }

    pub fn cg_tolerance(mut self, cg_tolerance: S) -> Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    pub fn build_laplacian<G, F, R>(
        self,
        graph: G,
        length: F,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = crate::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let result = rdmds.eigendecomposition(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let eigenvectors = result.eigenvectors;

        Ok(LowRankMultiscaleDiffusionKernel::new(
            self.alpha,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_normalized_laplacian<G, F, R>(
        self,
        graph: G,
        mut length: F,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = crate::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let mut degrees = Array1::<S>::zeros(n);
        let mut two_m = S::zero();
        for node in graph.node_identifiers() {
            for edge in graph.edges(node) {
                let i = graph.to_index(edge.source());
                let j = graph.to_index(edge.target());
                let w = (&mut length)(edge);
                if i != j {
                    degrees[i] = degrees[i] + w;
                    two_m = two_m + w;
                }
            }
        }

        let result = rdmds.eigendecomposition_symmetric_normalized(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let mut eigenvectors = result.eigenvectors;

        if two_m > S::zero() {
            for i in 0..n {
                let stat = (degrees[i] / two_m).sqrt();
                if stat > S::zero() {
                    let mut row = eigenvectors.row_mut(i);
                    for j in 0..rank {
                        row[j] = row[j] / stat;
                    }
                }
            }
        } else if n > 0 {
            let inv_sqrt_n = S::one() / S::from_usize(n).unwrap().sqrt();
            for i in 0..n {
                let mut row = eigenvectors.row_mut(i);
                for j in 0..rank {
                    row[j] = row[j] / inv_sqrt_n;
                }
            }
        }

        Ok(LowRankMultiscaleDiffusionKernel::new(
            self.alpha,
            eigenvalues,
            eigenvectors,
        ))
    }
}
