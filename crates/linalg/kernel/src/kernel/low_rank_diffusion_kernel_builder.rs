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

/// Builder for LowRankDiffusionKernel
#[derive(Clone)]
pub struct LowRankDiffusionKernelBuilder<S> {
    pub t: S,
    pub rank: usize,
    pub shift: S,
    pub eigenvalue_max_iterations: usize,
    pub cg_max_iterations: usize,
    pub eigenvalue_tolerance: S,
    pub cg_tolerance: S,
}

impl<S> LowRankDiffusionKernelBuilder<S>
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
            t: S::one(),
            rank: 10,
            shift: S::from_f64(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f64(1e-2).unwrap(),
            cg_tolerance: S::from_f64(1e-4).unwrap(),
        }
    }

    pub fn t(mut self, t: S) -> Self {
        self.t = t;
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
    ) -> Result<LowRankDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        if !self.t.is_finite() || self.t < S::zero() {
            return Err("t must be finite and >= 0".to_string());
        }
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

        Ok(LowRankDiffusionKernel::new(
            self.t,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_normalized_laplacian<G, F, R>(
        self,
        graph: G,
        mut length: F,
        rng: &mut R,
    ) -> Result<LowRankDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        if !self.t.is_finite() || self.t < S::zero() {
            return Err("t must be finite and >= 0".to_string());
        }
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

        Ok(LowRankDiffusionKernel::new(
            self.t,
            eigenvalues,
            eigenvectors,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_low_rank_diffusion_kernel_builder_rejects_invalid_t() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n0, ());

        let mut rng = StdRng::seed_from_u64(7);
        for bad_t in [-1.0, f64::NAN, f64::INFINITY] {
            let err_lap = LowRankDiffusionKernelBuilder::new()
                .t(bad_t)
                .rank(2)
                .build_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_lap.contains("t"));

            let err_norm = LowRankDiffusionKernelBuilder::new()
                .t(bad_t)
                .rank(2)
                .build_normalized_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_norm.contains("t"));
        }
    }
}
