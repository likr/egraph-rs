use crate::*;
use ndarray::Array1;
use ndarray::ScalarOperand;
use num_traits::Float;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;

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

    fn validate(&self) -> Result<(), String> {
        // (I - alpha P)^{-1} diverges at alpha = 1.0 and requires 0 <= alpha < 1 for a valid damping kernel.
        if !self.alpha.is_finite() || self.alpha < S::zero() || self.alpha >= S::one() {
            return Err("alpha must be finite and in [0, 1)".to_string());
        }
        if !self.shift.is_finite() || self.shift <= S::zero() {
            return Err("shift must be finite and > 0".to_string());
        }
        if !self.eigenvalue_tolerance.is_finite() || self.eigenvalue_tolerance <= S::zero() {
            return Err("eigenvalue_tolerance must be finite and > 0".to_string());
        }
        if !self.cg_tolerance.is_finite() || self.cg_tolerance <= S::zero() {
            return Err("cg_tolerance must be finite and > 0".to_string());
        }
        Ok(())
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
        self.validate()?;
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
        self.validate()?;
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
                let w = length(edge);
                if i != j {
                    degrees[i] += w;
                    two_m += w;
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
                        row[j] /= stat;
                    }
                }
            }
        } else if n > 0 {
            let inv_sqrt_n = S::one() / S::from_usize(n).unwrap().sqrt();
            for i in 0..n {
                let mut row = eigenvectors.row_mut(i);
                for j in 0..rank {
                    row[j] /= inv_sqrt_n;
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

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_low_rank_multiscale_builder_rejects_invalid_alpha() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n0, ());

        let mut rng = StdRng::seed_from_u64(7);
        for bad_alpha in [1.0, 1.5, -0.1, f64::NAN, f64::INFINITY] {
            let err_lap = LowRankMultiscaleDiffusionKernelBuilder::new()
                .alpha(bad_alpha)
                .rank(2)
                .build_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_lap.contains("alpha"));

            let err_norm = LowRankMultiscaleDiffusionKernelBuilder::new()
                .alpha(bad_alpha)
                .rank(2)
                .build_normalized_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_norm.contains("alpha"));
        }

        let ok = LowRankMultiscaleDiffusionKernelBuilder::new()
            .alpha(0.0)
            .rank(2)
            .build_normalized_laplacian(&g, |_| 1.0, &mut rng)
            .unwrap();
        assert!(ok.get(0, 1).unwrap().is_finite());
    }

    #[test]
    fn test_low_rank_multiscale_builder_rejects_invalid_shift_and_tolerances() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n0, ());

        let mut rng = StdRng::seed_from_u64(7);
        for bad_val in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            let err_shift = LowRankMultiscaleDiffusionKernelBuilder::new()
                .shift(bad_val)
                .rank(2)
                .build_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_shift.contains("shift"));

            let err_shift_norm = LowRankMultiscaleDiffusionKernelBuilder::new()
                .shift(bad_val)
                .rank(2)
                .build_normalized_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_shift_norm.contains("shift"));

            let err_eig = LowRankMultiscaleDiffusionKernelBuilder::new()
                .eigenvalue_tolerance(bad_val)
                .rank(2)
                .build_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_eig.contains("eigenvalue_tolerance"));

            let err_cg = LowRankMultiscaleDiffusionKernelBuilder::new()
                .cg_tolerance(bad_val)
                .rank(2)
                .build_normalized_laplacian(&g, |_| 1.0, &mut rng)
                .unwrap_err();
            assert!(err_cg.contains("cg_tolerance"));
        }
    }
}
