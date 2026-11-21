use crate::*;
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;
use std::hash::Hash;

/// A pivoted distance matrix of shape `(N, h)` computing normalized negative log similarity:
///
/// `s(j, p) = max(K(p, j), eps) / sqrt(max(K_pp, eps) * max(K_jj, eps))`
/// `d(j, p) = scale * max(-ln(s(j, p)), 0)^power`
///
/// Uses `K.get_from_pivot(p_idx, pivot_node)` for `K_pp` and `K.diagonal(j)` for `K_jj`
/// (falling back to `1.0` when `diagonal(j)` returns `None`, and returning `0.0` when `j == pivot_node`).
#[derive(Debug, Clone)]
pub struct PivotedNegLogSimDistance<N, S, K> {
    pub kernel: K,
    pub epsilon: S,
    pub scale: S,
    pub power: S,
    pub node_indices: HashMap<N, usize>,
    pivot_indices: HashMap<usize, usize>,
}

impl<N, S, K> PivotedNegLogSimDistance<N, S, K>
where
    K: PivotedKernel<S>,
{
    /// Returns the pivot node indices stored in the underlying kernel.
    pub fn pivots(&self) -> &[usize] {
        self.kernel.pivots()
    }
}

impl<N, S, K> Distance<N, S> for PivotedNegLogSimDistance<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float,
    K: PivotedKernel<S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        if let (Some(j), Some(p_idx)) = (self.row_index(u), self.col_index(v)) {
            Some(self.get_by_index(j, p_idx))
        } else if let (Some(j), Some(p_idx)) = (self.row_index(v), self.col_index(u)) {
            // Allow symmetric queries such as SparseSgd's get(pivot, node).
            Some(self.get_by_index(j, p_idx))
        } else {
            None
        }
    }

    fn get_by_index(&self, j: usize, p_idx: usize) -> S {
        let pivot_node = self.kernel.pivots()[p_idx];
        if j == pivot_node {
            return S::zero();
        }
        let k_pj = self.kernel.get_from_pivot(p_idx, j);
        let k_pp = self.kernel.get_from_pivot(p_idx, pivot_node);
        let k_jj = if let Some(&j_p_idx) = self.pivot_indices.get(&j) {
            self.kernel.get_from_pivot(j_p_idx, j)
        } else {
            self.kernel.diagonal(j).unwrap_or_else(S::one)
        };

        let denom = (k_pp.max(self.epsilon) * k_jj.max(self.epsilon)).sqrt();
        let sim = k_pj.max(self.epsilon) / denom;
        let neg_log = (-sim.ln()).max(S::zero());
        self.scale * neg_log.powf(self.power)
    }

    fn shape(&self) -> (usize, usize) {
        (self.node_indices.len(), self.kernel.pivots().len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices
            .get(&u)
            .and_then(|&node_idx| self.pivot_indices.get(&node_idx).copied())
    }
}

/// Builder for [`PivotedNegLogSimDistance`].
#[derive(Clone, Debug)]
pub struct PivotedNegLogSimDistanceBuilder<S> {
    pub epsilon: S,
    pub scale: S,
    pub power: S,
}

impl<S: Float + num_traits::FromPrimitive> PivotedNegLogSimDistanceBuilder<S> {
    pub fn new() -> Self {
        Self {
            epsilon: S::from_f64(1e-12).unwrap(),
            scale: S::one(),
            power: S::from_f64(0.5).unwrap(),
        }
    }

    pub fn epsilon(mut self, epsilon: S) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn scale(mut self, scale: S) -> Self {
        self.scale = scale;
        self
    }

    pub fn power(mut self, power: S) -> Self {
        self.power = power;
        self
    }

    pub fn build<G, N, K>(
        self,
        graph: G,
        kernel: K,
    ) -> Result<PivotedNegLogSimDistance<N, S, K>, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
        K: PivotedKernel<S>,
    {
        if !self.epsilon.is_finite() || self.epsilon <= S::zero() {
            return Err("epsilon must be finite and positive".to_string());
        }
        if !self.scale.is_finite() || self.scale < S::zero() {
            return Err("scale must be finite and non-negative".to_string());
        }
        if !self.power.is_finite() || self.power < S::zero() {
            return Err("power must be finite and non-negative".to_string());
        }

        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();
        let n = node_indices.len();
        if let Some(kn) = kernel.node_count() {
            if kn != n {
                return Err(format!(
                    "Kernel node count ({kn}) must match graph node count ({n})"
                ));
            }
        }

        let mut pivot_indices = HashMap::new();
        for (p_idx, &p_node) in kernel.pivots().iter().enumerate() {
            if p_node >= n {
                return Err("Pivot index out of bounds".to_string());
            }
            pivot_indices.entry(p_node).or_insert(p_idx);
        }

        Ok(PivotedNegLogSimDistance {
            kernel,
            epsilon: self.epsilon,
            scale: self.scale,
            power: self.power,
            node_indices,
            pivot_indices,
        })
    }
}

impl<S: Float + num_traits::FromPrimitive> Default for PivotedNegLogSimDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{NodeIndex, UnGraph};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_pivoted_neg_log_sim_matches_full_on_pivot_columns() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| g.add_node(())).collect();
        for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)] {
            g.add_edge(nodes[u], nodes[v], ());
        }

        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0);
        let pivots = vec![0, 3];
        let t = 0.8;

        let mut rng1 = StdRng::seed_from_u64(42);
        let full_kernel = DiffusionKernelBuilder::new(&lap, t, 25)
            .lambda_max(2.0)
            .build(&mut rng1)
            .unwrap();
        // NegLogSimDistance(alpha=0.5, beta=0, p=0.5) computes sqrt(-ln(K_ij / sqrt(K_ii K_jj)))
        let full_dist: NegLogSimDistance<NodeIndex, f64, _> = NegLogSimDistanceBuilder::new()
            .alpha(0.5)
            .beta(0.0)
            .p(0.5)
            .build(&g, full_kernel.clone())
            .unwrap();

        let mut rng2 = StdRng::seed_from_u64(42);
        let pivoted_kernel = PivotedDiffusionKernelBuilder::new(&lap, t, 25, pivots.clone())
            .lambda_max(2.0)
            .num_hutchinson(Some(1024))
            .build(&mut rng2)
            .unwrap();
        let pivoted_dist: PivotedNegLogSimDistance<NodeIndex, f64, _> =
            PivotedNegLogSimDistanceBuilder::new()
                .epsilon(1e-12)
                .scale(1.0)
                .power(0.5)
                .build(&g, pivoted_kernel)
                .unwrap();

        assert_eq!(pivoted_dist.shape(), (6, 2));

        for (p_idx, &p) in pivots.iter().enumerate() {
            assert_eq!(pivoted_dist.get_by_index(p, p_idx), 0.0);
            assert_eq!(pivoted_dist.get(nodes[p], nodes[p]), Some(0.0));
            for j in 0..6 {
                let d_piv = pivoted_dist.get_by_index(j, p_idx);
                let d_full = full_dist.get_by_index(j, p);
                assert!(
                    (d_piv - d_full).abs() < 0.08,
                    "mismatch at j={j}, p={p}: pivoted={d_piv}, full={d_full}"
                );
                // Symmetric get(pivot, node) and get(node, pivot), including pivot-pivot pairs
                assert_eq!(pivoted_dist.get(nodes[j], nodes[p]), Some(d_piv));
                assert!(
                    (pivoted_dist.get(nodes[p], nodes[j]).unwrap() - d_piv).abs() < 1e-12,
                    "asymmetry at pivot pair ({p}, {j})"
                );
            }
        }
        // Non-pivot pair returns None
        assert_eq!(pivoted_dist.get(nodes[1], nodes[2]), None);
    }

    #[test]
    fn test_pivoted_neg_log_sim_fallback_and_negative_clamping() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());

        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0);
        let mut rng = StdRng::seed_from_u64(123);

        // No Hutchinson diagonal -> diagonal(j) returns None -> fallback 1.0
        let pk_no_diag = PivotedDiffusionKernelBuilder::new(&lap, 0.5, 20, vec![0])
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();
        assert!(pk_no_diag.diagonal(1).is_none());

        let dist_no_diag: PivotedNegLogSimDistance<NodeIndex, f64, _> =
            PivotedNegLogSimDistanceBuilder::new()
                .epsilon(1e-6)
                .scale(2.0)
                .power(0.5)
                .build(&g, pk_no_diag.clone())
                .unwrap();
        let k_00 = pk_no_diag.get_from_pivot(0, 0);
        let k_01 = pk_no_diag.get_from_pivot(0, 1);
        let expected_sim = k_01.max(1e-6) / (k_00.max(1e-6) * 1.0).sqrt();
        let expected_d = 2.0 * (-expected_sim.ln()).max(0.0).powf(0.5);
        assert!((dist_no_diag.get_by_index(1, 0) - expected_d).abs() < 1e-12);

        // BandPass DoG can have negative values; verify epsilon clamping yields finite distance
        let bp = PivotedBandPassDiffusionKernelBuilder::dog(&lap, 0.2, 2.0, 0.9, 0.0, 20, vec![0])
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();
        let dist_bp: PivotedNegLogSimDistance<NodeIndex, f64, _> =
            PivotedNegLogSimDistanceBuilder::new()
                .epsilon(1e-6)
                .scale(1.5)
                .power(0.5)
                .build(&g, bp)
                .unwrap();
        for j in 0..3 {
            let d = dist_bp.get_by_index(j, 0);
            assert!(d.is_finite() && d >= 0.0);
        }
    }

    #[test]
    fn test_pivoted_distance_rejects_mismatched_graph_node_count() {
        let mut g6 = UnGraph::<(), ()>::new_undirected();
        let n6: Vec<_> = (0..6).map(|_| g6.add_node(())).collect();
        for i in 0..5 {
            g6.add_edge(n6[i], n6[i + 1], ());
        }
        let mut g10 = UnGraph::<(), ()>::new_undirected();
        let n10: Vec<_> = (0..10).map(|_| g10.add_node(())).collect();
        for i in 0..9 {
            g10.add_edge(n10[i], n10[i + 1], ());
        }
        let lap6 = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g6, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        let pk6 = PivotedDiffusionKernelBuilder::new(&lap6, 0.5, 10, vec![0])
            .build(&mut rng)
            .unwrap();
        assert!(PivotedNegLogSimDistanceBuilder::new()
            .build::<_, petgraph::graph::NodeIndex, _>(&g10, pk6.clone())
            .is_err());
        assert!(PivotedNegLogDistanceBuilder::new()
            .build::<_, petgraph::graph::NodeIndex, _>(&g10, pk6)
            .is_err());
    }
}
