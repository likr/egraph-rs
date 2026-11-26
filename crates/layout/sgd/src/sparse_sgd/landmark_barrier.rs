use super::two_hop::{
    compute_adaptive_perplexity_from_pairs, compute_median_f64, extract_canonical_edges_and_adj,
    extract_hub_safe_2hop, round_half_to_even, sample_non2hop_pairs,
};
use super::{BarrierBuildStats, BarrierSymmetry, SparseSgd};
use crate::Sgd;
use ndarray::ArrayView2;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingIndex;
use std::collections::{HashMap, HashSet};

impl SparseSgd {
    /// Builds a 3-tier Sparse Barrier SGD schedule directly from precomputed `(N, k_lm)`
    /// multi-scale Varadhan landmark tables `(d_ms, d_coarse)` in `O(|E| + N k_lm)` time.
    ///
    /// Executes canonical 1-hop calibration, hub-safe `O(|E| * max_2hop_per_node)` 2-hop
    /// registration, adaptive or explicit non-2-hop repulsion sampling, relative-margin
    /// `R_2` barrier enforcement, and native one-sided hinge repulsion on Tier 3 pairs.
    pub fn build_barrier_from_landmark_tables<G>(
        &self,
        graph: G,
        d_ms: ArrayView2<'_, f64>,
        d_coarse: Option<ArrayView2<'_, f64>>,
        pivots: &[usize],
        neg_per_node: Option<usize>,
    ) -> Result<(Sgd<f64>, BarrierBuildStats), String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
    {
        if !self.delta_rel.is_finite() || self.delta_rel < 0.0 {
            return Err(format!(
                "delta_rel must be finite and >= 0, got {}",
                self.delta_rel
            ));
        }
        if !self.omega_1.is_finite() || self.omega_1 < 0.0 {
            return Err(format!(
                "omega_1 must be finite and >= 0, got {}",
                self.omega_1
            ));
        }
        if !self.omega_2.is_finite() || self.omega_2 < 0.0 {
            return Err(format!(
                "omega_2 must be finite and >= 0, got {}",
                self.omega_2
            ));
        }
        if !self.gamma_non2.is_finite() || self.gamma_non2 < 0.0 {
            return Err(format!(
                "gamma_non2 must be finite and >= 0, got {}",
                self.gamma_non2
            ));
        }
        if !self.min_ratio.is_finite() || self.min_ratio < 0.0 {
            return Err(format!(
                "min_ratio must be finite and >= 0, got {}",
                self.min_ratio
            ));
        }
        if !self.eps_dist.is_finite() || self.eps_dist <= 0.0 {
            return Err(format!(
                "eps_dist must be finite and > 0, got {}",
                self.eps_dist
            ));
        }

        let n = graph.node_count();
        if d_ms.nrows() != n {
            return Err(format!(
                "d_ms row count ({}) must match node count ({n})",
                d_ms.nrows()
            ));
        }
        if n > 0 && d_ms.ncols() == 0 {
            return Err("d_ms must have at least 1 landmark column when n > 0".to_string());
        }
        if d_ms.iter().any(|&v| !v.is_finite()) {
            return Err("d_ms must contain only finite float64 values".to_string());
        }
        if let Some(dc) = d_coarse {
            if dc.nrows() != n {
                return Err(format!(
                    "d_coarse row count ({}) must match node count ({n})",
                    dc.nrows()
                ));
            }
            if n > 0 && dc.ncols() == 0 {
                return Err("d_coarse must have at least 1 landmark column when n > 0".to_string());
            }
            if dc.iter().any(|&v| !v.is_finite()) {
                return Err("d_coarse must contain only finite float64 values".to_string());
            }
        }
        for &p in pivots {
            if p >= n {
                return Err(format!(
                    "pivot index {p} is out of bounds for graph with {n} nodes"
                ));
            }
        }
        let mut seen_pivots = HashSet::with_capacity(pivots.len());
        let unique_pivots: Vec<usize> = pivots
            .iter()
            .copied()
            .filter(|&p| seen_pivots.insert(p))
            .collect();
        let pivots = unique_pivots.as_slice();

        if n <= 1 {
            let (perplexity_val, per_node_neg_used, perplexity_source) = match neg_per_node {
                Some(k) => (k as f64, k, "explicit"),
                None => (1.0, 1, "adaptive_2hop"),
            };
            return Ok((
                Sgd::new(vec![]),
                BarrierBuildStats {
                    n_pairs_1hop: 0,
                    n_pairs_2hop: 0,
                    n_pairs_non2: 0,
                    n_pairs_hinge: 0,
                    n_pairs_two_sided: 0,
                    perplexity_val,
                    per_node_neg_used,
                    perplexity_source,
                    d_floor: self.eps_dist,
                },
            ));
        }

        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let (edges_1hop, adj) = extract_canonical_edges_and_adj(graph, &indices, &mut |_| 1.0_f64);
        let pairs_1hop: Vec<(usize, usize)> = edges_1hop.iter().map(|&(u, v, _)| (u, v)).collect();
        let pairs_2hop = if self.max_2hop_per_node > 0 && n > 2 {
            extract_hub_safe_2hop(&adj, self.max_2hop_per_node, self.seed)
        } else {
            Vec::new()
        };

        let (perplexity_val, per_node_neg_used, perplexity_source) = match neg_per_node {
            Some(k) => (k as f64, k, "explicit"),
            None => {
                let p_val = compute_adaptive_perplexity_from_pairs(n, &pairs_1hop, &pairs_2hop);
                let neg = if n > 3 {
                    round_half_to_even(p_val).max(1)
                } else {
                    1
                };
                (p_val, neg, "adaptive_2hop")
            }
        };

        let pairs_non2 = if n > 2 && per_node_neg_used > 0 {
            sample_non2hop_pairs(
                &adj,
                pivots,
                &pairs_1hop,
                &pairs_2hop,
                per_node_neg_used,
                self.seed,
            )
        } else {
            Vec::new()
        };

        let d1_raw: Vec<f64> = pairs_1hop
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();
        let d2_raw: Vec<f64> = pairs_2hop
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();
        let d3_raw: Vec<f64> = pairs_non2
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();

        let mut pos_ref: Vec<f64> = d1_raw.iter().copied().filter(|&d| d > 0.0).collect();
        if pos_ref.is_empty() {
            pos_ref = d2_raw.iter().copied().filter(|&d| d > 0.0).collect();
        }
        if pos_ref.is_empty() {
            pos_ref = d3_raw.iter().copied().filter(|&d| d > 0.0).collect();
        }
        let ref_med = compute_median_f64(&mut pos_ref).unwrap_or(1.0);
        let d_floor = self.eps_dist.max(self.min_ratio * ref_med);

        let mut r2 = vec![0.0_f64; n];
        let mut node_pairs = Vec::with_capacity(pairs_1hop.len() + pairs_2hop.len());
        let mut hinge_pairs = Vec::new();

        for (idx, &(u, v)) in pairs_1hop.iter().enumerate() {
            let d1 = d1_raw[idx].max(d_floor);
            if d1 > r2[u] {
                r2[u] = d1;
            }
            if d1 > r2[v] {
                r2[v] = d1;
            }
            if self.omega_1 > 0.0 {
                let w1 = self.omega_1 / (d1 * d1);
                node_pairs.push((u, v, d1, d1, w1, w1));
            }
        }

        for (idx, &(u, v)) in pairs_2hop.iter().enumerate() {
            let d2 = d2_raw[idx].max(d_floor);
            if d2 > r2[u] {
                r2[u] = d2;
            }
            if d2 > r2[v] {
                r2[v] = d2;
            }
            if self.omega_2 > 0.0 {
                let w2 = self.omega_2 / (d2 * d2);
                node_pairs.push((u, v, d2, d2, w2, w2));
            }
        }

        let barrier_mult = 1.0 + self.delta_rel;
        for (idx, &(u, v)) in pairs_non2.iter().enumerate() {
            let d3 = d3_raw[idx].max(d_floor);
            let r_pair = match self.barrier_sym {
                BarrierSymmetry::SymMax => r2[u].max(r2[v]),
                BarrierSymmetry::SymAvg => 0.5 * (r2[u] + r2[v]),
            };
            let d3_bar = d3.max(barrier_mult * r_pair).max(self.eps_dist);
            if self.gamma_non2 > 0.0 {
                let w3 = self.gamma_non2 / (d3_bar * d3_bar);
                let pair = (u, v, d3_bar, d3_bar, w3, w3);
                if self.hinge_repulsion {
                    hinge_pairs.push(pair);
                } else {
                    node_pairs.push(pair);
                }
            }
        }

        let n_pairs_two_sided = node_pairs.len();
        let n_pairs_hinge = hinge_pairs.len();
        let sgd = Sgd::new_with_hinge_pairs(node_pairs, hinge_pairs);
        let stats = BarrierBuildStats {
            n_pairs_1hop: pairs_1hop.len(),
            n_pairs_2hop: pairs_2hop.len(),
            n_pairs_non2: pairs_non2.len(),
            n_pairs_hinge,
            n_pairs_two_sided,
            perplexity_val,
            per_node_neg_used,
            perplexity_source,
            d_floor,
        };
        Ok((sgd, stats))
    }
}

#[inline]
fn alt_max_pair_distance(
    d_ms: ArrayView2<'_, f64>,
    d_coarse: Option<ArrayView2<'_, f64>>,
    u: usize,
    v: usize,
) -> f64 {
    let mut d_val = 0.0_f64;
    let row_u = d_ms.row(u);
    let row_v = d_ms.row(v);
    for (&au, &av) in row_u.iter().zip(row_v.iter()) {
        let diff = (au - av).abs();
        if diff > d_val {
            d_val = diff;
        }
    }
    if let Some(dc) = d_coarse {
        let cu = dc.row(u);
        let cv = dc.row(v);
        for (&bu, &bv) in cu.iter().zip(cv.iter()) {
            let diff = (bu - bv).abs();
            if diff > d_val {
                d_val = diff;
            }
        }
    }
    d_val
}
