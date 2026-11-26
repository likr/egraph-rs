use super::two_hop::{compute_median_f64, extract_canonical_edges_and_adj, extract_hub_safe_2hop};
use super::{BarrierSymmetry, SparseSgd};
use crate::Sgd;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use std::collections::{HashMap, HashSet};

impl SparseSgd {
    pub fn build_with_pivot_and_distance_matrix<G, F, S>(
        &self,
        graph: G,
        mut length: F,
        pivot: &[G::NodeId],
        distance_matrix: &dyn Distance<G::NodeId, S>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        // Deduplicate pivots preserving first-seen order so duplicate pivots never overwrite
        // `pivot_slot` or leak 2-hop pairs into Tier 3 hinge repulsion.
        let mut seen_pivots = HashSet::with_capacity(pivot.len());
        let unique_pivots: Vec<G::NodeId> = pivot
            .iter()
            .copied()
            .filter(|u| seen_pivots.insert(*u))
            .collect();
        let pivot = &unique_pivots;
        let h = pivot.len();
        let is_legacy = !self.calibrate_one_hop
            && self.max_2hop_per_node == 0
            && !self.hinge_repulsion
            && self.delta_rel == 0.0
            && self.min_ratio == 0.0
            && self.omega_1 == 1.0
            && self.gamma_non2 == 1.0;

        if is_legacy {
            let mut node_pairs = vec![];
            let mut edges = HashSet::new();
            for edge in graph.edge_references() {
                let i = indices[&edge.source()];
                let j = indices[&edge.target()];
                let dij = length(edge);
                let wij = S::one() / (dij * dij);
                node_pairs.push((i, j, dij, dij, wij, wij));
                edges.insert((i, j));
                edges.insert((j, i));
            }
            if h == 0 || n <= 1 {
                return Sgd::new(node_pairs);
            }

            // Cache h x N pivot distances once so nearest-pivot assignment and Voronoi counting
            // avoid O(N^2) distance queries.
            let mut piv_dists = vec![S::infinity(); h * n];
            for (k, &u) in pivot.iter().enumerate() {
                for j in 0..n {
                    piv_dists[k * n + j] =
                        distance_matrix.get(u, nodes[j]).unwrap_or(S::infinity());
                }
            }

            let mut r_dists = vec![vec![]; h];
            for j in 0..n {
                let best_k = (0..h)
                    .min_by_key(|&k| OrderedFloat(piv_dists[k * n + j]))
                    .unwrap();
                r_dists[best_k].push(piv_dists[best_k * n + j]);
            }
            for dists_k in &mut r_dists {
                dists_k.sort_by_key(|&d| OrderedFloat(d));
            }

            let two = S::from_usize(2).unwrap();
            for (k, &u) in pivot.iter().enumerate() {
                let p = indices[&u];
                for i in 0..n {
                    if edges.contains(&(p, i)) || p == i {
                        continue;
                    }
                    let dpi = piv_dists[k * n + i];
                    if !dpi.is_finite() || dpi <= S::zero() {
                        continue;
                    }
                    let wpi = S::one() / (dpi * dpi);
                    let count =
                        r_dists[k].partition_point(|&dpj| dpj.is_finite() && two * dpj <= dpi);
                    let spi = S::from_usize(count).unwrap();
                    node_pairs.push((p, i, dpi, dpi, S::zero(), spi * wpi));
                }
            }
            return Sgd::new(node_pairs);
        }

        if n <= 1 {
            return Sgd::new(vec![]);
        }

        // Step A: Canonical undirected 1-hop edges and sorted CSR-like adjacency lists.
        let (edges_1hop, adj) = extract_canonical_edges_and_adj(graph, &indices, &mut length);

        // Step B: Cache h x N pivot distances in O(N h) memory without O(N^2) allocations.
        let mut pivot_slot: Vec<Option<usize>> = vec![None; n];
        let mut pivot_node_indices = Vec::with_capacity(h);
        let mut piv_dists = vec![S::infinity(); h * n];
        for (k, &u) in pivot.iter().enumerate() {
            let p = indices[&u];
            pivot_slot[p] = Some(k);
            pivot_node_indices.push(p);
            for j in 0..n {
                piv_dists[k * n + j] = distance_matrix
                    .get(u, nodes[j])
                    .or_else(|| distance_matrix.get(nodes[j], u))
                    .unwrap_or(S::infinity());
            }
        }

        let resolve_pair_distance = |u: usize, v: usize| -> Option<S> {
            if let Some(k) = pivot_slot[u] {
                let d = piv_dists[k * n + v];
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            if let Some(k) = pivot_slot[v] {
                let d = piv_dists[k * n + u];
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            if let Some(d) = distance_matrix.get(nodes[u], nodes[v]) {
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            // Landmark ALT lower bound across cached pivot rows for non-pivot pairs.
            let mut best = S::zero();
            for k in 0..h {
                let du = piv_dists[k * n + u];
                let dv = piv_dists[k * n + v];
                if du.is_finite() && dv.is_finite() {
                    let diff = (du - dv).abs();
                    if diff > best {
                        best = diff;
                    }
                }
            }
            if best > S::zero() {
                Some(best)
            } else {
                None
            }
        };

        // Step C: 1-hop anchor calibration and relative distance floor.
        let eps_s = S::from_f64(self.eps_dist.max(1e-12)).unwrap();
        let mut d1_raw: Vec<S> = Vec::with_capacity(edges_1hop.len());
        let mut scale_1hop_f64 = 1.0_f64;

        if self.calibrate_one_hop {
            let mut ratios = Vec::with_capacity(edges_1hop.len());
            let mut metrics_only = Vec::with_capacity(edges_1hop.len());
            let mut d_metrics = Vec::with_capacity(edges_1hop.len());
            for &(u, v, raw_len) in &edges_1hop {
                let dm = resolve_pair_distance(u, v);
                if let Some(d_val) = dm {
                    let df = d_val.to_f64().unwrap_or(0.0);
                    if df.is_finite() && df > 0.0 {
                        metrics_only.push(df);
                        let rl = raw_len.to_f64().unwrap_or(0.0);
                        if rl.is_finite() && rl > 0.0 {
                            ratios.push(df / rl);
                        }
                    }
                }
                d_metrics.push(dm);
            }
            scale_1hop_f64 = compute_median_f64(&mut ratios)
                .or_else(|| compute_median_f64(&mut metrics_only))
                .unwrap_or(1.0);
            let scale_s = S::from_f64(scale_1hop_f64).unwrap();
            for (idx, &(_, _, raw_len)) in edges_1hop.iter().enumerate() {
                let val = match d_metrics[idx] {
                    Some(dm) if dm.is_finite() && dm > S::zero() => dm,
                    _ => {
                        let rl = if raw_len.is_finite() && raw_len > S::zero() {
                            raw_len
                        } else {
                            S::one()
                        };
                        scale_s * rl
                    }
                };
                d1_raw.push(val);
            }
        } else {
            for &(_, _, raw_len) in &edges_1hop {
                let val = if raw_len.is_finite() && raw_len > S::zero() {
                    raw_len
                } else {
                    eps_s
                };
                d1_raw.push(val);
            }
        }

        let mut pos_d1: Vec<f64> = d1_raw
            .iter()
            .filter_map(|d| {
                let df = d.to_f64()?;
                if df.is_finite() && df > 0.0 {
                    Some(df)
                } else {
                    None
                }
            })
            .collect();
        let ref_med = compute_median_f64(&mut pos_d1).unwrap_or(1.0);
        let d_floor_f64 = self.eps_dist.max(self.min_ratio.max(0.0) * ref_med);
        let d_floor = S::from_f64(d_floor_f64).unwrap();

        let omega_1_s = S::from_f64(self.omega_1.max(0.0)).unwrap();
        let omega_2_s = S::from_f64(self.omega_2.max(0.0)).unwrap();
        let gamma_non2_s = S::from_f64(self.gamma_non2.max(0.0)).unwrap();

        let mut node_pairs = Vec::with_capacity(edges_1hop.len());
        let mut r2 = vec![S::zero(); n];
        for (idx, &(u, v, _)) in edges_1hop.iter().enumerate() {
            let d1 = d1_raw[idx].max(d_floor);
            if d1 > r2[u] {
                r2[u] = d1;
            }
            if d1 > r2[v] {
                r2[v] = d1;
            }
            if omega_1_s > S::zero() {
                let w1 = omega_1_s / (d1 * d1);
                node_pairs.push((u, v, d1, d1, w1, w1));
            }
        }

        // Step D: Hub-safe O(|E| * max_2hop_per_node) 2-hop registration.
        let mut pivot_2hop = vec![vec![]; h];
        if self.max_2hop_per_node > 0 && n > 2 {
            let pairs_2hop = extract_hub_safe_2hop(&adj, self.max_2hop_per_node, self.seed);
            let fallback_2hop = S::from_f64(2.0 * scale_1hop_f64).unwrap().max(d_floor);
            for (u, v) in pairs_2hop {
                if let Some(k) = pivot_slot[u] {
                    pivot_2hop[k].push(v);
                }
                if let Some(k) = pivot_slot[v] {
                    pivot_2hop[k].push(u);
                }
                let d2 = resolve_pair_distance(u, v)
                    .unwrap_or(fallback_2hop)
                    .max(d_floor);
                if d2 > r2[u] {
                    r2[u] = d2;
                }
                if d2 > r2[v] {
                    r2[v] = d2;
                }
                if omega_2_s > S::zero() {
                    let w2 = omega_2_s / (d2 * d2);
                    node_pairs.push((u, v, d2, d2, w2, w2));
                }
            }
        }

        if h == 0 {
            return Sgd::new(node_pairs);
        }

        // Step E: O(N h log N) Voronoi counting + relative-margin barrier + optional hinge repulsion.
        let mut r_dists = vec![vec![]; h];
        for j in 0..n {
            let best_k = (0..h)
                .min_by_key(|&k| OrderedFloat(piv_dists[k * n + j]))
                .unwrap();
            r_dists[best_k].push(piv_dists[best_k * n + j]);
        }
        for dists_k in &mut r_dists {
            dists_k.sort_by_key(|&d| OrderedFloat(d));
        }

        let two = S::from_usize(2).unwrap();
        let half = S::from_f64(0.5).unwrap();
        let barrier_mult = S::from_f64(1.0 + self.delta_rel.max(0.0)).unwrap();
        let mut p_1hop_stamp = vec![0usize; n];
        let mut p_2hop_stamp = vec![0usize; n];
        let mut hinge_pairs = Vec::new();
        let cap = self.max_2hop_per_node;

        for (k, &p) in pivot_node_indices.iter().enumerate() {
            let stamp_val = k + 1;
            p_1hop_stamp[p] = stamp_val;
            for &w in &adj[p] {
                p_1hop_stamp[w] = stamp_val;
            }
            for &v2 in &pivot_2hop[k] {
                p_2hop_stamp[v2] = stamp_val;
            }
            let is_hub_pivot =
                cap > 0 && (adj[p].len() > cap || adj[p].iter().any(|&w| adj[w].len() > cap));

            for i in 0..n {
                if p_1hop_stamp[i] == stamp_val || p_2hop_stamp[i] == stamp_val {
                    continue;
                }
                if is_hub_pivot && adj[i].iter().any(|&w| p_1hop_stamp[w] == stamp_val) {
                    continue;
                }
                let dpi = piv_dists[k * n + i];
                if !dpi.is_finite() || dpi <= S::zero() {
                    continue;
                }
                let r_pair = match self.barrier_sym {
                    BarrierSymmetry::SymMax => r2[p].max(r2[i]),
                    BarrierSymmetry::SymAvg => half * (r2[p] + r2[i]),
                };
                let dpi_bar = dpi.max(barrier_mult * r_pair).max(d_floor);
                if gamma_non2_s <= S::zero() {
                    continue;
                }
                let wpi = gamma_non2_s / (dpi_bar * dpi_bar);
                let count = r_dists[k].partition_point(|&dpj| dpj.is_finite() && two * dpj <= dpi);
                let spi = S::from_usize(count).unwrap();
                let pair = (p, i, dpi_bar, dpi_bar, S::zero(), spi * wpi);
                if self.hinge_repulsion {
                    hinge_pairs.push(pair);
                } else {
                    node_pairs.push(pair);
                }
            }
        }

        Sgd::new_with_hinge_pairs(node_pairs, hinge_pairs)
    }
}
