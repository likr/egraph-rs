use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges};
use petgraph_drawing::DrawingValue;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

pub(super) type CanonicalEdgesAndAdj<S> = (Vec<(usize, usize, S)>, Vec<Vec<usize>>);

pub(super) fn extract_canonical_edges_and_adj<G, F, S>(
    graph: G,
    indices: &HashMap<G::NodeId, usize>,
    length: &mut F,
) -> CanonicalEdgesAndAdj<S>
where
    G: IntoEdges,
    G::NodeId: Eq + std::hash::Hash,
    F: FnMut(G::EdgeRef) -> S,
    S: DrawingValue,
{
    let n = indices.len();
    let mut adj = vec![Vec::new(); n];
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut edges_1hop = Vec::new();
    for edge in graph.edge_references() {
        let i = indices[&edge.source()];
        let j = indices[&edge.target()];
        if i == j {
            continue;
        }
        let (u, v) = if i < j { (i, j) } else { (j, i) };
        let len = length(edge);
        if seen.insert((u, v)) {
            edges_1hop.push((u, v, len));
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    edges_1hop.sort_unstable_by_key(|&(u, v, _)| (u, v));
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    (edges_1hop, adj)
}

pub(super) fn extract_hub_safe_2hop(
    adj: &[Vec<usize>],
    cap: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let n = adj.len();
    if cap == 0 || n <= 2 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stamp = vec![0usize; n];
    let mut raw_2hop = Vec::new();

    for u in 0..n {
        if adj[u].is_empty() {
            continue;
        }
        let stamp_val = u + 1;
        stamp[u] = stamp_val;
        for &w in &adj[u] {
            stamp[w] = stamp_val;
        }

        let mut collected = 0usize;
        for &w in &adj[u] {
            // Skipping degree-1 neighbors prevents O(N) scans at star hubs without losing 2-hop pairs.
            let deg_w = adj[w].len();
            if deg_w <= 1 {
                continue;
            }
            if deg_w <= cap {
                for &v in &adj[w] {
                    if stamp[v] != stamp_val {
                        stamp[v] = stamp_val;
                        raw_2hop.push(if u < v { (u, v) } else { (v, u) });
                        collected += 1;
                        if collected >= cap {
                            break;
                        }
                    }
                }
            } else {
                let n_draws = deg_w.min(cap * 2);
                for _ in 0..n_draws {
                    let v = adj[w][rng.gen_range(0..deg_w)];
                    if stamp[v] != stamp_val {
                        stamp[v] = stamp_val;
                        raw_2hop.push(if u < v { (u, v) } else { (v, u) });
                        collected += 1;
                        if collected >= cap {
                            break;
                        }
                    }
                }
            }
            if collected >= cap {
                break;
            }
        }
    }

    raw_2hop.sort_unstable();
    raw_2hop.dedup();
    raw_2hop
}

pub(super) fn compute_median_f64(vals: &mut [f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by_key(|&v| OrderedFloat(v));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some(0.5 * (vals[mid - 1] + vals[mid]))
    }
}

pub(super) fn round_half_to_even(x: f64) -> usize {
    x.round_ties_even().max(0.0) as usize
}

pub(super) fn compute_adaptive_perplexity_from_pairs(
    n_nodes: usize,
    pairs_1hop: &[(usize, usize)],
    pairs_2hop: &[(usize, usize)],
) -> f64 {
    if n_nodes <= 3 {
        return 1.0;
    }
    let upper = 32.0_f64.min((n_nodes - 1) as f64 / 3.0);
    if upper < 1.0 {
        return 1.0;
    }
    let lower = 8.0_f64.min(upper);
    let mut deg2_counts = vec![0usize; n_nodes];
    for &(u, v) in pairs_1hop.iter().chain(pairs_2hop.iter()) {
        deg2_counts[u] += 1;
        deg2_counts[v] += 1;
    }
    deg2_counts.sort_unstable();
    let mid = n_nodes / 2;
    let med = if n_nodes % 2 == 1 {
        deg2_counts[mid] as f64
    } else {
        0.5 * (deg2_counts[mid - 1] + deg2_counts[mid]) as f64
    };
    (1.5 * med).clamp(lower, upper)
}

pub(super) fn has_common_neighbor(
    n1_u: &[usize],
    n1_v: &[usize],
    stamp: &[usize],
    stamp_val: usize,
) -> bool {
    if n1_v.is_empty() || n1_u.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| stamp[x] == stamp_val);
    }
    n1_u.iter().any(|x| n1_v.binary_search(x).is_ok())
}

pub(super) fn sample_non2hop_pairs(
    adj: &[Vec<usize>],
    pivots: &[usize],
    pairs_1hop: &[(usize, usize)],
    pairs_2hop: &[(usize, usize)],
    budget: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let n = adj.len();
    if n <= 2 || budget == 0 {
        return Vec::new();
    }
    let n_piv = pivots.len();
    let piv_quota = if n_piv > 0 {
        (budget / 2).max(1).min(n_piv)
    } else {
        0
    };

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(101));
    let mut stamp = vec![0usize; n];
    let mut neg_stamp = vec![0usize; n];
    let mut raw_non2 = Vec::new();
    let max_trials = 8usize.max(3 * budget);

    for u in 0..n {
        let stamp_val = u + 1;
        stamp[u] = stamp_val;
        for &w in &adj[u] {
            stamp[w] = stamp_val;
        }

        let mut collected = 0usize;
        if piv_quota > 0 {
            let start_off = if n_piv > piv_quota {
                (u * 7) % n_piv
            } else {
                0
            };
            let mut checked = 0usize;
            let mut idx_p = start_off;
            while checked < n_piv && collected < piv_quota {
                let v = pivots[idx_p];
                idx_p = (idx_p + 1) % n_piv;
                checked += 1;
                if stamp[v] == stamp_val || neg_stamp[v] == stamp_val {
                    continue;
                }
                if has_common_neighbor(&adj[u], &adj[v], &stamp, stamp_val) {
                    continue;
                }
                neg_stamp[v] = stamp_val;
                raw_non2.push(if u < v { (u, v) } else { (v, u) });
                collected += 1;
            }
        }

        if collected < budget {
            for _ in 0..max_trials {
                let v = rng.gen_range(0..n);
                if stamp[v] == stamp_val || neg_stamp[v] == stamp_val {
                    continue;
                }
                if has_common_neighbor(&adj[u], &adj[v], &stamp, stamp_val) {
                    continue;
                }
                neg_stamp[v] = stamp_val;
                raw_non2.push(if u < v { (u, v) } else { (v, u) });
                collected += 1;
                if collected >= budget {
                    break;
                }
            }
        }
    }

    raw_non2.sort_unstable();
    raw_non2.dedup();
    raw_non2.retain(|pair| {
        pairs_1hop.binary_search(pair).is_err() && pairs_2hop.binary_search(pair).is_err()
    });
    raw_non2
}
