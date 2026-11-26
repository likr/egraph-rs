use egraph_core::{CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};
use egraph_neighborhood::Sparse12HopPairs;

#[derive(Debug, Clone)]
pub(super) struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    pub(super) fn from_weights(weights: &[f64]) -> Option<Self> {
        let k = weights.len();
        if k == 0 {
            return None;
        }
        let mut total = 0.0_f64;
        for &w in weights {
            if w.is_finite() && w > 0.0 {
                total += w;
            }
        }
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let scale = (k as f64) / total;
        let mut scaled = vec![0.0_f64; k];
        let mut small = Vec::with_capacity(k);
        let mut large = Vec::with_capacity(k);
        for (i, &w) in weights.iter().enumerate() {
            let p = if w.is_finite() && w > 0.0 {
                w * scale
            } else {
                0.0
            };
            scaled[i] = p;
            if p < 1.0 {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }
        let mut prob = vec![1.0_f64; k];
        let mut alias: Vec<u32> = (0..k as u32).collect();
        while let (Some(s_idx), Some(l_idx)) = (small.pop(), large.pop()) {
            let s = s_idx as usize;
            let l = l_idx as usize;
            prob[s] = scaled[s].clamp(0.0, 1.0);
            alias[s] = l_idx;
            scaled[l] = ((scaled[l] + scaled[s]) - 1.0).max(0.0);
            if scaled[l] < 1.0 {
                small.push(l_idx);
            } else {
                large.push(l_idx);
            }
        }
        while let Some(l_idx) = large.pop() {
            prob[l_idx as usize] = 1.0;
        }
        while let Some(s_idx) = small.pop() {
            prob[s_idx as usize] = 1.0;
        }
        Some(Self { prob, alias })
    }

    #[inline]
    pub(super) fn sample(&self, rng: &mut DeterministicRng) -> usize {
        let k = self.prob.len();
        let col = rng.gen_range_usize(k);
        let coin = rng.next_f64();
        if coin < self.prob[col] {
            col
        } else {
            self.alias[col] as usize
        }
    }
}

/// Builds canonical sorted simple symmetric unweighted CSR `(indptr, indices)` from sorted
/// deduplicated upper-triangular pairs `(u1[k], v1[k])` (`u1[k] < v1[k]`) in $O(N + |E|)$.
pub(super) fn build_symmetric_csr_from_sorted_pairs(
    num_nodes: usize,
    u1: &[u32],
    v1: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut deg = vec![0u32; num_nodes];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        indptr.push(acc);
    }
    let mut cursor = indptr[..num_nodes].to_vec();
    let mut indices = vec![0u32; acc as usize];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        let pu = cursor[u as usize] as usize;
        indices[pu] = v;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize] as usize;
        indices[pv] = u;
        cursor[v as usize] += 1;
    }
    for u in 0..num_nodes {
        let row = &mut indices[indptr[u] as usize..indptr[u + 1] as usize];
        if !row.windows(2).all(|w| w[0] < w[1]) {
            row.sort_unstable();
        }
    }
    (indptr, indices)
}

/// Extracts canonical deduplicated undirected 1-hop edges `(u, v, weight)` (`u < v`) from `graph`.
pub(super) fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> Vec<(u32, u32, f64)> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Vec::new();
    }
    let weights_opt = graph.weights();
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    edges.push((u32_u, v, w));
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return edges;
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            raw.push((a, b, w));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 = last.2.min(w);
        } else {
            out.push((a, b, w));
        }
    }
    out
}

/// Collects sorted deduplicated pairs `(u, v)` (`u < v`) combining all 1-hop edges and
/// `k` deterministically sampled random partners per node.
pub(super) fn sample_edges_and_random_pairs<W: Scalar>(
    graph: &CsrGraph<W>,
    k: usize,
    seed: u64,
) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let base_edges = extract_canonical_undirected_edges(graph);
    let mut raw_pairs: Vec<(u32, u32)> = Vec::with_capacity(base_edges.len() + n.saturating_mul(k));
    for &(u, v, _) in &base_edges {
        raw_pairs.push((u, v));
    }
    if k > 0 && n > 1 {
        let mut sample_rng = DeterministicRng::from_seed_u64(seed);
        for i in 0..n {
            let i_u32 = i as u32;
            for _ in 0..k {
                let j = sample_rng.gen_range_usize(n) as u32;
                if j != i_u32 {
                    raw_pairs.push(if i_u32 < j { (i_u32, j) } else { (j, i_u32) });
                }
            }
        }
    }
    raw_pairs.sort_unstable();
    raw_pairs.dedup();
    raw_pairs
}

pub(super) fn compute_adaptive_perplexity_from_pairs(
    n_nodes: usize,
    pairs: &Sparse12HopPairs,
) -> f64 {
    if n_nodes <= 3 {
        return 1.0;
    }
    let upper = 32.0_f64.min((n_nodes - 1) as f64 / 3.0);
    if upper < 1.0 {
        return 1.0;
    }
    let lower = 8.0_f64.min(upper);
    let mut deg2_counts = vec![0u32; n_nodes];
    for slice in [&pairs.u1, &pairs.v1, &pairs.u2, &pairs.v2] {
        for &node in slice {
            deg2_counts[node as usize] += 1;
        }
    }
    deg2_counts.sort_unstable();
    let mid = n_nodes / 2;
    let med = if n_nodes % 2 == 1 {
        deg2_counts[mid] as f64
    } else {
        0.5 * ((deg2_counts[mid - 1] as f64) + (deg2_counts[mid] as f64))
    };
    (1.5 * med).clamp(lower, upper)
}

#[inline]
pub(super) fn has_common_neighbor(n1_u: &[u32], n1_v: &[u32], stamp: &VisitStamp) -> bool {
    if n1_u.is_empty() || n1_v.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| stamp.is_visited(x as usize));
    }
    n1_u.iter().any(|x| n1_v.binary_search(x).is_ok())
}

#[inline]
pub(super) fn has_common_neighbor_non_hub(
    n1_u: &[u32],
    n1_v: &[u32],
    stamp: &VisitStamp,
    adj_indptr: &[u32],
    non_hub_thresh: usize,
) -> bool {
    if n1_u.is_empty() || n1_v.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| {
            let xi = x as usize;
            stamp.is_visited(xi)
                && ((adj_indptr[xi + 1] - adj_indptr[xi]) as usize) <= non_hub_thresh
        });
    }
    n1_u.iter().any(|&x| {
        let xi = x as usize;
        ((adj_indptr[xi + 1] - adj_indptr[xi]) as usize) <= non_hub_thresh
            && n1_v.binary_search(&x).is_ok()
    })
}

pub(super) fn sample_non2hop_pairs(
    n_nodes: usize,
    adj_indptr: &[u32],
    adj_indices: &[u32],
    pivots: &[u32],
    pairs: &Sparse12HopPairs,
    budget: usize,
    seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    if n_nodes <= 2 || budget == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_piv = pivots.len();
    let piv_quota = if n_piv > 0 {
        (budget / 2).max(1).min(n_piv)
    } else {
        0
    };

    let mut rng = DeterministicRng::from_seed_u64(seed.wrapping_add(101));
    let mut stamp = VisitStamp::new(n_nodes);
    let mut neg_stamp = VisitStamp::new(n_nodes);
    let mut raw_non2: Vec<(u32, u32)> = Vec::new();
    let max_trials = 8usize.max(3 * budget);

    for u in 0..n_nodes {
        stamp.clear();
        neg_stamp.clear();
        stamp.mark(u);
        let n1_u = &adj_indices[adj_indptr[u] as usize..adj_indptr[u + 1] as usize];
        for &w in n1_u {
            stamp.mark(w as usize);
        }

        let u_u32 = u as u32;
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
                let vi = v as usize;
                if stamp.is_visited(vi) || neg_stamp.is_visited(vi) {
                    continue;
                }
                let n1_v = &adj_indices[adj_indptr[vi] as usize..adj_indptr[vi + 1] as usize];
                if has_common_neighbor(n1_u, n1_v, &stamp) {
                    continue;
                }
                neg_stamp.mark(vi);
                raw_non2.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                collected += 1;
            }
        }

        if collected < budget {
            for _ in 0..max_trials {
                let v = rng.gen_range_usize(n_nodes) as u32;
                let vi = v as usize;
                if stamp.is_visited(vi) || neg_stamp.is_visited(vi) {
                    continue;
                }
                let n1_v = &adj_indices[adj_indptr[vi] as usize..adj_indptr[vi + 1] as usize];
                if has_common_neighbor(n1_u, n1_v, &stamp) {
                    continue;
                }
                neg_stamp.mark(vi);
                raw_non2.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                collected += 1;
                if collected >= budget {
                    break;
                }
            }
        }
    }

    raw_non2.sort_unstable();
    raw_non2.dedup();

    // Linear two-pointer disjointness check against sorted 1-hop and 2-hop pair slices.
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut u3 = Vec::with_capacity(raw_non2.len());
    let mut v3 = Vec::with_capacity(raw_non2.len());
    for pair in raw_non2 {
        while i1 < pairs.u1.len() && (pairs.u1[i1], pairs.v1[i1]) < pair {
            i1 += 1;
        }
        if i1 < pairs.u1.len() && (pairs.u1[i1], pairs.v1[i1]) == pair {
            continue;
        }
        while i2 < pairs.u2.len() && (pairs.u2[i2], pairs.v2[i2]) < pair {
            i2 += 1;
        }
        if i2 < pairs.u2.len() && (pairs.u2[i2], pairs.v2[i2]) == pair {
            continue;
        }
        u3.push(pair.0);
        v3.push(pair.1);
    }

    (u3, v3)
}

pub(super) type CanonicalPositiveGraph = (CsrGraph<f64>, Vec<(u32, u32, f64)>);

pub(super) fn build_positive_canonical_csr<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<CanonicalPositiveGraph> {
    let n = graph.num_nodes();
    let raw_edges = extract_canonical_undirected_edges(graph);
    let weighted = graph.is_weighted();
    let mut positive_edges = Vec::with_capacity(raw_edges.len());
    for (u, v, w) in raw_edges {
        let w_pos = if weighted { w } else { 1.0 };
        if w_pos.is_finite() && w_pos > 0.0 {
            positive_edges.push((u, v, w_pos));
        }
    }
    let mut pair_uv = Vec::with_capacity(positive_edges.len());
    let mut pair_w = Vec::with_capacity(positive_edges.len());
    for &(u, v, w) in &positive_edges {
        pair_uv.push((u, v));
        pair_w.push(w);
    }
    let weights_opt = if weighted {
        Some(pair_w.as_slice())
    } else {
        None
    };
    let sym_graph = CsrGraph::<f64>::from_edges(n, &pair_uv, weights_opt, false, false)?;
    Ok((sym_graph, positive_edges))
}

pub(super) fn compute_mssp_dists(graph: &CsrGraph<f64>, pivots: &[u32]) -> CoreResult<Vec<f64>> {
    let n = graph.num_nodes();
    let h = pivots.len();
    let mut piv_dists = vec![f64::INFINITY; h * n];
    if graph.is_weighted() {
        let mut dijk = DijkstraWorkspace::new(n);
        for (k, &p) in pivots.iter().enumerate() {
            dijk.dijkstra_from(graph, p as usize, &mut piv_dists[k * n..(k + 1) * n])?;
        }
    } else {
        let mut bfs = BfsWorkspace::new(n);
        for (k, &p) in pivots.iter().enumerate() {
            bfs.bfs_from(graph, p as usize, &mut piv_dists[k * n..(k + 1) * n])?;
        }
    }
    Ok(piv_dists)
}
