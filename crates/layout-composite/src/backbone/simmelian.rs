use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};

pub(super) const WEIGHT_TIE_EPS: f64 = 1e-12;

pub(super) struct SimpleUndirectedGraph {
    pub(super) num_nodes: usize,
    pub(super) edges: Vec<(u32, u32)>,
    pub(super) input_weights: Option<Vec<f64>>,
    pub(super) adj_indptr: Vec<u32>,
    pub(super) adj_indices: Vec<u32>,
    pub(super) adj_edge_id: Vec<u32>,
}

pub(super) fn build_simple_undirected_graph<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<SimpleUndirectedGraph> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "extract_quadrilateral_backbone requires an undirected graph",
        ));
    }
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("backbone::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let (edges, input_weights) = if let Some(ws) = graph.weights() {
        let indptr = graph.indptr();
        let mut weighted_edges: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.num_edges());
        for (u, &start_u32) in indptr.iter().enumerate().take(n) {
            let u32_u = u as u32;
            let row_start = start_u32 as usize;
            for (offset, &v_u32) in graph.neighbors(u).iter().enumerate() {
                if u32_u < v_u32 {
                    weighted_edges.push((u32_u, v_u32, ws[row_start + offset].to_f64()));
                }
            }
        }
        weighted_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut dedup_edges = Vec::with_capacity(weighted_edges.len());
        let mut dedup_weights = Vec::with_capacity(weighted_edges.len());
        let mut idx = 0usize;
        while idx < weighted_edges.len() {
            let (u, v, _) = weighted_edges[idx];
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            while idx < weighted_edges.len()
                && weighted_edges[idx].0 == u
                && weighted_edges[idx].1 == v
            {
                sum += weighted_edges[idx].2;
                cnt += 1;
                idx += 1;
            }
            dedup_edges.push((u, v));
            dedup_weights.push(sum / (cnt as f64));
        }
        (dedup_edges, Some(dedup_weights))
    } else {
        (collect_unweighted_simple_edges(graph, n), None)
    };

    build_simple_undirected_from_edges(n, edges, input_weights)
}

pub(super) fn build_simple_undirected_graph_unweighted<W>(
    graph: &CsrGraph<W>,
) -> CoreResult<SimpleUndirectedGraph> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "extract_quadrilateral_backbone_unweighted requires an undirected graph",
        ));
    }
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    let edges = collect_unweighted_simple_edges(graph, n);
    build_simple_undirected_from_edges(n, edges, None)
}

fn collect_unweighted_simple_edges<W>(graph: &CsrGraph<W>, n: usize) -> Vec<(u32, u32)> {
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(graph.num_edges());
    for u in 0..n {
        let u32_u = u as u32;
        for &v_u32 in graph.neighbors(u) {
            if u32_u < v_u32 {
                edges.push((u32_u, v_u32));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

fn build_simple_undirected_from_edges(
    n: usize,
    edges: Vec<(u32, u32)>,
    input_weights: Option<Vec<f64>>,
) -> CoreResult<SimpleUndirectedGraph> {
    let m = edges.len();
    let two_m = m
        .checked_mul(2)
        .ok_or_else(|| CoreError::invalid_csr("backbone half-edge count overflow"))?;
    if u32::try_from(two_m).is_err() {
        return Err(CoreError::invalid_csr("half-edge count exceeds u32::MAX"));
    }

    let mut deg = vec![0u32; n];
    for &(u, v) in &edges {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }

    let mut adj_indptr = Vec::with_capacity(n + 1);
    adj_indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        adj_indptr.push(acc);
    }

    let mut cursor = adj_indptr[..n].to_vec();
    let mut adj_indices = vec![0u32; two_m];
    let mut adj_edge_id = vec![0u32; two_m];

    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        let e_u32 = e_idx as u32;
        let pu = cursor[u as usize] as usize;
        adj_indices[pu] = v;
        adj_edge_id[pu] = e_u32;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize] as usize;
        adj_indices[pv] = u;
        adj_edge_id[pv] = e_u32;
        cursor[v as usize] += 1;
    }

    Ok(SimpleUndirectedGraph {
        num_nodes: n,
        edges,
        input_weights,
        adj_indptr,
        adj_indices,
        adj_edge_id,
    })
}

/// Chiba–Nishizeki $O(m\,\alpha(G))$ quadrangle ($C_4$) counting per edge.
pub(super) fn count_quadrangles(g: &SimpleUndirectedGraph) -> Vec<u64> {
    let n = g.num_nodes;
    let m = g.edges.len();
    let mut q_edge = vec![0u64; m];
    if n < 4 || m < 4 {
        return q_edge;
    }

    // Sort vertices by non-increasing degree (tie-break deterministically by vertex ID).
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = g.adj_indptr[a as usize + 1] - g.adj_indptr[a as usize];
        let db = g.adj_indptr[b as usize + 1] - g.adj_indptr[b as usize];
        db.cmp(&da).then_with(|| a.cmp(&b))
    });

    let mut rank = vec![0u32; n];
    for (r, &v) in order.iter().enumerate() {
        rank[v as usize] = r as u32;
    }

    // P[c] stores (b, e_ab, e_bc) for common neighbors b of a and c with rank(b), rank(c) > rank(a).
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    let mut touched_c: Vec<usize> = Vec::new();

    for &a_u32 in &order {
        let a = a_u32 as usize;
        let rank_a = rank[a];
        let a_start = g.adj_indptr[a] as usize;
        let a_end = g.adj_indptr[a + 1] as usize;
        if a_end - a_start < 2 {
            continue;
        }

        for pos_ab in a_start..a_end {
            let b = g.adj_indices[pos_ab] as usize;
            if rank[b] <= rank_a {
                continue;
            }
            let e_ab = g.adj_edge_id[pos_ab];
            let b_start = g.adj_indptr[b] as usize;
            let b_end = g.adj_indptr[b + 1] as usize;

            for pos_bc in b_start..b_end {
                let c = g.adj_indices[pos_bc] as usize;
                if c != a && rank[c] > rank_a {
                    if buckets[c].is_empty() {
                        touched_c.push(c);
                    }
                    buckets[c].push((e_ab, g.adj_edge_id[pos_bc]));
                }
            }
        }

        for &c in &touched_c {
            let list = &mut buckets[c];
            let k = list.len();
            if k >= 2 {
                let delta = (k - 1) as u64;
                for &(e_ab, e_bc) in list.iter() {
                    q_edge[e_ab as usize] += delta;
                    q_edge[e_bc as usize] += delta;
                }
            }
            list.clear();
        }
        touched_c.clear();
    }

    q_edge
}

#[derive(Debug, Clone, Copy)]
struct RankedNeighbor {
    nbr: u32,
    q_norm: f64,
    tie_start: usize,
    tie_end: usize,
}

#[cfg(test)]
pub(super) fn compute_simmelian_prefix_jaccard(
    g: &SimpleUndirectedGraph,
    q_edge: &[u64],
) -> Vec<f64> {
    compute_simmelian_prefix_jaccard_with_weights(g, q_edge, false)
}

pub(super) fn compute_simmelian_prefix_jaccard_with_weights(
    g: &SimpleUndirectedGraph,
    q_edge: &[u64],
    use_weights: bool,
) -> Vec<f64> {
    let n = g.num_nodes;
    let m = g.edges.len();
    if m == 0 {
        return Vec::new();
    }

    let mut q_node = vec![0u128; n];
    for (e_idx, &(u, v)) in g.edges.iter().enumerate() {
        let qv = q_edge[e_idx] as u128;
        q_node[u as usize] += qv;
        q_node[v as usize] += qv;
    }

    let mut q_norm = vec![0.0_f64; m];
    let mut has_nonzero_q = false;
    for (e_idx, &(u, v)) in g.edges.iter().enumerate() {
        let q_uv = q_edge[e_idx];
        if q_uv > 0 {
            let qu = q_node[u as usize] as f64;
            let qv = q_node[v as usize] as f64;
            let denom = qu.sqrt() * qv.sqrt();
            if denom > 0.0 && denom.is_finite() {
                q_norm[e_idx] = (q_uv as f64) / denom;
                has_nonzero_q = true;
            }
        }
    }

    if use_weights && let Some(ref iw) = g.input_weights {
        let max_iw = iw.iter().copied().fold(0.0_f64, f64::max);
        if max_iw > 0.0 {
            for e_idx in 0..m {
                let iw_norm = iw[e_idx] / max_iw;
                if has_nonzero_q {
                    q_norm[e_idx] *= iw_norm;
                } else {
                    q_norm[e_idx] = iw_norm;
                }
            }
        }
    }

    // Build per-vertex Q-ranked neighbor lists with tie-closed group bounds.
    let mut ranked_lists: Vec<Vec<RankedNeighbor>> = Vec::with_capacity(n);
    let mut max_deg = 0usize;

    for u in 0..n {
        let start = g.adj_indptr[u] as usize;
        let end = g.adj_indptr[u + 1] as usize;
        let deg = end - start;
        max_deg = max_deg.max(deg);
        let mut list = Vec::with_capacity(deg);
        for pos in start..end {
            list.push(RankedNeighbor {
                nbr: g.adj_indices[pos],
                q_norm: q_norm[g.adj_edge_id[pos] as usize],
                tie_start: 0,
                tie_end: 0,
            });
        }
        list.sort_unstable_by(|a, b| {
            b.q_norm
                .total_cmp(&a.q_norm)
                .then_with(|| a.nbr.cmp(&b.nbr))
        });

        let mut grp_start = 0usize;
        while grp_start < deg {
            let anchor = list[grp_start].q_norm;
            let mut grp_end = grp_start + 1;
            while grp_end < deg && (anchor - list[grp_end].q_norm).abs() <= WEIGHT_TIE_EPS {
                grp_end += 1;
            }
            for item in &mut list[grp_start..grp_end] {
                item.tie_start = grp_start;
                item.tie_end = grp_end;
            }
            grp_start = grp_end;
        }
        ranked_lists.push(list);
    }

    // Compute exact tie-invariant maximum prefix Jaccard in O(m * Delta(G)) without hash maps.
    let mut simmelian_weights = vec![0.0_f64; m];
    let mut in_u = VisitStamp::new(n);
    let mut u_tie_start = vec![0usize; n];
    let mut enter_at = vec![0usize; max_deg + 1];

    for u in 0..n {
        let list_u = &ranked_lists[u];
        let deg_u = list_u.len();
        if deg_u == 0 {
            continue;
        }
        in_u.clear();
        for item in list_u {
            let w = item.nbr as usize;
            in_u.mark(w);
            u_tie_start[w] = item.tie_start;
        }

        let start = g.adj_indptr[u] as usize;
        let end = g.adj_indptr[u + 1] as usize;
        for pos in start..end {
            let v = g.adj_indices[pos] as usize;
            if u >= v {
                continue;
            }
            let e_idx = g.adj_edge_id[pos] as usize;
            let list_v = &ranked_lists[v];
            let deg_v = list_v.len();
            let limit = deg_u.min(deg_v);
            if limit == 0 {
                continue;
            }

            enter_at[..=limit].fill(0);
            for item_v in list_v {
                let w = item_v.nbr as usize;
                if in_u.is_visited(w) {
                    let ku = u_tie_start[w] + 1;
                    let kv = item_v.tie_start + 1;
                    let k_enter = ku.max(kv);
                    if k_enter <= limit {
                        enter_at[k_enter] += 1;
                    }
                }
            }

            let mut intersect = 0usize;
            let mut best = 0.0_f64;
            for k in 1..=limit {
                intersect += enter_at[k];
                let size_a = list_u[k - 1].tie_end;
                let size_b = list_v[k - 1].tie_end;
                let union = size_a + size_b - intersect;
                if union > 0 {
                    let jaccard = (intersect as f64) / (union as f64);
                    if jaccard > best {
                        best = jaccard;
                    }
                }
            }
            simmelian_weights[e_idx] = best;
        }
    }

    simmelian_weights
}
