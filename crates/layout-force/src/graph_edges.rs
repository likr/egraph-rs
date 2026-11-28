use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

/// Extracts canonical sorted loopless undirected edges `(edges_u, edges_v, edges_w)` and
/// per-vertex degrees from a weighted or unweighted [`CsrGraph<W>`].
#[allow(clippy::type_complexity)]
pub(crate) fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
    context: &'static str,
) -> CoreResult<(usize, Vec<u32>, Vec<u32>, Vec<f64>, Vec<usize>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite(context));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let mut edges_u = Vec::new();
    let mut edges_v = Vec::new();
    let mut edges_w = Vec::new();
    let mut deg = vec![0usize; n];

    if !graph.is_directed() {
        for (u, deg_u) in deg.iter_mut().enumerate() {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
            *deg_u = nbrs.len();
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                if u32_u < v_u32 {
                    edges_u.push(u32_u);
                    edges_v.push(v_u32);
                    edges_w.push(ws.map_or(1.0, |w| w[pos].to_f64()));
                }
            }
        }
    } else {
        let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                if u32_u == v_u32 {
                    continue;
                }
                let (a, b) = if u32_u < v_u32 {
                    (u32_u, v_u32)
                } else {
                    (v_u32, u32_u)
                };
                raw.push((a, b, ws.map_or(1.0, |w| w[pos].to_f64())));
            }
        }
        raw.sort_unstable_by_key(|x| (x.0, x.1));

        let weighted = graph.is_weighted();
        let mut idx = 0usize;
        while idx < raw.len() {
            let (a, b, first_w) = raw[idx];
            let mut sum_w = first_w;
            let mut cnt = 1usize;
            idx += 1;
            while idx < raw.len() && raw[idx].0 == a && raw[idx].1 == b {
                sum_w += raw[idx].2;
                cnt += 1;
                idx += 1;
            }
            let merged_w = if weighted { sum_w / (cnt as f64) } else { 1.0 };
            edges_u.push(a);
            edges_v.push(b);
            edges_w.push(merged_w);
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
    }

    Ok((n, edges_u, edges_v, edges_w, deg))
}

/// Extracts canonical loopless undirected edges with unit weights (`1.0`), ignoring any stored
/// edge weights so `CsrGraph<()>` and arbitrary `W` types are supported without `W: Scalar`.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_canonical_undirected_unweighted_edges<W>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<u32>, Vec<u32>, Vec<f64>, Vec<usize>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }

    let pairs = graph.canonical_undirected_edges();
    let mut edges_u = Vec::with_capacity(pairs.len());
    let mut edges_v = Vec::with_capacity(pairs.len());
    let mut edges_w = Vec::with_capacity(pairs.len());
    let mut deg = vec![0usize; n];
    for (u, v) in pairs {
        edges_u.push(u);
        edges_v.push(v);
        edges_w.push(1.0);
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }

    Ok((n, edges_u, edges_v, edges_w, deg))
}

/// Builds symmetric undirected CSR adjacency `(num_nodes, indptr, indices, weights)` from canonical
/// undirected edge lists.
#[cfg(feature = "baselines")]
#[allow(clippy::type_complexity)]
pub(crate) fn build_undirected_adj_from_edges(
    n: usize,
    edges_u: &[u32],
    edges_v: &[u32],
    edges_w: &[f64],
) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut directed_pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(edges_u.len() * 2);
    for idx in 0..edges_u.len() {
        let ua = edges_u[idx] as usize;
        let ub = edges_v[idx] as usize;
        let w = edges_w[idx];
        directed_pairs.push((ua, ub, w));
        directed_pairs.push((ub, ua, w));
    }
    directed_pairs.sort_unstable_by_key(|x| (x.0, x.1));

    let mut indptr = vec![0usize; n + 1];
    for &(u, _, _) in &directed_pairs {
        indptr[u + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let mut indices = Vec::with_capacity(directed_pairs.len());
    let mut weights = Vec::with_capacity(directed_pairs.len());
    for &(_, v, w) in &directed_pairs {
        indices.push(v);
        weights.push(w);
    }

    (n, indptr, indices, weights)
}
