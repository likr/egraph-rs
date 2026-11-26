use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

#[derive(Debug, Clone, Copy)]
pub(super) struct GridEntry {
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) node: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GridCell {
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) start: usize,
    pub(super) end: usize,
}

#[inline]
pub(super) fn clamp_cell_coord(v: f64) -> i64 {
    if !v.is_finite() {
        return 0;
    }
    let min_c = -1_000_000_000_000.0_f64;
    let max_c = 1_000_000_000_000.0_f64;
    v.clamp(min_c, max_c) as i64
}

#[allow(clippy::type_complexity)]
pub(super) fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<u32>, Vec<u32>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("fr::graph_weights"));
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

    if !graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
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
        }
    }

    Ok((n, edges_u, edges_v, edges_w))
}
