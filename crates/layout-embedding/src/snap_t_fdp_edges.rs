use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

pub(crate) type SnapEdgeBuffers = (Vec<u32>, Vec<u32>, Option<Vec<f32>>);

pub(crate) fn extract_snap_edges<W: Scalar>(graph: &CsrGraph<W>) -> CoreResult<SnapEdgeBuffers> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Ok((Vec::new(), Vec::new(), None));
    }

    let (edges_u, edges_v, edge_weights) = if !graph.is_directed() {
        // Stream directly into flat u32 arrays without temporary tuple vectors to respect
        // strict multi-million-node Peak RSS budgets.
        let m_est = graph.num_edges();
        let mut eu = Vec::with_capacity(m_est);
        let mut ev = Vec::with_capacity(m_est);
        let mut ew_mut = graph.is_weighted().then(|| Vec::with_capacity(m_est));
        let weights_opt = graph.weights();
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos];
                if u_u32 < v {
                    eu.push(u_u32);
                    ev.push(v);
                    if let (Some(ws_out), Some(ws_in)) = (ew_mut.as_mut(), weights_opt) {
                        ws_out.push(ws_in[pos].to_f32());
                    }
                }
            }
        }
        (eu, ev, ew_mut)
    } else if let Some(ws_in) = graph.weights() {
        let mut raw: Vec<(u32, u32, f32)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for (&v, &w) in graph.indices()[start..end]
                .iter()
                .zip(ws_in[start..end].iter())
            {
                if u_u32 == v {
                    continue;
                }
                let (a, b) = if u_u32 < v { (u_u32, v) } else { (v, u_u32) };
                raw.push((a, b, w.to_f32()));
            }
        }
        raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        let mut eu = Vec::with_capacity(raw.len());
        let mut ev = Vec::with_capacity(raw.len());
        let mut ws_out: Vec<f32> = Vec::with_capacity(raw.len());
        for (a, b, w) in raw {
            if eu.last().copied() == Some(a) && ev.last().copied() == Some(b) {
                if let Some(last_w) = ws_out.last_mut() {
                    *last_w = (*last_w).max(w);
                }
            } else {
                eu.push(a);
                ev.push(b);
                ws_out.push(w);
            }
        }
        (eu, ev, Some(ws_out))
    } else {
        let mut raw: Vec<(u32, u32)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for &v in &graph.indices()[start..end] {
                if u_u32 != v {
                    raw.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                }
            }
        }
        raw.sort_unstable();
        raw.dedup();
        let mut eu = Vec::with_capacity(raw.len());
        let mut ev = Vec::with_capacity(raw.len());
        for (a, b) in raw {
            eu.push(a);
            ev.push(b);
        }
        (eu, ev, None)
    };

    if u32::try_from(edges_u.len()).is_err() {
        return Err(CoreError::invalid_csr(
            "SNAP-tFDP undirected edge count exceeds u32::MAX",
        ));
    }
    Ok((edges_u, edges_v, edge_weights))
}
