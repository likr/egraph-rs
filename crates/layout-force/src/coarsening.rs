use std::cmp::Reverse;

use egraph_core::{CoreError, CoreResult, CsrGraph, VisitStamp};

pub(crate) struct CoarsenMapping {
    pub(crate) coarse_n: usize,
    pub(crate) coarse_of: Vec<u32>,
    pub(crate) cluster_counts: Vec<u32>,
}

pub(crate) fn ec_coarsen(graph: &CsrGraph<f64>) -> CoarsenMapping {
    let n = graph.num_nodes();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&u| (graph.degree(u as usize), u));

    let mut coarse_of = vec![u32::MAX; n];
    let mut cluster_counts = Vec::with_capacity(n);
    let mut next_cid = 0u32;

    for &u_u32 in &order {
        let u = u_u32 as usize;
        if coarse_of[u] != u32::MAX {
            continue;
        }

        let nbrs = graph.neighbors(u);
        let weights = graph.neighbor_weights(u);
        let mut best_v = None;
        let mut best_score = (f64::NEG_INFINITY, Reverse(usize::MAX), Reverse(u32::MAX));

        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if v == u || coarse_of[v] != u32::MAX {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[pos]);
            let score = (w, Reverse(graph.degree(v)), Reverse(v_u32));
            if best_v.is_none() || score > best_score {
                best_score = score;
                best_v = Some(v);
            }
        }

        coarse_of[u] = next_cid;
        if let Some(v) = best_v {
            coarse_of[v] = next_cid;
            cluster_counts.push(2);
        } else {
            cluster_counts.push(1);
        }
        next_cid += 1;
    }

    CoarsenMapping {
        coarse_n: next_cid as usize,
        coarse_of,
        cluster_counts,
    }
}

pub(crate) fn mivs_coarsen(graph: &CsrGraph<f64>) -> CoarsenMapping {
    let n = graph.num_nodes();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&u| (graph.degree(u as usize), u));

    let mut covered = VisitStamp::new(n);
    let mut is_indep = vec![false; n];
    let mut coarse_of = vec![u32::MAX; n];
    let mut next_cid = 0u32;

    for &u_u32 in &order {
        let u = u_u32 as usize;
        if !covered.is_visited(u) {
            is_indep[u] = true;
            coarse_of[u] = next_cid;
            next_cid += 1;
            covered.mark(u);
            for &v in graph.neighbors(u) {
                covered.mark(v as usize);
            }
        }
    }

    let coarse_n = next_cid as usize;
    let mut cluster_counts = vec![1u32; coarse_n];

    for u in 0..n {
        if is_indep[u] {
            continue;
        }
        let nbrs = graph.neighbors(u);
        let weights = graph.neighbor_weights(u);
        let mut best_cid = None;
        let mut best_score = (f64::NEG_INFINITY, Reverse(u32::MAX));

        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if !is_indep[v] {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[pos]);
            let score = (w, Reverse(v_u32));
            if best_cid.is_none() || score > best_score {
                best_score = score;
                best_cid = Some(coarse_of[v]);
            }
        }

        if let Some(cid) = best_cid {
            coarse_of[u] = cid;
            cluster_counts[cid as usize] += 1;
        } else {
            let cid = cluster_counts.len() as u32;
            coarse_of[u] = cid;
            cluster_counts.push(1);
        }
    }

    CoarsenMapping {
        coarse_n: cluster_counts.len(),
        coarse_of,
        cluster_counts,
    }
}

pub(crate) fn build_coarse_csr(
    fine: &CsrGraph<f64>,
    coarse_n: usize,
    coarse_of: &[u32],
    cluster_counts: &[u32],
    stamp: &mut VisitStamp,
) -> CoreResult<CsrGraph<f64>> {
    let fine_n = fine.num_nodes();
    let mut group_indptr = Vec::with_capacity(coarse_n + 1);
    group_indptr.push(0usize);
    let mut acc = 0usize;
    for &cnt in cluster_counts {
        acc += cnt as usize;
        group_indptr.push(acc);
    }

    let mut cursor = group_indptr[..coarse_n].to_vec();
    let mut group_nodes = vec![0u32; fine_n];
    for (u, &c) in coarse_of.iter().enumerate() {
        let pos = cursor[c as usize];
        group_nodes[pos] = u as u32;
        cursor[c as usize] = pos + 1;
    }

    stamp.resize_and_clear(coarse_n);
    let mut edge_acc = vec![0.0_f64; coarse_n];
    let mut row_nbrs: Vec<u32> = Vec::new();
    let mut deg = vec![0u32; coarse_n];
    let mut upper_edges: Vec<(u32, u32, f64)> = Vec::new();

    // Sum each undirected coarse edge once in canonical (c < cv) order so both (c, cv) and (cv, c)
    // receive bit-identical f64 weights without non-associative 1-ULP drift.
    for c in 0..coarse_n {
        stamp.clear();
        row_nbrs.clear();

        let start = group_indptr[c];
        let end = group_indptr[c + 1];
        for &u_u32 in &group_nodes[start..end] {
            let u = u_u32 as usize;
            let nbrs = fine.neighbors(u);
            let weights = fine.neighbor_weights(u);
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                let cv = coarse_of[v_u32 as usize] as usize;
                if cv <= c {
                    continue;
                }
                let w = weights.map_or(1.0, |ws| ws[pos]);
                if stamp.visit(cv) {
                    row_nbrs.push(cv as u32);
                    edge_acc[cv] = w;
                } else {
                    edge_acc[cv] += w;
                }
            }
        }

        row_nbrs.sort_unstable();
        let c_u32 = c as u32;
        for &cv_u32 in &row_nbrs {
            let w = edge_acc[cv_u32 as usize];
            upper_edges.push((c_u32, cv_u32, w));
            deg[c] += 1;
            deg[cv_u32 as usize] += 1;
        }
    }

    let mut out_indptr = Vec::with_capacity(coarse_n + 1);
    out_indptr.push(0u32);
    let mut total_nnz = 0u32;
    for &d in &deg {
        total_nnz = total_nnz
            .checked_add(d)
            .ok_or_else(|| CoreError::invalid_csr("coarse edge count exceeds u32::MAX"))?;
        out_indptr.push(total_nnz);
    }

    let nnz = total_nnz as usize;
    let mut out_indices = vec![0u32; nnz];
    let mut out_weights = vec![0.0_f64; nnz];
    let mut write_pos = out_indptr[..coarse_n].to_vec();

    // Since `upper_edges` is ordered by ascending `c` and then ascending `cv` (`c < cv`),
    // appending `cv` to row `c` and `c` to row `cv` yields strictly sorted column indices per row.
    for &(c_u32, cv_u32, w) in &upper_edges {
        let c = c_u32 as usize;
        let cv = cv_u32 as usize;
        let p_c = write_pos[c] as usize;
        out_indices[p_c] = cv_u32;
        out_weights[p_c] = w;
        write_pos[c] = (p_c + 1) as u32;

        let p_cv = write_pos[cv] as usize;
        out_indices[p_cv] = c_u32;
        out_weights[p_cv] = w;
        write_pos[cv] = (p_cv + 1) as u32;
    }

    CsrGraph::from_csr(&out_indptr, &out_indices, Some(&out_weights), false, false)
}
