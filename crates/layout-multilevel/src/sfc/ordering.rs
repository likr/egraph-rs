use std::cmp::Ordering;
use std::collections::VecDeque;

use egraph_clustering::{CoarseLouvainConfig, louvain_coarsen};
use egraph_core::{CoreResult, CsrGraph, Scalar};

pub(super) fn canonical_undirected_graph<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<CsrGraph<f64>> {
    let n = graph.num_nodes();
    let weights_opt = graph.weights();
    let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());

    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            if !graph.is_directed() && u32_u > v {
                continue;
            }
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            if w <= 0.0 {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            raw.push((a, b, w));
        }
    }

    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut pairs = Vec::with_capacity(raw.len());
    let mut weights = Vec::with_capacity(raw.len());
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
        pairs.push((a, b));
        weights.push(sum_w / (cnt as f64));
    }

    CsrGraph::from_edges(n, &pairs, Some(&weights), false, false)
}

pub(super) fn compute_community_vertex_order(
    graph: &CsrGraph<f64>,
    seed: u64,
) -> CoreResult<Vec<u32>> {
    let n = graph.num_nodes();
    let hier = louvain_coarsen(
        graph,
        &CoarseLouvainConfig {
            seed,
            ..Default::default()
        },
    )?;

    let num_clusters = hier.coarsest.num_clusters;
    let coarse_graph = &hier.coarsest.graph;
    let coarse_sizes = &hier.coarsest.coarse_sizes;
    let node_to_coarsest = &hier.coarsest.node_to_coarse;
    let node_to_fine_cid = hier
        .levels
        .first()
        .map(|lvl| lvl.coarse.node_to_coarse.as_slice())
        .unwrap_or(node_to_coarsest.as_slice());

    // 1. Order coarsest clusters via BFS prioritizing largest clusters and strongest inter-cluster
    // edges so strongly connected communities sit adjacent along the space-filling curve.
    let mut cluster_seeds: Vec<usize> = (0..num_clusters).collect();
    cluster_seeds.sort_unstable_by(|&a, &b| {
        coarse_sizes[b]
            .cmp(&coarse_sizes[a])
            .then_with(|| a.cmp(&b))
    });

    let mut cluster_visited = vec![false; num_clusters];
    let mut coarse_rank = vec![0usize; num_clusters];
    let mut next_rank = 0usize;
    let mut c_queue = VecDeque::new();
    let mut c_nbrs: Vec<(u32, f64)> = Vec::new();
    let coarse_weights = coarse_graph.weights();

    for &seed_cid in &cluster_seeds {
        if cluster_visited[seed_cid] {
            continue;
        }
        cluster_visited[seed_cid] = true;
        c_queue.push_back(seed_cid);
        while let Some(c) = c_queue.pop_front() {
            coarse_rank[c] = next_rank;
            next_rank += 1;
            c_nbrs.clear();
            let start = coarse_graph.indptr()[c] as usize;
            let end = coarse_graph.indptr()[c + 1] as usize;
            for pos in start..end {
                let nc = coarse_graph.indices()[pos];
                if !cluster_visited[nc as usize] {
                    let w = coarse_weights.map_or(1.0, |ws| ws[pos]);
                    c_nbrs.push((nc, w));
                }
            }
            c_nbrs.sort_unstable_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| coarse_sizes[b.0 as usize].cmp(&coarse_sizes[a.0 as usize]))
                    .then_with(|| a.0.cmp(&b.0))
            });
            for &(nc, _) in &c_nbrs {
                let nci = nc as usize;
                if !cluster_visited[nci] {
                    cluster_visited[nci] = true;
                    c_queue.push_back(nci);
                }
            }
        }
    }

    // 2. Group vertices by (coarse_rank[coarsest_cid], fine_cid), then order within each
    // coarsest community via intra-community BFS starting from the highest-degree node.
    let mut grouped: Vec<(usize, u32, usize, u32)> = (0..n)
        .map(|u| {
            let c_cid = node_to_coarsest[u] as usize;
            let f_cid = node_to_fine_cid[u];
            (coarse_rank[c_cid], f_cid, graph.degree(u), u as u32)
        })
        .collect();
    grouped.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.3.cmp(&b.3))
    });

    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut v_queue = VecDeque::new();
    let mut v_nbrs: Vec<(u32, f64)> = Vec::new();
    let fine_weights = graph.weights();

    let mut start = 0usize;
    while start < n {
        let cr = grouped[start].0;
        let mut end = start + 1;
        while end < n && grouped[end].0 == cr {
            end += 1;
        }
        for &(_, _, _, seed_u) in &grouped[start..end] {
            let su = seed_u as usize;
            if visited[su] {
                continue;
            }
            visited[su] = true;
            v_queue.push_back(su);
            while let Some(u) = v_queue.pop_front() {
                order.push(u as u32);
                let cu = node_to_coarsest[u];
                v_nbrs.clear();
                let s = graph.indptr()[u] as usize;
                let e = graph.indptr()[u + 1] as usize;
                for pos in s..e {
                    let v = graph.indices()[pos];
                    let vi = v as usize;
                    if !visited[vi] && node_to_coarsest[vi] == cu {
                        let w = fine_weights.map_or(1.0, |ws| ws[pos]);
                        v_nbrs.push((v, w));
                    }
                }
                v_nbrs.sort_unstable_by(|a, b| match b.1.total_cmp(&a.1) {
                    Ordering::Equal => graph
                        .degree(b.0 as usize)
                        .cmp(&graph.degree(a.0 as usize))
                        .then_with(|| a.0.cmp(&b.0)),
                    other => other,
                });
                for &(v, _) in &v_nbrs {
                    let vi = v as usize;
                    if !visited[vi] {
                        visited[vi] = true;
                        v_queue.push_back(vi);
                    }
                }
            }
        }
        start = end;
    }

    Ok(order)
}
