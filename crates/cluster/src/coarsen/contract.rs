use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use super::config::CoarsenedGraph;

/// Compacts arbitrary `u32` labels into canonical `0..num_clusters` in first-appearance order
/// without hash maps.
fn compact_cluster_labels(labels: &[u32]) -> (Vec<u32>, Vec<u32>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut node_to_coarse = vec![0u32; n];

    // Use a flat lookup array when labels are bounded; otherwise sort (label, node) pairs.
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &l) in labels.iter().enumerate() {
            let idx = l as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            node_to_coarse[i] = remap[idx];
        }
        let num_clusters = next_id as usize;
        let mut coarse_sizes = vec![0u32; num_clusters];
        for &c in &node_to_coarse {
            coarse_sizes[c as usize] += 1;
        }
        return (node_to_coarse, coarse_sizes, num_clusters);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let key = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == key {
            end += 1;
        }
        groups.push((pairs[start].1, start, end));
        start = end;
    }

    // Order clusters by their smallest (first-appearing) original vertex index.
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let num_clusters = groups.len();
    let mut coarse_sizes = Vec::with_capacity(num_clusters);
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        coarse_sizes.push((g_end - g_start) as u32);
        for &(_, orig_node) in &pairs[g_start..g_end] {
            node_to_coarse[orig_node as usize] = cid_u32;
        }
    }
    (node_to_coarse, coarse_sizes, num_clusters)
}

/// Coarsens an undirected [`CsrGraph`] by contracting vertices sharing a cluster label.
///
/// Arbitrary `labels` values are canonically compacted to `0..num_clusters` in ascending
/// order of first appearance. Parallel inter-cluster edges are summed into a single
/// weighted undirected edge in `CoarsenedGraph::graph`, and intra-cluster edges are omitted.
pub fn coarsen_graph<W: Scalar>(graph: &CsrGraph<W>, labels: &[u32]) -> CoreResult<CoarsenedGraph> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "coarsen_graph requires an undirected graph",
        ));
    }
    let n = graph.num_nodes();
    if labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            labels.len(),
            "coarsen_graph::labels",
        ));
    }
    if n == 0 {
        let empty = CsrGraph::from_csr(&[0], &[], None, false, false)?;
        return Ok(CoarsenedGraph {
            graph: empty,
            node_to_coarse: Vec::new(),
            coarse_sizes: Vec::new(),
            num_clusters: 0,
        });
    }

    let weights_opt = graph.weights();
    if let Some(ws) = weights_opt {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("coarsen_graph::graph::weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "coarsen_graph edge weights must be non-negative",
                ));
            }
        }
    }

    let (node_to_coarse, coarse_sizes, num_clusters) = compact_cluster_labels(labels);

    let mut inter_edges: Vec<(u32, u32, f64)> = Vec::new();
    for u in 0..n {
        let cu = node_to_coarse[u];
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos] as usize;
            // Visit each undirected edge once in canonical endpoint order.
            if v <= u {
                continue;
            }
            let cv = node_to_coarse[v];
            if cu == cv {
                continue;
            }
            let w = match weights_opt {
                Some(ws) => ws[pos].to_f64(),
                None => 1.0,
            };
            let (a, b) = if cu < cv { (cu, cv) } else { (cv, cu) };
            inter_edges.push((a, b, w));
        }
    }

    inter_edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut coarse_pairs: Vec<(u32, u32)> = Vec::new();
    let mut coarse_weights: Vec<f64> = Vec::new();
    for &(a, b, w) in &inter_edges {
        if let Some(&last_pair) = coarse_pairs.last()
            && last_pair == (a, b)
        {
            if let Some(last_w) = coarse_weights.last_mut() {
                *last_w += w;
            }
        } else {
            coarse_pairs.push((a, b));
            coarse_weights.push(w);
        }
    }

    let coarse_graph = CsrGraph::from_edges(
        num_clusters,
        &coarse_pairs,
        Some(&coarse_weights),
        false,
        false,
    )?;

    Ok(CoarsenedGraph {
        graph: coarse_graph,
        node_to_coarse,
        coarse_sizes,
        num_clusters,
    })
}
