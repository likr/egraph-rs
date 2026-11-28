use std::collections::HashSet;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, EuclideanLayout, MetricSpace, VisitStamp,
};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use crate::build_undirected_adjacency;
use crate::spatial_index::{try_build_spatial_index, validate_graph_layout, validate_index_layout};

use super::{collect_2hop_count, k_nearest_metric_excluding_self_into};

/// Samples `k` distinct vertex indices from `0..n` in ascending order without replacement.
///
/// Uses Robert Floyd's $O(k)$ sampling algorithm when $k \le n / 4$ to avoid $O(N)$ auxiliary
/// allocations on multi-million-node graphs, and partial Fisher-Yates when $k > n / 4$.
pub(super) fn sample_distinct_vertices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    debug_assert!(k > 0 && k < n);
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut samples = if k <= n / 4 {
        let mut chosen = HashSet::with_capacity(k);
        let mut out = Vec::with_capacity(k);
        for j in (n - k)..n {
            let t = rng.gen_range_usize(j + 1);
            let picked = if chosen.insert(t) {
                t
            } else {
                chosen.insert(j);
                j
            };
            out.push(picked);
        }
        out
    } else {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + rng.gen_range_usize(n - i);
            perm.swap(i, j);
        }
        perm.truncate(k);
        perm
    };
    samples.sort_unstable();
    samples
}

/// Computes deterministic sampled 1-hop neighborhood preservation (`Sampled NP1`) in `[0.0, 1.0]`
/// across any [`MetricSpace`].
///
/// Samples `min(num_samples, n)` distinct vertices without replacement using `seed` and computes
/// the micro-averaged directed-pair Jaccard similarity
/// $\frac{\sum_{u \in S} |N_G(u) \cap N_D(u)|}{\sum_{u \in S} |N_G(u) \cup N_D(u)|}$.
/// When `num_samples >= n`, all `0..n` vertices are evaluated in order for bit-exact parity
/// with [`super::neighborhood_preservation_1hop`]. Returns `0.0` for empty, single-vertex, or
/// edgeless graphs, or when all sampled vertices are isolated.
pub fn sampled_neighborhood_preservation_1hop<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_neighborhood_preservation_1hop")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return sampled_np1_with_index_inner(graph, n, index.as_ref(), num_samples, seed);
    }
    sampled_np1_metric_inner(graph, layout, num_samples, seed)
}

/// Computes [`sampled_neighborhood_preservation_1hop`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sampled_neighborhood_preservation_1hop_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_1hop_with_index",
    )?;
    validate_index_layout(
        index,
        layout,
        "sampled_neighborhood_preservation_1hop_with_index",
    )?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    sampled_np1_with_index_inner(graph, n, index, num_samples, seed)
}

fn sampled_np1_with_index_inner<W, I: NearestNeighborIndex + ?Sized>(
    graph: &CsrGraph<W>,
    n: usize,
    index: &I,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    let mut eval_vertex = |i: usize| -> CoreResult<()> {
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        in_ng.clear();
        in_ng.mark(i);
        let mut deg = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg += 1;
            }
        }
        if deg == 0 {
            return Ok(());
        }
        cup += deg;
        index.query_knn_excluding_self_with_scratch(i, deg, &mut kd_scratch, &mut knn_buf)?;
        for &(j, _) in &knn_buf {
            if in_ng.is_visited(j) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
        Ok(())
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i)?;
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i)?;
            }
        }
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

fn sampled_np1_metric_inner<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let n = layout.len();
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    let mut eval_vertex = |i: usize| {
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        in_ng.clear();
        in_ng.mark(i);
        let mut deg = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg += 1;
            }
        }
        if deg == 0 {
            return;
        }
        cup += deg;
        k_nearest_metric_excluding_self_into(layout, i, deg, &mut knn_buf);
        for &(j, _) in &knn_buf {
            if in_ng.is_visited(j) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i);
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i);
            }
        }
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

/// Computes deterministic sampled 2-hop neighborhood preservation (`Sampled NP2`) in `[0.0, 1.0]`
/// across any [`MetricSpace`].
///
/// Samples `min(num_samples, n)` distinct vertices without replacement using `seed` and averages
/// node-wise 2-hop Jaccard coefficients $|N_G(u, 2) \cap N_D(u, 2)| / |N_G(u, 2) \cup N_D(u, 2)|$
/// (defined as `1.0` when $N_G(u, 2) = \emptyset$, and `0.0` when `n == 0`).
/// When `num_samples >= n`, all `0..n` vertices are evaluated in order for bit-exact parity
/// with [`super::neighborhood_preservation_2hop`].
pub fn sampled_neighborhood_preservation_2hop<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_neighborhood_preservation_2hop")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return sampled_np2_with_index_inner(graph, n, index.as_ref(), num_samples, seed);
    }
    sampled_np2_metric_inner(graph, layout, num_samples, seed)
}

/// Computes [`sampled_neighborhood_preservation_2hop`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sampled_neighborhood_preservation_2hop_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_2hop_with_index",
    )?;
    validate_index_layout(
        index,
        layout,
        "sampled_neighborhood_preservation_2hop_with_index",
    )?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    sampled_np2_with_index_inner(graph, n, index, num_samples, seed)
}

fn sampled_np2_with_index_inner<W, I: NearestNeighborIndex + ?Sized>(
    graph: &CsrGraph<W>,
    n: usize,
    index: &I,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sample_count = num_samples.min(n);
    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    let mut eval_vertex = |i: usize| -> CoreResult<()> {
        let k = collect_2hop_count(graph, directed_adj.as_deref(), i, &mut in_ng2);
        if k == 0 {
            sum_jaccard += 1.0;
            return Ok(());
        }

        index.query_knn_excluding_self_with_scratch(i, k, &mut kd_scratch, &mut knn_buf)?;
        let mut intersection = 0usize;
        for &(j, _) in &knn_buf {
            if in_ng2.is_visited(j) {
                intersection += 1;
            }
        }
        let union = k + knn_buf.len() - intersection;
        sum_jaccard += intersection as f64 / union as f64;
        Ok(())
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i)?;
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i)?;
            }
        }
    }

    Ok(sum_jaccard / sample_count as f64)
}

fn sampled_np2_metric_inner<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let n = layout.len();
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sample_count = num_samples.min(n);
    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    let mut eval_vertex = |i: usize| {
        let k = collect_2hop_count(graph, directed_adj.as_deref(), i, &mut in_ng2);
        if k == 0 {
            sum_jaccard += 1.0;
            return;
        }

        k_nearest_metric_excluding_self_into(layout, i, k, &mut knn_buf);
        let mut intersection = 0usize;
        for &(j, _) in &knn_buf {
            if in_ng2.is_visited(j) {
                intersection += 1;
            }
        }
        let union = k + knn_buf.len() - intersection;
        sum_jaccard += intersection as f64 / union as f64;
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i);
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i);
            }
        }
    }

    Ok(sum_jaccard / sample_count as f64)
}
