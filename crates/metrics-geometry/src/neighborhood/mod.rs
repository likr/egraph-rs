//! 1-hop and 2-hop neighborhood preservation metrics across arbitrary [`MetricSpace`] layouts,
//! accelerated by [`NearestNeighborIndex`] for Euclidean spaces.

mod sampled;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_spaces;

use egraph_core::{CoreResult, CsrGraph, EuclideanLayout, MetricSpace, VisitStamp};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use crate::build_undirected_adjacency;
use crate::spatial_index::{try_build_spatial_index, validate_graph_layout, validate_index_layout};

pub use sampled::{
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_with_index,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_with_index,
};

pub(super) fn k_nearest_metric_excluding_self_into<L: MetricSpace + ?Sized>(
    layout: &L,
    query: usize,
    k: usize,
    buf: &mut Vec<(usize, f64)>,
) {
    let n = layout.len();
    buf.clear();
    if k == 0 || n <= 1 {
        return;
    }
    let k_eff = k.min(n - 1);
    buf.reserve(n - 1);
    for j in 0..n {
        if j == query {
            continue;
        }
        buf.push((j, layout.distance(query, j)));
    }
    let cmp = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
    if k_eff < buf.len() {
        buf.select_nth_unstable_by(k_eff - 1, cmp);
        buf.truncate(k_eff);
    }
    buf.sort_unstable_by(cmp);
}

/// Computes 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` across any [`MetricSpace`].
///
/// Compares each vertex's $d_u$ graph neighbors against its $d_u$ nearest spatial neighbors
/// in `layout` via global directed-pair Jaccard similarity:
/// $\frac{\sum_u |N_G(u) \cap N_D(u)|}{\sum_u |N_G(u) \cup N_D(u)|}$.
/// Uses a spatial KD-tree automatically for Euclidean layouts and falls back to exact pairwise
/// [`MetricSpace::distance`] otherwise. Returns `0.0` for empty or edgeless graphs.
pub fn neighborhood_preservation_1hop<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_1hop")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return np1_with_index_inner(graph, n, index.as_ref());
    }
    np1_metric_inner(graph, layout)
}

/// Computes [`neighborhood_preservation_1hop`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn neighborhood_preservation_1hop_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_1hop_with_index")?;
    validate_index_layout(index, layout, "neighborhood_preservation_1hop_with_index")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    np1_with_index_inner(graph, n, index)
}

fn np1_with_index_inner<W, I: NearestNeighborIndex + ?Sized>(
    graph: &CsrGraph<W>,
    n: usize,
    index: &I,
) -> CoreResult<f64> {
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    for i in 0..n {
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
            continue;
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
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

fn np1_metric_inner<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    let n = layout.len();
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    for i in 0..n {
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
            continue;
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
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

/// Computes 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` across any [`MetricSpace`].
///
/// For each vertex $u$, computes its 2-hop graph neighborhood $N_G(u, 2)$ (vertices at
/// shortest-path distance $1$ or $2$), finds its $|N_G(u, 2)|$ nearest spatial neighbors
/// $N_D(u, 2)$ in `layout`, and averages node-wise Jaccard coefficients
/// $|N_G(u, 2) \cap N_D(u, 2)| / |N_G(u, 2) \cup N_D(u, 2)|$ (defined as `1.0` when
/// $N_G(u, 2) = \emptyset$, and `0.0` when `n == 0`).
pub fn neighborhood_preservation_2hop<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_2hop")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return np2_with_index_inner(graph, n, index.as_ref());
    }
    np2_metric_inner(graph, layout)
}

/// Computes [`neighborhood_preservation_2hop`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn neighborhood_preservation_2hop_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_2hop_with_index")?;
    validate_index_layout(index, layout, "neighborhood_preservation_2hop_with_index")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    np2_with_index_inner(graph, n, index)
}

fn collect_2hop_count<W>(
    graph: &CsrGraph<W>,
    directed_adj: Option<&[Vec<u32>]>,
    i: usize,
    in_ng2: &mut VisitStamp,
) -> usize {
    in_ng2.clear();
    in_ng2.mark(i);
    let mut k = 0usize;

    let nbrs_1: &[u32] = match directed_adj {
        Some(adj) => &adj[i],
        None => graph.neighbors(i),
    };
    for &v in nbrs_1 {
        let vi = v as usize;
        if in_ng2.visit(vi) {
            k += 1;
        }
        let nbrs_2: &[u32] = match directed_adj {
            Some(adj) => &adj[vi],
            None => graph.neighbors(vi),
        };
        for &w in nbrs_2 {
            if in_ng2.visit(w as usize) {
                k += 1;
            }
        }
    }
    k
}

fn np2_with_index_inner<W, I: NearestNeighborIndex + ?Sized>(
    graph: &CsrGraph<W>,
    n: usize,
    index: &I,
) -> CoreResult<f64> {
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        let k = collect_2hop_count(graph, directed_adj.as_deref(), i, &mut in_ng2);
        if k == 0 {
            sum_jaccard += 1.0;
            continue;
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
    }

    Ok(sum_jaccard / n as f64)
}

fn np2_metric_inner<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    let n = layout.len();
    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        let k = collect_2hop_count(graph, directed_adj.as_deref(), i, &mut in_ng2);
        if k == 0 {
            sum_jaccard += 1.0;
            continue;
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
    }

    Ok(sum_jaccard / n as f64)
}
