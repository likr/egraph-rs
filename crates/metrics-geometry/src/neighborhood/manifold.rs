use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldLayout2d, Scalar, VisitStamp};

use crate::{build_undirected_adjacency, validate_graph_manifold_layout};

use super::sampled::sample_distinct_vertices;

/// Selects the `k` nearest distinct vertices to `query` in `layout` by exact geodesic distance,
/// breaking ties deterministically by ascending vertex index.
#[doc(hidden)]
pub fn k_nearest_manifold_excluding_self_into<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    query: usize,
    k: usize,
    buf: &mut Vec<(usize, f64)>,
) -> CoreResult<()> {
    let n = layout.len();
    if query >= n {
        return Err(CoreError::index_out_of_bounds(
            query,
            n,
            "k_nearest_manifold_excluding_self_into",
        ));
    }
    buf.clear();
    if k == 0 || n <= 1 {
        return Ok(());
    }
    let k_eff = k.min(n - 1);
    buf.reserve(n - 1);
    for j in 0..n {
        if j == query {
            continue;
        }
        let d = layout.geodesic_distance(query, j)?.to_f64();
        buf.push((j, d));
    }
    let cmp = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
    if k_eff < buf.len() {
        buf.select_nth_unstable_by(k_eff - 1, cmp);
        buf.truncate(k_eff);
    }
    buf.sort_unstable_by(cmp);
    Ok(())
}

/// Computes 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` for a [`ManifoldLayout2d`].
pub fn neighborhood_preservation_1hop_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "neighborhood_preservation_1hop_manifold")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

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
        k_nearest_manifold_excluding_self_into(layout, i, deg, &mut knn_buf)?;
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

/// Computes 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` for a [`ManifoldLayout2d`].
pub fn neighborhood_preservation_2hop_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "neighborhood_preservation_2hop_manifold")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        in_ng2.clear();
        in_ng2.mark(i);
        let mut k = 0usize;

        let nbrs_1: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        for &v in nbrs_1 {
            let vi = v as usize;
            if in_ng2.visit(vi) {
                k += 1;
            }
            let nbrs_2: &[u32] = match &directed_adj {
                Some(adj) => &adj[vi],
                None => graph.neighbors(vi),
            };
            for &w in nbrs_2 {
                if in_ng2.visit(w as usize) {
                    k += 1;
                }
            }
        }

        if k == 0 {
            sum_jaccard += 1.0;
            continue;
        }

        k_nearest_manifold_excluding_self_into(layout, i, k, &mut knn_buf)?;
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

/// Computes deterministic sampled 1-hop neighborhood preservation (`Sampled NP1`) in `[0.0, 1.0]`
/// for a [`ManifoldLayout2d`].
pub fn sampled_neighborhood_preservation_1hop_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_manifold_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_1hop_manifold",
    )?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

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
        k_nearest_manifold_excluding_self_into(layout, i, deg, &mut knn_buf)?;
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

/// Computes deterministic sampled 2-hop neighborhood preservation (`Sampled NP2`) in `[0.0, 1.0]`
/// for a [`ManifoldLayout2d`].
pub fn sampled_neighborhood_preservation_2hop_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_manifold_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_2hop_manifold",
    )?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }

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

    let mut eval_vertex = |i: usize| -> CoreResult<()> {
        in_ng2.clear();
        in_ng2.mark(i);
        let mut k = 0usize;

        let nbrs_1: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        for &v in nbrs_1 {
            let vi = v as usize;
            if in_ng2.visit(vi) {
                k += 1;
            }
            let nbrs_2: &[u32] = match &directed_adj {
                Some(adj) => &adj[vi],
                None => graph.neighbors(vi),
            };
            for &w in nbrs_2 {
                if in_ng2.visit(w as usize) {
                    k += 1;
                }
            }
        }

        if k == 0 {
            sum_jaccard += 1.0;
            return Ok(());
        }

        k_nearest_manifold_excluding_self_into(layout, i, k, &mut knn_buf)?;
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
