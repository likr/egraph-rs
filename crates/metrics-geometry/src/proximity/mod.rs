//! Spatial proximity metrics (`gabriel_graph_property`, `node_resolution`, and Eades et al.
//! 2017 unweighted shape-based `S_RNG` / `S_GG`) accelerated via `StaticKdTree<2, f64>`.

pub(crate) mod exact_predicates;
mod nd;
mod sector_pruning;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nd;

pub use nd::{
    gabriel_graph_edges_nd, gabriel_graph_edges_nd_with_index, gabriel_graph_preservation_nd,
    gabriel_graph_preservation_nd_with_index, gabriel_graph_property_nd,
    gabriel_graph_property_nd_with_index, node_resolution_nd, node_resolution_nd_with_index,
    relative_neighborhood_edges_nd, relative_neighborhood_edges_nd_with_index,
    relative_neighborhood_preservation_nd, relative_neighborhood_preservation_nd_with_index,
};

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};
use egraph_spatial_trees::StaticKdTree;

use crate::{
    build_kdtree_f64, build_undirected_adjacency, extract_unique_edges, validate_graph_layout,
    validate_tree_layout,
};

use exact_predicates::euclidean_diameter_2d;
use sector_pruning::{ProximityMode, proximity_edges_with_tree};

/// Evaluates violations of the Gabriel graph condition across all unique edges yielded by
/// `extract_unique_edges(graph)`.
///
/// For each edge `(u, v)`, computes the disk with diameter $\overline{uv}$ (center $c_{uv}$,
/// radius $r_{uv}$) and accumulates $\max(0, r_{uv} - \|p_i - c_{uv}\|)^2$ over vertices
/// inside the disk via [`StaticKdTree::within_radius_into`]. On directed graphs, every stored
/// directed edge `(u, v)` is evaluated independently (so reciprocal pairs `(u, v)` and `(v, u)`
/// are both summed), and zero-length edges (`r_{uv} <= 0.0`) are skipped.
pub fn gabriel_graph_property<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property")?;
    if layout.len() <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_property_with_tree(graph, layout, &tree)
}

/// Computes [`gabriel_graph_property`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_property_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property_with_tree")?;
    validate_tree_layout(tree, layout, "gabriel_graph_property_with_tree")?;
    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let edges = extract_unique_edges(graph);
    let mut disk_buf: Vec<(usize, f64)> = Vec::new();
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let x1 = xs[ui].to_f64();
        let y1 = ys[ui].to_f64();
        let x2 = xs[vi].to_f64();
        let y2 = ys[vi].to_f64();
        let r = 0.5 * (x1 - x2).hypot(y1 - y2);
        if r <= 0.0 {
            continue;
        }
        let cx = 0.5 * (x1 + x2);
        let cy = 0.5 * (y1 + y2);
        tree.within_radius_into(&[cx, cy], r, &mut disk_buf)?;
        for &(idx, d_sq) in &disk_buf {
            if idx == ui || idx == vi {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < r {
                let viol = r - dist;
                violation_sum += viol * viol;
            }
        }
    }

    Ok(violation_sum)
}

/// Computes the one-sided hinge node-resolution violation penalty:
/// $\sum_{0 \le i < j < N} \max\left(0, 1 - \frac{\|x_i - x_j\|}{r \cdot d_{\max}}\right)^2$,
/// where $r = 1 / \sqrt{N}$ and $d_{\max}$ is the exact Euclidean diameter of `layout`.
pub fn node_resolution<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    node_resolution_with_tree(layout, &tree)
}

/// Computes [`node_resolution`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn node_resolution_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "node_resolution_with_tree")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let d_max = euclidean_diameter_2d(layout);
    if d_max <= 0.0 {
        // Completely collapsed layout: every pair sits at distance 0 < ideal spacing.
        let num_pairs = (n as f64) * ((n - 1) as f64) * 0.5;
        return Ok(num_pairs);
    }

    let threshold = d_max / (n as f64).sqrt();
    let xs = layout.x();
    let ys = layout.y();
    let mut buf: Vec<(usize, f64)> = Vec::new();
    let mut sum = 0.0_f64;

    for i in 0..n {
        let q = [xs[i].to_f64(), ys[i].to_f64()];
        tree.within_radius_into(&q, threshold, &mut buf)?;
        for &(j, d_sq) in &buf {
            if j <= i {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < threshold {
                let hinge = 1.0 - dist / threshold;
                sum += hinge * hinge;
            }
        }
    }

    Ok(sum)
}

/// Extracts exact undirected Relative Neighborhood Graph (`RNG`) edges `(u, v)` (`u < v`)
/// over the 2D coordinates in `layout`.
pub fn relative_neighborhood_edges<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<Vec<(u32, u32)>> {
    let tree = build_kdtree_f64(layout)?;
    relative_neighborhood_edges_with_tree(layout, &tree)
}

/// Extracts [`relative_neighborhood_edges`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn relative_neighborhood_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "relative_neighborhood_edges_with_tree")?;
    proximity_edges_with_tree(layout, tree, ProximityMode::RelativeNeighborhood)
}

/// Extracts exact undirected Gabriel Graph (`GG`) edges `(u, v)` (`u < v`)
/// over the 2D coordinates in `layout`.
pub fn gabriel_graph_edges<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<Vec<(u32, u32)>> {
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_edges_with_tree(layout, &tree)
}

/// Extracts [`gabriel_graph_edges`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "gabriel_graph_edges_with_tree")?;
    proximity_edges_with_tree(layout, tree, ProximityMode::Gabriel)
}

/// Computes Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
/// (`S_RNG`) in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph
/// 1-hop neighborhoods and spatial `RNG(layout)`.
///
/// Note: This implements the classical unweighted `S_RNG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dRNG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcRNG` (both deferred pending primary specification PDFs). Input graphs
/// are deduplicated as undirected simple neighborhoods (directed graphs are symmetrized and
/// self-loops are excluded). Returns `0.0` for `N == 0`, and scores `N == 1` or mutually
/// isolated vertices (`deg_G == 0 && deg_RNG == 0`) as `1.0` (`0/0` convention).
pub fn relative_neighborhood_preservation<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "relative_neighborhood_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    relative_neighborhood_preservation_with_tree(graph, layout, &tree)
}

/// Computes [`relative_neighborhood_preservation`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn relative_neighborhood_preservation_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(
        graph,
        layout,
        "relative_neighborhood_preservation_with_tree",
    )?;
    validate_tree_layout(tree, layout, "relative_neighborhood_preservation_with_tree")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges = proximity_edges_with_tree(layout, tree, ProximityMode::RelativeNeighborhood)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`)
/// in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph 1-hop
/// neighborhoods and spatial `GG(layout)`.
///
/// Note: This implements the classical unweighted `S_GG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dGG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcGG` (both deferred pending primary specification PDFs). Input graphs
/// are deduplicated as undirected simple neighborhoods (directed graphs are symmetrized and
/// self-loops are excluded). Returns `0.0` for `N == 0`, and scores `N == 1` or mutually
/// isolated vertices (`deg_G == 0 && deg_GG == 0`) as `1.0` (`0/0` convention).
pub fn gabriel_graph_preservation<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_preservation_with_tree(graph, layout, &tree)
}

/// Computes [`gabriel_graph_preservation`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_preservation_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation_with_tree")?;
    validate_tree_layout(tree, layout, "gabriel_graph_preservation_with_tree")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = proximity_edges_with_tree(layout, tree, ProximityMode::Gabriel)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &gg_edges))
}

/// Eades et al. (2017) Eq. (2) unweighted Mean Jaccard Similarity (`MJS`) between graph 1-hop
/// and proximity edges.
pub(super) fn mean_jaccard_graph_vs_edges<W>(
    graph: &CsrGraph<W>,
    prox_edges: &[(u32, u32)],
) -> f64 {
    let n = graph.num_nodes();
    if n == 0 {
        return 0.0;
    }

    let mut prox_adj = vec![Vec::new(); n];
    for &(u, v) in prox_edges {
        prox_adj[u as usize].push(v);
        prox_adj[v as usize].push(u);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        in_ng.clear();
        in_ng.mark(i);
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        let mut deg_g = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg_g += 1;
            }
        }
        let prox_row = &prox_adj[i];
        let deg_p = prox_row.len();
        if deg_g == 0 && deg_p == 0 {
            sum_jaccard += 1.0;
            continue;
        }
        let mut intersection = 0usize;
        for &v in prox_row {
            if in_ng.is_visited(v as usize) {
                intersection += 1;
            }
        }
        let union = deg_g + deg_p - intersection;
        sum_jaccard += intersection as f64 / union as f64;
    }

    sum_jaccard / n as f64
}
