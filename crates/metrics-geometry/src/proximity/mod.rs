//! Spatial proximity metrics (`gabriel_graph_property`, `node_resolution`, and Eades et al.
//! 2017 unweighted shape-based `S_RNG` / `S_GG`) across arbitrary [`MetricSpace`] layouts,
//! accelerated via [`NearestNeighborIndex`] for Euclidean spaces.

mod euclidean_pruning;
pub(crate) mod exact_predicates;
mod metric_fallback;
mod sector_pruning;
#[cfg(test)]
mod tests;

use egraph_core::{CoreResult, CsrGraph, EuclideanLayout, MetricSpace, VisitStamp};
use egraph_spatial_trees::NearestNeighborIndex;

use crate::build_undirected_adjacency;
use crate::spatial_index::{try_build_spatial_index, validate_graph_layout, validate_index_layout};

use euclidean_pruning::{
    gabriel_graph_property_euclidean_with_index, node_resolution_euclidean_with_index,
    proximity_edges_euclidean_with_index,
};
use metric_fallback::{
    gabriel_graph_edges_metric, gabriel_graph_property_metric, node_resolution_metric,
    relative_neighborhood_edges_metric,
};
use sector_pruning::ProximityMode;

/// Evaluates violations of the Gabriel graph condition across all unique edges yielded by
/// `extract_unique_edges(graph)` on any [`MetricSpace`].
///
/// For Euclidean layouts, accelerates diametral-ball queries via [`NearestNeighborIndex`]. For
/// `ManifoldLayout2d`, uses exact geodesic midpoints and manifold distances. On directed graphs,
/// every stored directed edge `(u, v)` is evaluated independently, and zero-length edges are
/// skipped.
pub fn gabriel_graph_property<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property")?;
    if layout.len() <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return gabriel_graph_property_metric_or_euclidean(graph, layout, index.as_ref());
    }
    gabriel_graph_property_metric(graph, layout)
}

fn gabriel_graph_property_metric_or_euclidean<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &dyn NearestNeighborIndex,
) -> CoreResult<f64> {
    let any: &dyn std::any::Any = layout;
    if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f64>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f32>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f64>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f32>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f64>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f32>>() {
        return gabriel_graph_property_euclidean_with_index(graph, l, index);
    }
    gabriel_graph_property_metric(graph, layout)
}

/// Computes [`gabriel_graph_property`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn gabriel_graph_property_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property_with_index")?;
    validate_index_layout(index, layout, "gabriel_graph_property_with_index")?;
    if layout.len() <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    gabriel_graph_property_euclidean_with_index(graph, layout, index)
}

/// Computes the one-sided hinge node-resolution violation penalty across any [`MetricSpace`]:
/// $\sum_{0 \le i < j < N} \max\left(0, 1 - \frac{d(x_i, x_j)}{r \cdot d_{\max}}\right)^2$,
/// where $r = 1 / \sqrt{N}$ and $d_{\max}$ is the exact diameter of `layout`.
pub fn node_resolution<L: MetricSpace + 'static>(layout: &L) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        let any: &dyn std::any::Any = layout;
        if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f64>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f32>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f64>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f32>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f64>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f32>>() {
            return node_resolution_euclidean_with_index(l, index.as_ref());
        }
    }
    Ok(node_resolution_metric(layout))
}

/// Computes [`node_resolution`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn node_resolution_with_index<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    layout.validate_finite()?;
    validate_index_layout(index, layout, "node_resolution_with_index")?;
    if layout.len() <= 1 {
        return Ok(0.0);
    }
    node_resolution_euclidean_with_index(layout, index)
}

fn dispatch_proximity_edges<L: MetricSpace + 'static>(
    layout: &L,
    mode: ProximityMode,
) -> CoreResult<Vec<(u32, u32)>> {
    if let Some(index) = try_build_spatial_index(layout)? {
        let any: &dyn std::any::Any = layout;
        if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f64>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout2d<f32>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f64>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
        if let Some(l) = any.downcast_ref::<egraph_core::Layout3d<f32>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
        if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f64>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
        if let Some(l) = any.downcast_ref::<egraph_core::LayoutNd<f32>>() {
            return proximity_edges_euclidean_with_index(l, index.as_ref(), mode);
        }
    }
    match mode {
        ProximityMode::RelativeNeighborhood => Ok(relative_neighborhood_edges_metric(layout)),
        ProximityMode::Gabriel => gabriel_graph_edges_metric(layout),
    }
}

/// Extracts exact undirected Relative Neighborhood Graph (`RNG`) edges `(u, v)` (`u < v`)
/// across any [`MetricSpace`].
pub fn relative_neighborhood_edges<L: MetricSpace + 'static>(
    layout: &L,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    dispatch_proximity_edges(layout, ProximityMode::RelativeNeighborhood)
}

/// Extracts [`relative_neighborhood_edges`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn relative_neighborhood_edges_with_index<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    layout: &L,
    index: &I,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_index_layout(index, layout, "relative_neighborhood_edges_with_index")?;
    proximity_edges_euclidean_with_index(layout, index, ProximityMode::RelativeNeighborhood)
}

/// Extracts exact undirected Gabriel Graph (`GG`) edges `(u, v)` (`u < v`)
/// across any [`MetricSpace`].
pub fn gabriel_graph_edges<L: MetricSpace + 'static>(layout: &L) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    dispatch_proximity_edges(layout, ProximityMode::Gabriel)
}

/// Extracts [`gabriel_graph_edges`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn gabriel_graph_edges_with_index<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    layout: &L,
    index: &I,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_index_layout(index, layout, "gabriel_graph_edges_with_index")?;
    proximity_edges_euclidean_with_index(layout, index, ProximityMode::Gabriel)
}

/// Computes Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
/// (`S_RNG`) in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph
/// 1-hop neighborhoods and spatial `RNG(layout)` across any [`MetricSpace`].
///
/// Note: This implements the classical unweighted `S_RNG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dRNG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcRNG`. Input graphs are deduplicated as undirected simple neighborhoods.
/// Returns `0.0` for `N == 0`, and scores `N == 1` or mutually isolated vertices as `1.0`.
pub fn relative_neighborhood_preservation<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "relative_neighborhood_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges = dispatch_proximity_edges(layout, ProximityMode::RelativeNeighborhood)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes [`relative_neighborhood_preservation`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn relative_neighborhood_preservation_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    validate_graph_layout(
        graph,
        layout,
        "relative_neighborhood_preservation_with_index",
    )?;
    validate_index_layout(
        index,
        layout,
        "relative_neighborhood_preservation_with_index",
    )?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges =
        proximity_edges_euclidean_with_index(layout, index, ProximityMode::RelativeNeighborhood)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`)
/// in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph 1-hop
/// neighborhoods and spatial `GG(layout)` across any [`MetricSpace`].
///
/// Note: This implements the classical unweighted `S_GG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dGG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcGG`. Input graphs are deduplicated as undirected simple neighborhoods.
/// Returns `0.0` for `N == 0`, and scores `N == 1` or mutually isolated vertices as `1.0`.
pub fn gabriel_graph_preservation<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = dispatch_proximity_edges(layout, ProximityMode::Gabriel)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &gg_edges))
}

/// Computes [`gabriel_graph_preservation`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn gabriel_graph_preservation_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation_with_index")?;
    validate_index_layout(index, layout, "gabriel_graph_preservation_with_index")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = proximity_edges_euclidean_with_index(layout, index, ProximityMode::Gabriel)?;
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
