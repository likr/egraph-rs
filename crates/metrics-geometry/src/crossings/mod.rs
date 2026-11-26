//! Edge-crossing metrics built on `egraph-spatial-crossings`'s sweep-line and sampling primitives.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};
use egraph_spatial_crossings::{
    SweepLineCrossings, sampled_crossing_number_edges, torus_crossing_edges_from_edges,
};

pub use egraph_spatial_crossings::{
    CrossingPair, CrossingSummary, TorusCrossing, TorusLayoutView, crossing_edges_hyperbolic,
    crossing_edges_spherical, hyperbolic_crossing_number, spherical_crossing_number,
    torus_crossing_number,
};

use crate::{extract_unique_edges, validate_graph_layout};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;

/// Counts the exact number of non-adjacent crossing edge pairs in `graph` under `layout`.
pub fn crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<usize> {
    validate_graph_layout(graph, layout, "crossing_number")?;
    SweepLineCrossings::count(graph, layout)
}

/// Computes the crossing-angle penalty $\sum \cos^2(\theta)$ over all crossing edge pairs.
///
/// Lower values indicate crossings closer to perpendicular (`0.0` = all perpendicular crossings).
pub fn crossing_angle<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    Ok(crossing_summary(graph, layout)?.angle_cos_sq_sum)
}

/// Computes both exact crossing count and crossing-angle statistics in a single sweep pass.
pub fn crossing_summary<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<CrossingSummary> {
    validate_graph_layout(graph, layout, "crossing_summary")?;
    SweepLineCrossings::summarize(graph, layout)
}

/// Returns all non-adjacent crossing edge pairs in `graph` under `layout`.
pub fn crossing_edges<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<Vec<CrossingPair>> {
    validate_graph_layout(graph, layout, "crossing_edges")?;
    SweepLineCrossings::find_crossings(graph, layout)
}

/// Computes crossing count directly from precomputed [`CrossingPair`]s.
#[inline]
pub fn crossing_number_from_pairs(pairs: &[CrossingPair]) -> usize {
    pairs.len()
}

/// Computes the crossing-angle sum $\sum \cos^2(\theta)$ directly from precomputed [`CrossingPair`]s.
#[inline]
pub fn crossing_angle_from_pairs(pairs: &[CrossingPair]) -> f64 {
    pairs.iter().map(|p| p.cos_sq).sum()
}

/// Finds all crossing edge sub-segment pairs for `graph` laid out on a 2D torus
/// (accepting `&Layout2d<S>` or `&ManifoldLayout2d<S>` of kind [`ManifoldKind::Torus2d`]).
pub fn crossing_edges_torus<W, S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<Vec<TorusCrossing>> {
    let coords = layout.torus_layout()?;
    validate_graph_layout(graph, coords, "crossing_edges_torus")?;
    let edges = extract_unique_edges(graph);
    torus_crossing_edges_from_edges(&edges, layout)
}

/// Finds all crossing edge sub-segment pairs for an explicit edge slice on a 2D torus
/// (accepting `&Layout2d<S>` or `&ManifoldLayout2d<S>` of kind [`ManifoldKind::Torus2d`]).
pub fn crossing_edges_torus_from_edges<S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    edges: &[(u32, u32)],
    layout: &L,
) -> CoreResult<Vec<TorusCrossing>> {
    torus_crossing_edges_from_edges(edges, layout)
}

/// Estimates total edge crossings in $O(\text{num\_samples})$ time using deterministic sampling.
pub fn sampled_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    sampled_crossing_number_with_rng(graph, layout, num_samples, &mut rng)
}

/// RNG-injecting variant of [`sampled_crossing_number`].
pub fn sampled_crossing_number_with_rng<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "sampled_crossing_number")?;
    let edges = extract_unique_edges(graph);
    sampled_crossing_number_edges(&edges, layout, num_samples, rng)
}

pub(super) fn max_crossing_pairs_from_edges(num_nodes: usize, edges: &[(u32, u32)]) -> u128 {
    let mut inc = vec![0_u128; num_nodes];
    let mut keys = Vec::with_capacity(edges.len());
    let mut m = 0_u128;
    for &(u, v) in edges {
        if u == v {
            continue;
        }
        let ui = u as usize;
        let vi = v as usize;
        if ui >= num_nodes || vi >= num_nodes {
            continue;
        }
        m += 1;
        inc[ui] += 1;
        inc[vi] += 1;
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        keys.push(((a as u64) << 32) | (b as u64));
    }
    if m < 2 {
        return 0;
    }
    let total_pairs = (m * (m - 1)) / 2;
    let adj_pairs: u128 = inc
        .into_iter()
        .map(|d| if d >= 2 { (d * (d - 1)) / 2 } else { 0 })
        .sum();
    keys.sort_unstable();
    let mut shared_both = 0_u128;
    let mut run = 1_u128;
    for i in 1..keys.len() {
        if keys[i] == keys[i - 1] {
            run += 1;
        } else {
            if run >= 2 {
                shared_both += (run * (run - 1)) / 2;
            }
            run = 1;
        }
    }
    if run >= 2 {
        shared_both += (run * (run - 1)) / 2;
    }
    (total_pairs + shared_both).saturating_sub(adj_pairs)
}

/// Computes the exact maximum number of non-adjacent edge pairs $c_{\max}$ in `graph`
/// using `u128` combinatorics (safe on 32-bit `wasm32` and accounting for reciprocal/parallel
/// edges while ignoring self-loops).
pub fn max_crossing_pairs<W>(graph: &CsrGraph<W>) -> u128 {
    let edges = extract_unique_edges(graph);
    max_crossing_pairs_from_edges(graph.num_nodes(), &edges)
}

#[inline]
fn crosslessness_from_cmax(crossings: f64, c_max: u128) -> f64 {
    if c_max == 0 {
        return 1.0;
    }
    let ratio = (crossings / (c_max as f64)).clamp(0.0, 1.0);
    (1.0 - ratio.sqrt()).clamp(0.0, 1.0)
}

/// Computes DRGraph Eq. (17) crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$ from a
/// precomputed crossing count `crossings` without re-running sweep-line detection.
#[inline]
pub fn crosslessness_from_count<W>(graph: &CsrGraph<W>, crossings: f64) -> CoreResult<f64> {
    if !crossings.is_finite() {
        return Err(CoreError::non_finite("crosslessness_from_count::crossings"));
    }
    if crossings < 0.0 {
        return Err(CoreError::invalid_input("crossings must be >= 0.0"));
    }
    let c_max = max_crossing_pairs(graph);
    Ok(crosslessness_from_cmax(crossings, c_max))
}

/// Computes exact DRGraph Eq. (17) crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$.
///
/// Returns `1.0` when $c_{\max} = 0$ (e.g., graphs with $< 2$ edges, stars, or triangles).
pub fn crosslessness<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "crosslessness")?;
    let c_max = max_crossing_pairs(graph);
    if c_max == 0 {
        return Ok(1.0);
    }
    let count = SweepLineCrossings::count(graph, layout)?;
    Ok(crosslessness_from_cmax(count as f64, c_max))
}

/// Estimates DRGraph Eq. (17) crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$
/// using deterministic edge-pair sampling (`sampled_crossing_number`).
pub fn sampled_crosslessness<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "sampled_crosslessness")?;
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    let edges = extract_unique_edges(graph);
    let c_max = max_crossing_pairs_from_edges(graph.num_nodes(), &edges);
    if c_max == 0 {
        return Ok(1.0);
    }
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let est = sampled_crossing_number_edges(&edges, layout, num_samples, &mut rng)?;
    Ok(crosslessness_from_cmax(est, c_max))
}
