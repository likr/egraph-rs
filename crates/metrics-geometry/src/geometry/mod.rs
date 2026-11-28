//! Geometric shape, edge-length, angular, and PRISM layout-preservation metrics.

mod angles;
mod degeneracy;
mod prism;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_spaces;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, MetricSpace, RiemannianLayout, Scalar,
};

use crate::extract_unique_edges;
use crate::spatial_index::validate_graph_layout;

pub use angles::{MinimumAngleSummary, minimum_angle, minimum_angle_summary};
pub use degeneracy::{LayoutDegeneracy, layout_degeneracy, layout_degeneracy_with_index};
pub use prism::{PrismDistortion, prism_distortion, prism_distortion_with_edges};

/// Computes the undirected angle in `[0.0, pi]` radians between 2D vectors `(x1, y1)` and `(x2, y2)`.
///
/// Returns `None` if either vector is non-finite or has zero length. Clamps the cosine
/// dot-product ratio to `[-1.0, 1.0]` so collinear vectors never produce `NaN`.
#[inline]
pub fn edge_angle<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> Option<f64> {
    let ax = x1.to_f64();
    let ay = y1.to_f64();
    let bx = x2.to_f64();
    let by = y2.to_f64();
    if !ax.is_finite() || !ay.is_finite() || !bx.is_finite() || !by.is_finite() {
        return None;
    }
    let norm1 = ax.hypot(ay);
    let norm2 = bx.hypot(by);
    if norm1 == 0.0 || norm2 == 0.0 {
        return None;
    }
    let cos_val = ((ax * bx + ay * by) / (norm1 * norm2)).clamp(-1.0, 1.0);
    let angle = cos_val.acos();
    if angle.is_finite() { Some(angle) } else { None }
}

#[inline]
fn norm_slice_f64(v: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for &x in v {
        if !x.is_finite() {
            return f64::NAN;
        }
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let inv = 1.0 / max_abs;
    let mut sum_sq = 0.0_f64;
    for &x in v {
        let s = x * inv;
        sum_sq += s * s;
    }
    max_abs * sum_sq.sqrt()
}

#[inline]
fn tangent_angle(a: &[f64], b: &[f64]) -> Option<f64> {
    debug_assert_eq!(a.len(), b.len());
    if a.len() == 2 {
        return edge_angle(a[0], a[1], b[0], b[1]);
    }
    let norm_a = norm_slice_f64(a);
    let norm_b = norm_slice_f64(b);
    if !norm_a.is_finite() || !norm_b.is_finite() || norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }
    let mut dot = 0.0_f64;
    for d in 0..a.len() {
        dot += (a[d] / norm_a) * (b[d] / norm_b);
    }
    let angle = dot.clamp(-1.0, 1.0).acos();
    if angle.is_finite() { Some(angle) } else { None }
}

/// Computes the PCA aspect ratio $\sigma_2 / \sigma_1 \in [0.0, 1.0]$ of 2D `layout`.
///
/// Returns `0.0` for layouts with `<= 1` vertex or zero variance along the major axis.
pub fn aspect_ratio<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let n_f64 = n as f64;
    let cx = xs.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let cy = ys.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;

    let mut xx = 0.0_f64;
    let mut xy = 0.0_f64;
    let mut yy = 0.0_f64;
    for i in 0..n {
        let dx = xs[i].to_f64() - cx;
        let dy = ys[i].to_f64() - cy;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }

    let tr = xx + yy;
    let diff = xx - yy;
    let disc = (diff * diff + 4.0 * xy * xy).sqrt();
    let lam1 = 0.5 * (tr + disc);
    let lam2 = (0.5 * (tr - disc)).max(0.0);
    if !lam1.is_finite() || lam1 <= 0.0 {
        return Ok(0.0);
    }
    let ratio = (lam2 / lam1).sqrt();
    if ratio.is_finite() {
        Ok(ratio.clamp(0.0, 1.0))
    } else {
        Ok(0.0)
    }
}

/// Computes the exponential angular-resolution penalty $\sum_{u \in V} \sum_{a < b} \exp(-\theta_{a u b})$
/// over pairs of incident edges sharing a vertex across any [`RiemannianLayout`].
pub fn angular_resolution<W, L: RiemannianLayout + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "angular_resolution")?;
    let n = layout.len();
    let dim = layout.tangent_dim();
    if n == 0 || dim == 0 {
        return Ok(0.0);
    }

    let mut t1 = vec![0.0_f64; dim];
    let mut t2 = vec![0.0_f64; dim];
    let mut sum = 0.0_f64;

    for u in 0..n {
        let nbrs = graph.neighbors(u);
        let deg = nbrs.len();
        for i in 1..deg {
            let v1 = nbrs[i] as usize;
            layout.log_map_into(u, v1, &mut t1)?;
            for &v2_u32 in &nbrs[..i] {
                let v2 = v2_u32 as usize;
                layout.log_map_into(u, v2, &mut t2)?;
                if let Some(angle) = tangent_angle(&t1, &t2) {
                    sum += (-angle).exp();
                }
            }
        }
    }

    Ok(sum)
}

/// Evaluates ideal edge-length preservation $\sum_{(u,v)\in E} ((d(x_u, x_v) - \ell_{uv}) / \ell_{uv})^2$
/// across any [`MetricSpace`].
///
/// Uses `graph.weights()` when `graph.is_weighted()` is true (skipping non-positive/non-finite
/// weights), and falls back to unit length `1.0` for unweighted graphs.
pub fn ideal_edge_lengths<W: Scalar, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "ideal_edge_lengths")?;
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit(graph, layout, 1.0);
    };

    let n = layout.len();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for pos in start..end {
            let v = indices[pos];
            if v == u32_u || (!directed && u32_u >= v) {
                continue;
            }
            let target_len = weights[pos].to_f64();
            if !target_len.is_finite() || target_len <= 0.0 {
                continue;
            }
            let actual = layout.distance(u, v as usize);
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths`] against a uniform `unit_edge_length > 0.0` across any
/// [`MetricSpace`].
pub fn ideal_edge_lengths_with_unit<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "ideal_edge_lengths_with_unit")?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_with_unit::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }

    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let actual = layout.distance(u as usize, v as usize);
        let err = (actual - unit_edge_length) / unit_edge_length;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths`] scaled by `unit_edge_length > 0.0` across any [`MetricSpace`].
///
/// Uses `graph.weights() * unit_edge_length` on weighted graphs (skipping edges whose scaled
/// target length is non-positive or non-finite), and delegates to [`ideal_edge_lengths_with_unit`]
/// on unweighted graphs.
pub fn ideal_edge_lengths_scaled<W: Scalar, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "ideal_edge_lengths_scaled")?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_scaled::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit(graph, layout, unit_edge_length);
    };

    let n = layout.len();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for pos in start..end {
            let v = indices[pos];
            if v == u32_u || (!directed && u32_u >= v) {
                continue;
            }
            let target_len = weights[pos].to_f64() * unit_edge_length;
            if !target_len.is_finite() || target_len <= 0.0 {
                continue;
            }
            let actual = layout.distance(u, v as usize);
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths`] using a flat row-major `N * N` target distance matrix across
/// any [`MetricSpace`].
pub fn ideal_edge_lengths_from_distances<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    distances: &[f64],
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "ideal_edge_lengths_from_distances")?;
    let n = layout.len();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "ideal_edge_lengths_from_distances::distances",
        ));
    }

    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let target_len = distances[ui * n + vi];
        if !target_len.is_finite() || target_len <= 0.0 {
            continue;
        }
        let actual = layout.distance(ui, vi);
        let err = (actual - target_len) / target_len;
        sum += err * err;
    }
    Ok(sum)
}
