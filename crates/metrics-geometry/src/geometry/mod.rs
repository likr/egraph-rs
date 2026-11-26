//! Geometric shape, edge-length, angular, and PRISM layout-preservation metrics.

mod angles_degeneracy;
mod prism;
#[cfg(test)]
mod tests;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};

use crate::{extract_unique_edges, validate_graph_layout};

pub use angles_degeneracy::{
    LayoutDegeneracy, MinimumAngleSummary, layout_degeneracy, minimum_angle, minimum_angle_summary,
};
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

/// Computes the PCA aspect ratio $\sigma_2 / \sigma_1 \in [0.0, 1.0]$ of `layout`.
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
/// over pairs of incident edges sharing a vertex.
pub fn angular_resolution<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "angular_resolution")?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let x0 = xs[u].to_f64();
        let y0 = ys[u].to_f64();
        let nbrs = graph.neighbors(u);
        let deg = nbrs.len();
        for i in 1..deg {
            let v1 = nbrs[i] as usize;
            let dx1 = xs[v1].to_f64() - x0;
            let dy1 = ys[v1].to_f64() - y0;
            for &v2_u32 in &nbrs[..i] {
                let v2 = v2_u32 as usize;
                let dx2 = xs[v2].to_f64() - x0;
                let dy2 = ys[v2].to_f64() - y0;
                if let Some(angle) = edge_angle(dx1, dy1, dx2, dy2) {
                    sum += (-angle).exp();
                }
            }
        }
    }

    Ok(sum)
}

/// Evaluates ideal edge-length preservation $\sum_{(u,v)\in E} ((\|x_u - x_v\| - \ell_{uv}) / \ell_{uv})^2$.
///
/// Uses `graph.weights()` when `graph.is_weighted()` is true (skipping non-positive/non-finite
/// weights), and falls back to unit length `1.0` for unweighted graphs.
pub fn ideal_edge_lengths<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "ideal_edge_lengths")?;
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit(graph, layout, 1.0);
    };

    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let xu = xs[u].to_f64();
        let yu = ys[u].to_f64();
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
            let vi = v as usize;
            let actual = (xu - xs[vi].to_f64()).hypot(yu - ys[vi].to_f64());
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths`] against a uniform `unit_edge_length > 0.0`.
pub fn ideal_edge_lengths_with_unit<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
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

    let xs = layout.x();
    let ys = layout.y();
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let actual = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        let err = (actual - unit_edge_length) / unit_edge_length;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths`] using a flat row-major `N * N` target distance matrix.
pub fn ideal_edge_lengths_from_distances<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
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

    let xs = layout.x();
    let ys = layout.y();
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let target_len = distances[ui * n + vi];
        if !target_len.is_finite() || target_len <= 0.0 {
            continue;
        }
        let actual = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        let err = (actual - target_len) / target_len;
        sum += err * err;
    }
    Ok(sum)
}
