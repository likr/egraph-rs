//! Geometric shape, edge-length, angular, and PRISM layout-preservation metrics.

use std::f64::consts::TAU;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};

use crate::{
    build_undirected_adjacency, extract_unique_edges, proximity::gabriel_graph_edges,
    validate_graph_layout,
};

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`) in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MinimumAngleSummary {
    /// A1 score using `atan2(0.0, 0.0) = 0.0` for zero-length incident edges.
    pub a1: f64,
    /// A2 score penalizing any `deg >= 2` vertex with at least one zero-length incident edge to `0.0`.
    pub a2: f64,
}

/// Computes DRGraph Eq. (19) minimum-angle resolution (`A1` and `A2`) in `[0.0, 1.0]`.
///
/// Vertices with effective undirected degree `< 2` receive score `1.0`, while empty graphs
/// (`N == 0`) return `MinimumAngleSummary { a1: 0.0, a2: 0.0 }`.
pub fn minimum_angle_summary<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<MinimumAngleSummary> {
    validate_graph_layout(graph, layout, "minimum_angle_summary")?;
    let n = layout.len();
    if n == 0 {
        return Ok(MinimumAngleSummary { a1: 0.0, a2: 0.0 });
    }

    let directed = graph.is_directed();
    let directed_adj = if directed {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let xs = layout.x();
    let ys = layout.y();
    let mut angles = Vec::<f64>::new();
    let mut sum_a1 = 0.0_f64;
    let mut sum_a2 = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let nbrs = match directed_adj.as_ref() {
            Some(adj) => adj[u].as_slice(),
            None => graph.neighbors(u),
        };
        angles.clear();
        let ux = xs[u].to_f64();
        let uy = ys[u].to_f64();
        let mut has_zero = false;
        for &v in nbrs {
            if v == u32_u {
                continue;
            }
            let vi = v as usize;
            let dx = xs[vi].to_f64() - ux;
            let dy = ys[vi].to_f64() - uy;
            if dx == 0.0 && dy == 0.0 {
                has_zero = true;
                angles.push(0.0);
            } else {
                angles.push(dy.atan2(dx));
            }
        }
        let d = angles.len();
        if d < 2 {
            sum_a1 += 1.0;
            sum_a2 += 1.0;
            continue;
        }
        angles.sort_unstable_by(f64::total_cmp);
        let mut min_diff = (TAU - (angles[d - 1] - angles[0])).max(0.0);
        for k in 0..(d - 1) {
            let gap = (angles[k + 1] - angles[k]).max(0.0);
            if gap < min_diff {
                min_diff = gap;
            }
        }
        let ideal = TAU / (d as f64);
        let score = (1.0 - (ideal - min_diff).abs() / ideal).clamp(0.0, 1.0);
        sum_a1 += score;
        if !has_zero {
            sum_a2 += score;
        }
    }

    let inv_n = 1.0 / (n as f64);
    Ok(MinimumAngleSummary {
        a1: (sum_a1 * inv_n).clamp(0.0, 1.0),
        a2: (sum_a2 * inv_n).clamp(0.0, 1.0),
    })
}

/// Computes DRGraph Eq. (19) strict minimum-angle resolution (`A2`) in `[0.0, 1.0]`.
///
/// Equivalent to `minimum_angle_summary(graph, layout)?.a2`.
pub fn minimum_angle<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    Ok(minimum_angle_summary(graph, layout)?.a2)
}

/// Coordinate-collapse and zero-length edge degeneracy diagnostics for a 2D layout.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayoutDegeneracy {
    /// Total number of vertices $N$.
    pub num_nodes: usize,
    /// Total number of unique non-self-loop edges $M$.
    pub num_edges: usize,
    /// Number of vertices sharing an exact `f32` `(x, y)` coordinate with at least one other vertex.
    pub duplicate_nodes_f32: usize,
    /// Fraction `duplicate_nodes_f32 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f32: f64,
    /// Number of vertices sharing an exact `f64` `(x, y)` coordinate with at least one other vertex.
    pub duplicate_nodes_f64: usize,
    /// Fraction `duplicate_nodes_f64 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f64: f64,
    /// Number of unique non-self-loop edges whose endpoints coincide (`hypot(dx, dy) == 0.0`).
    pub zero_length_edges: usize,
    /// Fraction `zero_length_edges / num_edges` (`0.0` when `num_edges == 0`).
    pub zero_length_edge_ratio: f64,
    /// Axis-aligned bounding-box width `max_x - min_x`.
    pub bbox_width: f64,
    /// Axis-aligned bounding-box height `max_y - min_y`.
    pub bbox_height: f64,
    /// `true` when `num_nodes > 0` and either `bbox_width <= 1e-12` or `bbox_height <= 1e-12`.
    pub is_bbox_degenerate: bool,
}

#[inline]
fn canonical_f32(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
fn canonical_f64(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

/// Evaluates duplicate-node (`f32`/`f64`), zero-length edge, and bounding-box degeneracies.
///
/// Canonicalizes `-0.0` to `+0.0` before bitwise key comparison.
pub fn layout_degeneracy<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout(graph, layout, "layout_degeneracy")?;
    let n = layout.len();
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    if n == 0 {
        return Ok(LayoutDegeneracy {
            num_nodes: 0,
            num_edges: m,
            ..LayoutDegeneracy::default()
        });
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut keys_f32 = Vec::with_capacity(n);
    let mut keys_f64 = Vec::with_capacity(n);
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..n {
        let x64 = canonical_f64(xs[i].to_f64());
        let y64 = canonical_f64(ys[i].to_f64());
        min_x = min_x.min(x64);
        max_x = max_x.max(x64);
        min_y = min_y.min(y64);
        max_y = max_y.max(y64);
        keys_f64.push((x64.to_bits(), y64.to_bits()));

        let x32 = canonical_f32(x64 as f32);
        let y32 = canonical_f32(y64 as f32);
        keys_f32.push(((x32.to_bits() as u64) << 32) | (y32.to_bits() as u64));
    }

    keys_f32.sort_unstable();
    let mut dup_f32 = 0usize;
    let mut run = 1usize;
    for i in 1..n {
        if keys_f32[i] == keys_f32[i - 1] {
            run += 1;
        } else {
            if run > 1 {
                dup_f32 += run;
            }
            run = 1;
        }
    }
    if run > 1 {
        dup_f32 += run;
    }

    keys_f64.sort_unstable();
    let mut dup_f64 = 0usize;
    let mut run64 = 1usize;
    for i in 1..n {
        if keys_f64[i] == keys_f64[i - 1] {
            run64 += 1;
        } else {
            if run64 > 1 {
                dup_f64 += run64;
            }
            run64 = 1;
        }
    }
    if run64 > 1 {
        dup_f64 += run64;
    }

    let mut zero_edges = 0usize;
    for &(u, v) in &edges {
        let ui = u as usize;
        let vi = v as usize;
        if (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64()) == 0.0 {
            zero_edges += 1;
        }
    }

    let bbox_width = (max_x - min_x).max(0.0);
    let bbox_height = (max_y - min_y).max(0.0);
    let is_bbox_degenerate = bbox_width <= 1e-12 || bbox_height <= 1e-12;

    Ok(LayoutDegeneracy {
        num_nodes: n,
        num_edges: m,
        duplicate_nodes_f32: dup_f32,
        duplicate_ratio_f32: dup_f32 as f64 / n as f64,
        duplicate_nodes_f64: dup_f64,
        duplicate_ratio_f64: dup_f64 as f64 / n as f64,
        zero_length_edges: zero_edges,
        zero_length_edge_ratio: if m == 0 {
            0.0
        } else {
            zero_edges as f64 / m as f64
        },
        bbox_width,
        bbox_height,
        is_bbox_degenerate,
    })
}

/// PRISM layout-preservation distortion metrics comparing `original` and `adjusted` layouts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PrismDistortion {
    /// Proximity edge length ratio coefficient of variation ($\sigma_{\mathrm{dist}}$):
    /// $\frac{1}{\bar{r}}\sqrt{\frac{1}{|E_P|}\sum_{(i,j)\in E_P}(r_{ij} - \bar{r})^2}$
    /// where $r_{ij} = \|x_i - x_j\| / \|x_i^0 - x_j^0\|$.
    pub sigma_dist: f64,
    /// Scale-, translation-, and rotation-invariant Procrustes dissimilarity ($\sigma_{\mathrm{disp}}$)
    /// normalized by centered `original` energy $\sum \|x_i^0 - \bar{x}^0\|^2$, lying in `[0.0, 1.0]`.
    pub sigma_disp: f64,
}

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

/// Computes PRISM layout-preservation distortion (`sigma_dist`, `sigma_disp`) between
/// `original` and `adjusted` layouts using the spatial Gabriel graph of `original` as
/// proximity scaffold $E_P$.
pub fn prism_distortion<S: Scalar>(
    original: &Layout2d<S>,
    adjusted: &Layout2d<S>,
) -> CoreResult<PrismDistortion> {
    if original.len() != adjusted.len() {
        return Err(CoreError::dimension_mismatch(
            original.len(),
            adjusted.len(),
            "prism_distortion",
        ));
    }
    original.validate_finite()?;
    adjusted.validate_finite()?;
    if original.len() <= 1 {
        return Ok(PrismDistortion::default());
    }

    let edges = gabriel_graph_edges(original)?;
    prism_distortion_with_edges(original, adjusted, &edges)
}

/// Computes PRISM layout-preservation distortion using caller-supplied proximity edges `edges`.
pub fn prism_distortion_with_edges<S: Scalar>(
    original: &Layout2d<S>,
    adjusted: &Layout2d<S>,
    edges: &[(u32, u32)],
) -> CoreResult<PrismDistortion> {
    if original.len() != adjusted.len() {
        return Err(CoreError::dimension_mismatch(
            original.len(),
            adjusted.len(),
            "prism_distortion_with_edges",
        ));
    }
    original.validate_finite()?;
    adjusted.validate_finite()?;
    let n = original.len();
    for &(u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
                n,
                "prism_distortion_with_edges",
            ));
        }
    }
    if n <= 1 {
        return Ok(PrismDistortion::default());
    }

    let sigma_dist = compute_sigma_dist(original, adjusted, edges);
    let sigma_disp = compute_sigma_disp(original, adjusted);
    Ok(PrismDistortion {
        sigma_dist,
        sigma_disp,
    })
}

fn compute_sigma_dist<S: Scalar>(
    original: &Layout2d<S>,
    adjusted: &Layout2d<S>,
    edges: &[(u32, u32)],
) -> f64 {
    let ox = original.x();
    let oy = original.y();
    let ax = adjusted.x();
    let ay = adjusted.y();

    let mut ratios = Vec::with_capacity(edges.len());
    let mut sum_r = 0.0_f64;
    for &(u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        if ui == vi {
            continue;
        }
        let d0 = (ox[ui].to_f64() - ox[vi].to_f64()).hypot(oy[ui].to_f64() - oy[vi].to_f64());
        if d0 <= 0.0 {
            continue;
        }
        let d1 = (ax[ui].to_f64() - ax[vi].to_f64()).hypot(ay[ui].to_f64() - ay[vi].to_f64());
        let r = d1 / d0;
        ratios.push(r);
        sum_r += r;
    }

    if ratios.is_empty() {
        return 0.0;
    }
    let mean_r = sum_r / ratios.len() as f64;
    if mean_r <= 0.0 {
        return 0.0;
    }
    let var = ratios
        .iter()
        .map(|&r| {
            let diff = r - mean_r;
            diff * diff
        })
        .sum::<f64>()
        / ratios.len() as f64;
    var.sqrt() / mean_r
}

fn compute_sigma_disp<S: Scalar>(original: &Layout2d<S>, adjusted: &Layout2d<S>) -> f64 {
    let n = original.len();
    let n_f64 = n as f64;
    let ox = original.x();
    let oy = original.y();
    let ax = adjusted.x();
    let ay = adjusted.y();

    let mean_ox = ox.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mean_oy = oy.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mean_ax = ax.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mean_ay = ay.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;

    let mut s00 = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut a_dot = 0.0_f64;
    let mut b_cross = 0.0_f64;

    for i in 0..n {
        let u0 = ox[i].to_f64() - mean_ox;
        let u1 = oy[i].to_f64() - mean_oy;
        let v0 = ax[i].to_f64() - mean_ax;
        let v1 = ay[i].to_f64() - mean_ay;
        s00 += u0 * u0 + u1 * u1;
        sxx += v0 * v0 + v1 * v1;
        a_dot += u0 * v0 + u1 * v1;
        b_cross += u1 * v0 - u0 * v1;
    }

    if s00 <= 0.0 {
        return 0.0;
    }
    if sxx <= 0.0 {
        return 1.0;
    }

    let norm_ab = a_dot.hypot(b_cross);
    if norm_ab <= 0.0 {
        return 1.0;
    }
    let cos_t = a_dot / norm_ab;
    let sin_t = b_cross / norm_ab;
    let r_opt = norm_ab / sxx;

    let mut err_sum = 0.0_f64;
    for i in 0..n {
        let u0 = ox[i].to_f64() - mean_ox;
        let u1 = oy[i].to_f64() - mean_oy;
        let v0 = ax[i].to_f64() - mean_ax;
        let v1 = ay[i].to_f64() - mean_ay;
        let rx = r_opt * (cos_t * v0 - sin_t * v1);
        let ry = r_opt * (sin_t * v0 + cos_t * v1);
        let dx = rx - u0;
        let dy = ry - u1;
        err_sum += dx * dx + dy * dy;
    }

    (err_sum / s00).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_edge_angle_aspect_ratio_and_angular_resolution() {
        assert!((edge_angle(1.0, 0.0, 0.0, 1.0).unwrap() - FRAC_PI_2).abs() < 1e-12);
        assert!(edge_angle(1.0_f64, 1e-8, 1.0, 1e-8).unwrap().abs() < 1e-12);
        assert!((edge_angle(1.0, 0.0, -1.0, 0.0).unwrap() - PI).abs() < 1e-12);
        assert!(edge_angle(0.0, 0.0, 1.0, 0.0).is_none());
        assert!(edge_angle(f64::NAN, 0.0, 1.0, 0.0).is_none());

        let square = Layout2d::from_xy(vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!((aspect_ratio(&square).unwrap() - 1.0).abs() < 1e-12);

        let collinear = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(aspect_ratio(&collinear).unwrap(), 0.0);

        let star =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
        let angles_star = Layout2d::from_xy(
            vec![0.0, 1.0, -0.5, -0.5],
            vec![0.0, 0.0, 0.5 * 3.0_f64.sqrt(), -0.5 * 3.0_f64.sqrt()],
        )
        .unwrap();
        let expected_ang = 3.0 * (-2.0 * PI / 3.0).exp();
        assert!((angular_resolution(&star, &angles_star).unwrap() - expected_ang).abs() < 1e-12);
    }

    #[test]
    fn test_ideal_edge_lengths_unit_weighted_and_distances() {
        let g_unw = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        let l = Layout2d::from_xy(vec![0.0, 2.0, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert!((ideal_edge_lengths(&g_unw, &l).unwrap() - 5.0).abs() < 1e-12);
        assert!((ideal_edge_lengths_with_unit(&g_unw, &l, 2.5).unwrap() - 0.08).abs() < 1e-12);

        let g_w =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[2.0, 3.0]), false, false)
                .unwrap();
        assert!(ideal_edge_lengths(&g_w, &l).unwrap().abs() < 1e-12);

        let dm = vec![0.0, 2.0, 5.0, 2.0, 0.0, 3.0, 5.0, 3.0, 0.0];
        assert!(
            ideal_edge_lengths_from_distances(&g_unw, &l, &dm)
                .unwrap()
                .abs()
                < 1e-12
        );

        assert!(matches!(
            ideal_edge_lengths_with_unit(&g_unw, &l, 0.0),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_prism_distortion_similarity_invariance_and_perturbation() {
        let orig = Layout2d::from_xy(vec![0.0, 1.0, 0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0, 1.0, 1.6])
            .unwrap();

        // Rigid similarity transform (scale 3.5, rotation 0.7 rad, translation (12, -4)).
        let theta = 0.7_f64;
        let (st, ct) = theta.sin_cos();
        let scale = 3.5_f64;
        let adj_x: Vec<f64> = (0..orig.len())
            .map(|i| scale * (ct * orig.x()[i] - st * orig.y()[i]) + 12.0)
            .collect();
        let adj_y: Vec<f64> = (0..orig.len())
            .map(|i| scale * (st * orig.x()[i] + ct * orig.y()[i]) - 4.0)
            .collect();
        let similar = Layout2d::from_xy(adj_x, adj_y).unwrap();

        let d_sim = prism_distortion(&orig, &similar).unwrap();
        assert!(d_sim.sigma_dist < 1e-12, "sigma_dist={}", d_sim.sigma_dist);
        assert!(d_sim.sigma_disp < 1e-12, "sigma_disp={}", d_sim.sigma_disp);

        // Non-uniform distortion produces positive dispersion & Procrustes disparity.
        let warped =
            Layout2d::from_xy(vec![0.0, 3.0, 0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0, 2.5, 2.0])
                .unwrap();
        let d_warp = prism_distortion(&orig, &warped).unwrap();
        assert!(d_warp.sigma_dist > 0.05);
        assert!(d_warp.sigma_disp > 0.01 && d_warp.sigma_disp <= 1.0);

        // Collapsed adjusted layout gives finite sigma_dist = 0.0 and sigma_disp = 1.0.
        let collapsed = Layout2d::zeros(orig.len());
        let d_col = prism_distortion(&orig, &collapsed).unwrap();
        assert_eq!(d_col.sigma_dist, 0.0);
        assert_eq!(d_col.sigma_disp, 1.0);
    }

    fn brute_minimum_angle_oracle<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> (f64, f64) {
        let n = layout.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let adj = build_undirected_adjacency(graph);
        let xs = layout.x();
        let ys = layout.y();
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for u in 0..n {
            let d = adj[u].len();
            if d < 2 {
                s1 += 1.0;
                s2 += 1.0;
                continue;
            }
            let ux = xs[u].to_f64();
            let uy = ys[u].to_f64();
            let mut has_zero = false;
            let mut vecs = Vec::with_capacity(d);
            for &v in &adj[u] {
                let dx = xs[v as usize].to_f64() - ux;
                let dy = ys[v as usize].to_f64() - uy;
                if dx == 0.0 && dy == 0.0 {
                    has_zero = true;
                    vecs.push((1.0_f64, 0.0_f64));
                } else {
                    vecs.push((dx, dy));
                }
            }
            let mut min_theta = TAU;
            for i in 0..d {
                for j in (i + 1)..d {
                    let ang = edge_angle(vecs[i].0, vecs[i].1, vecs[j].0, vecs[j].1).unwrap();
                    if ang < min_theta {
                        min_theta = ang;
                    }
                }
            }
            let ideal = TAU / (d as f64);
            let score = (1.0 - (ideal - min_theta).abs() / ideal).clamp(0.0, 1.0);
            s1 += score;
            if !has_zero {
                s2 += score;
            }
        }
        (s1 / n as f64, s2 / n as f64)
    }

    #[test]
    fn test_minimum_angle_brute_force_oracle_branch_cut_and_zero_edges() {
        // Empty graph -> (0.0, 0.0).
        let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let empty_l = Layout2d::<f64>::zeros(0);
        assert_eq!(
            minimum_angle_summary(&empty_g, &empty_l).unwrap(),
            MinimumAngleSummary { a1: 0.0, a2: 0.0 }
        );
        assert_eq!(minimum_angle(&empty_g, &empty_l).unwrap(), 0.0);

        // Regular 3-star: ideal 120 deg at center, deg 1 leaves -> both A1 and A2 = 1.0.
        let star =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
        let star_l = Layout2d::from_xy(
            vec![0.0, 1.0, -0.5, -0.5],
            vec![0.0, 0.0, 0.5 * 3.0_f64.sqrt(), -0.5 * 3.0_f64.sqrt()],
        )
        .unwrap();
        let s_star = minimum_angle_summary(&star, &star_l).unwrap();
        assert!((s_star.a1 - 1.0).abs() < 1e-12);
        assert!((s_star.a2 - 1.0).abs() < 1e-12);

        // Branch cut around -pi / +pi: center 0 with neighbors near (-1, +eps) and (-1, -eps).
        let branch_g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
        let branch_l =
            Layout2d::from_xy(vec![0.0, -1.0, -1.0, 1.0], vec![0.0, 1e-4, -1e-4, 0.0]).unwrap();
        let s_branch = minimum_angle_summary(&branch_g, &branch_l).unwrap();
        let (o1, o2) = brute_minimum_angle_oracle(&branch_g, &branch_l);
        assert!((s_branch.a1 - o1).abs() < 1e-12);
        assert!((s_branch.a2 - o2).abs() < 1e-12);

        // Zero-length edge divergence between A1 (atan2(0,0)=0) and A2 (score=0.0).
        // Vertex 0 has 3 neighbors: (1, 0), (-1, 0), and coincident (0, 0) -> ideal 2pi/3.
        // A1 sees angles [0, 0, pi] -> min_diff = 0 -> score(0) = 0; wait, pick neighbor at (0, 1):
        // angles [0 (zero-len), pi/2, pi] -> min_diff = pi/2, ideal = 2pi/3 -> A1 > 0 while A2 = 0 at vertex 0!
        let zero_l =
            Layout2d::from_xy(vec![0.0, 0.0, 0.0, -1.0], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let s_zero = minimum_angle_summary(&star, &zero_l).unwrap();
        let (z1, z2) = brute_minimum_angle_oracle(&star, &zero_l);
        assert!((s_zero.a1 - z1).abs() < 1e-12);
        assert!((s_zero.a2 - z2).abs() < 1e-12);
        assert!(s_zero.a1 > s_zero.a2);
        assert_eq!(minimum_angle(&star, &zero_l).unwrap(), s_zero.a2);

        // Multi-seed random graphs (directed + undirected, f32 + f64) vs O(sum d_v^2) oracle.
        for seed in 1..=20u64 {
            let mut rng = DeterministicRng::from_seed_u64(500 + seed);
            let n = 16usize;
            let directed = seed % 2 == 0;
            let mut edges = Vec::new();
            for _ in 0..32 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                edges.push((u, v));
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, directed, true).unwrap();
            let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let l64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let l32 = Layout2d::from_xy(
                xs.iter().map(|&v| v as f32).collect(),
                ys.iter().map(|&v| v as f32).collect(),
            )
            .unwrap();

            let s64 = minimum_angle_summary(&g, &l64).unwrap();
            let (b64_1, b64_2) = brute_minimum_angle_oracle(&g, &l64);
            assert!((s64.a1 - b64_1).abs() < 1e-12);
            assert!((s64.a2 - b64_2).abs() < 1e-12);

            let s32 = minimum_angle_summary(&g, &l32).unwrap();
            let (b32_1, b32_2) = brute_minimum_angle_oracle(&g, &l32);
            assert!((s32.a1 - b32_1).abs() < 1e-12);
            assert!((s32.a2 - b32_2).abs() < 1e-12);
        }
    }

    #[test]
    fn test_layout_degeneracy_brute_force_oracle_and_edge_cases() {
        // 1. Canonicalization of -0.0 -> +0.0 and f32 vs f64 collision divergence.
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
        // Nodes 0 and 1 are (+0.0, -0.0) and (-0.0, +0.0) -> identical in both f32 and f64.
        // Nodes 2 and 3 differ by 1e-12 -> distinct in f64, identical in f32!
        let l = Layout2d::from_xy(
            vec![0.0_f64, -0.0_f64, 1.0, 1.0 + 1e-12],
            vec![-0.0_f64, 0.0_f64, 1.0, 1.0],
        )
        .unwrap();
        let deg = layout_degeneracy(&g, &l).unwrap();
        assert_eq!(deg.num_nodes, 4);
        assert_eq!(deg.num_edges, 3);
        assert_eq!(deg.duplicate_nodes_f32, 4);
        assert!((deg.duplicate_ratio_f32 - 1.0).abs() < 1e-12);
        assert_eq!(deg.duplicate_nodes_f64, 2);
        assert!((deg.duplicate_ratio_f64 - 0.5).abs() < 1e-12);
        assert_eq!(deg.zero_length_edges, 1);
        assert!((deg.zero_length_edge_ratio - (1.0 / 3.0)).abs() < 1e-12);
        assert!(!deg.is_bbox_degenerate);

        // 2. 1D collinear layout: bbox_height == 0 -> is_bbox_degenerate = true.
        let line = Layout2d::from_xy(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4]).unwrap();
        let deg_line = layout_degeneracy(&g, &line).unwrap();
        assert_eq!(deg_line.duplicate_nodes_f32, 0);
        assert_eq!(deg_line.bbox_width, 3.0);
        assert_eq!(deg_line.bbox_height, 0.0);
        assert!(deg_line.is_bbox_degenerate);

        // 3. Empty graph -> zeros, not degenerate.
        let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let empty_l = Layout2d::<f64>::zeros(0);
        let deg_empty = layout_degeneracy(&empty_g, &empty_l).unwrap();
        assert_eq!(deg_empty.num_nodes, 0);
        assert_eq!(deg_empty.duplicate_ratio_f32, 0.0);
        assert!(!deg_empty.is_bbox_degenerate);

        // 4. Multi-seed O(N^2) brute-force oracle for duplicate_nodes_f32/f64 and zero_length_edges.
        for seed in 1..=15u64 {
            let mut rng = DeterministicRng::from_seed_u64(600 + seed);
            let n = 25usize;
            let mut edges = Vec::new();
            for _ in 0..40 {
                edges.push((rng.gen_range_usize(n) as u32, rng.gen_range_usize(n) as u32));
            }
            let rg = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();
            let xs: Vec<f64> = (0..n)
                .map(|_| {
                    let v = rng.gen_range_usize(5) as f64 - 2.0;
                    if v == 0.0 && rng.gen_range_usize(2) == 0 {
                        -0.0
                    } else {
                        v
                    }
                })
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(5) as f64) * 0.5)
                .collect();
            let rl = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let got = layout_degeneracy(&rg, &rl).unwrap();

            let mut brute_dup64 = 0usize;
            let mut brute_dup32 = 0usize;
            for i in 0..n {
                let has64 = (0..n).any(|j| j != i && xs[i] == xs[j] && ys[i] == ys[j]);
                let has32 = (0..n).any(|j| {
                    j != i && (xs[i] as f32) == (xs[j] as f32) && (ys[i] as f32) == (ys[j] as f32)
                });
                if has64 {
                    brute_dup64 += 1;
                }
                if has32 {
                    brute_dup32 += 1;
                }
            }
            let uedges = extract_unique_edges(&rg);
            let brute_zero = uedges
                .iter()
                .filter(|&&(u, v)| {
                    xs[u as usize] == xs[v as usize] && ys[u as usize] == ys[v as usize]
                })
                .count();
            assert_eq!(got.duplicate_nodes_f64, brute_dup64);
            assert_eq!(got.duplicate_nodes_f32, brute_dup32);
            assert_eq!(got.zero_length_edges, brute_zero);
        }
    }
}
