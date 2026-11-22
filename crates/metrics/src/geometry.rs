//! Geometric shape, edge-length, angular, and PRISM layout-preservation metrics.

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};

use crate::{extract_unique_edges, proximity::gabriel_graph_edges, validate_graph_layout};

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
}
