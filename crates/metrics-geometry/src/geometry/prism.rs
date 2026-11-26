use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use crate::proximity::gabriel_graph_edges;

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
