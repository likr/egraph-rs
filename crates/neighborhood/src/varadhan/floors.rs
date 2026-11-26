//! Horvitz-Thompson repulsion scaling and relative pair distance flooring utilities.

use egraph_core::{CoreError, CoreResult};

/// Computes R3 parameter-free Horvitz-Thompson repulsion density scaling:
/// $\gamma_{\mathrm{eff}} = \gamma_{\mathrm{non2}} \sqrt{N / \max(k_{\mathrm{neg}}, 1)}$.
pub fn compute_horvitz_thompson_gamma(
    gamma_non2: f64,
    num_nodes: usize,
    neg_per_node: usize,
) -> CoreResult<f64> {
    if !gamma_non2.is_finite() {
        return Err(CoreError::non_finite(
            "compute_horvitz_thompson_gamma::gamma_non2",
        ));
    }
    if gamma_non2 < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "gamma_non2 must be >= 0, got {gamma_non2}"
        )));
    }
    if num_nodes == 0 {
        return Ok(0.0);
    }
    let k_neg = neg_per_node.max(1) as f64;
    Ok(gamma_non2 * ((num_nodes as f64) / k_neg).sqrt())
}

/// Clamps pair distances `d1`, `d2`, and `d3` in-place using a relative floor derived from
/// positive 1-hop medians (`max(eps_dist, min_ratio * median_pos)`), returning the applied floor.
pub fn apply_relative_distance_floor(
    d1: &mut [f64],
    d2: &mut [f64],
    d3: &mut [f64],
    eps_dist: f64,
    min_ratio: f64,
) -> CoreResult<f64> {
    if !eps_dist.is_finite() {
        return Err(CoreError::non_finite(
            "apply_relative_distance_floor::eps_dist",
        ));
    }
    if eps_dist <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_dist must be > 0, got {eps_dist}"
        )));
    }
    if !min_ratio.is_finite() {
        return Err(CoreError::non_finite(
            "apply_relative_distance_floor::min_ratio",
        ));
    }
    if min_ratio < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "min_ratio must be >= 0, got {min_ratio}"
        )));
    }
    for (ctx, slice) in [
        ("apply_relative_distance_floor::d1", &*d1),
        ("apply_relative_distance_floor::d2", &*d2),
        ("apply_relative_distance_floor::d3", &*d3),
    ] {
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(ctx));
        }
    }

    let mut pos: Vec<f64> = d1.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        pos = d2.iter().copied().filter(|&v| v > 0.0).collect();
    }
    if pos.is_empty() {
        pos = d3.iter().copied().filter(|&v| v > 0.0).collect();
    }

    let ref_med = if pos.is_empty() {
        1.0
    } else {
        pos.sort_by(f64::total_cmp);
        let mid = pos.len() / 2;
        if pos.len() % 2 == 1 {
            pos[mid]
        } else {
            0.5 * (pos[mid - 1] + pos[mid])
        }
    };

    let d_floor = eps_dist.max(min_ratio * ref_med);
    for slice in [d1, d2, d3] {
        for v in slice.iter_mut() {
            if *v < d_floor {
                *v = d_floor;
            }
        }
    }
    Ok(d_floor)
}
