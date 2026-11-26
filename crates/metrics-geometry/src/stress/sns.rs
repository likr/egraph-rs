use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};

use crate::validate_graph_layout;

use super::StressEvaluation;
use super::bfs::{BfsScratch, bfs_single_source};

#[derive(Default)]
pub(super) struct SnsMoments {
    pub(super) sum_w_r2: f64,
    pub(super) sum_w_d_r: f64,
    pub(super) sum_w_d2: f64,
    pub(super) pair_count: usize,
}

impl SnsMoments {
    #[inline]
    pub(super) fn push(&mut self, r: f64, d: f64, w: f64) {
        self.sum_w_r2 += w * r * r;
        self.sum_w_d_r += w * d * r;
        self.sum_w_d2 += w * d * d;
        self.pair_count += 1;
    }
}

#[inline]
pub(super) fn finalize_moments(m: &SnsMoments) -> Option<StressEvaluation> {
    if m.pair_count == 0 || m.sum_w_d2 <= 0.0 {
        return Some(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: m.pair_count,
            weight_distance_sq_sum: m.sum_w_d2,
        });
    }
    if m.sum_w_r2 <= 0.0 {
        return Some(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: m.sum_w_d2,
            alpha_star: 0.0,
            pair_count: m.pair_count,
            weight_distance_sq_sum: m.sum_w_d2,
        });
    }
    let alpha_star = m.sum_w_d_r / m.sum_w_r2;
    let one_pass_norm = (1.0 - (m.sum_w_d_r / m.sum_w_r2) * (m.sum_w_d_r / m.sum_w_d2)).max(0.0);
    // Use two-pass direct residual summation when stress is small so cancellation near zero
    // never drops precision on near-isometric embeddings.
    if one_pass_norm < 1e-4 {
        return None;
    }
    Some(StressEvaluation {
        normalized_stress: one_pass_norm.clamp(0.0, 1.0),
        raw_stress: (one_pass_norm * m.sum_w_d2).max(0.0),
        alpha_star,
        pair_count: m.pair_count,
        weight_distance_sq_sum: m.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and `layout`.
///
/// Uses unweighted BFS hop distances (`1.0` per edge) with stress weights $w_{ij} = d_{ij}^{-2}$,
/// $O(N)$ scratch memory via [`BfsScratch`], and `f64` accumulation.
pub fn scale_normalized_stress<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    Ok(scale_normalized_stress_detail(graph, layout)?.normalized_stress)
}

/// Computes full [`StressEvaluation`] details for closed-form `SNS` over `graph` and `layout`.
pub fn scale_normalized_stress_detail<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<StressEvaluation> {
    validate_graph_layout(graph, layout, "scale_normalized_stress")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();
    let directed = graph.is_directed();

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let w = 1.0 / (d * d);
            moments.push(r, d, w);
        }
    }

    if let Some(eval) = finalize_moments(&moments) {
        return Ok(eval);
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let err = (alpha_star * r - d) / d;
            raw_stress += err * err;
        }
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) from a precomputed flat row-major `N * N`
/// `distances` matrix and optional `weights` matrix (`w_ij = 1 / d_ij^2` when `None`).
pub fn scale_normalized_stress_from_distances<S: Scalar>(
    layout: &Layout2d<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<f64> {
    Ok(
        scale_normalized_stress_from_distances_detail(layout, distances, weights)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details from precomputed `N * N` `distances` and optional `weights`.
pub fn scale_normalized_stress_from_distances_detail<S: Scalar>(
    layout: &Layout2d<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<StressEvaluation> {
    layout.validate_finite()?;
    let n = layout.len();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "scale_normalized_stress_from_distances::distances",
        ));
    }
    if let Some(ws) = weights
        && ws.len() != expected
    {
        return Err(CoreError::dimension_mismatch(
            expected,
            ws.len(),
            "scale_normalized_stress_from_distances::weights",
        ));
    }
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut moments = SnsMoments::default();

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let row = i * n;
        for j in (i + 1)..n {
            let d = distances[row + j];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let w = match weights {
                Some(ws) => {
                    let val = ws[row + j];
                    if !val.is_finite() {
                        return Err(CoreError::non_finite(
                            "scale_normalized_stress_from_distances::weights",
                        ));
                    }
                    if val < 0.0 {
                        return Err(CoreError::invalid_input(
                            "stress weights must be non-negative",
                        ));
                    }
                    if val == 0.0 {
                        continue;
                    }
                    val
                }
                None => 1.0 / (d * d),
            };
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            moments.push(r, d, w);
        }
    }

    if moments.pair_count == 0 || moments.sum_w_d2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }
    if moments.sum_w_r2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: moments.sum_w_d2,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let row = i * n;
        for j in (i + 1)..n {
            let d = distances[row + j];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let w = match weights {
                Some(ws) => {
                    let val = ws[row + j];
                    if val == 0.0 {
                        continue;
                    }
                    val
                }
                None => 1.0 / (d * d),
            };
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let diff = alpha_star * r - d;
            raw_stress += w * diff * diff;
        }
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) from sparse `(i, j, d_ij, w_ij)` pairs.
pub fn scale_normalized_stress_from_pairs<S: Scalar>(
    layout: &Layout2d<S>,
    pairs: &[(u32, u32, f64, f64)],
) -> CoreResult<StressEvaluation> {
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut moments = SnsMoments::default();

    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
                n,
                "scale_normalized_stress_from_pairs",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite(
                "scale_normalized_stress_from_pairs::weight",
            ));
        }
        if w < 0.0 {
            return Err(CoreError::invalid_input(
                "stress weights must be non-negative",
            ));
        }
        if ui == vi || !d.is_finite() || d <= 0.0 || w == 0.0 {
            continue;
        }
        let r = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        moments.push(r, d, w);
    }

    if moments.pair_count == 0 || moments.sum_w_d2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }
    if moments.sum_w_r2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: moments.sum_w_d2,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui == vi || !d.is_finite() || d <= 0.0 || w == 0.0 {
            continue;
        }
        let r = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        let diff = alpha_star * r - d;
        raw_stress += w * diff * diff;
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}
