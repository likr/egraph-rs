use egraph_core::{CoreError, CoreResult, CsrGraph, LayoutNd, Scalar};

use crate::nd_index::{slice_dist_nd, validate_graph_layout_nd, validate_layout_nd};

use super::StressEvaluation;
use super::bfs::{BfsScratch, bfs_single_source};
use super::sampled::select_degree_stratified_pivots;
use super::sns::{SnsMoments, finalize_moments, validate_distance_matrix_symmetry};

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and
/// N-D `layout`.
pub fn scale_normalized_stress_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<f64> {
    Ok(scale_normalized_stress_detail_nd(graph, layout)?.normalized_stress)
}

/// Computes full [`StressEvaluation`] details for closed-form `SNS` over `graph` and N-D `layout`.
pub fn scale_normalized_stress_detail_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<StressEvaluation> {
    validate_graph_layout_nd(graph, layout)?;
    let n = layout.num_nodes();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let dims = layout.dims();
    let coords = layout.coords();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();
    let directed = graph.is_directed();

    for i in 0..n {
        let pi = &coords[i * dims..(i + 1) * dims];
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pi, pj);
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
        let pi = &coords[i * dims..(i + 1) * dims];
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pi, pj);
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
/// `distances` matrix and optional `weights` matrix over an N-D `layout`.
pub fn scale_normalized_stress_from_distances_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<f64> {
    Ok(
        scale_normalized_stress_from_distances_detail_nd(layout, distances, weights)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details from precomputed `N * N` `distances` and optional
/// `weights` over an N-D `layout`.
pub fn scale_normalized_stress_from_distances_detail_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<StressEvaluation> {
    validate_layout_nd(layout)?;
    let n = layout.num_nodes();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "scale_normalized_stress_from_distances_nd::distances",
        ));
    }
    if let Some(ws) = weights
        && ws.len() != expected
    {
        return Err(CoreError::dimension_mismatch(
            expected,
            ws.len(),
            "scale_normalized_stress_from_distances_nd::weights",
        ));
    }
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }
    validate_distance_matrix_symmetry(distances, n)?;

    let dims = layout.dims();
    let coords = layout.coords();
    let mut moments = SnsMoments::default();

    for i in 0..n {
        let pi = &coords[i * dims..(i + 1) * dims];
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
                            "scale_normalized_stress_from_distances_nd::weights",
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
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pi, pj);
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
        let pi = &coords[i * dims..(i + 1) * dims];
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
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pi, pj);
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

/// Computes closed-form Scale-Normalized Stress (`SNS`) from sparse `(i, j, d_ij, w_ij)` pairs
/// over an N-D `layout`.
pub fn scale_normalized_stress_from_pairs_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    pairs: &[(u32, u32, f64, f64)],
) -> CoreResult<StressEvaluation> {
    validate_layout_nd(layout)?;
    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let mut moments = SnsMoments::default();

    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
                n,
                "scale_normalized_stress_from_pairs_nd",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite(
                "scale_normalized_stress_from_pairs_nd::weight",
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
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let r = slice_dist_nd(pu, pv);
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
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let r = slice_dist_nd(pu, pv);
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

/// Computes $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`) over an N-D `layout`.
pub fn sampled_scale_normalized_stress_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(
        sampled_scale_normalized_stress_detail_nd(graph, layout, num_pivots, seed)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details for $O(kN)$ pivot-sampled `SNS` over an N-D `layout`.
pub fn sampled_scale_normalized_stress_detail_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<StressEvaluation> {
    if num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    validate_graph_layout_nd(graph, layout)?;
    let n = layout.num_nodes();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let pivots = select_degree_stratified_pivots(graph, num_pivots, seed);
    let mut is_pivot = vec![false; n];
    for &p in &pivots {
        is_pivot[p] = true;
    }

    let dims = layout.dims();
    let coords = layout.coords();
    let directed = graph.is_directed();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();

    for &p in &pivots {
        let pp = &coords[p * dims..(p + 1) * dims];
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pp, pj);
            let w = 1.0 / (d * d);
            moments.push(r, d, w);
        }
    }

    if let Some(eval) = finalize_moments(&moments) {
        return Ok(eval);
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for &p in &pivots {
        let pp = &coords[p * dims..(p + 1) * dims];
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let pj = &coords[j * dims..(j + 1) * dims];
            let r = slice_dist_nd(pp, pj);
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
