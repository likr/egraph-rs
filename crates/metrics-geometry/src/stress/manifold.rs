use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldLayout2d, Scalar};
use egraph_linalg::DijkstraWorkspace;

use crate::validate_graph_manifold_layout;

use super::StressEvaluation;
use super::bfs::{BfsScratch, bfs_single_source};
use super::sampled::select_degree_stratified_pivots;
use super::sns::{SnsMoments, finalize_moments, validate_distance_matrix_symmetry};

/// Computes Miller et al. (2024) Eq. (2) Geodesic Distortion in `[0.0, inf)` over `graph` and
/// `layout` without post-hoc optimal $\alpha^*$ rescaling:
/// $$\frac{1}{|P|} \sum_{(i, j) \in P} \frac{|d_{\mathcal{M}_s}(x_i, x_j) - D_{ij}|}{D_{ij}}.$$
///
/// Uses BFS hop distances (`1.0` per edge) on unweighted graphs and Dijkstra shortest-path
/// distances on weighted graphs, matching `scale_normalized_stress`. Evaluates undirected pairs
/// `i < j` or directed pairs `i != j` with `0 < D_{ij} < inf`, returning `0.0` when $|P| = 0$.
pub fn geodesic_distortion<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "geodesic_distortion")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let directed = graph.is_directed();
    let weighted = graph.is_weighted();
    let mut bfs_scratch = if !weighted {
        Some(BfsScratch::new(n))
    } else {
        None
    };
    let mut dijkstra_ws = if weighted {
        Some(DijkstraWorkspace::new(n))
    } else {
        None
    };
    let mut dijkstra_buf = if weighted {
        vec![f64::INFINITY; n]
    } else {
        Vec::new()
    };

    let mut sum_rel_err = 0.0_f64;
    let mut pair_count = 0usize;

    for i in 0..n {
        let dists: &[f64] = if let Some(ws) = dijkstra_ws.as_mut() {
            ws.dijkstra_from(graph, i, &mut dijkstra_buf)?;
            &dijkstra_buf
        } else if let Some(scratch) = bfs_scratch.as_mut() {
            bfs_single_source(graph, i, 1.0, scratch)?
        } else {
            &[]
        };

        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = layout.geodesic_distance(i, j)?.to_f64();
            sum_rel_err += (r - d).abs() / d;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        Ok(0.0)
    } else {
        Ok(sum_rel_err / pair_count as f64)
    }
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and
/// [`ManifoldLayout2d`] using exact geodesic distances.
pub fn scale_normalized_stress_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    Ok(scale_normalized_stress_detail_manifold(graph, layout)?.normalized_stress)
}

/// Computes full [`StressEvaluation`] details for closed-form geodesic `SNS` over `graph` and
/// [`ManifoldLayout2d`].
pub fn scale_normalized_stress_detail_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<StressEvaluation> {
    validate_graph_manifold_layout(graph, layout, "scale_normalized_stress_manifold")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();
    let directed = graph.is_directed();

    for i in 0..n {
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = layout.geodesic_distance(i, j)?.to_f64();
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
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = layout.geodesic_distance(i, j)?.to_f64();
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

/// Computes closed-form geodesic Scale-Normalized Stress (`SNS`) from a precomputed flat
/// row-major `N * N` `distances` matrix and optional `weights` matrix over [`ManifoldLayout2d`].
pub fn scale_normalized_stress_from_distances_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<f64> {
    Ok(
        scale_normalized_stress_from_distances_detail_manifold(layout, distances, weights)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details from precomputed `N * N` `distances` and optional
/// `weights` over [`ManifoldLayout2d`].
pub fn scale_normalized_stress_from_distances_detail_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
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
            "scale_normalized_stress_from_distances_manifold::distances",
        ));
    }
    if let Some(ws) = weights
        && ws.len() != expected
    {
        return Err(CoreError::dimension_mismatch(
            expected,
            ws.len(),
            "scale_normalized_stress_from_distances_manifold::weights",
        ));
    }
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }
    validate_distance_matrix_symmetry(distances, n)?;

    let mut moments = SnsMoments::default();
    for i in 0..n {
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
                            "scale_normalized_stress_from_distances_manifold::weights",
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
            let r = layout.geodesic_distance(i, j)?.to_f64();
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
            let r = layout.geodesic_distance(i, j)?.to_f64();
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

/// Computes closed-form geodesic Scale-Normalized Stress (`SNS`) from sparse
/// `(i, j, d_ij, w_ij)` pairs over [`ManifoldLayout2d`].
pub fn scale_normalized_stress_from_pairs_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    pairs: &[(u32, u32, f64, f64)],
) -> CoreResult<StressEvaluation> {
    layout.validate_finite()?;
    let n = layout.len();
    let mut moments = SnsMoments::default();

    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
                n,
                "scale_normalized_stress_from_pairs_manifold",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite(
                "scale_normalized_stress_from_pairs_manifold::weight",
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
        let r = layout.geodesic_distance(ui, vi)?.to_f64();
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
        let r = layout.geodesic_distance(ui, vi)?.to_f64();
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

/// Computes $O(kN)$ pivot-sampled geodesic Scale-Normalized Stress (`SNS`) over [`ManifoldLayout2d`].
pub fn sampled_scale_normalized_stress_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(
        sampled_scale_normalized_stress_detail_manifold(graph, layout, num_pivots, seed)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details for $O(kN)$ pivot-sampled geodesic `SNS` over
/// [`ManifoldLayout2d`].
pub fn sampled_scale_normalized_stress_detail_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<StressEvaluation> {
    if num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    validate_graph_manifold_layout(graph, layout, "sampled_scale_normalized_stress_manifold")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let pivots = select_degree_stratified_pivots(graph, num_pivots, seed);
    let mut is_pivot = vec![false; n];
    for &p in &pivots {
        is_pivot[p] = true;
    }

    let directed = graph.is_directed();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();

    for &p in &pivots {
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let r = layout.geodesic_distance(p, j)?.to_f64();
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
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let r = layout.geodesic_distance(p, j)?.to_f64();
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
