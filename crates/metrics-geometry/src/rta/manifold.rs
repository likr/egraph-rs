use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldLayout2d, Scalar};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};

use crate::validate_graph_manifold_layout;

use super::RtaEvaluation;
use super::evaluate::evaluate_rta_core;
use super::row_stats::{RowProvider, is_manifold_collapsed};

/// Evaluates Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`
/// over a [`ManifoldLayout2d`], returning detailed evaluation metadata ([`RtaEvaluation`]).
pub fn landmark_stratified_rta_detail_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_landmarks: usize,
    num_triplets: usize,
    min_degree: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    validate_graph_manifold_layout(graph, layout, "landmark_stratified_rta_manifold")?;
    if num_landmarks == 0 {
        return Err(CoreError::invalid_input("num_landmarks must be >= 1"));
    }
    if num_triplets == 0 {
        return Err(CoreError::invalid_input("num_triplets must be >= 1"));
    }
    let n = layout.len();
    if n < 3 || graph.num_edges() == 0 || is_manifold_collapsed(layout) {
        return Ok(RtaEvaluation::default());
    }

    let landmark_ids = select_maxmin_sp_pivots(graph, num_landmarks.min(n), seed, min_degree)?;
    let provider = RowProvider::Graph {
        graph,
        ws: BfsWorkspace::new(n),
        buf: vec![u32::MAX; n],
    };
    evaluate_rta_core(layout, &landmark_ids, provider, num_triplets, seed)
}

/// Evaluates Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`
/// over a [`ManifoldLayout2d`].
pub fn landmark_stratified_rta_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    num_landmarks: usize,
    num_triplets: usize,
    min_degree: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(landmark_stratified_rta_detail_manifold(
        graph,
        layout,
        num_landmarks,
        num_triplets,
        min_degree,
        seed,
    )?
    .rta)
}

/// Evaluates Landmark-Stratified Random Triplet Accuracy over a [`ManifoldLayout2d`] from a
/// precomputed flat row-major `(num_landmarks, N)` integer hop-distance table (`u32::MAX`
/// denotes unreachable).
pub fn landmark_stratified_rta_from_landmarks_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    landmark_ids: &[u32],
    landmark_dists: &[u32],
    num_triplets: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    layout.validate_finite()?;
    if num_triplets == 0 {
        return Err(CoreError::invalid_input("num_triplets must be >= 1"));
    }
    let n = layout.len();
    let p_l = landmark_ids.len();
    let expected = p_l
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("landmark distance table size overflow"))?;
    if landmark_dists.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            landmark_dists.len(),
            "landmark_stratified_rta_from_landmarks_manifold::landmark_dists",
        ));
    }
    for &lid in landmark_ids {
        if (lid as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lid as usize,
                n,
                "landmark_stratified_rta_from_landmarks_manifold::landmark_ids",
            ));
        }
    }
    if n < 3 || p_l == 0 || is_manifold_collapsed(layout) {
        return Ok(RtaEvaluation {
            rta: 0.0,
            num_landmarks: p_l,
            num_triplets: 0,
            num_strata: 0,
            is_exact: false,
        });
    }

    let provider = RowProvider::Table::<f64> {
        dists: landmark_dists,
        n,
    };
    evaluate_rta_core(layout, landmark_ids, provider, num_triplets, seed)
}

/// Evaluates Landmark-Stratified Random Triplet Accuracy over a [`ManifoldLayout2d`] from a
/// precomputed flat row-major `(num_landmarks, N)` `f64` distance table, mapping
/// `f64::INFINITY` to the unreachable sentinel (`u32::MAX`).
pub fn landmark_stratified_rta_from_landmarks_f64_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    landmark_ids: &[u32],
    landmark_dists: &[f64],
    num_triplets: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    layout.validate_finite()?;
    if num_triplets == 0 {
        return Err(CoreError::invalid_input("num_triplets must be >= 1"));
    }
    let n = layout.len();
    let p_l = landmark_ids.len();
    let expected = p_l
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("landmark distance table size overflow"))?;
    if landmark_dists.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            landmark_dists.len(),
            "landmark_stratified_rta_from_landmarks_f64_manifold::landmark_dists",
        ));
    }
    for &lid in landmark_ids {
        if (lid as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lid as usize,
                n,
                "landmark_stratified_rta_from_landmarks_f64_manifold::landmark_ids",
            ));
        }
    }

    let mut dists_u32 = Vec::with_capacity(landmark_dists.len());
    for &d in landmark_dists {
        if d.is_nan() {
            return Err(CoreError::non_finite(
                "landmark_stratified_rta_from_landmarks_f64_manifold::landmark_dists",
            ));
        }
        if d == f64::INFINITY {
            dists_u32.push(u32::MAX);
        } else if !d.is_finite() || d < 0.0 || d > (u32::MAX as f64) || d.fract() != 0.0 {
            return Err(CoreError::invalid_input(format!(
                "landmark_dists entries must be non-negative finite integers <= u32::MAX or +inf, got {d}"
            )));
        } else {
            dists_u32.push(d as u32);
        }
    }

    landmark_stratified_rta_from_landmarks_manifold(
        layout,
        landmark_ids,
        &dists_u32,
        num_triplets,
        seed,
    )
}
