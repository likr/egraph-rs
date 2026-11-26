//! Landmark-Stratified Random Triplet Accuracy (`landmark_stratified_rta`).
//!
//! Uses $O(N)$-RAM 2-pass streaming BFS from Core-First MaxMin shortest-path pivots,
//! stratifies comparable `(u, v)` pairs into `floor(log2(d))` octave bins `(b1 <= b2)`,
//! and evaluates exact enumeration deterministically whenever `total_comparable <= num_triplets`.

mod evaluate;
mod manifold;
mod nd;
mod row_stats;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};

use crate::validate_graph_layout;

use evaluate::evaluate_rta_core;
pub use manifold::{
    landmark_stratified_rta_detail_manifold, landmark_stratified_rta_from_landmarks_f64_manifold,
    landmark_stratified_rta_from_landmarks_manifold, landmark_stratified_rta_manifold,
};
pub use nd::{
    landmark_stratified_rta_detail_nd, landmark_stratified_rta_from_landmarks_f64_nd,
    landmark_stratified_rta_from_landmarks_nd, landmark_stratified_rta_nd,
};
use row_stats::{RowProvider, is_layout_collapsed};

/// Detailed result returned by [`landmark_stratified_rta_detail`] and
/// [`landmark_stratified_rta_from_landmarks`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RtaEvaluation {
    /// Landmark-Stratified Random Triplet Accuracy in `[0.0, 1.0]`.
    pub rta: f64,
    /// Number of landmark vertices evaluated.
    pub num_landmarks: usize,
    /// Number of triplets evaluated (or all comparable triplets when `is_exact` is `true`).
    pub num_triplets: usize,
    /// Number of non-empty `(b1 <= b2)` log-distance octave strata.
    pub num_strata: usize,
    /// `true` when `total_comparable <= num_triplets` triggered exact enumeration.
    pub is_exact: bool,
}

/// Evaluates Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`,
/// returning detailed evaluation metadata ([`RtaEvaluation`]).
pub fn landmark_stratified_rta_detail<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_landmarks: usize,
    num_triplets: usize,
    min_degree: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    validate_graph_layout(graph, layout, "landmark_stratified_rta")?;
    if num_landmarks == 0 {
        return Err(CoreError::invalid_input("num_landmarks must be >= 1"));
    }
    if num_triplets == 0 {
        return Err(CoreError::invalid_input("num_triplets must be >= 1"));
    }
    let n = layout.len();
    if n < 3 || graph.num_edges() == 0 || is_layout_collapsed(layout) {
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

/// Evaluates Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`.
pub fn landmark_stratified_rta<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_landmarks: usize,
    num_triplets: usize,
    min_degree: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(landmark_stratified_rta_detail(
        graph,
        layout,
        num_landmarks,
        num_triplets,
        min_degree,
        seed,
    )?
    .rta)
}

/// Evaluates Landmark-Stratified Random Triplet Accuracy from a precomputed flat row-major
/// `(num_landmarks, N)` integer hop-distance table (`u32::MAX` denotes unreachable).
pub fn landmark_stratified_rta_from_landmarks<S: Scalar>(
    layout: &Layout2d<S>,
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
            "landmark_stratified_rta_from_landmarks::landmark_dists",
        ));
    }
    for &lid in landmark_ids {
        if (lid as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lid as usize,
                n,
                "landmark_stratified_rta_from_landmarks::landmark_ids",
            ));
        }
    }
    if n < 3 || p_l == 0 || is_layout_collapsed(layout) {
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

/// Evaluates Landmark-Stratified Random Triplet Accuracy from a precomputed flat row-major
/// `(num_landmarks, N)` `f64` distance table, mapping `f64::INFINITY` to the unreachable
/// sentinel (`u32::MAX`) and rejecting `NaN`, negative values, `-inf`, non-integer finite values,
/// or finite values exceeding `u32::MAX`.
pub fn landmark_stratified_rta_from_landmarks_f64<S: Scalar>(
    layout: &Layout2d<S>,
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
            "landmark_stratified_rta_from_landmarks_f64::landmark_dists",
        ));
    }
    for &lid in landmark_ids {
        if (lid as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lid as usize,
                n,
                "landmark_stratified_rta_from_landmarks_f64::landmark_ids",
            ));
        }
    }

    let mut dists_u32 = Vec::with_capacity(landmark_dists.len());
    for &d in landmark_dists {
        if d.is_nan() {
            return Err(CoreError::non_finite(
                "landmark_stratified_rta_from_landmarks_f64::landmark_dists",
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

    landmark_stratified_rta_from_landmarks(layout, landmark_ids, &dists_u32, num_triplets, seed)
}
