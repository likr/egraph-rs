use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, MetricSpace};

use crate::spatial_index::validate_graph_layout;

use super::StressEvaluation;
use super::bfs::{BfsScratch, bfs_single_source};
use super::sns::{SnsMoments, finalize_moments};

/// Computes $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`) from `num_pivots`
/// deterministic [`DeterministicRng`] source vertices across any [`MetricSpace`].
pub fn sampled_scale_normalized_stress<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(sampled_scale_normalized_stress_detail(graph, layout, num_pivots, seed)?.normalized_stress)
}

/// Selects up to `num_pivots` deterministic pivots stratified by vertex degree.
///
/// Ignores isolated vertices (`degree == 0`) whenever at least `num_pivots` non-isolated
/// vertices exist, sorting non-isolated vertices by `(degree, id)` into `k` contiguous
/// quantile strata and drawing one pivot per stratum with `DeterministicRng`. When
/// `num_pivots >= non_isolated_count`, all non-isolated vertices are selected (and topped
/// up with isolated vertices if `non_isolated_count < min(num_pivots, n)`), guaranteeing
/// exact parity with [`super::scale_normalized_stress`].
pub(super) fn select_degree_stratified_pivots<W>(
    graph: &CsrGraph<W>,
    num_pivots: usize,
    seed: u64,
) -> Vec<usize> {
    let n = graph.num_nodes();
    let target = num_pivots.min(n);
    if target == 0 {
        return Vec::new();
    }

    let mut non_iso: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        let deg = graph.degree(u);
        if deg > 0 {
            non_iso.push((deg, u));
        }
    }
    let m = non_iso.len();
    if m <= target {
        let mut pivots: Vec<usize> = non_iso.into_iter().map(|(_, u)| u).collect();
        if pivots.len() < target {
            for u in 0..n {
                if graph.degree(u) == 0 {
                    pivots.push(u);
                    if pivots.len() == target {
                        break;
                    }
                }
            }
        }
        return pivots;
    }

    non_iso.sort_unstable();
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut pivots = Vec::with_capacity(target);
    let m_u128 = m as u128;
    let k_u128 = target as u128;
    for b in 0..target {
        let start = (((b as u128) * m_u128) / k_u128) as usize;
        let end = ((((b + 1) as u128) * m_u128) / k_u128) as usize;
        let pick = start + rng.gen_range_usize(end - start);
        pivots.push(non_iso[pick].1);
    }
    pivots
}

/// Computes full [`StressEvaluation`] details for $O(kN)$ pivot-sampled `SNS` across any [`MetricSpace`].
pub fn sampled_scale_normalized_stress_detail<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<StressEvaluation> {
    if num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_scale_normalized_stress")?;
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
            let r = layout.distance(p, j);
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
            let r = layout.distance(p, j);
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
