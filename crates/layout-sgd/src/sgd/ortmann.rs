use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::select_maxmin_sp_pivots;

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::graph_helpers::{build_positive_canonical_csr, compute_mssp_dists};
use super::ortmann_pivots::{select_maxmin_random_sp_pivots, select_weighted_maxmin_sp_pivots};
use super::pair_buffer::{SgdPairKind, SgdPairRecord};
use super::pairs::{
    DirectedPairAccumulator, MAX_PACKED_NODES, PairStorage, validate_graph_weights,
};
use super::variant_config::{OrtmannPivotStrategy, OrtmannSparseStressConfig};

fn select_kmeans_sp_pivots(
    graph: &CsrGraph<f64>,
    target_k: usize,
    seed: u64,
) -> CoreResult<(Vec<u32>, Vec<f64>)> {
    let n = graph.num_nodes();
    if target_k == 0 || n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut pivots = if graph.is_weighted() {
        select_weighted_maxmin_sp_pivots(graph, target_k, seed, 2)?
    } else {
        select_maxmin_sp_pivots(graph, target_k, seed, 2)?
    };
    let h = pivots.len();
    let mut piv_dists = compute_mssp_dists(graph, &pivots)?;

    if h < 2 || h >= n || h.saturating_mul(n) > 2_000_000 {
        return Ok((pivots, piv_dists));
    }

    let mut max_finite = 0.0_f64;
    for &d in &piv_dists {
        if d.is_finite() {
            max_finite = max_finite.max(d);
        }
    }
    if max_finite <= 0.0 {
        return Ok((pivots, piv_dists));
    }
    let fallback = 2.0 * max_finite;

    let mut coords = vec![0.0_f64; n * h];
    for j in 0..n {
        for k in 0..h {
            let d = piv_dists[k * n + j];
            coords[j * h + k] = if d.is_finite() { d } else { fallback };
        }
    }

    let mut assign = vec![0usize; n];
    for j in 0..n {
        let mut best_k = 0usize;
        let mut best_d = f64::INFINITY;
        for k in 0..h {
            let d = piv_dists[k * n + j];
            if d.is_finite() && d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        assign[j] = if best_d.is_finite() { best_k } else { j % h };
    }

    let mut centroids = vec![0.0_f64; h * h];
    let mut counts = vec![0usize; h];
    let mut new_pivots = vec![0u32; h];
    let mut used = vec![false; n];

    for round in 0..5 {
        centroids.fill(0.0);
        counts.fill(0);
        for j in 0..n {
            let c = assign[j];
            counts[c] += 1;
            let row_j = &coords[j * h..(j + 1) * h];
            let row_c = &mut centroids[c * h..(c + 1) * h];
            for d_idx in 0..h {
                row_c[d_idx] += row_j[d_idx];
            }
        }
        for c in 0..h {
            let cnt = counts[c];
            if cnt > 0 {
                let inv = 1.0 / (cnt as f64);
                let row_c = &mut centroids[c * h..(c + 1) * h];
                for val in row_c.iter_mut() {
                    *val *= inv;
                }
            } else {
                let p = pivots[c] as usize;
                centroids[c * h..(c + 1) * h].copy_from_slice(&coords[p * h..(p + 1) * h]);
            }
        }

        if round > 0 {
            for j in 0..n {
                let row_j = &coords[j * h..(j + 1) * h];
                let mut best_c = 0usize;
                let mut best_sq = f64::INFINITY;
                for c in 0..h {
                    let row_c = &centroids[c * h..(c + 1) * h];
                    let mut sq = 0.0_f64;
                    for d_idx in 0..h {
                        let diff = row_j[d_idx] - row_c[d_idx];
                        sq += diff * diff;
                    }
                    if sq < best_sq {
                        best_sq = sq;
                        best_c = c;
                    }
                }
                assign[j] = best_c;
            }
        }

        used.fill(false);
        for c in 0..h {
            let row_c = &centroids[c * h..(c + 1) * h];
            let mut best_in_cluster: Option<(usize, f64)> = None;
            let mut best_any: Option<(usize, f64)> = None;
            for j in 0..n {
                if used[j] {
                    continue;
                }
                let row_j = &coords[j * h..(j + 1) * h];
                let mut sq = 0.0_f64;
                for d_idx in 0..h {
                    let diff = row_j[d_idx] - row_c[d_idx];
                    sq += diff * diff;
                }
                if assign[j] == c && best_in_cluster.is_none_or(|(_, bsq)| sq < bsq) {
                    best_in_cluster = Some((j, sq));
                }
                if best_any.is_none_or(|(_, bsq)| sq < bsq) {
                    best_any = Some((j, sq));
                }
            }
            if let Some((best_j, _)) = best_in_cluster.or(best_any) {
                used[best_j] = true;
                new_pivots[c] = best_j as u32;
            }
        }

        if new_pivots == pivots {
            break;
        }
        pivots.copy_from_slice(&new_pivots);
    }

    piv_dists = compute_mssp_dists(graph, &pivots)?;
    Ok((pivots, piv_dists))
}

pub(super) fn select_ortmann_pivots_and_dists(
    graph: &CsrGraph<f64>,
    target_k: usize,
    strategy: OrtmannPivotStrategy,
    seed: u64,
) -> CoreResult<(Vec<u32>, Vec<f64>)> {
    match strategy {
        OrtmannPivotStrategy::MaxMinSp => {
            let pivots = if graph.is_weighted() {
                select_weighted_maxmin_sp_pivots(graph, target_k, seed, 2)?
            } else {
                select_maxmin_sp_pivots(graph, target_k, seed, 2)?
            };
            let piv_dists = compute_mssp_dists(graph, &pivots)?;
            Ok((pivots, piv_dists))
        }
        OrtmannPivotStrategy::MaxMinRandomSp => {
            select_maxmin_random_sp_pivots(graph, target_k, seed)
        }
        OrtmannPivotStrategy::KMeansSp => select_kmeans_sp_pivots(graph, target_k, seed),
    }
}

/// Decoupled Problem IR for Ortmann et al. (2016) Sparse Stress SGD.
///
/// Separates read-only pivot-to-non-pivot pairs (`one_sided`, with `w_uv == 0`) from symmetric or
/// mutually weighted `two_sided` pairs (1-hop edges and merged non-edge pivot-pivot pairs).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct OrtmannSparseProblem<S: Scalar = f64> {
    /// Expected number of vertices in the layout.
    pub num_nodes: usize,
    /// Selected shortest-path pivot vertex indices.
    pub pivots: Vec<u32>,
    /// One-sided pivot-to-non-pivot spring pairs (`u` in `pivots`, `v` non-pivot, `w_uv == 0`).
    pub one_sided: Vec<SgdPairRecord<S>>,
    /// Two-sided spring pairs (canonical 1-hop edges and merged non-edge pivot-pivot pairs `u < v`).
    pub two_sided: Vec<SgdPairRecord<S>>,
    /// Minimum positive stored weight across `one_sided` and `two_sided` (`0.0` if empty).
    pub w_min: f64,
    /// Maximum positive stored weight across `one_sided` and `two_sided` (`0.0` if empty).
    pub w_max: f64,
    /// Derived SGD learning rate schedule configuration.
    pub schedule_cfg: SgdScheduleConfig,
}

#[inline]
fn record_positive_weight<S: Scalar>(w: S, w_min: &mut f64, w_max: &mut f64) {
    let wf = w.to_f64();
    if wf.is_finite() && wf > 0.0 {
        *w_min = (*w_min).min(wf);
        *w_max = (*w_max).max(wf);
    }
}

/// Builds the decoupled [`OrtmannSparseProblem`] Problem IR for Ortmann et al. (2016) Sparse Stress.
///
/// Separates read-only pivot-to-non-pivot pairs (`one_sided`) from 1-hop edges and merged
/// non-edge pivot-pivot pairs (`two_sided`), preserving 1-hop edge precedence and Voronoi region
/// weights $w'_{ip} = |\{j \in R(p) : 2 d_{jp} \le d_{ip}\}| / d_{ip}^2$.
pub fn build_ortmann_sparse_problem<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: OrtmannSparseStressConfig,
) -> CoreResult<OrtmannSparseProblem<S>> {
    config.validate()?;
    validate_graph_weights(graph, "build_ortmann_sparse_problem")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }

    let schedule_cfg = SgdScheduleConfig::new(config.iterations)
        .with_eps(config.eps)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);
    if n <= 1 {
        return Ok(OrtmannSparseProblem {
            num_nodes: n,
            pivots: Vec::new(),
            one_sided: Vec::new(),
            two_sided: Vec::new(),
            w_min: 0.0,
            w_max: 0.0,
            schedule_cfg,
        });
    }

    let (sym_graph, positive_edges) = build_positive_canonical_csr(graph)?;
    if positive_edges.is_empty() {
        return Ok(OrtmannSparseProblem {
            num_nodes: n,
            pivots: Vec::new(),
            one_sided: Vec::new(),
            two_sided: Vec::new(),
            w_min: 0.0,
            w_max: 0.0,
            schedule_cfg,
        });
    }

    let target_k = config.num_pivots.min(n);
    let (pivots, piv_dists) =
        select_ortmann_pivots_and_dists(&sym_graph, target_k, config.pivot_strategy, config.seed)?;
    let h = pivots.len();

    let mut pivot_slot: Vec<Option<usize>> = vec![None; n];
    for (k, &p) in pivots.iter().enumerate() {
        debug_assert!(pivot_slot[p as usize].is_none());
        pivot_slot[p as usize] = Some(k);
    }

    let mut w_min_raw = f64::INFINITY;
    let mut w_max_raw = 0.0_f64;
    let mut two_sided: Vec<SgdPairRecord<S>> =
        Vec::with_capacity(positive_edges.len() + h.saturating_mul(h.saturating_sub(1)) / 2);
    let mut edges_1hop: Vec<(u32, u32)> = Vec::with_capacity(positive_edges.len());
    for &(u, v, d_uv) in &positive_edges {
        edges_1hop.push((u, v));
        let d_s = S::from_f64(d_uv);
        let w_s = S::from_f64(1.0 / (d_uv * d_uv));
        if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
            two_sided.push(SgdPairRecord::new(
                u,
                v,
                SgdPairKind::Spring,
                d_s,
                d_s,
                w_s,
                w_s,
            ));
            record_positive_weight(w_s, &mut w_min_raw, &mut w_max_raw);
        }
    }
    if !edges_1hop.windows(2).all(|w| w[0] < w[1]) {
        edges_1hop.sort_unstable();
    }

    // Incremental Voronoi assignment R(p): break distance ties on smallest current |R(p)|,
    // then lowest pivot slot k (Ortmann et al. 2016, Section 3).
    let mut r_counts = vec![0usize; h];
    let mut r_dists: Vec<Vec<f64>> = vec![Vec::new(); h];
    for j in 0..n {
        let mut best: Option<(f64, usize, usize)> = None;
        for k in 0..h {
            let d = piv_dists[k * n + j];
            if !d.is_finite() {
                continue;
            }
            let cand = (d, r_counts[k], k);
            if best.is_none_or(|b| {
                cand.0
                    .total_cmp(&b.0)
                    .then_with(|| cand.1.cmp(&b.1))
                    .then_with(|| cand.2.cmp(&b.2))
                    .is_lt()
            }) {
                best = Some(cand);
            }
        }
        if let Some((d, _, best_k)) = best {
            r_counts[best_k] += 1;
            r_dists[best_k].push(d);
        }
    }
    for dists_k in &mut r_dists {
        dists_k.sort_unstable_by(f64::total_cmp);
    }

    // Merge non-edge pivot-pivot pairs (u, v) with u < v into two_sided:
    // w_uv moves u toward v (governed by R(v)), while w_vu moves v toward u (governed by R(u)).
    for k1 in 0..h {
        for k2 in (k1 + 1)..h {
            let p1 = pivots[k1];
            let p2 = pivots[k2];
            let (u, v, ku, kv) = if p1 < p2 {
                (p1, p2, k1, k2)
            } else {
                (p2, p1, k2, k1)
            };
            if edges_1hop.binary_search(&(u, v)).is_ok() {
                continue;
            }
            let d = piv_dists[ku * n + (v as usize)];
            let d_rev = piv_dists[kv * n + (u as usize)];
            if !d.is_finite() || d <= 0.0 || !d_rev.is_finite() || d_rev <= 0.0 {
                continue;
            }
            let count_v = r_dists[kv].partition_point(|&dpj| dpj.is_finite() && 2.0 * dpj <= d);
            let count_u = r_dists[ku].partition_point(|&dpj| dpj.is_finite() && 2.0 * dpj <= d);
            let d_s = S::from_f64(d);
            let w_uv_s = S::from_f64((count_v as f64) / (d * d));
            let w_vu_s = S::from_f64((count_u as f64) / (d * d));
            if !d_s.is_finite()
                || d_s <= S::ZERO
                || !w_uv_s.is_finite()
                || w_uv_s < S::ZERO
                || !w_vu_s.is_finite()
                || w_vu_s < S::ZERO
                || (w_uv_s == S::ZERO && w_vu_s == S::ZERO)
            {
                continue;
            }
            two_sided.push(SgdPairRecord::new(
                u,
                v,
                SgdPairKind::Spring,
                d_s,
                d_s,
                w_uv_s,
                w_vu_s,
            ));
            record_positive_weight(w_uv_s, &mut w_min_raw, &mut w_max_raw);
            record_positive_weight(w_vu_s, &mut w_min_raw, &mut w_max_raw);
        }
    }

    let mut one_sided: Vec<SgdPairRecord<S>> = Vec::with_capacity(h * n.saturating_sub(h));
    for (k, &p_u32) in pivots.iter().enumerate() {
        for i in 0..n {
            if pivot_slot[i].is_some() {
                continue;
            }
            let i_u32 = i as u32;
            let edge_key = if p_u32 < i_u32 {
                (p_u32, i_u32)
            } else {
                (i_u32, p_u32)
            };
            if edges_1hop.binary_search(&edge_key).is_ok() {
                continue;
            }
            let dpi = piv_dists[k * n + i];
            if !dpi.is_finite() || dpi <= 0.0 {
                continue;
            }
            let count = r_dists[k].partition_point(|&dpj| dpj.is_finite() && 2.0 * dpj <= dpi);
            if count == 0 {
                continue;
            }
            let d_s = S::from_f64(dpi);
            let w_s = S::from_f64((count as f64) / (dpi * dpi));
            if !d_s.is_finite() || d_s <= S::ZERO || !w_s.is_finite() || w_s <= S::ZERO {
                continue;
            }
            one_sided.push(SgdPairRecord::new(
                p_u32,
                i_u32,
                SgdPairKind::Spring,
                d_s,
                d_s,
                S::ZERO,
                w_s,
            ));
            record_positive_weight(w_s, &mut w_min_raw, &mut w_max_raw);
        }
    }

    let (w_min, w_max) = if w_max_raw > 0.0 && w_min_raw.is_finite() {
        (w_min_raw, w_max_raw)
    } else {
        (0.0, 0.0)
    };

    Ok(OrtmannSparseProblem {
        num_nodes: n,
        pivots,
        one_sided,
        two_sided,
        w_min,
        w_max,
        schedule_cfg,
    })
}

pub(super) fn ortmann_problem_into_storage<S: Scalar>(
    problem: &OrtmannSparseProblem<S>,
) -> PairStorage<S> {
    let mut acc = DirectedPairAccumulator::<S>::with_capacity(
        problem.two_sided.len() + problem.one_sided.len(),
    );
    for rec in problem.two_sided.iter().chain(problem.one_sided.iter()) {
        acc.push_s(rec.u, rec.v, rec.d_uv, rec.d_vu, rec.w_uv, rec.w_vu);
    }
    acc.into_storage()
}
