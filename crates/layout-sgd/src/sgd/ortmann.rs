use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::select_maxmin_sp_pivots;

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::graph_sampling::{build_positive_canonical_csr, compute_mssp_dists};
use super::ortmann_pivots::{select_maxmin_random_sp_pivots, select_weighted_maxmin_sp_pivots};
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

pub(super) struct BuiltOrtmann<S: Scalar> {
    pub(super) num_nodes: usize,
    pub(super) storage: PairStorage<S>,
    pub(super) schedule_cfg: SgdScheduleConfig,
}

pub(super) fn build_ortmann_storage<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: OrtmannSparseStressConfig,
) -> CoreResult<BuiltOrtmann<S>> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::ortmann_sparse_stress")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }

    let schedule_cfg = SgdScheduleConfig {
        iterations: config.iterations,
        eps: config.eps,
        kind: SgdScheduleKind::Exponential,
        step_scale: 1.0,
    };
    if n <= 1 {
        return Ok(BuiltOrtmann {
            num_nodes: n,
            storage: PairStorage::Symmetric(Vec::new()),
            schedule_cfg,
        });
    }

    let (sym_graph, positive_edges) = build_positive_canonical_csr(graph)?;
    if positive_edges.is_empty() {
        return Ok(BuiltOrtmann {
            num_nodes: n,
            storage: PairStorage::Symmetric(Vec::new()),
            schedule_cfg,
        });
    }

    let target_k = config.num_pivots.min(n);
    let (pivots, piv_dists) =
        select_ortmann_pivots_and_dists(&sym_graph, target_k, config.pivot_strategy, config.seed)?;
    let h = pivots.len();

    let mut acc = DirectedPairAccumulator::<S>::with_capacity(positive_edges.len() + h * n);
    let mut edges_1hop: Vec<(u32, u32)> = Vec::with_capacity(positive_edges.len());
    for &(u, v, d_uv) in &positive_edges {
        edges_1hop.push((u, v));
        let w_uv = 1.0 / (d_uv * d_uv);
        acc.push_f64(u, v, d_uv, d_uv, w_uv, w_uv);
    }
    // `extract_canonical_undirected_edges` may emit unsorted entries if input rows are unsorted.
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

    for (k, &p_u32) in pivots.iter().enumerate() {
        let p_usize = p_u32 as usize;
        for i in 0..n {
            if i == p_usize {
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
            let w_pi = (count as f64) / (dpi * dpi);
            acc.push_f64(p_u32, i_u32, dpi, dpi, 0.0, w_pi);
        }
    }

    Ok(BuiltOrtmann {
        num_nodes: n,
        storage: acc.into_storage(),
        schedule_cfg,
    })
}
