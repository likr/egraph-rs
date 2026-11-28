use egraph_core::{CsrGraph, DeterministicRng, VisitStamp};

use crate::sampling_utils::has_common_neighbor_non_hub;

pub(crate) enum LandmarkTable<'a> {
    None,
    RowMajor {
        data: &'a [f64],
        num_landmarks: usize,
    },
    ColMajor {
        data: Vec<f64>,
        num_landmarks: usize,
    },
}

impl LandmarkTable<'_> {
    #[inline]
    pub(crate) fn num_landmarks(&self) -> usize {
        match self {
            Self::None => 0,
            Self::RowMajor { num_landmarks, .. } | Self::ColMajor { num_landmarks, .. } => {
                *num_landmarks
            }
        }
    }

    #[inline]
    pub(crate) fn dist(&self, node: usize, lm: usize, num_nodes: usize) -> f64 {
        match self {
            Self::None => f64::INFINITY,
            Self::RowMajor {
                data,
                num_landmarks,
            } => data[node * (*num_landmarks) + lm],
            Self::ColMajor { data, .. } => data[lm * num_nodes + node],
        }
    }
}

pub(crate) struct PacmapTierPairs {
    pub(crate) nb_pairs: Vec<(u32, u32)>,
    pub(crate) mn_pairs: Vec<(u32, u32)>,
    pub(crate) fp_pairs: Vec<(u32, u32)>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_pacmap_tier_pairs(
    sym_graph: &CsrGraph<f64>,
    adj_indptr: &[u32],
    weighted_rows: Option<&[Vec<(f64, u32)>]>,
    lm_table: &LandmarkTable<'_>,
    n: usize,
    n_canon_edges: usize,
    n_nb: usize,
    n_mn: usize,
    n_fp: usize,
    keep_all_1hop: bool,
    max_1hop_per_node: Option<usize>,
    rng: &mut DeterministicRng,
) -> PacmapTierPairs {
    let num_lm = lm_table.num_landmarks();
    let mut hop1_stamp = VisitStamp::new(n);
    let mut exclude_stamp = VisitStamp::new(n);

    let hop1_cap: usize = match (keep_all_1hop, max_1hop_per_node) {
        (true, None) => usize::MAX,
        (true, Some(cap)) => cap,
        (false, None) => n_nb,
        (false, Some(cap)) => n_nb.min(cap),
    };

    let nb_cap = if keep_all_1hop {
        let capped_edges = match max_1hop_per_node {
            Some(cap) => n_canon_edges.min(n.saturating_mul(cap)),
            None => n_canon_edges,
        };
        (n * n_nb).max(capped_edges.saturating_mul(2))
    } else {
        n * n_nb
    };
    let mut nb_pairs = Vec::with_capacity(nb_cap);
    let mut mn_pairs = Vec::with_capacity(n * n_mn);
    let mut fp_pairs = Vec::with_capacity(n * n_fp);

    let hub_thresh = 64usize.max(n.isqrt());
    let non_hub_thresh = 64usize.min(16usize.max(n.isqrt()));
    let mut nb_u: Vec<u32> = Vec::with_capacity(n_nb);
    let mut n1_sorted_scratch: Vec<(usize, u32)> = Vec::new();
    let mut row1_scratch: Vec<u32> = Vec::new();
    let mut hop2_raw: Vec<(u32, f64)> = Vec::new();
    let mut hop2_ranked: Vec<(f64, u32)> = Vec::new();

    for u in 0..n {
        let u_u32 = u as u32;
        let n1_u = sym_graph.neighbors(u);

        hop1_stamp.clear();
        hop1_stamp.mark(u);
        for &w in n1_u {
            hop1_stamp.mark(w as usize);
        }

        let need_sorted_n1 = !n1_u.is_empty()
            && ((hop1_cap != usize::MAX && weighted_rows.is_none() && n1_u.len() > hop1_cap)
                || n1_u.len().min(hop1_cap) < n_nb);
        if need_sorted_n1 {
            n1_sorted_scratch.clear();
            for &w in n1_u {
                n1_sorted_scratch.push((sym_graph.degree(w as usize), w));
            }
            n1_sorted_scratch.sort_unstable();
        }

        nb_u.clear();
        if hop1_cap == usize::MAX {
            if let Some(w_rows) = weighted_rows {
                for &(_, v) in &w_rows[u] {
                    nb_u.push(v);
                }
            } else {
                nb_u.extend_from_slice(n1_u);
            }
        } else if let Some(w_rows) = weighted_rows {
            for &(_, v) in w_rows[u].iter().take(hop1_cap) {
                nb_u.push(v);
            }
        } else if n1_u.len() <= hop1_cap {
            nb_u.extend_from_slice(n1_u);
        } else if hop1_cap > 0 {
            // Keep all neighbors strictly below the cutoff degree deterministically, and only
            // shuffle the tied-degree boundary slice when it overflows `hop1_cap`.
            let d_cut = n1_sorted_scratch[hop1_cap - 1].0;
            let tie_start = n1_sorted_scratch.partition_point(|&(d, _)| d < d_cut);
            let tie_end = n1_sorted_scratch.partition_point(|&(d, _)| d <= d_cut);
            for &(_, w) in &n1_sorted_scratch[..tie_start] {
                nb_u.push(w);
            }
            if tie_end == hop1_cap {
                for &(_, w) in &n1_sorted_scratch[tie_start..hop1_cap] {
                    nb_u.push(w);
                }
            } else {
                row1_scratch.clear();
                for &(_, w) in &n1_sorted_scratch[tie_start..tie_end] {
                    row1_scratch.push(w);
                }
                rng.shuffle(&mut row1_scratch);
                nb_u.extend_from_slice(&row1_scratch[..(hop1_cap - tie_start)]);
            }
        }

        if nb_u.len() < n_nb && !n1_u.is_empty() {
            hop2_raw.clear();
            let cand_cap = (n_nb * 8).max(64);
            let hard_cand_cap = cand_cap.saturating_mul(16);
            let mut prev_deg = 0usize;
            for &(deg_w, w) in &n1_sorted_scratch {
                if deg_w <= 1 {
                    continue;
                }
                if deg_w > hub_thresh {
                    break;
                }
                if deg_w != prev_deg && hop2_raw.len() >= cand_cap {
                    break;
                }
                if n1_u.len() >= n_nb && hop2_raw.len() >= hard_cand_cap {
                    break;
                }
                prev_deg = deg_w;
                let aa_w = 1.0_f64 / (deg_w as f64).ln();
                for &v in sym_graph.neighbors(w as usize) {
                    if !hop1_stamp.is_visited(v as usize) {
                        hop2_raw.push((v, aa_w));
                    }
                }
            }
            if !hop2_raw.is_empty() {
                // Canonical `(v asc, aa_w desc)` order makes floating-point Adamic-Adar
                // summation bit-exact invariant under arbitrary vertex permutations.
                hop2_raw.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
                hop2_ranked.clear();
                let mut idx = 0usize;
                while idx < hop2_raw.len() {
                    let v = hop2_raw[idx].0;
                    let mut aa_sum = 0.0_f64;
                    while idx < hop2_raw.len() && hop2_raw[idx].0 == v {
                        aa_sum += hop2_raw[idx].1;
                        idx += 1;
                    }
                    hop2_ranked.push((aa_sum, v));
                }
                hop2_ranked
                    .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                let rem = n_nb - nb_u.len();
                for &(_, v) in hop2_ranked.iter().take(rem) {
                    nb_u.push(v);
                }
            }
        }

        // Canonicalize undirected NB pairs `(min, max)` so reciprocal selections are deduplicated
        // instead of applying double attractive gradients.
        for &v in &nb_u {
            nb_pairs.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
        }

        exclude_stamp.clear();
        exclude_stamp.mark(u);
        for &v in n1_u {
            exclude_stamp.mark(v as usize);
        }
        for &v in &nb_u {
            exclude_stamp.mark(v as usize);
        }

        // Mid-Near (`MN`): sample up to 6 non-1/2-hop candidates per slot and pick the
        // nearest landmark-estimated candidate.
        if n_mn > 0 {
            let mut mn_collected = 0usize;
            let mut slot_failures = 0usize;
            while mn_collected < n_mn && slot_failures < 4 {
                let mut best_cand: Option<(f64, u32)> = None;
                let mut pool_count = 0usize;
                let mut trials = 0usize;
                while pool_count < 6 && trials < 18 {
                    trials += 1;
                    let c = rng.gen_range_usize(n);
                    if exclude_stamp.is_visited(c)
                        || has_common_neighbor_non_hub(
                            n1_u,
                            sym_graph.neighbors(c),
                            &hop1_stamp,
                            adj_indptr,
                            non_hub_thresh,
                        )
                    {
                        continue;
                    }
                    pool_count += 1;
                    let mut lb = 0.0_f64;
                    let mut ub = f64::INFINITY;
                    for l in 0..num_lm {
                        let du = lm_table.dist(u, l, n);
                        let dc = lm_table.dist(c, l, n);
                        if du.is_finite() && dc.is_finite() {
                            lb = lb.max((du - dc).abs());
                            ub = ub.min(du + dc);
                        }
                    }
                    let est = if ub.is_finite() {
                        0.5 * (lb + ub)
                    } else {
                        f64::INFINITY
                    };
                    let c_u32 = c as u32;
                    if best_cand.is_none_or(|(b_est, b_c)| {
                        est.total_cmp(&b_est).then_with(|| c_u32.cmp(&b_c)).is_lt()
                    }) {
                        best_cand = Some((est, c_u32));
                    }
                }
                if let Some((_, c_u32)) = best_cand {
                    exclude_stamp.mark(c_u32 as usize);
                    mn_pairs.push((u_u32, c_u32));
                    mn_collected += 1;
                    slot_failures = 0;
                } else {
                    slot_failures += 1;
                }
            }
        }

        // Further (`FP`): sample non-neighbor repulsive pairs excluding u, 1-hop, and NB.
        if n_fp > 0 {
            exclude_stamp.clear();
            let mut n_excluded = 0usize;
            if exclude_stamp.visit(u) {
                n_excluded += 1;
            }
            for &v in n1_u {
                if exclude_stamp.visit(v as usize) {
                    n_excluded += 1;
                }
            }
            for &v in &nb_u {
                if exclude_stamp.visit(v as usize) {
                    n_excluded += 1;
                }
            }
            if n_excluded < n {
                let mut fp_collected = 0usize;
                let max_fp_trials = (n_fp * 4).max(16);
                let mut trials = 0usize;
                while fp_collected < n_fp && n_excluded < n && trials < max_fp_trials {
                    trials += 1;
                    let c = rng.gen_range_usize(n);
                    if exclude_stamp.is_visited(c)
                        || has_common_neighbor_non_hub(
                            n1_u,
                            sym_graph.neighbors(c),
                            &hop1_stamp,
                            adj_indptr,
                            non_hub_thresh,
                        )
                    {
                        continue;
                    }
                    if exclude_stamp.visit(c) {
                        n_excluded += 1;
                    }
                    fp_pairs.push((u_u32, c as u32));
                    fp_collected += 1;
                }

                // Two-stage bounded fallback relaxing only the 2-hop check while keeping u,
                // 1-hop, and NB strictly excluded.
                if fp_collected < n_fp && n_excluded < n {
                    let rem = (n_fp - fp_collected).min(n - n_excluded);
                    let fallback_trials = (rem * 4).max(16);
                    for _ in 0..fallback_trials {
                        if fp_collected >= n_fp || n_excluded >= n {
                            break;
                        }
                        let c = rng.gen_range_usize(n);
                        if exclude_stamp.visit(c) {
                            n_excluded += 1;
                            fp_pairs.push((u_u32, c as u32));
                            fp_collected += 1;
                        }
                    }
                    if fp_collected < n_fp && n_excluded < n {
                        let start = rng.gen_range_usize(n);
                        for offset in 0..n {
                            if fp_collected >= n_fp || n_excluded >= n {
                                break;
                            }
                            let c = (start + offset) % n;
                            if exclude_stamp.visit(c) {
                                n_excluded += 1;
                                fp_pairs.push((u_u32, c as u32));
                                fp_collected += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    PacmapTierPairs {
        nb_pairs,
        mn_pairs,
        fp_pairs,
    }
}
