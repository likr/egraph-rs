use egraph_core::{CoreResult, DeterministicRng};

use super::RtaEvaluation;
use super::row_stats::{
    RowProvider, RowScratch, RtaCoordSpace, counting_sort_row, scan_row_octaves,
};

#[inline]
fn gen_range_u64(rng: &mut DeterministicRng, upper_exclusive: u64) -> u64 {
    if upper_exclusive <= 1 {
        return 0;
    }
    if let Ok(u32_lim) = u32::try_from(upper_exclusive) {
        return u64::from(rng.gen_range_u32(u32_lim));
    }
    let s = u128::from(upper_exclusive);
    let mut m = u128::from(rng.next_u64()) * s;
    let mut l = m as u64;
    if l < upper_exclusive {
        let t = upper_exclusive.wrapping_neg() % upper_exclusive;
        while l < t {
            m = u128::from(rng.next_u64()) * s;
            l = m as u64;
        }
    }
    (m >> 64) as u64
}

#[inline]
fn gen_range_u128(rng: &mut DeterministicRng, upper_exclusive: u128) -> u128 {
    if let Ok(u64_lim) = u64::try_from(upper_exclusive) {
        return u128::from(gen_range_u64(rng, u64_lim));
    }
    let mask = u128::MAX >> upper_exclusive.leading_zeros();
    loop {
        let r = ((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) & mask;
        if r < upper_exclusive {
            return r;
        }
    }
}

pub(super) fn evaluate_rta_core<W, L: RtaCoordSpace>(
    layout: &L,
    landmark_ids: &[u32],
    mut provider: RowProvider<'_, W>,
    num_triplets: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    let n = layout.num_nodes();
    let p_l = landmark_ids.len();
    if n < 3 || p_l == 0 || layout.is_collapsed() {
        return Ok(RtaEvaluation {
            rta: 0.0,
            num_landmarks: p_l,
            num_triplets: 0,
            num_strata: 0,
            is_exact: false,
        });
    }

    let mut scratch = RowScratch::new(n);
    let mut oct_counts_all = vec![[0_u128; 32]; p_l];
    let mut same_pairs_all = vec![[0_u128; 32]; p_l];
    let mut max_oct = -1_isize;

    // Pass 1: streaming row-by-row octave histogramming.
    for (i, &lid) in landmark_ids.iter().enumerate() {
        let row = provider.row(i, lid)?;
        let (oc, sp, row_max) = scan_row_octaves(row, &mut scratch);
        oct_counts_all[i] = oc;
        same_pairs_all[i] = sp;
        if row_max > max_oct {
            max_oct = row_max;
        }
    }

    if max_oct < 0 {
        return Ok(RtaEvaluation {
            rta: 0.0,
            num_landmarks: p_l,
            num_triplets: 0,
            num_strata: 0,
            is_exact: false,
        });
    }

    let n_oct = (max_oct as usize) + 1;
    let mut strata = Vec::<(usize, usize)>::new();
    let mut strata_lm_counts = Vec::<Vec<u128>>::new();
    let mut total_comparable = 0_u128;

    for b1 in 0..n_oct {
        for b2 in b1..n_oct {
            let mut lm_cnt = vec![0_u128; p_l];
            let mut s_tot = 0_u128;
            for i in 0..p_l {
                let c = if b1 < b2 {
                    oct_counts_all[i][b1] * oct_counts_all[i][b2]
                } else {
                    same_pairs_all[i][b1]
                };
                lm_cnt[i] = c;
                s_tot += c;
            }
            if s_tot > 0 {
                strata.push((b1, b2));
                strata_lm_counts.push(lm_cnt);
                total_comparable += s_tot;
            }
        }
    }

    if total_comparable == 0 {
        return Ok(RtaEvaluation {
            rta: 0.0,
            num_landmarks: p_l,
            num_triplets: 0,
            num_strata: 0,
            is_exact: false,
        });
    }

    let n_strata = strata.len();

    // Exact deterministic enumeration fallback when comparable triplets <= num_triplets.
    if total_comparable <= (num_triplets as u128) {
        let mut total_ok = 0_u128;
        let mut total_eval = 0_u128;
        let mut valid_nodes = Vec::<u32>::new();
        let mut valid_dists = Vec::<u32>::new();

        for (i, &lid) in landmark_ids.iter().enumerate() {
            let row = provider.row(i, lid)?;
            let src = lid as usize;

            valid_nodes.clear();
            valid_dists.clear();
            for (u, &d) in row.iter().enumerate() {
                if (1..u32::MAX).contains(&d) {
                    valid_nodes.push(u as u32);
                    valid_dists.push(d);
                }
            }
            let m_v = valid_nodes.len();
            if m_v < 2 {
                continue;
            }
            for a in 0..(m_v - 1) {
                let u = valid_nodes[a] as usize;
                let d1 = valid_dists[a];
                let r1_sq = layout.dist_sq(src, u);
                for b in (a + 1)..m_v {
                    let d2 = valid_dists[b];
                    if d1 == d2 {
                        continue;
                    }
                    let v = valid_nodes[b] as usize;
                    let r2_sq = layout.dist_sq(src, v);
                    total_eval += 1;
                    if (d1 < d2 && r1_sq < r2_sq) || (d1 > d2 && r1_sq > r2_sq) {
                        total_ok += 1;
                    }
                }
            }
        }

        let rta = if total_eval > 0 {
            (total_ok as f64) / (total_eval as f64)
        } else {
            0.0
        };
        return Ok(RtaEvaluation {
            rta,
            num_landmarks: p_l,
            num_triplets: total_eval as usize,
            num_strata: n_strata,
            is_exact: true,
        });
    }

    // Stratified sampling across active (b1 <= b2) octave strata using u128 prefix sums.
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let base_q = num_triplets / n_strata;
    let rem_q = num_triplets % n_strata;

    let mut lm_alloc = vec![0usize; p_l * n_strata];
    let mut prefix = vec![0_u128; p_l];
    for s_idx in 0..n_strata {
        let k_s = base_q + usize::from(s_idx < rem_q);
        if k_s == 0 {
            continue;
        }
        let weights = &strata_lm_counts[s_idx];
        let mut acc = 0_u128;
        for i in 0..p_l {
            acc += weights[i];
            prefix[i] = acc;
        }
        if acc == 0 {
            continue;
        }
        for _ in 0..k_s {
            let draw = gen_range_u128(&mut rng, acc);
            let i = prefix.partition_point(|&c| c <= draw);
            lm_alloc[i * n_strata + s_idx] += 1;
        }
    }

    let mut stratum_correct = vec![0_u64; n_strata];
    let mut stratum_evaluated = vec![0_u64; n_strata];
    let mut grp_starts = Vec::<usize>::new();
    let mut grp_counts = Vec::<usize>::new();
    let mut grp_prefix = Vec::<u128>::new();

    // Pass 2: stream active landmarks only and counting-sort once per landmark.
    for (i, &lid) in landmark_ids.iter().enumerate() {
        let row_alloc = &lm_alloc[i * n_strata..(i + 1) * n_strata];
        if row_alloc.iter().all(|&c| c == 0) {
            continue;
        }

        let row = provider.row(i, lid)?;
        let oct_offsets = counting_sort_row(row, &mut scratch);
        let src = lid as usize;

        for (s_idx, &k_draw) in row_alloc.iter().enumerate() {
            if k_draw == 0 {
                continue;
            }
            let (b1, b2) = strata[s_idx];
            let s1 = oct_offsets[b1];
            let e1 = oct_offsets[b1 + 1];
            let s2 = oct_offsets[b2];
            let e2 = oct_offsets[b2 + 1];

            let mut ok_cnt = 0_u64;
            let mut eval_cnt = 0_u64;

            if b1 < b2 {
                let len1 = e1 - s1;
                let len2 = e2 - s2;
                if len1 == 0 || len2 == 0 {
                    continue;
                }
                for _ in 0..k_draw {
                    let pos1 = s1 + rng.gen_range_usize(len1);
                    let pos2 = s2 + rng.gen_range_usize(len2);
                    let u = scratch.sorted_nodes[pos1] as usize;
                    let v = scratch.sorted_nodes[pos2] as usize;
                    let d1 = scratch.sorted_dists[pos1];
                    let d2 = scratch.sorted_dists[pos2];
                    let r1_sq = layout.dist_sq(src, u);
                    let r2_sq = layout.dist_sq(src, v);
                    eval_cnt += 1;
                    if (d1 < d2 && r1_sq < r2_sq) || (d1 > d2 && r1_sq > r2_sq) {
                        ok_cnt += 1;
                    }
                }
            } else {
                let tot_n = e1 - s1;
                if tot_n < 2 {
                    continue;
                }
                grp_starts.clear();
                grp_counts.clear();
                grp_prefix.clear();

                let mut pos = s1;
                while pos < e1 {
                    let cur_d = scratch.sorted_dists[pos];
                    let g_start = pos;
                    while pos < e1 && scratch.sorted_dists[pos] == cur_d {
                        pos += 1;
                    }
                    grp_starts.push(g_start);
                    grp_counts.push(pos - g_start);
                }
                if grp_starts.len() < 2 {
                    continue;
                }
                let mut acc = 0_u128;
                for &cnt in &grp_counts {
                    acc += (cnt as u128) * ((tot_n - cnt) as u128);
                    grp_prefix.push(acc);
                }
                if acc == 0 {
                    continue;
                }

                for _ in 0..k_draw {
                    let draw = gen_range_u128(&mut rng, acc);
                    let g1 = grp_prefix.partition_point(|&c| c <= draw);
                    let g_start = grp_starts[g1];
                    let g_cnt = grp_counts[g1];
                    let pos1 = g_start + rng.gen_range_usize(g_cnt);

                    let rem_len = tot_n - g_cnt;
                    let raw_p2 = s1 + rng.gen_range_usize(rem_len);
                    let pos2 = if raw_p2 >= g_start {
                        raw_p2 + g_cnt
                    } else {
                        raw_p2
                    };

                    let u = scratch.sorted_nodes[pos1] as usize;
                    let v = scratch.sorted_nodes[pos2] as usize;
                    let d1 = scratch.sorted_dists[pos1];
                    let d2 = scratch.sorted_dists[pos2];
                    let r1_sq = layout.dist_sq(src, u);
                    let r2_sq = layout.dist_sq(src, v);
                    eval_cnt += 1;
                    if (d1 < d2 && r1_sq < r2_sq) || (d1 > d2 && r1_sq > r2_sq) {
                        ok_cnt += 1;
                    }
                }
            }

            stratum_correct[s_idx] += ok_cnt;
            stratum_evaluated[s_idx] += eval_cnt;
        }
    }

    let mut active_count = 0usize;
    let mut sum_acc = 0.0_f64;
    let mut total_eval = 0_u64;
    for s_idx in 0..n_strata {
        let ev = stratum_evaluated[s_idx];
        if ev > 0 {
            active_count += 1;
            sum_acc += (stratum_correct[s_idx] as f64) / (ev as f64);
            total_eval += ev;
        }
    }

    let rta = if active_count > 0 {
        sum_acc / (active_count as f64)
    } else {
        0.0
    };
    Ok(RtaEvaluation {
        rta,
        num_landmarks: p_l,
        num_triplets: total_eval as usize,
        num_strata: n_strata,
        is_exact: false,
    })
}
