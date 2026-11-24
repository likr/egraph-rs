//! Landmark-Stratified Random Triplet Accuracy (`landmark_stratified_rta`).
//!
//! Uses $O(N)$-RAM 2-pass streaming BFS from Core-First MaxMin shortest-path pivots,
//! stratifies comparable `(u, v)` pairs into `floor(log2(d))` octave bins `(b1 <= b2)`,
//! and evaluates exact enumeration deterministically whenever `total_comparable <= num_triplets`.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};

use crate::validate_graph_layout;

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

#[inline]
fn is_layout_collapsed<S: Scalar>(layout: &Layout2d<S>) -> bool {
    let n = layout.len();
    if n == 0 {
        return true;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x) <= 1e-15 && (max_y - min_y) <= 1e-15
}

#[inline]
fn use_counting_sort(n: usize, max_d: u32) -> bool {
    (max_d as usize) <= n.max(1024).saturating_mul(4)
}

struct RowScratch {
    dist_counts: Vec<u128>,
    dist_offsets: Vec<usize>,
    cursor: Vec<usize>,
    sorted_nodes: Vec<u32>,
    sorted_dists: Vec<u32>,
    sorted_pairs: Vec<(u32, u32)>,
}

impl RowScratch {
    fn new(n: usize) -> Self {
        Self {
            dist_counts: Vec::new(),
            dist_offsets: Vec::new(),
            cursor: Vec::new(),
            sorted_nodes: Vec::with_capacity(n),
            sorted_dists: Vec::with_capacity(n),
            sorted_pairs: Vec::new(),
        }
    }
}

fn scan_row_octaves(row: &[u32], scratch: &mut RowScratch) -> ([u128; 32], [u128; 32], isize) {
    let mut oct_counts = [0_u128; 32];
    let mut same_pairs = [0_u128; 32];

    let mut max_d = 0_u32;
    for &d in row {
        if (1..u32::MAX).contains(&d) && d > max_d {
            max_d = d;
        }
    }
    if max_d == 0 {
        return (oct_counts, same_pairs, -1);
    }

    let mut sum_sq = [0_u128; 32];
    let mut max_oct = 0_isize;

    if use_counting_sort(row.len(), max_d) {
        let max_d_usize = max_d as usize;
        scratch.dist_counts.clear();
        scratch.dist_counts.resize(max_d_usize + 1, 0);
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.dist_counts[d as usize] += 1;
            }
        }

        for (d, &cnt) in scratch
            .dist_counts
            .iter()
            .enumerate()
            .take(max_d_usize + 1)
            .skip(1)
        {
            if cnt > 0 {
                let b = (31 - (d as u32).leading_zeros()) as usize;
                if (b as isize) > max_oct {
                    max_oct = b as isize;
                }
                oct_counts[b] += cnt;
                sum_sq[b] += cnt * cnt;
            }
        }
    } else {
        // Fallback for large sparse/non-hop integer distances: O(N log N) sort in O(N) RAM.
        scratch.sorted_dists.clear();
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.sorted_dists.push(d);
            }
        }
        scratch.sorted_dists.sort_unstable();
        let mut idx = 0usize;
        let len = scratch.sorted_dists.len();
        while idx < len {
            let d = scratch.sorted_dists[idx];
            let start = idx;
            while idx < len && scratch.sorted_dists[idx] == d {
                idx += 1;
            }
            let cnt = (idx - start) as u128;
            let b = (31 - d.leading_zeros()) as usize;
            if (b as isize) > max_oct {
                max_oct = b as isize;
            }
            oct_counts[b] += cnt;
            sum_sq[b] += cnt * cnt;
        }
    }

    for b in 0..=(max_oct as usize) {
        let cb = oct_counts[b];
        if cb >= 2 {
            same_pairs[b] = (cb * cb - sum_sq[b]) / 2;
        }
    }

    (oct_counts, same_pairs, max_oct)
}

fn counting_sort_row(row: &[u32], scratch: &mut RowScratch) -> [usize; 33] {
    let mut oct_offsets = [0usize; 33];
    let mut max_d = 0_u32;
    let mut n_valid = 0usize;
    for &d in row {
        if (1..u32::MAX).contains(&d) {
            n_valid += 1;
            if d > max_d {
                max_d = d;
            }
        }
    }
    scratch.sorted_nodes.clear();
    scratch.sorted_dists.clear();
    if n_valid == 0 {
        return oct_offsets;
    }

    if use_counting_sort(row.len(), max_d) {
        let max_d_usize = max_d as usize;
        scratch.dist_counts.clear();
        scratch.dist_counts.resize(max_d_usize + 1, 0);
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.dist_counts[d as usize] += 1;
            }
        }

        scratch.dist_offsets.clear();
        scratch.dist_offsets.resize(max_d_usize + 2, 0);
        let mut oct_counts = [0usize; 32];
        for d in 1..=max_d_usize {
            let cnt = scratch.dist_counts[d] as usize;
            scratch.dist_offsets[d + 1] = scratch.dist_offsets[d] + cnt;
            if cnt > 0 {
                let b = (31 - (d as u32).leading_zeros()) as usize;
                oct_counts[b] += cnt;
            }
        }
        for b in 0..32 {
            oct_offsets[b + 1] = oct_offsets[b] + oct_counts[b];
        }

        scratch.cursor.clear();
        scratch
            .cursor
            .extend_from_slice(&scratch.dist_offsets[..=max_d_usize]);
        scratch.sorted_nodes.resize(n_valid, 0);
        scratch.sorted_dists.resize(n_valid, 0);

        for (u, &d) in row.iter().enumerate() {
            if (1..u32::MAX).contains(&d) {
                let di = d as usize;
                let pos = scratch.cursor[di];
                scratch.sorted_nodes[pos] = u as u32;
                scratch.sorted_dists[pos] = d;
                scratch.cursor[di] = pos + 1;
            }
        }
    } else {
        // Fallback for large sparse/non-hop integer distances: sort (dist, node) in O(N) RAM.
        scratch.sorted_pairs.clear();
        scratch.sorted_pairs.reserve(n_valid);
        for (u, &d) in row.iter().enumerate() {
            if (1..u32::MAX).contains(&d) {
                scratch.sorted_pairs.push((d, u as u32));
            }
        }
        scratch.sorted_pairs.sort_unstable();
        scratch.sorted_nodes.reserve(n_valid);
        scratch.sorted_dists.reserve(n_valid);
        let mut oct_counts = [0usize; 32];
        for &(d, u) in &scratch.sorted_pairs {
            scratch.sorted_dists.push(d);
            scratch.sorted_nodes.push(u);
            let b = (31 - d.leading_zeros()) as usize;
            oct_counts[b] += 1;
        }
        for b in 0..32 {
            oct_offsets[b + 1] = oct_offsets[b] + oct_counts[b];
        }
    }

    oct_offsets
}

enum RowProvider<'a, W> {
    Graph {
        graph: &'a CsrGraph<W>,
        ws: BfsWorkspace,
        buf: Vec<u32>,
    },
    Table {
        dists: &'a [u32],
        n: usize,
    },
}

impl<'a, W> RowProvider<'a, W> {
    fn row(&mut self, landmark_idx: usize, landmark_id: u32) -> CoreResult<&[u32]> {
        match self {
            RowProvider::Graph { graph, ws, buf } => {
                ws.bfs_from_u32(graph, landmark_id as usize, buf)?;
                Ok(buf.as_slice())
            }
            RowProvider::Table { dists, n } => {
                let start = landmark_idx * *n;
                Ok(&dists[start..start + *n])
            }
        }
    }
}

fn evaluate_rta_core<W, S: Scalar>(
    layout: &Layout2d<S>,
    landmark_ids: &[u32],
    mut provider: RowProvider<'_, W>,
    num_triplets: usize,
    seed: u64,
) -> CoreResult<RtaEvaluation> {
    let n = layout.len();
    let p_l = landmark_ids.len();
    if n < 3 || p_l == 0 || is_layout_collapsed(layout) {
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

    let xs = layout.x();
    let ys = layout.y();
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
            let px = xs[src].to_f64();
            let py = ys[src].to_f64();

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
                let dx1 = xs[u].to_f64() - px;
                let dy1 = ys[u].to_f64() - py;
                let r1_sq = dx1 * dx1 + dy1 * dy1;
                for b in (a + 1)..m_v {
                    let d2 = valid_dists[b];
                    if d1 == d2 {
                        continue;
                    }
                    let v = valid_nodes[b] as usize;
                    let dx2 = xs[v].to_f64() - px;
                    let dy2 = ys[v].to_f64() - py;
                    let r2_sq = dx2 * dx2 + dy2 * dy2;
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
        let px = xs[src].to_f64();
        let py = ys[src].to_f64();

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
                    let dx1 = xs[u].to_f64() - px;
                    let dy1 = ys[u].to_f64() - py;
                    let dx2 = xs[v].to_f64() - px;
                    let dy2 = ys[v].to_f64() - py;
                    let r1_sq = dx1 * dx1 + dy1 * dy1;
                    let r2_sq = dx2 * dx2 + dy2 * dy2;
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
                    let dx1 = xs[u].to_f64() - px;
                    let dy1 = ys[u].to_f64() - py;
                    let dx2 = xs[v].to_f64() - px;
                    let dy2 = ys[v].to_f64() - py;
                    let r1_sq = dx1 * dx1 + dy1 * dy1;
                    let r2_sq = dx2 * dx2 + dy2 * dy2;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::all_pairs_bfs;

    fn brute_force_all_comparable_rta<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
        landmark_ids: &[u32],
    ) -> (f64, usize) {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut ws = BfsWorkspace::new(n);
        let mut row = vec![u32::MAX; n];
        let mut ok = 0usize;
        let mut eval = 0usize;

        for &lid in landmark_ids {
            ws.bfs_from_u32(graph, lid as usize, &mut row).unwrap();
            let px = xs[lid as usize].to_f64();
            let py = ys[lid as usize].to_f64();
            for u in 0..n {
                let d1 = row[u];
                if !(1..u32::MAX).contains(&d1) {
                    continue;
                }
                let r1_sq = (xs[u].to_f64() - px).powi(2) + (ys[u].to_f64() - py).powi(2);
                for v in (u + 1)..n {
                    let d2 = row[v];
                    if !(1..u32::MAX).contains(&d2) || d1 == d2 {
                        continue;
                    }
                    let r2_sq = (xs[v].to_f64() - px).powi(2) + (ys[v].to_f64() - py).powi(2);
                    eval += 1;
                    if (d1 < d2 && r1_sq < r2_sq) || (d1 > d2 && r1_sq > r2_sq) {
                        ok += 1;
                    }
                }
            }
        }
        let rta = if eval > 0 {
            ok as f64 / eval as f64
        } else {
            0.0
        };
        (rta, eval)
    }

    #[test]
    fn test_rta_exact_oracle_similarity_invariance_and_stratified_sampling() {
        // 1. Path graph laid out cleanly on x-axis -> RTA == 1.0, exact fallback.
        let path_edges: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
        let path = CsrGraph::<f64>::from_edges(12, &path_edges, None, false, false).unwrap();
        let path_x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let path_y = vec![0.0; 12];
        let path_l = Layout2d::from_xy(path_x.clone(), path_y.clone()).unwrap();

        let det_path = landmark_stratified_rta_detail(&path, &path_l, 4, 100_000, 1, 42).unwrap();
        assert!(det_path.is_exact);
        assert!((det_path.rta - 1.0).abs() < 1e-12);

        // Compare directly against O(P_L * N^2) brute-force oracle on a warped path + grid.
        let warped_l = Layout2d::from_xy(
            path_x
                .iter()
                .enumerate()
                .map(|(i, &x)| x + if i % 3 == 0 { 2.5 } else { -0.4 })
                .collect(),
            (0..12).map(|i| ((i * 7) % 5) as f64 * 0.6).collect(),
        )
        .unwrap();
        let lm_ids = select_maxmin_sp_pivots(&path, 4, 42, 1).unwrap();
        let (oracle_rta, oracle_cnt) = brute_force_all_comparable_rta(&path, &warped_l, &lm_ids);
        let got_warped =
            landmark_stratified_rta_detail(&path, &warped_l, 4, 100_000, 1, 42).unwrap();
        assert!(got_warped.is_exact);
        assert_eq!(got_warped.num_triplets, oracle_cnt);
        assert!((got_warped.rta - oracle_rta).abs() < 1e-12);

        // Similarity transform invariance (scale + rotate + translate).
        let theta = 0.65_f64;
        let (st, ct) = theta.sin_cos();
        let sim_x: Vec<f64> = (0..12)
            .map(|i| 4.2 * (ct * warped_l.x()[i] - st * warped_l.y()[i]) + 15.0)
            .collect();
        let sim_y: Vec<f64> = (0..12)
            .map(|i| 4.2 * (st * warped_l.x()[i] + ct * warped_l.y()[i]) - 9.0)
            .collect();
        let sim_l = Layout2d::from_xy(sim_x, sim_y).unwrap();
        let sim_rta = landmark_stratified_rta(&path, &sim_l, 4, 100_000, 1, 42).unwrap();
        assert!((sim_rta - oracle_rta).abs() < 1e-12);

        // Verify landmark_stratified_rta_from_landmarks parity with graph evaluation.
        let apsp = all_pairs_bfs(&path, 1.0).unwrap();
        let mut flat_u32 = Vec::with_capacity(lm_ids.len() * 12);
        for &lid in &lm_ids {
            let row_off = (lid as usize) * 12;
            for j in 0..12 {
                let d = apsp[row_off + j];
                flat_u32.push(if d.is_finite() { d as u32 } else { u32::MAX });
            }
        }
        let from_tbl =
            landmark_stratified_rta_from_landmarks(&warped_l, &lm_ids, &flat_u32, 100_000, 42)
                .unwrap();
        assert_eq!(from_tbl, got_warped);

        // Stratified sampling branch (total_comparable > num_triplets): grid 8x8 (N=64).
        let mut g_edges = Vec::new();
        let mut gx = Vec::new();
        let mut gy = Vec::new();
        for r in 0..8u32 {
            for c in 0..8u32 {
                let id = r * 8 + c;
                gx.push(c as f64);
                gy.push(r as f64);
                if c + 1 < 8 {
                    g_edges.push((id, id + 1));
                }
                if r + 1 < 8 {
                    g_edges.push((id, id + 8));
                }
            }
        }
        let grid = CsrGraph::<f64>::from_edges(64, &g_edges, None, false, false).unwrap();
        let grid_l64 = Layout2d::from_xy(gx.clone(), gy.clone()).unwrap();
        let grid_l32 = Layout2d::from_xy(
            gx.iter().map(|&v| v as f32).collect(),
            gy.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();

        let samp_a = landmark_stratified_rta_detail(&grid, &grid_l64, 8, 500, 2, 42).unwrap();
        let samp_b = landmark_stratified_rta_detail(&grid, &grid_l64, 8, 500, 2, 42).unwrap();
        let samp_32 = landmark_stratified_rta_detail(&grid, &grid_l32, 8, 500, 2, 42).unwrap();
        assert!(!samp_a.is_exact);
        assert_eq!(samp_a, samp_b);
        assert!((samp_a.rta - samp_32.rta).abs() < 1e-12);
        assert!(samp_a.rta > 0.85, "grid L-Stratified-RTA={}", samp_a.rta);

        // Random layout on grid should be near ~0.5, collapsed should be 0.0.
        let mut rng = DeterministicRng::from_seed_u64(777);
        let rand_l = Layout2d::from_xy(
            (0..64).map(|_| rng.next_f64()).collect(),
            (0..64).map(|_| rng.next_f64()).collect(),
        )
        .unwrap();
        let rand_rta = landmark_stratified_rta(&grid, &rand_l, 8, 2000, 2, 42).unwrap();
        assert!((0.35..0.65).contains(&rand_rta), "rand_rta={rand_rta}");

        let collapsed = Layout2d::<f64>::zeros(64);
        assert_eq!(
            landmark_stratified_rta(&grid, &collapsed, 8, 500, 2, 42).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_rta_validation_errors() {
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
        let l = Layout2d::from_xy(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4]).unwrap();
        assert!(matches!(
            landmark_stratified_rta(&g, &l, 0, 100, 1, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            landmark_stratified_rta(&g, &l, 2, 0, 1, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            landmark_stratified_rta_from_landmarks(&l, &[0, 1], &[0; 5], 100, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            landmark_stratified_rta_from_landmarks(&l, &[4], &[0; 4], 100, 42),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_rta_huge_synthetic_distances_no_oom_and_oracle_parity() {
        // Synthetic non-hop integer distances up to u32::MAX - 1 must use the O(N)-RAM sort
        // fallback instead of allocating O(max_d) memory, while preserving exact RTA results.
        let n = 8usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys = vec![0.0; n];
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let landmark_ids = [0_u32];
        let huge_dists = [
            0_u32,
            10_000_000,
            10_000_000,
            250_000_000,
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            u32::MAX - 1,
        ];

        // Exact branch (total_comparable <= num_triplets).
        let exact =
            landmark_stratified_rta_from_landmarks(&layout, &landmark_ids, &huge_dists, 1000, 42)
                .unwrap();
        assert!(exact.is_exact);
        assert!((exact.rta - 1.0).abs() < 1e-12);
        assert_eq!(exact.num_triplets, 20); // C(7,2) - 1 equal pair (nodes 1,2)

        // Stratified sampling branch (total_comparable > num_triplets).
        let sampled =
            landmark_stratified_rta_from_landmarks(&layout, &landmark_ids, &huge_dists, 10, 42)
                .unwrap();
        assert!(!sampled.is_exact);
        assert!((sampled.rta - 1.0).abs() < 1e-12);

        // Verify fallback helper matches counting sort bit-for-bit on small distances too.
        let mut s_cnt = RowScratch::new(n);
        let mut s_sort = RowScratch::new(n);
        let small_row = [0_u32, 3, 1, 3, 2, u32::MAX, 5, 1];
        let oct_cnt = counting_sort_row(&small_row, &mut s_cnt);
        s_sort.sorted_pairs.clear();
        for (u, &d) in small_row.iter().enumerate() {
            if (1..u32::MAX).contains(&d) {
                s_sort.sorted_pairs.push((d, u as u32));
            }
        }
        s_sort.sorted_pairs.sort_unstable();
        let mut exp_nodes = Vec::new();
        let mut exp_dists = Vec::new();
        let mut oct_counts = [0usize; 32];
        for &(d, u) in &s_sort.sorted_pairs {
            exp_dists.push(d);
            exp_nodes.push(u);
            oct_counts[(31 - d.leading_zeros()) as usize] += 1;
        }
        let mut exp_oct = [0usize; 33];
        for b in 0..32 {
            exp_oct[b + 1] = exp_oct[b] + oct_counts[b];
        }
        assert_eq!(oct_cnt, exp_oct);
        assert_eq!(s_cnt.sorted_nodes, exp_nodes);
        assert_eq!(s_cnt.sorted_dists, exp_dists);
    }
}
