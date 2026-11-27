use egraph_core::{CoreError, CoreResult, DeterministicRng};
use egraph_linalg::CsrMatrix;

use super::config::AmgPcgConfig;

#[derive(Debug, Clone)]
pub(super) struct AmgLevel {
    pub(super) a: CsrMatrix,
    pub(super) inv_diag: Vec<f64>,
    pub(super) agg_id: Vec<u32>,
    pub(super) p_val: Vec<f64>,
}

pub(super) struct BuiltLevel {
    pub(super) agg_id: Vec<u32>,
    pub(super) p_val: Vec<f64>,
    pub(super) next_ns: Vec<f64>,
    pub(super) next_a: CsrMatrix,
}

#[derive(Default)]
pub(super) struct GalerkinScratch {
    stamp: Vec<u32>,
    epoch: u32,
    col_pos: Vec<u32>,
    row_pairs: Vec<(u32, f64)>,
}

pub(super) fn build_double_pairwise_level(
    a: &CsrMatrix,
    near_nullspace: &[f64],
    config: &AmgPcgConfig,
    rng: &mut DeterministicRng,
    scratch: &mut GalerkinScratch,
) -> CoreResult<Option<BuiltLevel>> {
    let n = a.num_nodes();
    let pass1_cap = (config.max_agg_size / 2).max(2);

    let (agg1, agg1_sizes) = pairwise_aggregate(
        a,
        &vec![1usize; n],
        config.theta,
        pass1_cap,
        true,
        config.randomize_ties,
        rng,
    );
    let n1 = agg1_sizes.len();
    if n1 == 0 || n1 >= n {
        return Ok(None);
    }

    let (p1_val, ns1) = build_prolongator_values(&agg1, n1, near_nullspace);
    let a1 = galerkin_ptap(a, &agg1, &p1_val, n1, scratch)?;

    // Stop after single pairwise pass if already within coarsest threshold.
    if n1 <= config.max_coarse_size {
        return Ok(Some(BuiltLevel {
            agg_id: agg1,
            p_val: p1_val,
            next_ns: ns1,
            next_a: a1,
        }));
    }

    let (agg2, agg2_sizes) = pairwise_aggregate(
        &a1,
        &agg1_sizes,
        config.theta,
        config.max_agg_size,
        true,
        config.randomize_ties,
        rng,
    );
    let n2 = agg2_sizes.len();

    let stall_limit = ((0.95 * (n as f64)).floor() as usize).max(1);
    if n2 == 0 || n2 >= n1 {
        if n1 <= stall_limit {
            return Ok(Some(BuiltLevel {
                agg_id: agg1,
                p_val: p1_val,
                next_ns: ns1,
                next_a: a1,
            }));
        }
        return Ok(None);
    }
    if n2 > stall_limit {
        return Ok(None);
    }

    let (p2_val, ns2) = build_prolongator_values(&agg2, n2, &ns1);
    let a2 = galerkin_ptap(&a1, &agg2, &p2_val, n2, scratch)?;

    // Compose P = P1 * P2 (still 1 nonzero per fine row, P^T P = I).
    let mut agg = Vec::with_capacity(n);
    let mut p_val = Vec::with_capacity(n);
    for i in 0..n {
        let a_mid = agg1[i] as usize;
        agg.push(agg2[a_mid]);
        p_val.push(p1_val[i] * p2_val[a_mid]);
    }

    Ok(Some(BuiltLevel {
        agg_id: agg,
        p_val,
        next_ns: ns2,
        next_a: a2,
    }))
}

fn pairwise_aggregate(
    a: &CsrMatrix,
    node_sizes: &[usize],
    theta: f64,
    max_agg_size: usize,
    allow_leaf_bypass: bool,
    randomize_ties: bool,
    rng: &mut DeterministicRng,
) -> (Vec<u32>, Vec<usize>) {
    let n = a.num_nodes();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();
    let diag = a.diagonal();

    // 1. Symmetric negative-off-diagonal strength of connection.
    let mut max_neg = vec![0.0_f64; n];
    for row in 0..n {
        if diag[row] <= 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut best = 0.0_f64;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row {
                let neg = -values[pos];
                if neg > best {
                    best = neg;
                }
            }
        }
        max_neg[row] = best;
    }

    let mut s_deg = vec![0usize; n];
    let mut max_s_deg = 0usize;
    for row in 0..n {
        let mn_row = max_neg[row];
        if mn_row <= 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut cnt = 0usize;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row && values[pos] < 0.0 {
                let mn_col = max_neg[col];
                if mn_col > 0.0 && (-values[pos]) >= theta * mn_row.min(mn_col) {
                    cnt += 1;
                }
            }
        }
        s_deg[row] = cnt;
        if cnt > max_s_deg {
            max_s_deg = cnt;
        }
    }

    // 2. O(n) counting sort in ascending strong-degree order.
    let mut counts = vec![0usize; max_s_deg + 1];
    for &d in &s_deg {
        counts[d] += 1;
    }
    let mut offsets = vec![0usize; max_s_deg + 1];
    let mut running = 0usize;
    for (d, &c) in counts.iter().enumerate() {
        offsets[d] = running;
        running += c;
    }
    let mut bucket_cursor = offsets.clone();
    let mut order = vec![0usize; n];
    for (i, &d) in s_deg.iter().enumerate() {
        order[bucket_cursor[d]] = i;
        bucket_cursor[d] += 1;
    }
    if randomize_ties {
        for d in 0..=max_s_deg {
            let start = offsets[d];
            let end = start + counts[d];
            if end - start > 1 {
                rng.shuffle(&mut order[start..end]);
            }
        }
    }

    // 3. Pairwise matching + size-capped leaf absorption.
    let mut agg_id = vec![u32::MAX; n];
    let mut agg_sizes: Vec<usize> = Vec::with_capacity(n / 2 + 1);

    for &u in &order {
        if agg_id[u] != u32::MAX {
            continue;
        }
        let mn_u = max_neg[u];
        let u_size = node_sizes[u];
        let u_is_small_leaf = allow_leaf_bypass && s_deg[u] <= 2 && u_size <= 2;

        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;

        let mut best_unagg: Option<(usize, f64, usize)> = None;
        let mut best_agg: Option<(usize, f64, usize)> = None;

        if mn_u > 0.0 {
            for pos in start..end {
                let v = indices[pos] as usize;
                if v == u || values[pos] >= 0.0 {
                    continue;
                }
                let mn_v = max_neg[v];
                if mn_v <= 0.0 {
                    continue;
                }
                let strength = -values[pos];
                if strength < theta * mn_u.min(mn_v) {
                    continue;
                }

                let v_agg = agg_id[v];
                if v_agg == u32::MAX {
                    let combined = u_size + node_sizes[v];
                    if combined <= max_agg_size || (u_is_small_leaf && node_sizes[v] <= 2) {
                        let replace = match best_unagg {
                            None => true,
                            Some((bv, bs, bdeg)) => {
                                strength > bs
                                    || (strength == bs
                                        && (s_deg[v] < bdeg || (s_deg[v] == bdeg && v < bv)))
                            }
                        };
                        if replace {
                            best_unagg = Some((v, strength, s_deg[v]));
                        }
                    }
                } else {
                    let a_idx = v_agg as usize;
                    let a_size = agg_sizes[a_idx];
                    if a_size + u_size <= max_agg_size || u_is_small_leaf {
                        let replace = match best_agg {
                            None => true,
                            Some((ba, bs, bas)) => {
                                strength > bs
                                    || (strength == bs
                                        && (a_size < bas || (a_size == bas && a_idx < ba)))
                            }
                        };
                        if replace {
                            best_agg = Some((a_idx, strength, a_size));
                        }
                    }
                }
            }
        }

        if let Some((v, _, _)) = best_unagg {
            let new_agg = agg_sizes.len() as u32;
            agg_id[u] = new_agg;
            agg_id[v] = new_agg;
            agg_sizes.push(u_size + node_sizes[v]);
        } else if let Some((a_idx, _, _)) = best_agg {
            agg_id[u] = a_idx as u32;
            agg_sizes[a_idx] += u_size;
        } else {
            let new_agg = agg_sizes.len() as u32;
            agg_id[u] = new_agg;
            agg_sizes.push(u_size);
        }
    }

    (agg_id, agg_sizes)
}

fn build_prolongator_values(
    agg_id: &[u32],
    num_coarse: usize,
    near_nullspace: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = agg_id.len();
    let mut norm_sq = vec![0.0_f64; num_coarse];
    let mut counts = vec![0usize; num_coarse];
    for (i, &a) in agg_id.iter().enumerate() {
        let a_idx = a as usize;
        let v = near_nullspace[i];
        norm_sq[a_idx] += v * v;
        counts[a_idx] += 1;
    }

    let mut ns_coarse = vec![0.0_f64; num_coarse];
    for a in 0..num_coarse {
        ns_coarse[a] = norm_sq[a].sqrt();
    }

    let mut p_val = Vec::with_capacity(n);
    for (i, &a) in agg_id.iter().enumerate() {
        let a_idx = a as usize;
        let s_a = ns_coarse[a_idx];
        if s_a > 1e-30 {
            p_val.push(near_nullspace[i] / s_a);
        } else {
            p_val.push(1.0 / (counts[a_idx].max(1) as f64).sqrt());
        }
    }

    (p_val, ns_coarse)
}

fn galerkin_ptap(
    a: &CsrMatrix,
    agg_id: &[u32],
    p_val: &[f64],
    num_coarse: usize,
    scratch: &mut GalerkinScratch,
) -> CoreResult<CsrMatrix> {
    let n_fine = a.num_nodes();
    let mut group_counts = vec![0u32; num_coarse];
    for &g in agg_id {
        group_counts[g as usize] += 1;
    }
    let mut group_ptr = Vec::with_capacity(num_coarse + 1);
    group_ptr.push(0u32);
    let mut running = 0u32;
    for &c in &group_counts {
        running += c;
        group_ptr.push(running);
    }
    let mut cursor = group_ptr[..num_coarse].to_vec();
    let mut group_nodes = vec![0u32; n_fine];
    for (i, &g) in agg_id.iter().enumerate() {
        let g_idx = g as usize;
        group_nodes[cursor[g_idx] as usize] = i as u32;
        cursor[g_idx] += 1;
    }

    if scratch.stamp.len() < num_coarse {
        scratch.stamp.resize(num_coarse, 0);
        scratch.col_pos.resize(num_coarse, 0);
    }

    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();

    let mut out_indptr = Vec::with_capacity(num_coarse + 1);
    out_indptr.push(0u32);
    let mut out_indices = Vec::new();
    let mut out_values = Vec::new();

    for row_c in 0..num_coarse {
        if scratch.epoch == u32::MAX {
            scratch.stamp.fill(0);
            scratch.epoch = 1;
        } else {
            scratch.epoch += 1;
        }
        let epoch = scratch.epoch;
        scratch.row_pairs.clear();

        let g_start = group_ptr[row_c] as usize;
        let g_end = group_ptr[row_c + 1] as usize;
        for &fine_u32 in &group_nodes[g_start..g_end] {
            let i = fine_u32 as usize;
            let pi = p_val[i];
            let a_start = indptr[i] as usize;
            let a_end = indptr[i + 1] as usize;
            for pos in a_start..a_end {
                let j = indices[pos] as usize;
                let col_c = agg_id[j];
                let contrib = pi * values[pos] * p_val[j];
                let col_idx = col_c as usize;
                if scratch.stamp[col_idx] != epoch {
                    scratch.stamp[col_idx] = epoch;
                    scratch.col_pos[col_idx] = scratch.row_pairs.len() as u32;
                    scratch.row_pairs.push((col_c, contrib));
                } else {
                    scratch.row_pairs[scratch.col_pos[col_idx] as usize].1 += contrib;
                }
            }
        }

        scratch.row_pairs.sort_unstable_by_key(|&(col, _)| col);
        for &(col, val) in &scratch.row_pairs {
            if val != 0.0 {
                out_indices.push(col);
                out_values.push(val);
            }
        }
        let ptr = u32::try_from(out_indices.len())
            .map_err(|_| CoreError::invalid_csr("Galerkin coarse nnz exceeds u32::MAX"))?;
        out_indptr.push(ptr);
    }

    let raw = CsrMatrix::from_csr(num_coarse, out_indptr, out_indices, out_values)?;
    raw.symmetrize()
}
