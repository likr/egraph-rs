use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};
use egraph_layout_sgd::sgd::PackedSymPair;

pub(super) const MAX_WEIGHT_RATIO: f64 = 8100.0;

pub(super) fn build_canonical_undirected_1hop<W>(
    graph: &CsrGraph<W>,
) -> CoreResult<(CsrGraph<f64>, Vec<u32>, Vec<u32>)> {
    let n = graph.num_nodes();
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (u, bucket) in buckets.iter_mut().enumerate() {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u && (v as usize) < n {
                bucket.push(v);
            }
        }
    }
    if graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v in graph.neighbors(u) {
                if v != u32_u && (v as usize) < n {
                    buckets[v as usize].push(u32_u);
                }
            }
        }
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    let mut u1 = Vec::new();
    let mut v1 = Vec::new();
    for (u, row) in buckets.iter_mut().enumerate() {
        row.sort_unstable();
        row.dedup();
        let u_u32 = u as u32;
        for &v in row.iter() {
            if u_u32 < v {
                u1.push(u_u32);
                v1.push(v);
            }
        }
        indices.extend_from_slice(row);
        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("canonical 1-hop nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    let adj1 = CsrGraph::from_csr(&indptr, &indices, None, false, false)?;
    Ok((adj1, u1, v1))
}

pub(super) fn compute_capped_2hop_sizes(
    adj_1hop: &CsrGraph<f64>,
    deg: &[usize],
    k_cap: usize,
) -> Vec<usize> {
    let n = adj_1hop.num_nodes();
    if n <= 1 || k_cap == 0 {
        return vec![0usize; n];
    }

    let mut k_arr = vec![0usize; n];
    let mut stamp = VisitStamp::new(n);

    for u in 0..n {
        let d_u = deg[u];
        if d_u == 0 {
            continue;
        }
        if d_u >= k_cap {
            k_arr[u] = k_cap;
            continue;
        }

        let n1_u = adj_1hop.neighbors(u);
        let mut max_nbr_deg = 0usize;
        for &w in n1_u {
            max_nbr_deg = max_nbr_deg.max(deg[w as usize]);
        }
        if max_nbr_deg >= k_cap {
            k_arr[u] = k_cap;
            continue;
        }

        stamp.clear();
        stamp.mark(u);
        for &w in n1_u {
            stamp.mark(w as usize);
        }
        let mut count = d_u;
        'outer: for &w in n1_u {
            for &v in adj_1hop.neighbors(w as usize) {
                if stamp.visit(v as usize) {
                    count += 1;
                    if count >= k_cap {
                        break 'outer;
                    }
                }
            }
        }
        k_arr[u] = count.min(k_cap);
    }
    k_arr
}

#[inline]
pub(super) fn fast_hypot_f64(dx: f64, dy: f64) -> f64 {
    let d2 = dx * dx + dy * dy;
    if d2 > 0.0 && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    }
}

#[inline]
pub(super) fn non_hub_degree_threshold(n_nodes: usize) -> usize {
    64usize.min(16usize.max(n_nodes.isqrt()))
}

pub(super) fn build_nonhub_2hop_csr_and_k_arr(
    adj1: &CsrGraph<f64>,
    deg: &[usize],
    non_hub_thresh: usize,
    k_eff: usize,
) -> CoreResult<(CsrGraph<f64>, Vec<usize>)> {
    let n = adj1.num_nodes();
    if n == 0 || adj1.nnz() == 0 {
        let indptr2 = vec![0u32; n + 1];
        let adj2 = CsrGraph::from_csr(&indptr2, &[], None, false, false)?;
        return Ok((adj2, vec![0usize; n]));
    }

    let mut stamp = VisitStamp::new(n);
    let mut indptr2 = Vec::with_capacity(n + 1);
    indptr2.push(0u32);
    let mut indices2 = Vec::new();
    let mut row2 = Vec::new();
    let mut k_arr = Vec::with_capacity(n);

    for u in 0..n {
        stamp.clear();
        stamp.mark(u);
        let n1 = adj1.neighbors(u);
        for &w in n1 {
            stamp.mark(w as usize);
        }

        row2.clear();
        for &w in n1 {
            if deg[w as usize] <= non_hub_thresh {
                for &v in adj1.neighbors(w as usize) {
                    if stamp.visit(v as usize) {
                        row2.push(v);
                    }
                }
            }
        }
        row2.sort_unstable();
        indices2.extend_from_slice(&row2);
        let ptr = u32::try_from(indices2.len())
            .map_err(|_| CoreError::invalid_csr("non-hub 2-hop CSR nnz exceeds u32::MAX"))?;
        indptr2.push(ptr);

        // Preserve exact capped 1+2-hop neighborhood size (including hub-bridged 2-hop nodes)
        // for spatial k_u-NN ball estimation.
        let mut ku = n1.len() + row2.len();
        if ku < k_eff {
            'hub_scan: for &w in n1 {
                if deg[w as usize] > non_hub_thresh {
                    for &v in adj1.neighbors(w as usize) {
                        if stamp.visit(v as usize) {
                            ku += 1;
                            if ku >= k_eff {
                                break 'hub_scan;
                            }
                        }
                    }
                }
            }
        }
        k_arr.push(ku.min(k_eff));
    }

    let adj2 = CsrGraph::from_csr(&indptr2, &indices2, None, false, false)?;
    Ok((adj2, k_arr))
}

#[inline]
pub(super) fn has_common_non_hub_neighbor(
    n1_u: &[u32],
    n1_v: &[u32],
    stamp: &VisitStamp,
    deg: &[usize],
    non_hub_thresh: usize,
) -> bool {
    if n1_u.is_empty() || n1_v.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| {
            let xi = x as usize;
            stamp.is_visited(xi) && deg[xi] <= non_hub_thresh
        });
    }
    n1_u.iter()
        .any(|&x| deg[x as usize] <= non_hub_thresh && n1_v.binary_search(&x).is_ok())
}

pub(super) fn compute_coords_std(coords: &[[f64; 2]]) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }
    let count = (coords.len() * 2) as f64;
    let mut sum = 0.0_f64;
    for pt in coords {
        sum += pt[0] + pt[1];
    }
    let mean = sum / count;
    let mut sq = 0.0_f64;
    for pt in coords {
        let dx = pt[0] - mean;
        let dy = pt[1] - mean;
        sq += dx * dx + dy * dy;
    }
    (sq / count).sqrt()
}

pub(super) fn compute_layout_scale(coords: &[[f64; 2]], u1: &[u32], v1: &[u32]) -> f64 {
    if !u1.is_empty() {
        let mut pos_d1: Vec<f64> = Vec::with_capacity(u1.len());
        for (&u, &v) in u1.iter().zip(v1.iter()) {
            let ui = u as usize;
            let vi = v as usize;
            let d = fast_hypot_f64(coords[ui][0] - coords[vi][0], coords[ui][1] - coords[vi][1]);
            if d.is_finite() && d > 1e-12 {
                pos_d1.push(d);
            }
        }
        if !pos_d1.is_empty() {
            pos_d1.sort_by(f64::total_cmp);
            let mid = pos_d1.len() / 2;
            return if pos_d1.len() % 2 == 1 {
                pos_d1[mid]
            } else {
                0.5 * (pos_d1[mid - 1] + pos_d1[mid])
            };
        }
    }
    let std_x = compute_coords_std(coords);
    if std_x.is_finite() && std_x > 1e-12 {
        std_x
    } else {
        1.0
    }
}

#[inline]
pub(super) fn push_packed_pair<S: Scalar>(
    packed: &mut Vec<PackedSymPair<S>>,
    u: u32,
    v: u32,
    is_hinge: bool,
    d: f64,
    w: f64,
) -> bool {
    if !(d.is_finite() && d > 0.0 && w.is_finite() && w > 0.0) {
        return false;
    }
    let d_s = S::from_f64(d);
    let w_s = S::from_f64(w);
    if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
        packed.push(PackedSymPair::new(u, v, is_hinge, d_s, w_s));
        true
    } else {
        false
    }
}

pub(super) fn condition_weights_packed<S: Scalar>(pairs: &mut [PackedSymPair<S>], max_ratio: f64) {
    if pairs.len() <= 1 {
        return;
    }
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    for p in pairs.iter() {
        let w = p.w.to_f64();
        if w < w_min {
            w_min = w;
        }
        if w > w_max {
            w_max = w;
        }
    }
    if w_max <= max_ratio * w_min || w_min <= 0.0 {
        return;
    }
    let half_span = max_ratio.sqrt();
    let mid = pairs.len() / 2;
    let (_, mid_elem, _) =
        pairs.select_nth_unstable_by(mid, |a, b| a.w.to_f64().total_cmp(&b.w.to_f64()));
    let mid_w = mid_elem.w.to_f64();
    let w_med = if pairs.len() % 2 == 1 {
        mid_w
    } else {
        let mut left_max = f64::NEG_INFINITY;
        for p in &pairs[..mid] {
            let w = p.w.to_f64();
            if w > left_max {
                left_max = w;
            }
        }
        0.5 * (left_max + mid_w)
    };
    let lo = w_med / half_span;
    let hi = w_med * half_span;
    for p in pairs.iter_mut() {
        let clamped = p.w.to_f64().clamp(lo, hi);
        let clamped_s = S::from_f64(clamped);
        if clamped_s.is_finite() && clamped_s > S::ZERO {
            p.w = clamped_s;
        }
    }
}

pub(super) fn sample_normal_pair(rng: &mut DeterministicRng, std_dev: f64) -> (f64, f64) {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    let mag = std_dev * (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (mag * c, mag * s)
}
