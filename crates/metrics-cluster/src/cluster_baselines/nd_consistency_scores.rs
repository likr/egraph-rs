use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};

use super::nd_proximity::euclidean_dist_nd;
use super::{compact_labels, validate_layout_nd_labels};

fn coords_to_f64_vec<S: Scalar>(layout: &LayoutNd<S>) -> Vec<f64> {
    layout.coords().iter().map(|v| v.to_f64()).collect()
}

/// Computes Sips et al. (2009) Distance Consistency (`DSC`) on an $N$-D [`LayoutNd`] in
/// `[0.0, 1.0]` (higher is better).
///
/// Counts the fraction of points whose squared Euclidean distance to their own class centroid
/// is **strictly smaller** than to every other class centroid (`N < 2 || K < 2` returns `0.0`).
pub fn distance_consistency_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "distance_consistency_nd")?;
    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 2 || k < 2 {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut centroids = vec![0.0_f64; k * d];
    for (&c, pt) in comp.iter().zip(coords.chunks_exact(d)) {
        let c_off = (c as usize) * d;
        for axis in 0..d {
            centroids[c_off + axis] += pt[axis];
        }
    }
    for (c, &sz) in sizes.iter().enumerate() {
        let inv = 1.0 / sz as f64;
        let c_off = c * d;
        for axis in 0..d {
            centroids[c_off + axis] *= inv;
        }
    }

    let mut consistent = 0usize;
    for i in 0..n {
        let own = comp[i] as usize;
        let pt = &coords[i * d..(i + 1) * d];
        let own_c = &centroids[own * d..(own + 1) * d];
        let mut own_d2 = 0.0_f64;
        for axis in 0..d {
            let diff = pt[axis] - own_c[axis];
            own_d2 += diff * diff;
        }

        let mut is_strictly_closest = true;
        for c in 0..k {
            if c == own {
                continue;
            }
            let other_c = &centroids[c * d..(c + 1) * d];
            let mut d2 = 0.0_f64;
            for axis in 0..d {
                let diff = pt[axis] - other_c[axis];
                d2 += diff * diff;
            }
            if d2 <= own_d2 {
                is_strictly_closest = false;
                break;
            }
        }
        if is_strictly_closest {
            consistent += 1;
        }
    }
    Ok(consistent as f64 / n as f64)
}

/// Computes Sips et al. (2009) Distribution Consistency (`DC`) on an $N$-D [`LayoutNd`] scaled
/// to `[0.0, 1.0]`, measuring weighted grid-cell class entropy purity over a `grid_size^D`
/// partition of the normalized `[0, 1]^D` layout box.
///
/// Uses sparse lexicographic sorting of occupied $D$-dimensional cell tuples in $O(N D \log N)$
/// time and $O(N D)$ memory without allocating a dense `grid_size^D` grid. Returns `0.0` when
/// `N == 0` or `K < 2`. Errors when `grid_size == 0`.
pub fn distribution_consistency_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "distribution_consistency_nd requires grid_size >= 1",
        ));
    }
    validate_layout_nd_labels(layout, labels, "distribution_consistency_nd")?;

    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, _, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for i in 0..n {
        let base = i * d;
        for axis in 0..d {
            let v = coords[base + axis];
            mins[axis] = mins[axis].min(v);
            maxs[axis] = maxs[axis].max(v);
        }
    }

    let g_f = grid_size as f64;
    let mut cell_coords = vec![0usize; n * d];
    for axis in 0..d {
        let min_v = mins[axis];
        let span = maxs[axis] - min_v;
        for i in 0..n {
            let u = if span > 1e-15 {
                (coords[i * d + axis] - min_v) / span
            } else {
                0.5
            };
            cell_coords[i * d + axis] = ((u * g_f).floor() as usize).min(grid_size - 1);
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        cell_coords[a * d..(a + 1) * d]
            .cmp(&cell_coords[b * d..(b + 1) * d])
            .then_with(|| comp[a].cmp(&comp[b]))
    });

    let mut weighted_entropy = 0.0_f64;
    let mut start = 0usize;
    while start < n {
        let ref_cell = &cell_coords[order[start] * d..(order[start] + 1) * d];
        let mut end = start + 1;
        while end < n {
            let cur_cell = &cell_coords[order[end] * d..(order[end] + 1) * d];
            if cur_cell != ref_cell {
                break;
            }
            end += 1;
        }
        let p_total = (end - start) as f64;
        let mut cell_h = 0.0_f64;
        let mut c_start = start;
        while c_start < end {
            let cls = comp[order[c_start]];
            let mut c_end = c_start + 1;
            while c_end < end && comp[order[c_end]] == cls {
                c_end += 1;
            }
            let prob = (c_end - c_start) as f64 / p_total;
            cell_h -= prob * prob.log2();
            c_start = c_end;
        }
        weighted_entropy += p_total * cell_h;
        start = end;
    }

    let denom = (k as f64).log2() * n as f64;
    Ok((1.0 - weighted_entropy / denom).clamp(0.0, 1.0))
}

/// Computes mean Silhouette coefficient on an $N$-D [`LayoutNd`] matching
/// `sklearn.metrics.silhouette_score` in `[-1.0, 1.0]` using $O(N + K)$ auxiliary memory
/// (singleton clusters score `0.0`).
///
/// Under the non-panic evaluation contract, returns `Ok(0.0)` when `N < 3`, `K < 2`, or `K >= N`.
pub fn silhouette_score_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "silhouette_score_nd")?;
    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut sum_dists = vec![0.0_f64; k];
    let mut total_s = 0.0_f64;

    for i in 0..n {
        let ci = comp[i] as usize;
        if sizes[ci] == 1 {
            continue;
        }
        sum_dists.fill(0.0);
        let pi = &coords[i * d..(i + 1) * d];
        for j in 0..n {
            let pj = &coords[j * d..(j + 1) * d];
            sum_dists[comp[j] as usize] += euclidean_dist_nd(pi, pj);
        }
        let a_i = sum_dists[ci] / (sizes[ci] - 1) as f64;
        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c != ci {
                b_i = b_i.min(sum_dists[c] / sizes[c] as f64);
            }
        }
        let denom = a_i.max(b_i);
        if denom > 0.0 {
            total_s += (b_i - a_i) / denom;
        }
    }

    Ok(total_s / n as f64)
}

/// Computes Calinski-Harabasz Index on an $N$-D [`LayoutNd`] matching
/// `sklearn.metrics.calinski_harabasz_score`.
///
/// Returns `Ok(1.0)` when `intra_disp == 0.0` (exact `sklearn` convention) and `Ok(0.0)` when
/// `N < 3`, `K < 2`, or `K >= N`.
pub fn calinski_harabasz_score_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "calinski_harabasz_score_nd")?;
    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut mean = vec![0.0_f64; d];
    let mut centroids = vec![0.0_f64; k * d];
    for (&c, pt) in comp.iter().zip(coords.chunks_exact(d)) {
        let c_off = (c as usize) * d;
        for axis in 0..d {
            let v = pt[axis];
            mean[axis] += v;
            centroids[c_off + axis] += v;
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut mean {
        *m *= inv_n;
    }
    for (c, &sz) in sizes.iter().enumerate() {
        let inv = 1.0 / sz as f64;
        let c_off = c * d;
        for axis in 0..d {
            centroids[c_off + axis] *= inv;
        }
    }

    let mut extra_disp = 0.0_f64;
    for (c, &sz) in sizes.iter().enumerate() {
        let c_off = c * d;
        let mut d2 = 0.0_f64;
        for axis in 0..d {
            let diff = centroids[c_off + axis] - mean[axis];
            d2 += diff * diff;
        }
        extra_disp += (sz as f64) * d2;
    }

    let mut intra_disp = 0.0_f64;
    for (&c, pt) in comp.iter().zip(coords.chunks_exact(d)) {
        let c_off = (c as usize) * d;
        for axis in 0..d {
            let diff = pt[axis] - centroids[c_off + axis];
            intra_disp += diff * diff;
        }
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

/// Computes Davies-Bouldin Index on an $N$-D [`LayoutNd`] matching
/// `sklearn.metrics.davies_bouldin_score` (lower is better).
///
/// Returns `Ok(0.0)` when `N < 3`, `K < 2`, `K >= N`, all intra-cluster spreads `<= 1e-8`, or all
/// inter-centroid distances `<= 1e-8`.
pub fn davies_bouldin_score_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "davies_bouldin_score_nd")?;
    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut centroids = vec![0.0_f64; k * d];
    for (&c, pt) in comp.iter().zip(coords.chunks_exact(d)) {
        let c_off = (c as usize) * d;
        for axis in 0..d {
            centroids[c_off + axis] += pt[axis];
        }
    }
    for (c, &sz) in sizes.iter().enumerate() {
        let inv = 1.0 / sz as f64;
        let c_off = c * d;
        for axis in 0..d {
            centroids[c_off + axis] *= inv;
        }
    }

    let mut s = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        let pt = &coords[i * d..(i + 1) * d];
        let cent = &centroids[c * d..(c + 1) * d];
        s[c] += euclidean_dist_nd(pt, cent);
    }
    for (c, &sz) in sizes.iter().enumerate() {
        s[c] /= sz as f64;
    }

    let all_s_zero = s.iter().all(|&v| v <= 1e-8);
    let mut all_d_zero = true;
    for i in 0..k {
        let ci = &centroids[i * d..(i + 1) * d];
        for j in (i + 1)..k {
            let cj = &centroids[j * d..(j + 1) * d];
            if euclidean_dist_nd(ci, cj) > 1e-8 {
                all_d_zero = false;
                break;
            }
        }
        if !all_d_zero {
            break;
        }
    }
    if all_s_zero || all_d_zero {
        return Ok(0.0);
    }

    let mut sum_max = 0.0_f64;
    for i in 0..k {
        let ci = &centroids[i * d..(i + 1) * d];
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let cj = &centroids[j * d..(j + 1) * d];
            let d_ij = euclidean_dist_nd(ci, cj);
            if d_ij > 0.0 {
                let r_ij = (s[i] + s[j]) / d_ij;
                if r_ij > max_r {
                    max_r = r_ij;
                }
            }
        }
        sum_max += max_r;
    }
    Ok(sum_max / k as f64)
}

/// Computes Nocaj et al. (2015) §4.3 Pairwise Distance-Overlap Error (`PDOE`) on an $N$-D
/// [`LayoutNd`] in `[0.0, 1.0]` (lower is better).
///
/// Partitions all $\binom{N}{2}$ pairs into intra-cluster and inter-cluster sets, constructs
/// normalized empirical distance histograms over `[0, d_max]` with `num_bins` uniform bins,
/// and returns $\sum_b \min(h_{\mathrm{intra}}(b), h_{\mathrm{inter}}(b))$.
pub fn pairwise_distance_overlap_error_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    num_bins: usize,
) -> CoreResult<f64> {
    if num_bins == 0 {
        return Err(CoreError::invalid_input(
            "pairwise_distance_overlap_error_nd requires num_bins >= 1",
        ));
    }
    validate_layout_nd_labels(layout, labels, "pairwise_distance_overlap_error_nd")?;

    let n = layout.num_nodes();
    let d = layout.dims();
    let (comp, _, k) = compact_labels(labels);
    if n < 2 || k <= 1 || k == n {
        return Ok(0.0);
    }

    let coords = coords_to_f64_vec(layout);
    let mut d_max = 0.0_f64;
    for i in 0..n {
        let pi = &coords[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &coords[j * d..(j + 1) * d];
            d_max = d_max.max(euclidean_dist_nd(pi, pj));
        }
    }
    if d_max == 0.0 {
        return Ok(1.0);
    }

    let mut hist_intra = vec![0u64; num_bins];
    let mut hist_inter = vec![0u64; num_bins];
    let mut count_intra = 0u64;
    let mut count_inter = 0u64;
    let num_bins_f = num_bins as f64;

    for i in 0..n {
        let pi = &coords[i * d..(i + 1) * d];
        let ci = comp[i];
        for j in (i + 1)..n {
            let pj = &coords[j * d..(j + 1) * d];
            let dist = euclidean_dist_nd(pi, pj);
            let bin = (((dist / d_max) * num_bins_f).floor() as usize).min(num_bins - 1);
            if ci == comp[j] {
                hist_intra[bin] += 1;
                count_intra += 1;
            } else {
                hist_inter[bin] += 1;
                count_inter += 1;
            }
        }
    }

    if count_intra == 0 || count_inter == 0 {
        return Ok(0.0);
    }

    let inv_intra = 1.0 / count_intra as f64;
    let inv_inter = 1.0 / count_inter as f64;
    let mut overlap = 0.0_f64;
    for b in 0..num_bins {
        let hi = hist_intra[b] as f64 * inv_intra;
        let hb = hist_inter[b] as f64 * inv_inter;
        overlap += hi.min(hb);
    }
    Ok(overlap.clamp(0.0, 1.0))
}
