use egraph_core::{CoreError, CoreResult, ManifoldLayout2d, MetricSpace, Scalar};

use super::compact_labels;
use super::manifold_ops::{
    compute_cluster_centroids, intrinsic_manifold_centroid, manifold_point_dist,
};
use super::space_dispatch::{
    SpaceKind, classify_space, euclidean_dist_slice, metric_diameter, validate_metric_labels,
};

/// Computes mean Silhouette coefficient matching `sklearn.metrics.silhouette_score` in
/// `[-1.0, 1.0]` using $O(N + K)$ memory across any [`MetricSpace`] (singleton clusters
/// score `0.0`).
///
/// Under the non-panic evaluation contract, returns `Ok(0.0)` when `N < 3`, `K < 2`, or `K >= N`.
pub fn silhouette_score<L: MetricSpace + ?Sized>(layout: &L, labels: &[u32]) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "silhouette_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let mut sum_dists = vec![0.0_f64; k];
    let mut total_s = 0.0_f64;

    for i in 0..n {
        let ci = comp[i] as usize;
        if sizes[ci] == 1 {
            continue;
        }
        sum_dists.fill(0.0);
        for j in 0..n {
            if i != j {
                sum_dists[comp[j] as usize] += layout.distance(i, j);
            }
        }
        let a_i = sum_dists[ci] / (sizes[ci] - 1) as f64;
        let mut b_i = f64::INFINITY;
        for (c, &sz) in sizes.iter().enumerate() {
            if c != ci {
                b_i = b_i.min(sum_dists[c] / sz as f64);
            }
        }
        let denom = a_i.max(b_i);
        if denom > 0.0 {
            total_s += (b_i - a_i) / denom;
        }
    }

    Ok(total_s / n as f64)
}

fn calinski_harabasz_euclidean(coords: &[f64], d: usize, labels: &[u32]) -> f64 {
    let n = labels.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return 0.0;
    }

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
        return 1.0;
    }
    extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64))
}

fn calinski_harabasz_manifold_impl<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let kind = layout.kind();
    let metric = layout.metric();
    let xs = layout.x();
    let ys = layout.y();
    let all_indices: Vec<usize> = (0..n).collect();
    let global_cent = intrinsic_manifold_centroid(kind, xs, ys, &all_indices)?;
    let centroids = compute_cluster_centroids(layout, &comp, &sizes, k)?;

    let mut extra_disp = 0.0_f64;
    for (c, &sz) in sizes.iter().enumerate() {
        let d = manifold_point_dist(kind, &metric, centroids[c], global_cent)?;
        extra_disp += (sz as f64) * d * d;
    }

    let mut intra_disp = 0.0_f64;
    for i in 0..n {
        let c = comp[i] as usize;
        let d = manifold_point_dist(
            kind,
            &metric,
            (xs[i].to_f64(), ys[i].to_f64()),
            centroids[c],
        )?;
        intra_disp += d * d;
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

/// Computes Calinski-Harabasz Index matching `sklearn.metrics.calinski_harabasz_score` across
/// any [`egraph_core::EuclideanLayout`] or [`ManifoldLayout2d`].
///
/// Returns `Ok(1.0)` when `intra_disp == 0.0` (exact `sklearn` convention) and `Ok(0.0)` when
/// `N < 3`, `K < 2`, or `K >= N`. Returns [`CoreError::UnsupportedOperation`] on non-Euclidean,
/// non-manifold [`MetricSpace`] types.
pub fn calinski_harabasz_score<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "calinski_harabasz_score")?;
    match classify_space(layout, "calinski_harabasz_score")? {
        SpaceKind::Euclidean { coords, dims } => {
            Ok(calinski_harabasz_euclidean(&coords, dims, labels))
        }
        SpaceKind::ManifoldF64(ml) => calinski_harabasz_manifold_impl(ml, labels),
        SpaceKind::ManifoldF32(ml) => calinski_harabasz_manifold_impl(ml, labels),
    }
}

fn davies_bouldin_euclidean(coords: &[f64], d: usize, labels: &[u32]) -> f64 {
    let n = labels.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return 0.0;
    }

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
        s[c] += euclidean_dist_slice(pt, cent);
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
            if euclidean_dist_slice(ci, cj) > 1e-8 {
                all_d_zero = false;
                break;
            }
        }
        if !all_d_zero {
            break;
        }
    }
    if all_s_zero || all_d_zero {
        return 0.0;
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
            let d_ij = euclidean_dist_slice(ci, cj);
            if d_ij > 0.0 {
                let r_ij = (s[i] + s[j]) / d_ij;
                if r_ij > max_r {
                    max_r = r_ij;
                }
            }
        }
        sum_max += max_r;
    }
    sum_max / k as f64
}

fn davies_bouldin_manifold_impl<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let kind = layout.kind();
    let metric = layout.metric();
    let xs = layout.x();
    let ys = layout.y();
    let centroids = compute_cluster_centroids(layout, &comp, &sizes, k)?;

    let mut s = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        s[c] += manifold_point_dist(
            kind,
            &metric,
            (xs[i].to_f64(), ys[i].to_f64()),
            centroids[c],
        )?;
    }
    for (c, &sz) in sizes.iter().enumerate() {
        s[c] /= sz as f64;
    }

    let all_s_zero = s.iter().all(|&v| v <= 1e-8);
    let mut all_d_zero = true;
    for i in 0..k {
        for j in (i + 1)..k {
            if manifold_point_dist(kind, &metric, centroids[i], centroids[j])? > 1e-8 {
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
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij = manifold_point_dist(kind, &metric, centroids[i], centroids[j])?;
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

/// Computes Davies-Bouldin Index matching `sklearn.metrics.davies_bouldin_score` (lower is better)
/// across any [`egraph_core::EuclideanLayout`] or [`ManifoldLayout2d`].
///
/// Returns `Ok(0.0)` when `N < 3`, `K < 2`, `K >= N`, all intra-cluster spreads `<= 1e-8`, or all
/// inter-centroid distances `<= 1e-8`. Returns [`CoreError::UnsupportedOperation`] on
/// non-Euclidean, non-manifold [`MetricSpace`] types.
pub fn davies_bouldin_score<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "davies_bouldin_score")?;
    match classify_space(layout, "davies_bouldin_score")? {
        SpaceKind::Euclidean { coords, dims } => {
            Ok(davies_bouldin_euclidean(&coords, dims, labels))
        }
        SpaceKind::ManifoldF64(ml) => davies_bouldin_manifold_impl(ml, labels),
        SpaceKind::ManifoldF32(ml) => davies_bouldin_manifold_impl(ml, labels),
    }
}

/// Computes Nocaj et al. (2015) §4.3 Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]`
/// (lower is better) across any [`MetricSpace`].
///
/// Partitions all $\binom{N}{2}$ pairs into intra-cluster and inter-cluster sets, constructs
/// normalized empirical distance histograms over `[0, d_max]` with `num_bins` uniform bins,
/// and returns $\sum_b \min(h_{\mathrm{intra}}(b), h_{\mathrm{inter}}(b))$.
pub fn pairwise_distance_overlap_error<L: MetricSpace + ?Sized>(
    layout: &L,
    labels: &[u32],
    num_bins: usize,
) -> CoreResult<f64> {
    if num_bins == 0 {
        return Err(CoreError::invalid_input(
            "pairwise_distance_overlap_error requires num_bins >= 1",
        ));
    }
    validate_metric_labels(layout, labels, "pairwise_distance_overlap_error")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n < 2 || k <= 1 || k == n {
        return Ok(0.0);
    }

    let d_max = metric_diameter(layout);
    if d_max == 0.0 {
        return Ok(1.0);
    }

    let mut hist_intra = vec![0u64; num_bins];
    let mut hist_inter = vec![0u64; num_bins];
    let mut count_intra = 0u64;
    let mut count_inter = 0u64;
    let num_bins_f = num_bins as f64;

    for (i, &ci) in comp.iter().enumerate() {
        for (j, &cj) in comp.iter().enumerate().skip(i + 1) {
            let d = layout.distance(i, j);
            let bin = (((d / d_max) * num_bins_f).floor() as usize).min(num_bins - 1);
            if ci == cj {
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
