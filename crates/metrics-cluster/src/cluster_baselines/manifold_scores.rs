use egraph_core::{CoreError, CoreResult, ManifoldLayout2d, Scalar};
use egraph_metrics_geometry::proximity::manifold_geodesic_diameter;

use super::compact_labels;
use super::manifold_centroid::{
    compute_cluster_centroids, intrinsic_manifold_centroid, manifold_point_dist,
    validate_manifold_labels,
};

/// Computes geodesic Distance Consistency (`DSC`) on a [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn distance_consistency_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "distance_consistency_manifold")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 2 || k < 2 {
        return Ok(0.0);
    }

    let centroids = compute_cluster_centroids(layout, &comp, &sizes, k)?;
    let kind = layout.kind();
    let metric = layout.metric();
    let xs = layout.x();
    let ys = layout.y();

    let mut consistent = 0usize;
    for i in 0..n {
        let own = comp[i] as usize;
        let pt = (xs[i].to_f64(), ys[i].to_f64());
        let own_d = manifold_point_dist::<S>(kind, metric, pt, centroids[own])?;
        let mut is_strictly_closest = true;
        for (c, &cent) in centroids.iter().enumerate() {
            if c == own {
                continue;
            }
            if manifold_point_dist::<S>(kind, metric, pt, cent)? <= own_d {
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

/// Computes Distribution Consistency (`DC`) on a [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn distribution_consistency_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "distribution_consistency_manifold requires grid_size >= 1",
        ));
    }
    validate_manifold_labels(layout, labels, "distribution_consistency_manifold")?;
    super::consistency::distribution_consistency(layout.as_layout(), labels, grid_size)
}

/// Computes geodesic mean Silhouette coefficient on a [`ManifoldLayout2d`] in `[-1.0, 1.0]`.
pub fn silhouette_score_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "silhouette_score_manifold")?;
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
                sum_dists[comp[j] as usize] += layout.geodesic_distance(i, j)?.to_f64();
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

/// Computes geodesic Calinski-Harabasz Index on a [`ManifoldLayout2d`].
pub fn calinski_harabasz_score_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "calinski_harabasz_score_manifold")?;
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
        let d = manifold_point_dist::<S>(kind, metric, centroids[c], global_cent)?;
        extra_disp += (sz as f64) * d * d;
    }

    let mut intra_disp = 0.0_f64;
    for i in 0..n {
        let c = comp[i] as usize;
        let d =
            manifold_point_dist::<S>(kind, metric, (xs[i].to_f64(), ys[i].to_f64()), centroids[c])?;
        intra_disp += d * d;
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

/// Computes geodesic Davies-Bouldin Index on a [`ManifoldLayout2d`] (lower is better).
pub fn davies_bouldin_score_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "davies_bouldin_score_manifold")?;
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
        s[c] +=
            manifold_point_dist::<S>(kind, metric, (xs[i].to_f64(), ys[i].to_f64()), centroids[c])?;
    }
    for (c, &sz) in sizes.iter().enumerate() {
        s[c] /= sz as f64;
    }

    let all_s_zero = s.iter().all(|&v| v <= 1e-8);
    let mut all_d_zero = true;
    for i in 0..k {
        for j in (i + 1)..k {
            if manifold_point_dist::<S>(kind, metric, centroids[i], centroids[j])? > 1e-8 {
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
            let d_ij = manifold_point_dist::<S>(kind, metric, centroids[i], centroids[j])?;
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

/// Computes geodesic Pairwise Distance-Overlap Error (`PDOE`) on a [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn pairwise_distance_overlap_error_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    num_bins: usize,
) -> CoreResult<f64> {
    if num_bins == 0 {
        return Err(CoreError::invalid_input(
            "pairwise_distance_overlap_error_manifold requires num_bins >= 1",
        ));
    }
    validate_manifold_labels(layout, labels, "pairwise_distance_overlap_error_manifold")?;
    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n < 2 || k <= 1 || k == n {
        return Ok(0.0);
    }

    let d_max = manifold_geodesic_diameter(layout)?;
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
            let d = layout.geodesic_distance(i, j)?.to_f64();
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
