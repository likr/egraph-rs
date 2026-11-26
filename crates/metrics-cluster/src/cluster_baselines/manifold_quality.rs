use egraph_core::{CoreError, CoreResult, DeterministicRng, ManifoldLayout2d, Scalar};
use egraph_metrics_geometry::proximity::manifold_geodesic_diameter;

use super::agreement::partition_agreement;
use super::cluster_quality::upgma_hac_labels;
use super::manifold_centroid::{
    intrinsic_manifold_centroid, manifold_point_dist, validate_manifold_labels,
};
use super::{PartitionAgreement, compact_labels};

fn run_single_manifold_kmeans<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    k: usize,
    max_iters: usize,
    tol_dist: f64,
    rng: &mut DeterministicRng,
) -> CoreResult<(Vec<u32>, f64)> {
    let n = layout.len();
    let kind = layout.kind();
    let metric = layout.metric();
    let xs = layout.x();
    let ys = layout.y();

    let first = rng.gen_range_usize(n);
    let mut centroids = vec![(0.0_f64, 0.0_f64); k];
    centroids[0] = (xs[first].to_f64(), ys[first].to_f64());
    let mut min_d2 = vec![0.0_f64; n];
    for (i, slot) in min_d2.iter_mut().enumerate() {
        let d = layout.geodesic_distance(i, first)?.to_f64();
        *slot = d * d;
    }

    for (c, cent_slot) in centroids.iter_mut().enumerate().take(k).skip(1) {
        let total: f64 = min_d2.iter().sum();
        let chosen = if total <= 0.0 || !total.is_finite() {
            rng.gen_range_usize(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cdf = 0.0_f64;
            let mut picked = n - 1;
            for (i, &d2) in min_d2.iter().enumerate() {
                if d2 > 0.0 {
                    picked = i;
                }
                cdf += d2;
                if cdf >= target {
                    picked = i;
                    break;
                }
            }
            picked
        };
        *cent_slot = (xs[chosen].to_f64(), ys[chosen].to_f64());
        if c + 1 < k {
            for (i, slot) in min_d2.iter_mut().enumerate() {
                let d = layout.geodesic_distance(i, chosen)?.to_f64();
                let d2 = d * d;
                if d2 < *slot {
                    *slot = d2;
                }
            }
        }
    }

    let mut pred = vec![0u32; n];
    let mut point_d2 = vec![0.0_f64; n];
    let mut counts = vec![0usize; k];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];

    for _ in 0..max_iters {
        counts.fill(0);
        for i in 0..n {
            let pt = (xs[i].to_f64(), ys[i].to_f64());
            let mut best_c = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (c, &cent) in centroids.iter().enumerate() {
                let d = manifold_point_dist::<S>(kind, metric, pt, cent)?;
                let d2 = d * d;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = c;
                }
            }
            pred[i] = best_c as u32;
            point_d2[i] = best_d2;
            counts[best_c] += 1;
        }

        let mut reseeded = false;
        for c in 0..k {
            if counts[c] > 0 {
                continue;
            }
            reseeded = true;
            let mut far_idx = 0usize;
            let mut far_d2 = -1.0_f64;
            for i in 0..n {
                let owner = pred[i] as usize;
                if counts[owner] > 1 && point_d2[i] > far_d2 {
                    far_d2 = point_d2[i];
                    far_idx = i;
                }
            }
            let old_c = pred[far_idx] as usize;
            counts[old_c] = counts[old_c].saturating_sub(1);
            pred[far_idx] = c as u32;
            point_d2[far_idx] = 0.0;
            counts[c] = 1;
        }

        for list in &mut members {
            list.clear();
        }
        for (i, &c) in pred.iter().enumerate() {
            members[c as usize].push(i);
        }

        let mut max_shift = 0.0_f64;
        for c in 0..k {
            if members[c].is_empty() {
                continue;
            }
            let next_c = intrinsic_manifold_centroid(kind, xs, ys, &members[c])?;
            let shift = manifold_point_dist::<S>(kind, metric, centroids[c], next_c)?;
            max_shift = max_shift.max(shift);
            centroids[c] = next_c;
        }
        if !reseeded && max_shift <= tol_dist {
            break;
        }
    }

    let mut inertia = 0.0_f64;
    for i in 0..n {
        let d = manifold_point_dist::<S>(
            kind,
            metric,
            (xs[i].to_f64(), ys[i].to_f64()),
            centroids[pred[i] as usize],
        )?;
        inertia += d * d;
    }
    Ok((pred, inertia))
}

/// Evaluates Cluster Quality via geodesic $k$-means++ clustering with intrinsic manifold
/// centroids (`CQ-kmeans`), returning [`PartitionAgreement`] against `labels`.
pub fn cq_kmeans_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    n_init: usize,
    max_iters: usize,
    seed: u64,
) -> CoreResult<PartitionAgreement> {
    if n_init == 0 {
        return Err(CoreError::invalid_input(
            "cq_kmeans_manifold requires n_init >= 1",
        ));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input(
            "cq_kmeans_manifold requires max_iters >= 1",
        ));
    }
    validate_manifold_labels(layout, labels, "cq_kmeans_manifold")?;
    let n = layout.len();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let d_max = manifold_geodesic_diameter(layout)?;
    if d_max <= 1e-15 {
        return Ok(PartitionAgreement::default());
    }

    let tol_dist = 1e-4 * d_max;
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let (mut best_labels, mut best_inertia) =
        run_single_manifold_kmeans(layout, k, max_iters, tol_dist, &mut rng)?;
    for _ in 1..n_init {
        let (cand_labels, cand_inertia) =
            run_single_manifold_kmeans(layout, k, max_iters, tol_dist, &mut rng)?;
        if cand_inertia < best_inertia {
            best_inertia = cand_inertia;
            best_labels = cand_labels;
        }
    }
    partition_agreement(labels, &best_labels)
}

/// Evaluates Cluster Quality via exact $O(N^2)$ UPGMA hierarchical agglomerative clustering over
/// exact geodesic distances (`CQ-HAC`), returning [`PartitionAgreement`] against `labels`.
pub fn cq_hac_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<PartitionAgreement> {
    validate_manifold_labels(layout, labels, "cq_hac_manifold")?;
    let n = layout.len();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let d_max = manifold_geodesic_diameter(layout)?;
    if d_max <= 1e-15 {
        return Ok(PartitionAgreement::default());
    }

    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("cq_hac_manifold N x N matrix size overflow"))?;
    let mut dist = vec![0.0_f64; matrix_len];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = layout.geodesic_distance(i, j)?.to_f64();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let pred_labels = upgma_hac_labels(n, k, &mut dist);
    partition_agreement(labels, &pred_labels)
}
