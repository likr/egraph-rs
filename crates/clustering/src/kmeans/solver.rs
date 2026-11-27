use egraph_core::{CoreError, CoreResult, DeterministicRng};

use super::{KMeansConfig, KMeansResult};

#[inline]
fn squared_euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    sum
}

/// Initializes `k` centroids via K-Means++ $D^2$ weighted sampling.
fn init_centroids_d2(
    points_flat: &[f64],
    n: usize,
    dims: usize,
    k: usize,
    rng: &mut DeterministicRng,
    centroids_out: &mut [f64],
    min_dist_sq: &mut [f64],
) {
    let first = rng.gen_range_usize(n);
    let first_src = &points_flat[first * dims..(first + 1) * dims];
    centroids_out[..dims].copy_from_slice(first_src);

    if k == 1 {
        return;
    }

    for i in 0..n {
        let pt = &points_flat[i * dims..(i + 1) * dims];
        min_dist_sq[i] = squared_euclidean_dist(pt, first_src);
    }

    for c in 1..k {
        let total: f64 = min_dist_sq.iter().sum();
        let chosen = if total <= 0.0 || !total.is_finite() {
            rng.gen_range_usize(n)
        } else {
            let target = rng.next_f64() * total;
            let mut cdf = 0.0_f64;
            let mut picked = n - 1;
            for (i, &d2) in min_dist_sq.iter().enumerate() {
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

        let c_off = c * dims;
        let center = &points_flat[chosen * dims..(chosen + 1) * dims];
        centroids_out[c_off..c_off + dims].copy_from_slice(center);

        if c + 1 < k {
            for i in 0..n {
                let pt = &points_flat[i * dims..(i + 1) * dims];
                let d2 = squared_euclidean_dist(pt, center);
                if d2 < min_dist_sq[i] {
                    min_dist_sq[i] = d2;
                }
            }
        }
    }
}

/// Runs a single Lloyd refinement trial starting from K-Means++ seeded centroids.
fn run_single_kmeans(
    points_flat: &[f64],
    n: usize,
    dims: usize,
    config: &KMeansConfig,
    rng: &mut DeterministicRng,
) -> KMeansResult {
    let k = config.k;
    let total_c = k * dims;
    let mut centroids = vec![0.0_f64; total_c];
    let mut next_centroids = vec![0.0_f64; total_c];
    let mut min_dist_sq = vec![0.0_f64; n];
    init_centroids_d2(
        points_flat,
        n,
        dims,
        k,
        rng,
        &mut centroids,
        &mut min_dist_sq,
    );

    let mut labels = vec![0u32; n];
    let mut point_dist_sq = vec![0.0_f64; n];
    let mut counts = vec![0usize; k];
    let tol_sq = config.tol * config.tol;

    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 1..=config.max_iterations {
        iterations = iter;
        counts.fill(0);
        next_centroids.fill(0.0);

        for i in 0..n {
            let pt = &points_flat[i * dims..(i + 1) * dims];
            let mut best_c = 0usize;
            let mut best_d2 = squared_euclidean_dist(pt, &centroids[..dims]);
            for c in 1..k {
                let d2 = squared_euclidean_dist(pt, &centroids[c * dims..(c + 1) * dims]);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = c;
                }
            }
            labels[i] = best_c as u32;
            point_dist_sq[i] = best_d2;
            counts[best_c] += 1;
            let dst = &mut next_centroids[best_c * dims..(best_c + 1) * dims];
            for (acc, &val) in dst.iter_mut().zip(pt.iter()) {
                *acc += val;
            }
        }

        // Reseed any empty cluster using the farthest point from a multi-member cluster
        // so cluster splitting never empties another cluster when avoidable.
        let mut reseeded = false;
        for c in 0..k {
            if counts[c] > 0 {
                continue;
            }
            reseeded = true;
            let mut far_idx = None;
            let mut far_d2 = -1.0_f64;
            for i in 0..n {
                let owner = labels[i] as usize;
                if counts[owner] > 1 && point_dist_sq[i] > far_d2 {
                    far_d2 = point_dist_sq[i];
                    far_idx = Some(i);
                }
            }
            let chosen = far_idx.unwrap_or_else(|| {
                let mut best_i = 0usize;
                let mut best_d = -1.0_f64;
                for (i, &d2) in point_dist_sq.iter().enumerate() {
                    if d2 > best_d {
                        best_d = d2;
                        best_i = i;
                    }
                }
                best_i
            });

            let old_c = labels[chosen] as usize;
            let pt = &points_flat[chosen * dims..(chosen + 1) * dims];
            counts[old_c] = counts[old_c].saturating_sub(1);
            let old_acc = &mut next_centroids[old_c * dims..(old_c + 1) * dims];
            for (acc, &val) in old_acc.iter_mut().zip(pt.iter()) {
                *acc -= val;
            }

            labels[chosen] = c as u32;
            point_dist_sq[chosen] = 0.0;
            counts[c] = 1;
            next_centroids[c * dims..(c + 1) * dims].copy_from_slice(pt);
        }

        let mut max_shift_sq = 0.0_f64;
        for (c, &count) in counts.iter().enumerate() {
            let c_off = c * dims;
            if count > 0 {
                let inv = 1.0 / (count as f64);
                for d in 0..dims {
                    next_centroids[c_off + d] *= inv;
                }
            } else {
                next_centroids[c_off..c_off + dims]
                    .copy_from_slice(&centroids[c_off..c_off + dims]);
            }
            let shift_sq = squared_euclidean_dist(
                &next_centroids[c_off..c_off + dims],
                &centroids[c_off..c_off + dims],
            );
            if shift_sq > max_shift_sq {
                max_shift_sq = shift_sq;
            }
        }

        centroids.copy_from_slice(&next_centroids);

        if !reseeded && max_shift_sq <= tol_sq {
            converged = true;
            break;
        }
    }

    // Compute exact inertia against the finalized centroids of the current label assignment.
    let mut inertia = 0.0_f64;
    for i in 0..n {
        let c = labels[i] as usize;
        let pt = &points_flat[i * dims..(i + 1) * dims];
        inertia += squared_euclidean_dist(pt, &centroids[c * dims..(c + 1) * dims]);
    }

    canonicalize_labels_and_centroids(&mut labels, &mut centroids, k, dims);

    KMeansResult {
        labels,
        centroids,
        k,
        dims,
        inertia,
        iterations,
        converged,
    }
}

/// Renumbers cluster labels in canonical `0..k` first-appearance order while permuting
/// the `(k, dims)` centroid rows identically.
fn canonicalize_labels_and_centroids(
    labels: &mut [u32],
    centroids: &mut [f64],
    k: usize,
    dims: usize,
) {
    let mut old_to_new = vec![u32::MAX; k];
    let mut next_id = 0u32;
    for label in labels.iter_mut() {
        let old = *label as usize;
        if old_to_new[old] == u32::MAX {
            old_to_new[old] = next_id;
            next_id += 1;
        }
        *label = old_to_new[old];
    }
    for slot in &mut old_to_new {
        if *slot == u32::MAX {
            *slot = next_id;
            next_id += 1;
        }
    }

    let mut reordered = vec![0.0_f64; k * dims];
    for (old, &new_c) in old_to_new.iter().enumerate() {
        let src = &centroids[old * dims..(old + 1) * dims];
        let dst = &mut reordered[new_c as usize * dims..(new_c as usize + 1) * dims];
        dst.copy_from_slice(src);
    }
    centroids.copy_from_slice(&reordered);
}

/// Clusters `n` points of dimension `dims` stored in flat row-major `points_flat` using
/// deterministic K-Means++ initialization and Lloyd's algorithm.
///
/// Labels in the returned [`KMeansResult`] are canonicalized in ascending order of first
/// appearance over `0..n`, and `centroids` rows are ordered to match `labels`.
pub fn kmeans_plus_plus(
    points_flat: &[f64],
    n: usize,
    dims: usize,
    config: &KMeansConfig,
) -> CoreResult<KMeansResult> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    if config.k == 0 {
        return Err(CoreError::invalid_input("k must be >= 1"));
    }
    if config.max_iterations == 0 {
        return Err(CoreError::invalid_input("max_iterations must be >= 1"));
    }
    if config.n_init == 0 {
        return Err(CoreError::invalid_input("n_init must be >= 1"));
    }
    if !config.tol.is_finite() {
        return Err(CoreError::non_finite("KMeansConfig::tol"));
    }
    if config.tol < 0.0 {
        return Err(CoreError::invalid_input("tol must be >= 0.0"));
    }
    if n == 0 {
        return Err(CoreError::invalid_input("n must be >= 1"));
    }
    if config.k > n {
        return Err(CoreError::invalid_input(format!(
            "k ({}) exceeds number of points n ({n})",
            config.k
        )));
    }

    let expected = n
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("kmeans_plus_plus input size overflow"))?;
    if points_flat.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            points_flat.len(),
            "kmeans_plus_plus::points_flat",
        ));
    }
    if points_flat.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite("kmeans_plus_plus::points_flat"));
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let mut best = run_single_kmeans(points_flat, n, dims, config, &mut rng);
    for _ in 1..config.n_init {
        let candidate = run_single_kmeans(points_flat, n, dims, config, &mut rng);
        if candidate.inertia < best.inertia {
            best = candidate;
        }
    }
    Ok(best)
}
