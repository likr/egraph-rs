use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::{compact_labels, convex_hull_ccw, cross_2d, validate_layout_labels};

/// Computes mean Silhouette coefficient matching `sklearn.metrics.silhouette_score` in
/// `[-1.0, 1.0]` using $O(N + K)$ memory (singleton clusters score `0.0`).
///
/// Under the non-panic evaluation contract, returns `Ok(0.0)` when `N < 3`, `K < 2`, or `K >= N`.
pub fn silhouette_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "silhouette_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut sum_dists = vec![0.0_f64; k];
    let mut total_s = 0.0_f64;

    for i in 0..n {
        let ci = comp[i] as usize;
        if sizes[ci] == 1 {
            continue;
        }
        sum_dists.fill(0.0);
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        for j in 0..n {
            let d = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            sum_dists[comp[j] as usize] += d;
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

/// Computes Calinski-Harabasz Index matching `sklearn.metrics.calinski_harabasz_score`.
///
/// Returns `Ok(1.0)` when `intra_disp == 0.0` (exact `sklearn` convention) and `Ok(0.0)` when
/// `N < 3`, `K < 2`, or `K >= N`.
pub fn calinski_harabasz_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "calinski_harabasz_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let c = comp[i] as usize;
        mean_x += x;
        mean_y += y;
        cx[c] += x;
        cy[c] += y;
    }
    mean_x /= n as f64;
    mean_y /= n as f64;
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut extra_disp = 0.0_f64;
    for c in 0..k {
        let dx = cx[c] - mean_x;
        let dy = cy[c] - mean_y;
        extra_disp += (sizes[c] as f64) * (dx * dx + dy * dy);
    }

    let mut intra_disp = 0.0_f64;
    for i in 0..n {
        let c = comp[i] as usize;
        let dx = xs[i].to_f64() - cx[c];
        let dy = ys[i].to_f64() - cy[c];
        intra_disp += dx * dx + dy * dy;
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

/// Computes Davies-Bouldin Index matching `sklearn.metrics.davies_bouldin_score` (lower is better).
///
/// Returns `Ok(0.0)` when `N < 3`, `K < 2`, `K >= N`, all intra-cluster spreads `<= 1e-8`, or all
/// inter-centroid distances `<= 1e-8`.
pub fn davies_bouldin_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "davies_bouldin_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        cx[c] += xs[i].to_f64();
        cy[c] += ys[i].to_f64();
    }
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut s = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        s[c] += (xs[i].to_f64() - cx[c]).hypot(ys[i].to_f64() - cy[c]);
    }
    for c in 0..k {
        s[c] /= sizes[c] as f64;
    }

    let all_s_zero = s.iter().all(|&v| v <= 1e-8);
    let mut all_d_zero = true;
    for i in 0..k {
        for j in (i + 1)..k {
            if (cx[i] - cx[j]).hypot(cy[i] - cy[j]) > 1e-8 {
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
            let d_ij = (cx[i] - cx[j]).hypot(cy[i] - cy[j]);
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

#[inline]
fn dist_sq_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn euclidean_diameter_2d_core(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n <= 1 {
        return 0.0;
    }
    if n <= 64 {
        let mut max_sq = 0.0_f64;
        for i in 0..n {
            let xi = xs[i];
            let yi = ys[i];
            for j in (i + 1)..n {
                let dx = xi - xs[j];
                let dy = yi - ys[j];
                max_sq = max_sq.max(dx * dx + dy * dy);
            }
        }
        return max_sq.sqrt();
    }

    let pts: Vec<[f64; 2]> = (0..n).map(|i| [xs[i], ys[i]]).collect();
    let hull = convex_hull_ccw(pts);
    let m = hull.len();
    if m <= 1 {
        return 0.0;
    }
    if m == 2 {
        return dist_sq_2d(hull[0], hull[1]).sqrt();
    }

    let mut max_sq = 0.0_f64;
    let mut j = 1usize;
    for i in 0..m {
        let next_i = (i + 1) % m;
        loop {
            let next_j = (j + 1) % m;
            let cur_area = cross_2d(hull[i], hull[next_i], hull[j]).abs();
            let next_area = cross_2d(hull[i], hull[next_i], hull[next_j]).abs();
            match next_area.total_cmp(&cur_area) {
                Ordering::Greater => j = next_j,
                Ordering::Equal => {
                    max_sq = max_sq.max(dist_sq_2d(hull[i], hull[next_j]));
                    max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[next_j]));
                    break;
                }
                Ordering::Less => break,
            }
        }
        max_sq = max_sq.max(dist_sq_2d(hull[i], hull[j]));
        max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[j]));
    }
    max_sq.sqrt()
}

pub(super) fn euclidean_diameter_2d<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs: Vec<f64> = layout.x().iter().map(|v| v.to_f64()).collect();
    let ys: Vec<f64> = layout.y().iter().map(|v| v.to_f64()).collect();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span = (max_x - min_x).max(max_y - min_y);
    if span == 0.0 {
        return 0.0;
    }
    if span < 1e-100 || !span.is_finite() || span > 1e100 {
        let (s_xs, s_ys, back_scale) = if !span.is_finite() || span > 1e100 {
            let half_span = (max_x * 0.5 - min_x * 0.5).max(max_y * 0.5 - min_y * 0.5);
            let s_xs: Vec<f64> = xs
                .iter()
                .map(|&x| (x * 0.5 - min_x * 0.5) / half_span)
                .collect();
            let s_ys: Vec<f64> = ys
                .iter()
                .map(|&y| (y * 0.5 - min_y * 0.5) / half_span)
                .collect();
            (s_xs, s_ys, 2.0 * half_span)
        } else {
            let s_xs: Vec<f64> = xs.iter().map(|&x| (x - min_x) / span).collect();
            let s_ys: Vec<f64> = ys.iter().map(|&y| (y - min_y) / span).collect();
            (s_xs, s_ys, span)
        };
        return euclidean_diameter_2d_core(&s_xs, &s_ys) * back_scale;
    }
    euclidean_diameter_2d_core(&xs, &ys)
}

/// Computes Nocaj et al. (2015) §4.3 Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]`
/// (lower is better).
///
/// Partitions all $\binom{N}{2}$ pairs into intra-cluster and inter-cluster sets, constructs
/// normalized empirical distance histograms over `[0, d_max]` with `num_bins` uniform bins,
/// and returns $\sum_b \min(h_{\mathrm{intra}}(b), h_{\mathrm{inter}}(b))$.
pub fn pairwise_distance_overlap_error<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    num_bins: usize,
) -> CoreResult<f64> {
    if num_bins == 0 {
        return Err(CoreError::invalid_input(
            "pairwise_distance_overlap_error requires num_bins >= 1",
        ));
    }
    validate_layout_labels(layout, labels, "pairwise_distance_overlap_error")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n < 2 || k <= 1 || k == n {
        return Ok(0.0);
    }

    let d_max = euclidean_diameter_2d(layout);
    if d_max == 0.0 {
        return Ok(1.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut hist_intra = vec![0u64; num_bins];
    let mut hist_inter = vec![0u64; num_bins];
    let mut count_intra = 0u64;
    let mut count_inter = 0u64;
    let num_bins_f = num_bins as f64;

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let ci = comp[i];
        for j in (i + 1)..n {
            let d = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let bin = (((d / d_max) * num_bins_f).floor() as usize).min(num_bins - 1);
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
