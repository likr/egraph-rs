use egraph_core::{CoreError, CoreResult, ManifoldLayout2d, MetricSpace, Scalar};

use super::compact_labels;
use super::manifold_ops::{compute_cluster_centroids, manifold_point_dist};
use super::space_dispatch::{SpaceKind, classify_space, validate_metric_labels};

fn distance_consistency_euclidean(coords: &[f64], d: usize, labels: &[u32]) -> f64 {
    let n = labels.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 2 || k < 2 {
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
    consistent as f64 / n as f64
}

fn distance_consistency_manifold_impl<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
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
        let own_d = manifold_point_dist(kind, &metric, pt, centroids[own])?;
        let mut is_strictly_closest = true;
        for (c, &cent) in centroids.iter().enumerate() {
            if c == own {
                continue;
            }
            if manifold_point_dist(kind, &metric, pt, cent)? <= own_d {
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

/// Computes Sips et al. (2009) Distance Consistency (`DSC`) in `[0.0, 1.0]` (higher is better)
/// across any [`egraph_core::EuclideanLayout`] or [`ManifoldLayout2d`].
///
/// Counts the fraction of points whose distance to their own class centroid is **strictly
/// smaller** than to every other class centroid (`N < 2 || K < 2` returns `0.0`). Returns
/// [`CoreError::UnsupportedOperation`] on non-Euclidean, non-manifold [`MetricSpace`] types.
pub fn distance_consistency<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "distance_consistency")?;
    match classify_space(layout, "distance_consistency")? {
        SpaceKind::Euclidean { coords, dims } => {
            Ok(distance_consistency_euclidean(&coords, dims, labels))
        }
        SpaceKind::ManifoldF64(ml) => distance_consistency_manifold_impl(ml, labels),
        SpaceKind::ManifoldF32(ml) => distance_consistency_manifold_impl(ml, labels),
    }
}

fn distribution_consistency_coords(
    coords: &[f64],
    d: usize,
    labels: &[u32],
    grid_size: usize,
) -> f64 {
    let n = labels.len();
    let (comp, _, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return 0.0;
    }

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
    (1.0 - weighted_entropy / denom).clamp(0.0, 1.0)
}

/// Computes Sips et al. (2009) Distribution Consistency (`DC`) scaled to `[0.0, 1.0]`
/// (`DC_paper = 100.0 * dc`) across any [`egraph_core::EuclideanLayout`] or
/// [`ManifoldLayout2d`] (evaluated over its 2D chart coordinates).
///
/// Uses sparse lexicographic sorting of occupied $D$-dimensional cell tuples in $O(N D \log N)$
/// time and $O(N D)$ memory over a `grid_size^D` partition of the normalized `[0, 1]^D` box.
/// Returns `0.0` when `N == 0` or `K < 2`. Errors when `grid_size == 0`, and returns
/// [`CoreError::UnsupportedOperation`] on non-Euclidean, non-manifold [`MetricSpace`] types.
pub fn distribution_consistency<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "distribution_consistency requires grid_size >= 1",
        ));
    }
    validate_metric_labels(layout, labels, "distribution_consistency")?;
    match classify_space(layout, "distribution_consistency")? {
        SpaceKind::Euclidean { coords, dims } => Ok(distribution_consistency_coords(
            &coords, dims, labels, grid_size,
        )),
        SpaceKind::ManifoldF64(ml) => distribution_consistency(ml.as_layout(), labels, grid_size),
        SpaceKind::ManifoldF32(ml) => distribution_consistency(ml.as_layout(), labels, grid_size),
    }
}
