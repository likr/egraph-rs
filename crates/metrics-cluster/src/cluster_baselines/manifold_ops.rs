use std::f64::consts::TAU;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, ManifoldKind, ManifoldLayout2d, ManifoldMetric,
    MetricSpace, Scalar, cartesian_to_spherical, lorentz_distance, lorentz_einstein_midpoint,
    lorentz_exp_map, lorentz_log_map, poincare_distance, poincare_einstein_midpoint,
    poincare_exp_map, poincare_log_map, project_spherical_coords, spherical_distance,
    spherical_exp_map, spherical_log_map, spherical_to_cartesian, torus_delta, wrap_torus_coord,
    wrap_torus_point,
};

use super::agreement::partition_agreement;
use super::space_dispatch::metric_diameter;
use super::{PartitionAgreement, compact_labels};

#[inline]
pub(super) fn manifold_point_dist(
    kind: ManifoldKind,
    metric: &ManifoldMetric,
    p1: (f64, f64),
    p2: (f64, f64),
) -> CoreResult<f64> {
    match kind {
        ManifoldKind::HyperbolicPoincare => {
            Ok(metric.scale * poincare_distance(p1.0, p1.1, p2.0, p2.1))
        }
        ManifoldKind::HyperbolicLorentz => {
            Ok(metric.scale * lorentz_distance(p1.0, p1.1, p2.0, p2.1))
        }
        ManifoldKind::Spherical => Ok(metric.scale * spherical_distance(p1.0, p1.1, p2.0, p2.1)),
        ManifoldKind::Torus2d => {
            let (dx, dy) = torus_delta(p2.0, p2.1, p1.0, p1.1);
            Ok((metric.scale * metric.aspect * dx).hypot(metric.scale * dy))
        }
        _ => Err(CoreError::unsupported("unsupported manifold kind")),
    }
}

pub(super) fn intrinsic_manifold_centroid<S: Scalar>(
    kind: ManifoldKind,
    xs: &[S],
    ys: &[S],
    indices: &[usize],
) -> CoreResult<(f64, f64)> {
    if indices.is_empty() {
        return Ok((0.0, 0.0));
    }
    match kind {
        ManifoldKind::HyperbolicPoincare => {
            let mut sub_x = Vec::with_capacity(indices.len());
            let mut sub_y = Vec::with_capacity(indices.len());
            for &i in indices {
                sub_x.push(xs[i].to_f64());
                sub_y.push(ys[i].to_f64());
            }
            poincare_einstein_midpoint(&sub_x, &sub_y)
        }
        ManifoldKind::HyperbolicLorentz => {
            let mut sub_x = Vec::with_capacity(indices.len());
            let mut sub_y = Vec::with_capacity(indices.len());
            for &i in indices {
                sub_x.push(xs[i].to_f64());
                sub_y.push(ys[i].to_f64());
            }
            lorentz_einstein_midpoint(&sub_x, &sub_y)
        }
        ManifoldKind::Spherical => {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut sz = 0.0_f64;
            for &i in indices {
                let p = spherical_to_cartesian(xs[i].to_f64(), ys[i].to_f64());
                sx += p[0];
                sy += p[1];
                sz += p[2];
            }
            let norm = (sx * sx + sy * sy + sz * sz).sqrt();
            if !norm.is_finite() || norm <= 1e-12 {
                let i0 = indices[0];
                Ok(project_spherical_coords(xs[i0].to_f64(), ys[i0].to_f64()))
            } else {
                Ok(cartesian_to_spherical(sx / norm, sy / norm, sz / norm))
            }
        }
        ManifoldKind::Torus2d => {
            let mut cx = 0.0_f64;
            let mut sx = 0.0_f64;
            let mut cy = 0.0_f64;
            let mut sy = 0.0_f64;
            for &i in indices {
                let wx = wrap_torus_coord(xs[i].to_f64());
                let wy = wrap_torus_coord(ys[i].to_f64());
                let (sin_x, cos_x) = (TAU * wx).sin_cos();
                let (sin_y, cos_y) = (TAU * wy).sin_cos();
                cx += cos_x;
                sx += sin_x;
                cy += cos_y;
                sy += sin_y;
            }
            let i0 = indices[0];
            let mx = if sx.hypot(cx) <= 1e-12 {
                wrap_torus_coord(xs[i0].to_f64())
            } else {
                wrap_torus_coord((sx.atan2(cx) / TAU).rem_euclid(1.0))
            };
            let my = if sy.hypot(cy) <= 1e-12 {
                wrap_torus_coord(ys[i0].to_f64())
            } else {
                wrap_torus_coord((sy.atan2(cy) / TAU).rem_euclid(1.0))
            };
            Ok((mx, my))
        }
        _ => Err(CoreError::unsupported("unsupported manifold kind")),
    }
}

pub(super) fn compute_cluster_centroids<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    comp: &[u32],
    sizes: &[usize],
    k: usize,
) -> CoreResult<Vec<(f64, f64)>> {
    let n = layout.len();
    let mut offsets = vec![0usize; k + 1];
    for c in 0..k {
        offsets[c + 1] = offsets[c] + sizes[c];
    }
    let mut cursor = offsets[..k].to_vec();
    let mut members = vec![0usize; n];
    for (i, &c) in comp.iter().enumerate() {
        let pos = cursor[c as usize];
        members[pos] = i;
        cursor[c as usize] = pos + 1;
    }

    let kind = layout.kind();
    let xs = layout.x();
    let ys = layout.y();
    let mut centroids = Vec::with_capacity(k);
    for c in 0..k {
        centroids.push(intrinsic_manifold_centroid(
            kind,
            xs,
            ys,
            &members[offsets[c]..offsets[c + 1]],
        )?);
    }
    Ok(centroids)
}

pub(super) fn manifold_geodesic_interpolate_point<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    t: f64,
) -> CoreResult<(f64, f64)> {
    let x1 = layout.x()[i].to_f64();
    let y1 = layout.y()[i].to_f64();
    let x2 = layout.x()[j].to_f64();
    let y2 = layout.y()[j].to_f64();
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare => {
            let (vx, vy) = poincare_log_map(x1, y1, x2, y2);
            Ok(poincare_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::HyperbolicLorentz => {
            let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
            Ok(lorentz_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::Spherical => {
            let (vx, vy) = spherical_log_map(x1, y1, x2, y2);
            Ok(spherical_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::Torus2d => {
            let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
            Ok(wrap_torus_point(x1 + t * dx_norm, y1 + t * dy_norm))
        }
        _ => Err(CoreError::unsupported("unsupported manifold kind")),
    }
}

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
        let d = layout.distance(i, first);
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
                let d = layout.distance(i, chosen);
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
                let d = manifold_point_dist(kind, &metric, pt, cent)?;
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
            let shift = manifold_point_dist(kind, &metric, centroids[c], next_c)?;
            max_shift = max_shift.max(shift);
            centroids[c] = next_c;
        }
        if !reseeded && max_shift <= tol_dist {
            break;
        }
    }

    let mut inertia = 0.0_f64;
    for i in 0..n {
        let d = manifold_point_dist(
            kind,
            &metric,
            (xs[i].to_f64(), ys[i].to_f64()),
            centroids[pred[i] as usize],
        )?;
        inertia += d * d;
    }
    Ok((pred, inertia))
}

pub(super) fn cq_kmeans_manifold_inner<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    n_init: usize,
    max_iters: usize,
    seed: u64,
) -> CoreResult<PartitionAgreement> {
    let n = layout.len();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let d_max = metric_diameter(layout);
    if !d_max.is_finite() || d_max <= 1e-15 {
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
