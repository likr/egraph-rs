use std::f64::consts::TAU;

use egraph_core::{
    CoreError, CoreResult, ManifoldKind, ManifoldLayout2d, ManifoldMetric, Scalar,
    cartesian_to_spherical, lorentz_distance, lorentz_einstein_midpoint, poincare_distance,
    poincare_einstein_midpoint, project_spherical_coords, spherical_distance,
    spherical_to_cartesian, torus_delta, wrap_torus_coord,
};

#[inline]
pub(super) fn validate_manifold_labels<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    context: &'static str,
) -> CoreResult<()> {
    if layout.len() != labels.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
            labels.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

#[inline]
pub(super) fn manifold_point_dist<S: Scalar>(
    kind: ManifoldKind,
    metric: ManifoldMetric,
    p1: (f64, f64),
    p2: (f64, f64),
) -> CoreResult<f64> {
    let _ = std::marker::PhantomData::<S>;
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
