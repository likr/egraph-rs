use std::f64::consts::FRAC_PI_2;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
    clamp_poincare_point, poincare_log_map,
};

use super::geometry::extract_csr_edges;
use super::{CrossingPair, SweepLineCrossings};

struct KleinProjection {
    poincare_pts: Vec<(f64, f64)>,
    klein: Layout2d<f64>,
}

/// Converts a Poincaré disk layout (or Lorentz hyperboloid layout via `to_poincare_layout`)
/// into clamped Poincaré coordinates and Beltrami-Klein disk chords where hyperbolic geodesics
/// are exact Euclidean line segments.
fn to_poincare_and_klein<S: Scalar>(layout: &ManifoldLayout2d<S>) -> CoreResult<KleinProjection> {
    layout.validate_finite()?;
    let poincare = match layout.kind() {
        ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => {
            layout.to_poincare_layout()?
        }
        _ => {
            return Err(CoreError::unsupported(
                "hyperbolic crossings require HyperbolicPoincare or HyperbolicLorentz",
            ));
        }
    };

    let n = poincare.len();
    let mut poincare_pts = Vec::with_capacity(n);
    let mut kx = Vec::with_capacity(n);
    let mut ky = Vec::with_capacity(n);
    for (&x, &y) in poincare.x().iter().zip(poincare.y().iter()) {
        let (cx, cy) = clamp_poincare_point(x.to_f64(), y.to_f64());
        poincare_pts.push((cx, cy));
        let denom = 1.0 + cx * cx + cy * cy;
        kx.push((2.0 * cx) / denom);
        ky.push((2.0 * cy) / denom);
    }
    let klein = Layout2d::from_xy(kx, ky)?;
    Ok(KleinProjection {
        poincare_pts,
        klein,
    })
}

#[inline]
fn hyperbolic_crossing_angle(
    poincare_pts: &[(f64, f64)],
    klein: &Layout2d<f64>,
    pair: &CrossingPair,
) -> (Option<f64>, f64) {
    let kxs = klein.x();
    let kys = klein.y();
    let ax = kxs[pair.u1 as usize];
    let ay = kys[pair.u1 as usize];
    let bx = kxs[pair.v1 as usize];
    let by = kys[pair.v1 as usize];
    let cx = kxs[pair.u2 as usize];
    let cy = kys[pair.u2 as usize];
    let dx = kxs[pair.v2 as usize];
    let dy = kys[pair.v2 as usize];

    let d1x = bx - ax;
    let d1y = by - ay;
    let d2x = dx - cx;
    let d2y = dy - cy;
    let len1 = d1x.hypot(d1y);
    let len2 = d2x.hypot(d2y);
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }

    let det = d1x * d2y - d1y * d2x;
    if det.abs() <= 1e-14 * len1 * len2 {
        // Collinear overlapping geodesics meet tangentially at angle 0.
        return (Some(0.0), 1.0);
    }

    let t = (((cx - ax) * d2y - (cy - ay) * d2x) / det).clamp(0.0, 1.0);
    let kix = ax + t * d1x;
    let kiy = ay + t * d1y;

    // Map Klein intersection k* back to Poincaré disk p* = k* / (1 + sqrt(1 - ||k*||^2)).
    let r2 = (kix * kix + kiy * kiy).min(1.0);
    let denom = 1.0 + (1.0 - r2).max(0.0).sqrt();
    let (px, py) = clamp_poincare_point(kix / denom, kiy / denom);

    let (pa_x, pa_y) = poincare_pts[pair.u1 as usize];
    let (pb_x, pb_y) = poincare_pts[pair.v1 as usize];
    let (pc_x, pc_y) = poincare_pts[pair.u2 as usize];
    let (pd_x, pd_y) = poincare_pts[pair.v2 as usize];

    let (t1x, t1y) = if (pb_x - px).hypot(pb_y - py) >= (pa_x - px).hypot(pa_y - py) {
        poincare_log_map(px, py, pb_x, pb_y)
    } else {
        poincare_log_map(px, py, pa_x, pa_y)
    };
    let (t2x, t2y) = if (pd_x - px).hypot(pd_y - py) >= (pc_x - px).hypot(pc_y - py) {
        poincare_log_map(px, py, pd_x, pd_y)
    } else {
        poincare_log_map(px, py, pc_x, pc_y)
    };

    let n1 = t1x.hypot(t1y);
    let n2 = t2x.hypot(t2y);
    if n1 == 0.0 || n2 == 0.0 {
        return (None, 0.0);
    }
    let cos_val = ((t1x * t2x + t1y * t2y) / (n1 * n2)).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    let cos_sq = cos_val * cos_val;
    (
        Some(if acute == 0.0 { 0.0 } else { acute }),
        if cos_sq == 0.0 { 0.0 } else { cos_sq },
    )
}

/// Counts exact hyperbolic geodesic crossings on [`ManifoldKind::HyperbolicPoincare`] and
/// [`ManifoldKind::HyperbolicLorentz`] (via `to_poincare_layout`) in $O((|E|+K)\log|E|)$ time.
///
/// Maps Poincaré geodesics bijectively to Euclidean chords in the Beltrami-Klein unit disk,
/// handling diameter geodesics, boundary clamping, and shared endpoints identically.
pub fn hyperbolic_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<usize> {
    let proj = to_poincare_and_klein(layout)?;
    let edges = extract_csr_edges(graph);
    SweepLineCrossings::count_edges(&edges, &proj.klein)
}

/// Finds all hyperbolic geodesic crossing pairs on [`ManifoldKind::HyperbolicPoincare`] and
/// [`ManifoldKind::HyperbolicLorentz`], sorted by `(edge_a, edge_b)` with conformal Poincaré
/// tangent intersection angles.
pub fn crossing_edges_hyperbolic<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<Vec<CrossingPair>> {
    let proj = to_poincare_and_klein(layout)?;
    let edges = extract_csr_edges(graph);
    let mut pairs = SweepLineCrossings::find_edge_crossings(&edges, &proj.klein)?;
    for pair in &mut pairs {
        let (angle_rad, cos_sq) = hyperbolic_crossing_angle(&proj.poincare_pts, &proj.klein, pair);
        pair.angle_rad = angle_rad;
        pair.cos_sq = cos_sq;
    }
    Ok(pairs)
}
