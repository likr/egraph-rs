use std::f64::consts::PI;

use egraph_core::{CoreResult, ManifoldKind, ManifoldLayout2d, Scalar};

/// Clamps a metric-scaled tangent displacement vector according to the manifold's geometry and
/// physical scale/aspect metric.
#[inline]
pub(crate) fn clamp_manifold_tangent_step<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    vx: f64,
    vy: f64,
) -> (f64, f64) {
    if !vx.is_finite() || !vy.is_finite() {
        return (0.0, 0.0);
    }
    let scale = layout.scale();
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => {
            let max_step = 8.0 * scale;
            let len = vx.hypot(vy);
            if !len.is_finite() || len == 0.0 {
                (0.0, 0.0)
            } else if len > max_step {
                let s = max_step / len;
                (vx * s, vy * s)
            } else {
                (vx, vy)
            }
        }
        ManifoldKind::Spherical => {
            let max_step = PI * scale;
            let len = vx.hypot(vy);
            if !len.is_finite() || len == 0.0 {
                (0.0, 0.0)
            } else if len > max_step {
                let s = max_step / len;
                (vx * s, vy * s)
            } else {
                (vx, vy)
            }
        }
        ManifoldKind::Torus2d => {
            let lx = scale * layout.aspect();
            let ly = scale;
            let norm_step = (vx / lx).hypot(vy / ly);
            if !norm_step.is_finite() || norm_step == 0.0 {
                (0.0, 0.0)
            } else if norm_step > 0.5 {
                let s = 0.5 / norm_step;
                (vx * s, vy * s)
            } else {
                (vx, vy)
            }
        }
        _ => {
            let max_step = 8.0 * scale;
            let len = vx.hypot(vy);
            if !len.is_finite() || len == 0.0 {
                (0.0, 0.0)
            } else if len > max_step {
                let s = max_step / len;
                (vx * s, vy * s)
            } else {
                (vx, vy)
            }
        }
    }
}

/// Applies a clamped tangent-space step to vertex `idx` and guards `f32` boundary rounding on
/// [`ManifoldKind::Spherical`].
#[inline]
pub(crate) fn apply_clamped_manifold_step<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    idx: usize,
    vx: f64,
    vy: f64,
) -> CoreResult<()> {
    let (cvx, cvy) = clamp_manifold_tangent_step(layout, vx, vy);
    layout.apply_tangent_step(idx, S::from_f64(cvx), S::from_f64(cvy))?;
    // Guard f32 rounding at Spherical [0, 2*pi) x [0, pi] boundary.
    if layout.kind() == ManifoldKind::Spherical {
        let px = layout.x()[idx];
        let py = layout.y()[idx];
        if !px.is_finite() || !py.is_finite() {
            layout.set_point(idx, S::ZERO, S::ZERO)?;
        } else if px >= S::TAU || px < S::ZERO || py < S::ZERO || py > S::PI {
            let safe_lon = if px >= S::TAU || px < S::ZERO {
                S::ZERO
            } else {
                px
            };
            let safe_colat = py.clamp(S::ZERO, S::PI);
            layout.set_point(idx, safe_lon, safe_colat)?;
        }
    }
    Ok(())
}
