use std::f64::consts::PI;

use egraph_core::{CoreError, CoreResult, ManifoldKind, ManifoldLayout2d, Scalar};

/// Tracks which layout representation a stateful solver instance is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLayoutKind {
    Unset,
    Euclidean2d,
    Nd(usize),
    Manifold(ManifoldKind),
}

impl ActiveLayoutKind {
    pub(crate) fn lock_2d(&mut self, solver_name: &str) -> CoreResult<()> {
        match *self {
            Self::Unset => {
                *self = Self::Euclidean2d;
                Ok(())
            }
            Self::Euclidean2d => Ok(()),
            Self::Nd(_) => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from LayoutNd to Layout2d on stateful solver"
            ))),
            Self::Manifold(_) => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from ManifoldLayout2d to Layout2d on stateful solver"
            ))),
        }
    }

    pub(crate) fn lock_nd(
        &mut self,
        dims: usize,
        solver_name: &'static str,
        dims_context: &'static str,
    ) -> CoreResult<()> {
        match *self {
            Self::Unset => {
                *self = Self::Nd(dims);
                Ok(())
            }
            Self::Nd(prev) if prev == dims => Ok(()),
            Self::Nd(prev) => Err(CoreError::dimension_mismatch(prev, dims, dims_context)),
            Self::Euclidean2d => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from Layout2d to LayoutNd on stateful solver"
            ))),
            Self::Manifold(_) => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from ManifoldLayout2d to LayoutNd on stateful solver"
            ))),
        }
    }

    pub(crate) fn lock_manifold(
        &mut self,
        kind: ManifoldKind,
        solver_name: &str,
    ) -> CoreResult<()> {
        match *self {
            Self::Unset => {
                *self = Self::Manifold(kind);
                Ok(())
            }
            Self::Manifold(prev) if prev == kind => Ok(()),
            Self::Manifold(_) => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch ManifoldKind on stateful solver"
            ))),
            Self::Euclidean2d => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from Layout2d to ManifoldLayout2d on stateful solver"
            ))),
            Self::Nd(_) => Err(CoreError::invalid_input(format!(
                "{solver_name}::step: cannot switch from LayoutNd to ManifoldLayout2d on stateful solver"
            ))),
        }
    }
}

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
