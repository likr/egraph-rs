use std::f64::consts::PI;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
};

use super::pairs::{HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair, PairSpec};

#[inline]
pub(super) fn fast_hypot<S: Scalar>(dx: S, dy: S) -> S {
    let d2 = dx * dx + dy * dy;
    if d2 > S::ZERO && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    }
}

#[inline]
pub(super) fn apply_euclidean_sym_pair<S: Scalar>(
    xs: &mut [S],
    ys: &mut [S],
    eta: S,
    p: PackedSymPair<S>,
) {
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let mu = if p.w.is_finite() && p.w > S::ZERO {
        (eta * p.w).min(S::ONE)
    } else {
        S::ZERO
    };
    if !p.d.is_finite() || p.d <= S::ZERO || !mu.is_finite() || mu <= S::ZERO {
        return;
    }
    let dx = xs[i] - xs[j];
    let dy = ys[i] - ys[j];
    let d2 = dx * dx + dy * dy;
    if is_hinge && d2.is_finite() && d2 >= p.d * p.d {
        return;
    }
    let norm = if d2 > S::ZERO && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    };
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    if is_hinge && norm >= p.d {
        return;
    }
    let coeff = S::from_f64(0.5) * (norm - p.d) / norm * mu;
    let step_x = dx * coeff;
    let step_y = dy * coeff;
    xs[i] -= step_x;
    ys[i] -= step_y;
    xs[j] += step_x;
    ys[j] += step_y;
}

#[inline]
pub(super) fn apply_euclidean_gen_pair<S: Scalar>(
    xs: &mut [S],
    ys: &mut [S],
    eta: S,
    p: &PackedGenPair<S>,
) {
    if !p.dij.is_finite() || !p.dji.is_finite() || p.dij <= S::ZERO || p.dji <= S::ZERO {
        return;
    }
    let mu_i = if p.wij.is_finite() && p.wij > S::ZERO {
        (eta * p.wij).min(S::ONE)
    } else {
        S::ZERO
    };
    let mu_j = if p.wji.is_finite() && p.wji > S::ZERO {
        (eta * p.wji).min(S::ONE)
    } else {
        S::ZERO
    };
    if mu_i <= S::ZERO && mu_j <= S::ZERO {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let dx = xs[i] - xs[j];
    let dy = ys[i] - ys[j];
    let d2 = dx * dx + dy * dy;
    if is_hinge && d2.is_finite() && d2 >= p.dij * p.dij && d2 >= p.dji * p.dji {
        return;
    }
    let norm = fast_hypot(dx, dy);
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    let update_i = mu_i > S::ZERO && (!is_hinge || norm < p.dij);
    let update_j = mu_j > S::ZERO && (!is_hinge || norm < p.dji);
    if !update_i && !update_j {
        return;
    }
    let half = S::from_f64(0.5);
    if update_i {
        let coeff_i = half * (norm - p.dij) / norm * mu_i;
        xs[i] -= dx * coeff_i;
        ys[i] -= dy * coeff_i;
    }
    if update_j {
        let coeff_j = half * (norm - p.dji) / norm * mu_j;
        xs[j] += dx * coeff_j;
        ys[j] += dy * coeff_j;
    }
}

#[inline]
fn clamp_manifold_tangent_step<S: Scalar>(kind: ManifoldKind, vx: S, vy: S) -> CoreResult<(S, S)> {
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();
    if !vxf.is_finite() || !vyf.is_finite() {
        return Ok((S::ZERO, S::ZERO));
    }
    let len = vxf.hypot(vyf);
    if !len.is_finite() || len == 0.0 {
        return Ok((S::ZERO, S::ZERO));
    }
    let max_step = match kind {
        ManifoldKind::HyperbolicPoincare => 8.0,
        ManifoldKind::Spherical => PI,
        ManifoldKind::Torus2d => 0.5,
        _ => return Err(CoreError::unsupported("unsupported manifold kind")),
    };
    if len > max_step {
        let scale = max_step / len;
        Ok((S::from_f64(vxf * scale), S::from_f64(vyf * scale)))
    } else {
        Ok((vx, vy))
    }
}

#[inline]
fn apply_clamped_manifold_vertex_step<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    idx: usize,
    vx: S,
    vy: S,
) -> CoreResult<()> {
    let kind = layout.kind();
    let (cvx, cvy) = clamp_manifold_tangent_step(kind, vx, vy)?;
    layout.apply_tangent_step(idx, cvx, cvy)?;
    // Guard f32 rounding at Spherical [0, 2*pi) x [0, pi] boundary.
    if kind == ManifoldKind::Spherical {
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

pub(super) fn apply_manifold_pair<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    eta: S,
    p: PairSpec<S>,
) -> CoreResult<()> {
    if !p.dij.is_finite() || !p.dji.is_finite() || p.dij <= S::ZERO || p.dji <= S::ZERO {
        return Ok(());
    }
    let mu_i = if p.wij.is_finite() && p.wij > S::ZERO {
        (eta * p.wij).min(S::ONE)
    } else {
        S::ZERO
    };
    let mu_j = if p.wji.is_finite() && p.wji > S::ZERO {
        (eta * p.wji).min(S::ONE)
    } else {
        S::ZERO
    };
    if mu_i <= S::ZERO && mu_j <= S::ZERO {
        return Ok(());
    }
    let (tix, tiy) = layout.tangent_delta(p.i, p.j)?;
    let norm = tix.hypot(tiy);
    if !norm.is_finite() || norm <= S::ZERO {
        return Ok(());
    }
    let update_i = mu_i > S::ZERO && (!p.is_hinge || norm < p.dij);
    let update_j = mu_j > S::ZERO && (!p.is_hinge || norm < p.dji);
    if !update_i && !update_j {
        return Ok(());
    }
    // Evaluate vertex j's tangent vector before mutating vertex i so both endpoints
    // compute gradients from the same pre-step configuration.
    let tj = if update_j {
        Some(layout.tangent_delta(p.j, p.i)?)
    } else {
        None
    };
    let half = S::from_f64(0.5);
    if update_i {
        let coeff_i = half * (norm - p.dij) / norm * mu_i;
        apply_clamped_manifold_vertex_step(layout, p.i, tix * coeff_i, tiy * coeff_i)?;
    }
    if let Some((tjx, tjy)) = tj {
        let norm_j = tjx.hypot(tjy);
        if norm_j.is_finite() && norm_j > S::ZERO {
            let coeff_j = half * (norm_j - p.dji) / norm_j * mu_j;
            apply_clamped_manifold_vertex_step(layout, p.j, tjx * coeff_j, tjy * coeff_j)?;
        }
    }
    Ok(())
}

pub(super) fn scatter_collapsed_layout_if_needed<S: Scalar>(
    layout: &mut Layout2d<S>,
    span: f64,
    rng: &mut DeterministicRng,
) {
    let n = layout.len();
    if n <= 1 {
        return;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            xs_mut[i] = S::from_f64(span * (rng.next_f64() - 0.5));
            ys_mut[i] = S::from_f64(span * (rng.next_f64() - 0.5));
        }
    }
}
