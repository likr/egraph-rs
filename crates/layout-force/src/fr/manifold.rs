use std::f64::consts::TAU;

use egraph_core::{
    CoreResult, ManifoldKind, ManifoldLayout2d, Scalar, cartesian_to_spherical, lorentz_log_map,
    poincare_log_map, spherical_log_map, spherical_to_cartesian, torus_delta,
};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::solver::{COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST, FrState};

fn scatter_collapsed_manifold_if_needed<S: Scalar>(
    state: &mut FrState,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.iter != 0 || n <= 1 {
        return Ok(());
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
        let xf = x.to_f64();
        let yf = y.to_f64();
        min_x = min_x.min(xf);
        max_x = max_x.max(xf);
        min_y = min_y.min(yf);
        max_y = max_y.max(yf);
    }
    if (max_x - min_x) >= 1e-12 || (max_y - min_y) >= 1e-12 {
        return Ok(());
    }

    let kind = layout.kind();
    let spread = 0.5 * state.k.min(layout.scale());
    match kind {
        ManifoldKind::Spherical => {
            for i in 0..n {
                let lon = TAU * state.rng.next_f64();
                let u = (2.0 * state.rng.next_f64() - 1.0).clamp(-1.0, 1.0);
                layout.set_point(i, S::from_f64(lon), S::from_f64(u.acos()))?;
            }
        }
        ManifoldKind::Torus2d => {
            for i in 0..n {
                layout.set_point(
                    i,
                    S::from_f64(state.rng.next_f64()),
                    S::from_f64(state.rng.next_f64()),
                )?;
            }
        }
        _ => {
            for i in 0..n {
                let vx = spread * (state.rng.next_f64() - 0.5);
                let vy = spread * (state.rng.next_f64() - 0.5);
                apply_clamped_manifold_step(layout, i, vx, vy)?;
            }
        }
    }
    Ok(())
}

fn apply_manifold_gravity<S: Scalar>(state: &mut FrState, layout: &ManifoldLayout2d<S>) {
    let g = state.config.gravity;
    if g <= 0.0 {
        return;
    }
    let n = state.num_nodes;
    let scale = layout.scale();
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare => {
            for i in 0..n {
                let (vx, vy) =
                    poincare_log_map(layout.x()[i].to_f64(), layout.y()[i].to_f64(), 0.0, 0.0);
                state.disp_x[i] += g * scale * vx;
                state.disp_y[i] += g * scale * vy;
            }
        }
        ManifoldKind::HyperbolicLorentz => {
            for i in 0..n {
                let (vx, vy) =
                    lorentz_log_map(layout.x()[i].to_f64(), layout.y()[i].to_f64(), 0.0, 0.0);
                state.disp_x[i] += g * scale * vx;
                state.disp_y[i] += g * scale * vy;
            }
        }
        ManifoldKind::Spherical => {
            let n_f64 = n as f64;
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut sz = 0.0_f64;
            for (&lon, &colat) in layout.x().iter().zip(layout.y().iter()) {
                let p = spherical_to_cartesian(lon.to_f64(), colat.to_f64());
                sx += p[0];
                sy += p[1];
                sz += p[2];
            }
            let norm = (sx * sx + sy * sy + sz * sz).sqrt();
            if norm > 1e-12 * n_f64 && norm.is_finite() {
                let (mlon, mcolat) = cartesian_to_spherical(sx, sy, sz);
                for i in 0..n {
                    let (vx, vy) = spherical_log_map(
                        layout.x()[i].to_f64(),
                        layout.y()[i].to_f64(),
                        mlon,
                        mcolat,
                    );
                    state.disp_x[i] += g * scale * vx;
                    state.disp_y[i] += g * scale * vy;
                }
            }
        }
        ManifoldKind::Torus2d => {
            let n_f64 = n as f64;
            let lx = scale * layout.aspect();
            let ly = scale;
            let mut cx = 0.0_f64;
            let mut sx = 0.0_f64;
            let mut cy = 0.0_f64;
            let mut sy = 0.0_f64;
            for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
                let (sin_x, cos_x) = (TAU * x.to_f64()).sin_cos();
                let (sin_y, cos_y) = (TAU * y.to_f64()).sin_cos();
                cx += cos_x;
                sx += sin_x;
                cy += cos_y;
                sy += sin_y;
            }
            let mx = (cx.hypot(sx) > 1e-12 * n_f64).then(|| (sx.atan2(cx) / TAU).rem_euclid(1.0));
            let my = (cy.hypot(sy) > 1e-12 * n_f64).then(|| (sy.atan2(cy) / TAU).rem_euclid(1.0));
            for i in 0..n {
                let dx = mx.map_or(0.0, |m| torus_delta(m, 0.0, layout.x()[i].to_f64(), 0.0).0);
                let dy = my.map_or(0.0, |m| torus_delta(0.0, m, 0.0, layout.y()[i].to_f64()).1);
                state.disp_x[i] += g * lx * dx;
                state.disp_y[i] += g * ly * dy;
            }
        }
        _ => {}
    }
}

pub(super) fn step_manifold<S: Scalar>(
    state: &mut FrState,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    if state.locked_manifold.is_none() {
        state.locked_manifold = Some(layout.kind());
    }

    let n = state.num_nodes;
    scatter_collapsed_manifold_if_needed(state, layout)?;

    state.disp_x.fill(0.0);
    state.disp_y.fill(0.0);
    let k_sq = state.k * state.k;

    // 1. Exact geodesic repulsion: f_r(d) = k^2 / d in tangent space.
    for i in 0..n {
        for j in (i + 1)..n {
            let (tix_s, tiy_s) = layout.tangent_delta(i, j)?;
            let (tjx_s, tjy_s) = layout.tangent_delta(j, i)?;
            let tx = tix_s.to_f64();
            let ty = tiy_s.to_f64();
            let d_sq = tx * tx + ty * ty;
            if d_sq > COINCIDENT_DIST_SQ {
                let mult = k_sq / d_sq;
                state.disp_x[i] -= mult * tx;
                state.disp_y[i] -= mult * ty;
                state.disp_x[j] -= mult * tjx_s.to_f64();
                state.disp_y[j] -= mult * tjy_s.to_f64();
            } else {
                let (ux, uy) = state.next_unit_vector();
                let mag = k_sq / COINCIDENT_PERTURB_DIST;
                state.disp_x[i] += mag * ux;
                state.disp_y[i] += mag * uy;
                state.disp_x[j] -= mag * ux;
                state.disp_y[j] -= mag * uy;
            }
        }
    }

    // 2. Geodesic edge attraction: f_a(d) = d^2 / k in tangent space.
    let inv_k = 1.0 / state.k;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let (tux_s, tuy_s) = layout.tangent_delta(u, v)?;
        let (tvx_s, tvy_s) = layout.tangent_delta(v, u)?;
        let ux = tux_s.to_f64();
        let uy = tuy_s.to_f64();
        let d = ux.hypot(uy);
        let factor = w * d * inv_k;
        state.disp_x[u] += factor * ux;
        state.disp_y[u] += factor * uy;
        state.disp_x[v] += factor * tvx_s.to_f64();
        state.disp_y[v] += factor * tvy_s.to_f64();
    }

    // 3. Optional manifold gravity.
    apply_manifold_gravity(state, layout);

    // 4. Temperature cooling, tangent clamping, and exponential map update.
    let temp = state.current_temperature();
    for i in 0..n {
        let dx = state.disp_x[i];
        let dy = state.disp_y[i];
        let len = dx.hypot(dy);
        if len > 1e-24 && temp > 0.0 {
            let s = len.min(temp) / len;
            apply_clamped_manifold_step(layout, i, dx * s, dy * s)?;
        }
    }

    Ok(())
}
