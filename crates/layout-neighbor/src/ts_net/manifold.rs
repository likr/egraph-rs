use std::f64::consts::TAU;

use egraph_core::{
    CoreError, CoreResult, ManifoldKind, ManifoldLayout2d, RiemannianLayout, Scalar,
    cartesian_to_spherical, lorentz_log_map, poincare_log_map, spherical_log_map,
    spherical_to_cartesian, torus_delta,
};

use crate::manifold_utils::{clamp_manifold_tangent_step, guard_spherical_boundary};

use super::config::{MIN_LAMBDA_SUM, compute_stage_lr};
use super::solver::TsNetState;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

/// Lazily allocated exact geodesic repulsion scratch buffers for [`TsNetSolver`](super::TsNetSolver)
/// on [`ManifoldLayout2d`].
#[derive(Debug, Clone)]
pub(super) struct TsNetManifoldScratch {
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    ent_x: Vec<f64>,
    ent_y: Vec<f64>,
}

enum CompressionCenter {
    Origin,
    SphericalMean(Option<(f64, f64)>),
    TorusMean { mx: Option<f64>, my: Option<f64> },
    None,
}

impl CompressionCenter {
    fn compute<S: Scalar>(layout: &ManifoldLayout2d<S>) -> Self {
        let n = layout.len();
        if n == 0 {
            return Self::None;
        }
        let n_f64 = n as f64;
        match layout.kind() {
            ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => Self::Origin,
            ManifoldKind::Spherical => {
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
                    Self::SphericalMean(Some((mlon, mcolat)))
                } else {
                    Self::SphericalMean(None)
                }
            }
            ManifoldKind::Torus2d => {
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
                let mx = if cx.hypot(sx) > 1e-12 * n_f64 {
                    Some((sx.atan2(cx) / TAU).rem_euclid(1.0))
                } else {
                    None
                };
                let my = if cy.hypot(sy) > 1e-12 * n_f64 {
                    Some((sy.atan2(cy) / TAU).rem_euclid(1.0))
                } else {
                    None
                };
                Self::TorusMean { mx, my }
            }
            _ => Self::None,
        }
    }

    fn gradient_at<S: Scalar>(&self, layout: &ManifoldLayout2d<S>, idx: usize) -> (f64, f64) {
        let n_f64 = layout.len() as f64;
        let xi = layout.x()[idx].to_f64();
        let yi = layout.y()[idx].to_f64();
        let scale = layout.scale();
        match (self, layout.kind()) {
            (Self::Origin, ManifoldKind::HyperbolicPoincare) => {
                let (vx, vy) = poincare_log_map(xi, yi, 0.0, 0.0);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::Origin, ManifoldKind::HyperbolicLorentz) => {
                let (vx, vy) = lorentz_log_map(xi, yi, 0.0, 0.0);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::SphericalMean(Some((mlon, mcolat))), ManifoldKind::Spherical) => {
                let (vx, vy) = spherical_log_map(xi, yi, *mlon, *mcolat);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::TorusMean { mx, my }, ManifoldKind::Torus2d) => {
                let lx = scale * layout.aspect();
                let ly = scale;
                let dx = mx.map_or(0.0, |m| torus_delta(m, 0.0, xi, 0.0).0);
                let dy = my.map_or(0.0, |m| torus_delta(0.0, m, 0.0, yi).1);
                (-lx * dx / n_f64, -ly * dy / n_f64)
            }
            _ => (0.0, 0.0),
        }
    }
}

fn ensure_manifold_scratch<S: Scalar>(state: &mut TsNetState<S>) {
    if state.manifold_scratch.is_some() {
        return;
    }
    let n = state.num_nodes;
    let any_entropy = state
        .config
        .stages
        .iter()
        .any(|s| s.iterations > 0 && s.lambda_r > 0.0);
    state.manifold_scratch = Some(TsNetManifoldScratch {
        rep_x: vec![0.0; n],
        rep_y: vec![0.0; n],
        ent_x: vec![0.0; if any_entropy { n } else { 0 }],
        ent_y: vec![0.0; if any_entropy { n } else { 0 }],
    });
}

fn enter_current_stage_manifold<S: Scalar>(state: &mut TsNetState<S>) {
    let stage = &state.config.stages[state.stage_idx];
    let base_lr = state.config.resolve_learning_rate(state.num_nodes);
    let stage_lr = compute_stage_lr(stage, state.num_nodes, base_lr, state.config.momentum);
    if stage_lr < state.prev_stage_lr {
        let vel_scale = stage_lr / state.prev_stage_lr;
        for v in &mut state.vel_x {
            *v *= vel_scale;
        }
        for v in &mut state.vel_y {
            *v *= vel_scale;
        }
    }
    state.prev_stage_lr = stage_lr;
    state.active_stage_lr = stage_lr;
    state.stage_entered = true;
}

fn compute_attractions_manifold<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &ManifoldLayout2d<S>,
    exaggeration: f64,
) -> CoreResult<()> {
    let n = state.num_nodes;
    for i in 0..n {
        let start = state.indptr[i] as usize;
        let end = state.indptr[i + 1] as usize;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        for pos in start..end {
            let j = state.indices[pos] as usize;
            if j == i {
                continue;
            }
            let (tx_s, ty_s) = layout.tangent_delta(i, j)?;
            let dx = -tx_s.to_f64();
            let dy = -ty_s.to_f64();
            let dist_sq = dx * dx + dy * dy;
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (state.p_vals[pos] * exaggeration) * q_unnorm;
            ax += mult * dx;
            ay += mult * dy;
        }
        state.attr_x[i] = ax;
        state.attr_y[i] = ay;
    }
    Ok(())
}

fn compute_exact_repulsion_manifold<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &ManifoldLayout2d<S>,
    compute_entropy: bool,
) -> CoreResult<()> {
    let scratch = state
        .manifold_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetManifoldScratch workspace missing"))?;
    let n = state.num_nodes;
    let eps_r = state.config.epsilon_r;
    let mut total_z = 0.0_f64;

    for i in 0..n {
        let mut z_i = 0.0_f64;
        let mut rfx = 0.0_f64;
        let mut rfy = 0.0_f64;
        let mut efx = 0.0_f64;
        let mut efy = 0.0_f64;

        for j in 0..n {
            if j == i {
                continue;
            }
            let (tx_s, ty_s) = layout.tangent_delta(i, j)?;
            let dx = -tx_s.to_f64();
            let dy = -ty_s.to_f64();
            let dist_sq = dx * dx + dy * dy;
            let w = 1.0 / (1.0 + dist_sq);
            z_i += w;
            let w2 = w * w;
            rfx += w2 * dx;
            rfy += w2 * dy;
            if compute_entropy {
                let dist = dist_sq.sqrt();
                if dist > EXACT_ENTROPY_MIN_DIST {
                    let mult = 1.0 / (dist * (dist + eps_r));
                    efx += mult * dx;
                    efy += mult * dy;
                }
            }
        }

        total_z += z_i;
        scratch.rep_x[i] = rfx;
        scratch.rep_y[i] = rfy;
        if compute_entropy {
            scratch.ent_x[i] = efx;
            scratch.ent_y[i] = efy;
        }
    }

    let denom = total_z.max(MIN_TOTAL_Z);
    for i in 0..n {
        scratch.rep_x[i] /= denom;
        scratch.rep_y[i] /= denom;
    }
    Ok(())
}

pub(super) fn step_manifold<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    if state.locked_manifold.is_none() {
        state.locked_manifold = Some(layout.kind());
    }

    ensure_manifold_scratch(state);
    if !state.stage_entered {
        enter_current_stage_manifold(state);
    }

    let stage = state.config.stages[state.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    compute_attractions_manifold(state, layout, stage.exaggeration)?;
    compute_exact_repulsion_manifold(state, layout, compute_entropy)?;

    let n = state.num_nodes;
    let n_sq_f64 = (n as f64) * (n as f64);
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = state.active_stage_lr;
    let momentum = state.config.momentum;
    let use_jacobi = state.config.jacobi_precondition;
    let center = if stage.lambda_c > 0.0 {
        CompressionCenter::compute(layout)
    } else {
        CompressionCenter::None
    };

    {
        let scratch = state.manifold_scratch.as_ref().ok_or_else(|| {
            CoreError::numerical_breakdown("TsNetManifoldScratch workspace missing")
        })?;
        for i in 0..n {
            let grad_kl_x = 4.0 * (state.attr_x[i] - scratch.rep_x[i]);
            let grad_kl_y = 4.0 * (state.attr_y[i] - scratch.rep_y[i]);
            let (grad_c_x, grad_c_y) = if stage.lambda_c > 0.0 {
                center.gradient_at(layout, i)
            } else {
                (0.0, 0.0)
            };
            let (grad_r_x, grad_r_y) = if compute_entropy {
                (-scratch.ent_x[i] / n_sq_f64, -scratch.ent_y[i] / n_sq_f64)
            } else {
                (0.0, 0.0)
            };

            let raw_gx = (stage.lambda_kl * grad_kl_x
                + stage.lambda_c * grad_c_x
                + stage.lambda_r * grad_r_x)
                / lambda_sum;
            let raw_gy = (stage.lambda_kl * grad_kl_y
                + stage.lambda_c * grad_c_y
                + stage.lambda_r * grad_r_y)
                / lambda_sum;
            let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
            let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };

            let lr_i = if use_jacobi {
                stage_lr * state.inv_jacobi[i]
            } else {
                stage_lr
            };
            let raw_vx = momentum * state.vel_x[i] - lr_i * gx;
            let raw_vy = momentum * state.vel_y[i] - lr_i * gy;
            let (cvx, cvy) = clamp_manifold_tangent_step(layout, raw_vx, raw_vy);
            state.vel_x[i] = cvx;
            state.vel_y[i] = cvy;
        }
    }

    for i in 0..n {
        let vx = state.vel_x[i];
        let vy = state.vel_y[i];
        if vx != 0.0 || vy != 0.0 {
            let delta = [vx, vy];
            let mut vel = [vx, vy];
            layout.exp_map_and_transport(i, &delta, &mut vel)?;
            guard_spherical_boundary(layout, i)?;
            state.vel_x[i] = if vel[0].is_finite() { vel[0] } else { 0.0 };
            state.vel_y[i] = if vel[1].is_finite() { vel[1] } else { 0.0 };
        }
    }

    Ok(())
}
