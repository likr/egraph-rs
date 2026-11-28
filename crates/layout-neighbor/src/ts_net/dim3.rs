use egraph_core::{CoreError, CoreResult, Layout3d, Scalar};

use super::config::{MIN_LAMBDA_SUM, TsNetBackend, compute_stage_lr};
use super::solver::TsNetState;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

#[derive(Debug, Clone)]
pub(super) struct TsNet3dScratch<S: Scalar> {
    attr_x: Vec<f64>,
    attr_y: Vec<f64>,
    attr_z: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    vel_z: Vec<f64>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    rep_z: Vec<S>,
    ent_x: Vec<S>,
    ent_y: Vec<S>,
    ent_z: Vec<S>,
}

fn ensure_3d_scratch<S: Scalar>(state: &mut TsNetState<S>) {
    if state.dim3_scratch.is_some() {
        return;
    }
    let n = state.num_nodes;
    let any_entropy = state
        .config
        .stages
        .iter()
        .any(|s| s.iterations > 0 && s.lambda_r > 0.0);
    let ent_len = if any_entropy { n } else { 0 };
    state.dim3_scratch = Some(TsNet3dScratch {
        attr_x: vec![0.0; n],
        attr_y: vec![0.0; n],
        attr_z: vec![0.0; n],
        vel_x: vec![0.0; n],
        vel_y: vec![0.0; n],
        vel_z: vec![0.0; n],
        rep_x: vec![S::ZERO; n],
        rep_y: vec![S::ZERO; n],
        rep_z: vec![S::ZERO; n],
        ent_x: vec![S::ZERO; ent_len],
        ent_y: vec![S::ZERO; ent_len],
        ent_z: vec![S::ZERO; ent_len],
    });
}

fn enter_current_stage_3d<S: Scalar>(state: &mut TsNetState<S>) {
    let stage = &state.config.stages[state.stage_idx];
    let base_lr = state.config.resolve_learning_rate(state.num_nodes);
    let stage_lr = compute_stage_lr(stage, state.num_nodes, base_lr, state.config.momentum);
    if stage_lr < state.prev_stage_lr {
        let vel_scale = stage_lr / state.prev_stage_lr;
        if let Some(scratch) = &mut state.dim3_scratch {
            for v in &mut scratch.vel_x {
                *v *= vel_scale;
            }
            for v in &mut scratch.vel_y {
                *v *= vel_scale;
            }
            for v in &mut scratch.vel_z {
                *v *= vel_scale;
            }
        }
    }
    state.prev_stage_lr = stage_lr;
    state.active_stage_lr = stage_lr;
    state.stage_entered = true;
}

fn compute_attractions_3d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &Layout3d<S>,
    exaggeration: f64,
) {
    let Some(scratch) = &mut state.dim3_scratch else {
        return;
    };
    let n = state.num_nodes;
    let xs = layout.x();
    let ys = layout.y();
    let zs = layout.z();
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let start = state.indptr[i] as usize;
        let end = state.indptr[i + 1] as usize;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;
        for pos in start..end {
            let j = state.indices[pos] as usize;
            if j == i {
                continue;
            }
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dz = zi - zs[j].to_f64();
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (state.p_vals[pos] * exaggeration) * q_unnorm;
            ax += mult * dx;
            ay += mult * dy;
            az += mult * dz;
        }
        scratch.attr_x[i] = ax;
        scratch.attr_y[i] = ay;
        scratch.attr_z[i] = az;
    }
}

fn compute_exact_repulsion_3d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &Layout3d<S>,
    compute_entropy: bool,
) {
    let Some(scratch) = &mut state.dim3_scratch else {
        return;
    };
    let n = state.num_nodes;
    let eps_r = state.config.epsilon_r;
    let xs = layout.x();
    let ys = layout.y();
    let zs = layout.z();
    let mut total_z = 0.0_f64;

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let mut z_i = 0.0_f64;
        let mut rfx = 0.0_f64;
        let mut rfy = 0.0_f64;
        let mut rfz = 0.0_f64;
        let mut efx = 0.0_f64;
        let mut efy = 0.0_f64;
        let mut efz = 0.0_f64;

        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dz = zi - zs[j].to_f64();
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let w = 1.0 / (1.0 + dist_sq);
            z_i += w;
            let w2 = w * w;
            rfx += w2 * dx;
            rfy += w2 * dy;
            rfz += w2 * dz;
            if compute_entropy {
                let dist = dist_sq.sqrt();
                if dist > EXACT_ENTROPY_MIN_DIST {
                    let mult = 1.0 / (dist * (dist + eps_r));
                    efx += mult * dx;
                    efy += mult * dy;
                    efz += mult * dz;
                }
            }
        }
        total_z += z_i;
        scratch.rep_x[i] = S::from_f64(rfx);
        scratch.rep_y[i] = S::from_f64(rfy);
        scratch.rep_z[i] = S::from_f64(rfz);
        if compute_entropy {
            scratch.ent_x[i] = S::from_f64(efx);
            scratch.ent_y[i] = S::from_f64(efy);
            scratch.ent_z[i] = S::from_f64(efz);
        }
    }

    let denom = total_z.max(MIN_TOTAL_Z);
    for i in 0..n {
        scratch.rep_x[i] = S::from_f64(scratch.rep_x[i].to_f64() / denom);
        scratch.rep_y[i] = S::from_f64(scratch.rep_y[i].to_f64() / denom);
        scratch.rep_z[i] = S::from_f64(scratch.rep_z[i].to_f64() / denom);
    }
}

fn compute_barnes_hut_repulsion_3d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &Layout3d<S>,
    compute_entropy: bool,
) -> CoreResult<()> {
    let scratch = state
        .dim3_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNet3dScratch workspace missing"))?;
    state.octree.build_3d(layout)?;
    let theta_s = S::from_f64(state.config.theta);
    state.octree.compute_kl_repulsion(
        theta_s,
        &mut scratch.rep_x,
        &mut scratch.rep_y,
        &mut scratch.rep_z,
    )?;
    if compute_entropy {
        let eps_s = S::from_f64(state.config.epsilon_r).max(S::MIN_POSITIVE);
        state.octree.compute_entropy_repulsion(
            theta_s,
            eps_s,
            &mut scratch.ent_x,
            &mut scratch.ent_y,
            &mut scratch.ent_z,
        )?;
    }
    Ok(())
}

pub(super) fn step_3d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &mut Layout3d<S>,
) -> CoreResult<()> {
    ensure_3d_scratch(state);
    if !state.stage_entered {
        enter_current_stage_3d(state);
    }

    let stage = state.config.stages[state.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    compute_attractions_3d(state, layout, stage.exaggeration);

    match state.resolved_backend {
        TsNetBackend::Exact => compute_exact_repulsion_3d(state, layout, compute_entropy),
        TsNetBackend::BarnesHut => {
            compute_barnes_hut_repulsion_3d(state, layout, compute_entropy)?;
        }
        other => {
            return Err(CoreError::unsupported(format!(
                "TsNetBackend::{other:?} is not supported on Layout3d"
            )));
        }
    }

    let n = state.num_nodes;
    let n_f64 = n as f64;
    let n_sq_f64 = n_f64 * n_f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = state.active_stage_lr;
    let momentum = state.config.momentum;
    let use_jacobi = state.config.jacobi_precondition;

    let scratch = state
        .dim3_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNet3dScratch workspace missing"))?;

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut mean_z = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let zi = zs[i].to_f64();

            let grad_kl_x = 4.0 * (scratch.attr_x[i] - scratch.rep_x[i].to_f64());
            let grad_kl_y = 4.0 * (scratch.attr_y[i] - scratch.rep_y[i].to_f64());
            let grad_kl_z = 4.0 * (scratch.attr_z[i] - scratch.rep_z[i].to_f64());

            let grad_c_x = xi / n_f64;
            let grad_c_y = yi / n_f64;
            let grad_c_z = zi / n_f64;

            let (grad_r_x, grad_r_y, grad_r_z) = if compute_entropy {
                (
                    -scratch.ent_x[i].to_f64() / n_sq_f64,
                    -scratch.ent_y[i].to_f64() / n_sq_f64,
                    -scratch.ent_z[i].to_f64() / n_sq_f64,
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            let raw_gx = (stage.lambda_kl * grad_kl_x
                + stage.lambda_c * grad_c_x
                + stage.lambda_r * grad_r_x)
                / lambda_sum;
            let raw_gy = (stage.lambda_kl * grad_kl_y
                + stage.lambda_c * grad_c_y
                + stage.lambda_r * grad_r_y)
                / lambda_sum;
            let raw_gz = (stage.lambda_kl * grad_kl_z
                + stage.lambda_c * grad_c_z
                + stage.lambda_r * grad_r_z)
                / lambda_sum;
            let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
            let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };
            let gz = if raw_gz.is_finite() { raw_gz } else { 0.0 };

            let lr_i = if use_jacobi {
                stage_lr * state.inv_jacobi[i]
            } else {
                stage_lr
            };
            let mut vx = momentum * scratch.vel_x[i] - lr_i * gx;
            let mut vy = momentum * scratch.vel_y[i] - lr_i * gy;
            let mut vz = momentum * scratch.vel_z[i] - lr_i * gz;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            if !vz.is_finite() {
                vz = 0.0;
            }
            scratch.vel_x[i] = vx;
            scratch.vel_y[i] = vy;
            scratch.vel_z[i] = vz;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let cand_z = zi + vz;
            let new_x = if cand_x.is_finite() { cand_x } else { xi };
            let new_y = if cand_y.is_finite() { cand_y } else { yi };
            let new_z = if cand_z.is_finite() { cand_z } else { zi };

            scratch.attr_x[i] = new_x;
            scratch.attr_y[i] = new_y;
            scratch.attr_z[i] = new_z;
            mean_x += new_x;
            mean_y += new_y;
            mean_z += new_z;
        }
    }

    mean_x /= n_f64;
    mean_y /= n_f64;
    mean_z /= n_f64;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };
    let shift_z = if mean_z.is_finite() { mean_z } else { 0.0 };

    let (xs_mut, ys_mut, zs_mut) = layout.xyz_mut();
    for i in 0..n {
        let cx = S::from_f64(scratch.attr_x[i] - shift_x);
        let cy = S::from_f64(scratch.attr_y[i] - shift_y);
        let cz = S::from_f64(scratch.attr_z[i] - shift_z);
        if cx.is_finite() {
            xs_mut[i] = cx;
        }
        if cy.is_finite() {
            ys_mut[i] = cy;
        }
        if cz.is_finite() {
            zs_mut[i] = cz;
        }
    }
    Ok(())
}
