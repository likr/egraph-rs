use egraph_core::{CoreError, CoreResult, Layout3d, LayoutStorage, Scalar};

use super::config::{MIN_LAMBDA_SUM, compute_stage_lr};
use super::repulsion::{TsNetCore, TsNetRepulsion, TsNetSpaceKind, TsNetWorkspace};

#[derive(Debug, Clone)]
pub(super) struct TsNet3dWorkspace {
    attr_x: Vec<f64>,
    attr_y: Vec<f64>,
    attr_z: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    vel_z: Vec<f64>,
    rep: Vec<f64>,
    ent: Vec<f64>,
}

impl TsNet3dWorkspace {
    pub(super) fn new(num_nodes: usize, any_entropy: bool) -> Self {
        let ent_len = if any_entropy { num_nodes * 3 } else { 0 };
        Self {
            attr_x: vec![0.0; num_nodes],
            attr_y: vec![0.0; num_nodes],
            attr_z: vec![0.0; num_nodes],
            vel_x: vec![0.0; num_nodes],
            vel_y: vec![0.0; num_nodes],
            vel_z: vec![0.0; num_nodes],
            rep: vec![0.0; num_nodes * 3],
            ent: vec![0.0; ent_len],
        }
    }
}

pub(super) fn validate_3d<S: Scalar>(core: &TsNetCore, layout: &Layout3d<S>) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TsNetSolver::validate_target",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

fn compute_attractions_3d<S: Scalar>(
    core: &TsNetCore,
    ws: &mut TsNet3dWorkspace,
    layout: &Layout3d<S>,
    exaggeration: f64,
) {
    let n = core.num_nodes;
    let xs = layout.x();
    let ys = layout.y();
    let zs = layout.z();
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let start = core.indptr[i] as usize;
        let end = core.indptr[i + 1] as usize;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;
        for pos in start..end {
            let j = core.indices[pos] as usize;
            if j == i {
                continue;
            }
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dz = zi - zs[j].to_f64();
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (core.p_vals[pos] * exaggeration) * q_unnorm;
            ax += mult * dx;
            ay += mult * dy;
            az += mult * dz;
        }
        ws.attr_x[i] = ax;
        ws.attr_y[i] = ay;
        ws.attr_z[i] = az;
    }
}

pub(super) fn step_3d<S: Scalar, B: TsNetRepulsion<Layout3d<S>>>(
    layout: &mut Layout3d<S>,
    core: &mut TsNetCore,
    workspace: &mut TsNetWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TsNetSpaceKind::Euclidean3d(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TsNet3dWorkspace variant mismatch",
        ));
    };

    let stage = core.config.stages[core.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    if !core.stage_entered {
        let base_lr = core.config.resolve_learning_rate(core.num_nodes);
        let stage_lr = compute_stage_lr(&stage, core.num_nodes, base_lr, core.config.momentum);
        if stage_lr < core.prev_stage_lr {
            let vel_scale = stage_lr / core.prev_stage_lr;
            for v in &mut ws.vel_x {
                *v *= vel_scale;
            }
            for v in &mut ws.vel_y {
                *v *= vel_scale;
            }
            for v in &mut ws.vel_z {
                *v *= vel_scale;
            }
        }
        core.prev_stage_lr = stage_lr;
        core.active_stage_lr = stage_lr;
        backend.on_stage_enter(backend_state, core.config.epsilon_r, compute_entropy)?;
        core.stage_entered = true;
    }

    compute_attractions_3d(core, ws, layout, stage.exaggeration);
    backend.compute_repulsion(
        backend_state,
        layout,
        core.config.epsilon_r,
        compute_entropy,
        &mut ws.rep,
        &mut ws.ent,
    )?;

    let n = core.num_nodes;
    let n_f64 = n as f64;
    let n_sq_f64 = n_f64 * n_f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = core.active_stage_lr;
    let momentum = core.config.momentum;
    let use_jacobi = core.config.jacobi_precondition;

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

            let grad_kl_x = 4.0 * (ws.attr_x[i] - ws.rep[3 * i]);
            let grad_kl_y = 4.0 * (ws.attr_y[i] - ws.rep[3 * i + 1]);
            let grad_kl_z = 4.0 * (ws.attr_z[i] - ws.rep[3 * i + 2]);

            let grad_c_x = xi / n_f64;
            let grad_c_y = yi / n_f64;
            let grad_c_z = zi / n_f64;

            let (grad_r_x, grad_r_y, grad_r_z) = if compute_entropy {
                (
                    -ws.ent[3 * i] / n_sq_f64,
                    -ws.ent[3 * i + 1] / n_sq_f64,
                    -ws.ent[3 * i + 2] / n_sq_f64,
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
                stage_lr * core.inv_jacobi[i]
            } else {
                stage_lr
            };
            let mut vx = momentum * ws.vel_x[i] - lr_i * gx;
            let mut vy = momentum * ws.vel_y[i] - lr_i * gy;
            let mut vz = momentum * ws.vel_z[i] - lr_i * gz;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            if !vz.is_finite() {
                vz = 0.0;
            }
            ws.vel_x[i] = vx;
            ws.vel_y[i] = vy;
            ws.vel_z[i] = vz;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let cand_z = zi + vz;
            let new_x = if cand_x.is_finite() { cand_x } else { xi };
            let new_y = if cand_y.is_finite() { cand_y } else { yi };
            let new_z = if cand_z.is_finite() { cand_z } else { zi };

            ws.attr_x[i] = new_x;
            ws.attr_y[i] = new_y;
            ws.attr_z[i] = new_z;
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
        let cx = S::from_f64(ws.attr_x[i] - shift_x);
        let cy = S::from_f64(ws.attr_y[i] - shift_y);
        let cz = S::from_f64(ws.attr_z[i] - shift_z);
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
