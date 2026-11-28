use egraph_core::{CoreError, CoreResult, Layout3d, LayoutStorage, Scalar};

use super::repulsion::{TFdpCore, TFdpRepulsion, TFdpSpaceKind, TFdpWorkspace};

#[derive(Debug, Clone)]
pub(super) struct TFdp3dWorkspace {
    pub(super) attr_x: Vec<f64>,
    attr_y: Vec<f64>,
    attr_z: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    vel_z: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    next_y: Vec<f64>,
    next_z: Vec<f64>,
    pub(super) rep: Vec<f64>,
}

impl TFdp3dWorkspace {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            attr_x: vec![0.0; num_nodes],
            attr_y: vec![0.0; num_nodes],
            attr_z: vec![0.0; num_nodes],
            vel_x: vec![0.0; num_nodes],
            vel_y: vec![0.0; num_nodes],
            vel_z: vec![0.0; num_nodes],
            next_x: vec![0.0; num_nodes],
            next_y: vec![0.0; num_nodes],
            next_z: vec![0.0; num_nodes],
            rep: vec![0.0; num_nodes * 3],
        }
    }
}

pub(super) fn validate_3d<S: Scalar>(core: &TFdpCore, layout: &Layout3d<S>) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TFdpSolver::validate_target",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

pub(super) fn step_3d<S: Scalar, B: TFdpRepulsion<Layout3d<S>>>(
    layout: &mut Layout3d<S>,
    core: &mut TFdpCore,
    workspace: &mut TFdpWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TFdpSpaceKind::Euclidean3d(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TFdp3dWorkspace variant mismatch",
        ));
    };

    let n = core.num_nodes;
    if core.iteration == 0 {
        let (xs, ys, zs) = layout.xyz_mut();
        if xs.iter().all(|&v| v == S::ZERO)
            && ys.iter().all(|&v| v == S::ZERO)
            && zs.iter().all(|&v| v == S::ZERO)
        {
            for idx in 0..n {
                xs[idx] = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
                ys[idx] = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
                zs[idx] = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let inv_alpha_sq = 1.0 / (core.config.alpha * core.config.alpha);
    {
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let zi = zs[i].to_f64();
            let start = core.indptr[i] as usize;
            let end = core.indptr[i + 1] as usize;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            let mut fz = 0.0_f64;
            for pos in start..end {
                let j = core.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dz = zi - zs[j].to_f64();
                let d2 = dx * dx + dy * dy + dz * dz;
                let mult = (core.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                fx -= mult * dx;
                fy -= mult * dy;
                fz -= mult * dz;
            }
            ws.attr_x[i] = fx;
            ws.attr_y[i] = fy;
            ws.attr_z[i] = fz;
        }
    }

    backend.compute_repulsion(backend_state, layout, core.config.gamma, &mut ws.rep)?;

    let rho = core.current_rho() * core.rep_scale;
    let step_lr = core.config.learning_rate
        * (1.0 - (core.iteration as f64) / (core.total_iterations as f64));
    let momentum = core.config.momentum;
    let max_step = 5.0 * core.config.alpha.max(1.0) * core.config.learning_rate.max(1.0);

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
            let raw_fx = ws.attr_x[i] + rho * ws.rep[3 * i];
            let raw_fy = ws.attr_y[i] + rho * ws.rep[3 * i + 1];
            let raw_fz = ws.attr_z[i] + rho * ws.rep[3 * i + 2];
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };
            let fz = if raw_fz.is_finite() { raw_fz } else { 0.0 };

            let mut vx = momentum * ws.vel_x[i] + step_lr * fx;
            let mut vy = momentum * ws.vel_y[i] + step_lr * fy;
            let mut vz = momentum * ws.vel_z[i] + step_lr * fz;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            if !vz.is_finite() {
                vz = 0.0;
            }
            let v_norm = (vx * vx + vy * vy + vz * vz).sqrt();
            if v_norm > max_step && v_norm > 0.0 {
                let scale = max_step / v_norm;
                vx *= scale;
                vy *= scale;
                vz *= scale;
            }
            ws.vel_x[i] = vx;
            ws.vel_y[i] = vy;
            ws.vel_z[i] = vz;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let cand_z = zi + vz;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            let nz_val = if cand_z.is_finite() { cand_z } else { zi };
            ws.next_x[i] = nx_val;
            ws.next_y[i] = ny_val;
            ws.next_z[i] = nz_val;
            mean_x += nx_val;
            mean_y += ny_val;
            mean_z += nz_val;
        }
    }

    let inv_n = 1.0 / (n as f64);
    mean_x *= inv_n;
    mean_y *= inv_n;
    mean_z *= inv_n;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };
    let shift_z = if mean_z.is_finite() { mean_z } else { 0.0 };

    let (xs_mut, ys_mut, zs_mut) = layout.xyz_mut();
    for i in 0..n {
        let cx = S::from_f64(ws.next_x[i] - shift_x);
        let cy = S::from_f64(ws.next_y[i] - shift_y);
        let cz = S::from_f64(ws.next_z[i] - shift_z);
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
