use egraph_core::{CoreError, CoreResult, Layout3d, Scalar};

use super::config::FieldRepulsionBackend;
use super::t_fdp::TFdpState;

pub(super) fn step_3d<S: Scalar>(
    state: &mut TFdpState<S>,
    layout: &mut Layout3d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.iteration == 0 {
        let (xs, ys, zs) = layout.xyz_mut();
        if xs.iter().all(|&v| v == S::ZERO)
            && ys.iter().all(|&v| v == S::ZERO)
            && zs.iter().all(|&v| v == S::ZERO)
        {
            for idx in 0..n {
                xs[idx] = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
                ys[idx] = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
                zs[idx] = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let inv_alpha_sq = 1.0 / (state.config.alpha * state.config.alpha);
    {
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let zi = zs[i].to_f64();
            let start = state.indptr[i] as usize;
            let end = state.indptr[i + 1] as usize;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            let mut fz = 0.0_f64;
            for pos in start..end {
                let j = state.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dz = zi - zs[j].to_f64();
                let d2 = dx * dx + dy * dy + dz * dz;
                let mult = (state.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                fx -= mult * dx;
                fy -= mult * dy;
                fz -= mult * dz;
            }
            state.attr_x[i] = fx;
            state.attr_y[i] = fy;
            state.attr_z[i] = fz;
        }
    }

    let gamma = state.config.gamma;
    match state.config.backend {
        FieldRepulsionBackend::BarnesHut => {
            state.octree.build_3d(layout)?;
            let theta_s = S::from_f64(state.config.theta);
            if (gamma - 1.0).abs() <= 1e-12 {
                state.octree.compute_coulomb_repulsion(
                    theta_s,
                    S::ONE,
                    &mut state.rep_x,
                    &mut state.rep_y,
                    &mut state.rep_z,
                )?;
            } else if (gamma - 2.0).abs() <= 1e-12 {
                let total_z = state.octree.compute_kl_repulsion(
                    theta_s,
                    &mut state.rep_x,
                    &mut state.rep_y,
                    &mut state.rep_z,
                )?;
                let z_f64 = total_z.to_f64();
                for i in 0..n {
                    state.rep_x[i] = S::from_f64(state.rep_x[i].to_f64() * z_f64);
                    state.rep_y[i] = S::from_f64(state.rep_y[i].to_f64() * z_f64);
                    state.rep_z[i] = S::from_f64(state.rep_z[i].to_f64() * z_f64);
                }
            } else {
                let q = 2.0 * gamma - 2.0;
                state.octree.compute_maxent_repulsion(
                    theta_s,
                    q,
                    S::ONE,
                    &mut state.rep_x,
                    &mut state.rep_y,
                    &mut state.rep_z,
                )?;
            }
        }
        FieldRepulsionBackend::Exact => {
            let xs = layout.x();
            let ys = layout.y();
            let zs = layout.z();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let zi = zs[i].to_f64();
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                let mut fz = 0.0_f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let dz = zi - zs[j].to_f64();
                    let d2 = dx * dx + dy * dy + dz * dz;
                    let k = 1.0 / (1.0 + d2).powf(gamma);
                    fx += k * dx;
                    fy += k * dy;
                    fz += k * dz;
                }
                state.rep_x[i] = S::from_f64(fx);
                state.rep_y[i] = S::from_f64(fy);
                state.rep_z[i] = S::from_f64(fz);
            }
        }
        FieldRepulsionBackend::IbFft => {
            return Err(CoreError::unsupported(
                "FieldRepulsionBackend::IbFft is not supported on Layout3d",
            ));
        }
    }

    let rho = state.current_rho() * state.rep_scale;
    let step_lr = state.config.learning_rate
        * (1.0 - (state.iteration as f64) / (state.total_iterations as f64));
    let momentum = state.config.momentum;
    let max_step = 5.0 * state.config.alpha.max(1.0) * state.config.learning_rate.max(1.0);

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
            let raw_fx = state.attr_x[i] + rho * state.rep_x[i].to_f64();
            let raw_fy = state.attr_y[i] + rho * state.rep_y[i].to_f64();
            let raw_fz = state.attr_z[i] + rho * state.rep_z[i].to_f64();
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };
            let fz = if raw_fz.is_finite() { raw_fz } else { 0.0 };

            let mut vx = momentum * state.vel_x[i] + step_lr * fx;
            let mut vy = momentum * state.vel_y[i] + step_lr * fy;
            let mut vz = momentum * state.vel_z[i] + step_lr * fz;
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
            state.vel_x[i] = vx;
            state.vel_y[i] = vy;
            state.vel_z[i] = vz;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let cand_z = zi + vz;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            let nz_val = if cand_z.is_finite() { cand_z } else { zi };
            state.next_x[i] = nx_val;
            state.next_y[i] = ny_val;
            state.next_z[i] = nz_val;
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
        let cx = S::from_f64(state.next_x[i] - shift_x);
        let cy = S::from_f64(state.next_y[i] - shift_y);
        let cz = S::from_f64(state.next_z[i] - shift_z);
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
