use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout3d, Scalar};

use super::solver::{COULOMB_EPS_SQ, LinLogState, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

pub(super) fn step_3d<S: Scalar>(
    state: &mut LinLogState,
    layout: &mut Layout3d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    {
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            let z = zs[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            state.z_buf[i] = z;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        if state.iter == 0
            && (max_x - min_x) < 1e-12
            && (max_y - min_y) < 1e-12
            && (max_z - min_z) < 1e-12
        {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for i in 0..n {
                state.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                state.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                state.z_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
            }
        }
    }

    let r_exp = state.config.repulsion_exponent;
    let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;

    if state.config.use_barnes_hut {
        state
            .octree
            .build_weighted_xyz(&state.x_buf, &state.y_buf, &state.z_buf, &state.masses)?;
        state.octree.compute_power_repulsion_flat(
            state.config.theta,
            r_exp,
            COULOMB_EPS_SQ,
            &mut state.rep_3d_flat,
        )?;
        for i in 0..n {
            let mi = state.masses[i];
            let base = i * 3;
            state.fx[i] = mi * state.rep_3d_flat[base];
            state.fy[i] = mi * state.rep_3d_flat[base + 1];
            state.fz[i] = mi * state.rep_3d_flat[base + 2];
        }
    } else {
        state.fx.fill(0.0);
        state.fy.fill(0.0);
        state.fz.fill(0.0);
        let pow_exp = 0.5 * (r_exp - 1.0);
        for i in 0..n {
            let xi = state.x_buf[i];
            let yi = state.y_buf[i];
            let zi = state.z_buf[i];
            let mi = state.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            let mut acc_z = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - state.x_buf[j];
                let dy = yi - state.y_buf[j];
                let dz = zi - state.z_buf[j];
                let d_sq_reg = dx * dx + dy * dy + dz * dz + COULOMB_EPS_SQ;
                let mult = if is_coulomb {
                    (mi * state.masses[j]) / d_sq_reg
                } else {
                    (mi * state.masses[j]) * d_sq_reg.powf(pow_exp)
                };
                let fx_ij = mult * dx;
                let fy_ij = mult * dy;
                let fz_ij = mult * dz;
                acc_x += fx_ij;
                acc_y += fy_ij;
                acc_z += fz_ij;
                state.fx[j] -= fx_ij;
                state.fy[j] -= fy_ij;
                state.fz[j] -= fz_ij;
            }
            state.fx[i] += acc_x;
            state.fy[i] += acc_y;
            state.fz[i] += acc_z;
        }
    }

    let a_exp = state.config.attraction_exponent;
    let a_is_zero = a_exp.abs() <= 1e-12;
    let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
    let a_hess_scale = a_exp.abs().max(1.0);
    state.att_hess.fill(0.0);

    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = state.x_buf[v] - state.x_buf[u];
        let dy = state.y_buf[v] - state.y_buf[u];
        let dz = state.z_buf[v] - state.z_buf[u];
        let d_sq = dx * dx + dy * dy + dz * dz;
        let (factor, hess) = if a_is_zero {
            if d_sq > 1e-24 {
                let d = d_sq.sqrt();
                (w / d, w / d)
            } else {
                (0.0, 0.0)
            }
        } else if a_is_one {
            (w, w)
        } else if a_exp > 0.0 && d_sq <= 1e-24 {
            (0.0, 0.0)
        } else {
            let d_reg = (d_sq + 1e-24).sqrt();
            let f = w * d_reg.powf(a_exp - 1.0);
            (f, a_hess_scale * f)
        };

        let ax = factor * dx;
        let ay = factor * dy;
        let az = factor * dz;
        state.fx[u] += ax;
        state.fy[u] += ay;
        state.fz[u] += az;
        state.fx[v] -= ax;
        state.fy[v] -= ay;
        state.fz[v] -= az;
        state.att_hess[u] += hess;
        state.att_hess[v] += hess;
    }

    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut bx = 0.0_f64;
        let mut by = 0.0_f64;
        let mut bz = 0.0_f64;
        for i in 0..n {
            let m = state.masses[i];
            bx += m * state.x_buf[i];
            by += m * state.y_buf[i];
            bz += m * state.z_buf[i];
        }
        bx /= state.total_mass;
        by /= state.total_mass;
        bz /= state.total_mass;

        let g = state.config.gravity;
        for i in 0..n {
            let dx = bx - state.x_buf[i];
            let dy = by - state.y_buf[i];
            let dz = bz - state.z_buf[i];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 1e-12 {
                let factor = if a_is_zero {
                    (g * state.masses[i]) / d
                } else if a_is_one {
                    g * state.masses[i]
                } else {
                    g * state.masses[i] * d.powf(a_exp - 1.0)
                };
                state.fx[i] += factor * dx;
                state.fy[i] += factor * dy;
                state.fz[i] += factor * dz;
                state.att_hess[i] += a_hess_scale * factor;
            }
        }
    }

    let mut total_swing = 0.0_f64;
    let mut total_traction = 0.0_f64;
    for i in 0..n {
        let fxi = if state.fx[i].is_finite() {
            state.fx[i]
        } else {
            0.0
        };
        let fyi = if state.fy[i].is_finite() {
            state.fy[i]
        } else {
            0.0
        };
        let fzi = if state.fz[i].is_finite() {
            state.fz[i]
        } else {
            0.0
        };
        state.fx[i] = fxi;
        state.fy[i] = fyi;
        state.fz[i] = fzi;

        let dx_s = fxi - state.prev_fx[i];
        let dy_s = fyi - state.prev_fy[i];
        let dz_s = fzi - state.prev_fz[i];
        let swg = (dx_s * dx_s + dy_s * dy_s + dz_s * dz_s).sqrt();
        let dx_t = fxi + state.prev_fx[i];
        let dy_t = fyi + state.prev_fy[i];
        let dz_t = fzi + state.prev_fz[i];
        let tra = 0.5 * (dx_t * dx_t + dy_t * dy_t + dz_t * dz_t).sqrt();
        state.swing_buf[i] = swg;
        let m = state.masses[i];
        total_swing += m * swg;
        total_traction += m * tra;
    }

    let target_speed = if total_swing > 1e-12 {
        (total_traction / total_swing).clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED)
    } else {
        (state.global_speed * 1.5).min(MAX_LINLOG_SPEED)
    };
    if state.iter > 0 {
        let max_rise = 0.5 * state.global_speed;
        state.global_speed += (target_speed - state.global_speed).min(max_rise);
    } else {
        state.global_speed = target_speed;
    }
    state.global_speed = state.global_speed.clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED);

    let speed = state.global_speed;
    let max_disp = state.max_displacement;
    let (xs_mut, ys_mut, zs_mut) = layout.xyz_mut();
    for i in 0..n {
        let fxi = state.fx[i];
        let fyi = state.fy[i];
        let fzi = state.fz[i];
        let local_factor =
            speed / (1.0 + (speed * state.swing_buf[i]).sqrt() + 0.5 * state.att_hess[i]);
        let mut dx = fxi * local_factor;
        let mut dy = fyi * local_factor;
        let mut dz = fzi * local_factor;
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        if disp > max_disp {
            let cap = max_disp / disp;
            dx *= cap;
            dy *= cap;
            dz *= cap;
        }
        let nx = S::from_f64(state.x_buf[i] + dx);
        let ny = S::from_f64(state.y_buf[i] + dy);
        let nz = S::from_f64(state.z_buf[i] + dz);
        if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "LinLogSolver 3D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        zs_mut[i] = nz;
        state.prev_fx[i] = fxi;
        state.prev_fy[i] = fyi;
        state.prev_fz[i] = fzi;
    }

    Ok(())
}
