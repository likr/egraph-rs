use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout3d, Scalar};

use super::overlap_quadtree::{OVERLAP_DPRIME_MIN, OVERLAP_REPULSION_K};
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

pub(super) fn step_3d<S: Scalar>(
    state: &mut ForceAtlas2State,
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

    let prevent_overlap = state.config.prevent_overlap;
    if prevent_overlap {
        compute_3d_overlap_exact(state);
    } else if state.config.use_barnes_hut {
        state
            .octree
            .build_weighted_xyz(&state.x_buf, &state.y_buf, &state.z_buf, &state.masses)?;
        state.octree.compute_coulomb_repulsion(
            state.config.theta,
            COULOMB_EPS_SQ,
            &mut state.rep_x,
            &mut state.rep_y,
            &mut state.rep_z,
        )?;
        let sr = state.config.scaling_ratio;
        for i in 0..n {
            let scale = sr * state.masses[i];
            state.fx[i] = scale * state.rep_x[i];
            state.fy[i] = scale * state.rep_y[i];
            state.fz[i] = scale * state.rep_z[i];
        }
    } else {
        state.fx.fill(0.0);
        state.fy.fill(0.0);
        state.fz.fill(0.0);
        let sr = state.config.scaling_ratio;
        for i in 0..n {
            let xi = state.x_buf[i];
            let yi = state.y_buf[i];
            let zi = state.z_buf[i];
            let mi_sr = sr * state.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            let mut acc_z = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - state.x_buf[j];
                let dy = yi - state.y_buf[j];
                let dz = zi - state.z_buf[j];
                let denom = dx * dx + dy * dy + dz * dz + COULOMB_EPS_SQ;
                let mult = (mi_sr * state.masses[j]) / denom;
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

    let linlog = state.config.linlog_mode;
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
        let factor = if prevent_overlap {
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let d_prime = d - (state.node_radii[u] + state.node_radii[v]);
            if d_prime > 0.0 && d > 1e-12 {
                if linlog {
                    w * d_prime.ln_1p() / d
                } else {
                    w * d_prime / d
                }
            } else {
                0.0
            }
        } else if linlog {
            let d_sq = dx * dx + dy * dy + dz * dz;
            if d_sq > 1e-24 {
                let d = d_sq.sqrt();
                w * d.ln_1p() / d
            } else {
                0.0
            }
        } else {
            w
        };
        if factor == 0.0 {
            continue;
        }
        let ax = factor * dx;
        let ay = factor * dy;
        let az = factor * dz;
        state.fx[u] += ax;
        state.fy[u] += ay;
        state.fz[u] += az;
        state.fx[v] -= ax;
        state.fy[v] -= ay;
        state.fz[v] -= az;
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for i in 0..n {
                let gf = g * state.masses[i];
                state.fx[i] -= gf * state.x_buf[i];
                state.fy[i] -= gf * state.y_buf[i];
                state.fz[i] -= gf * state.z_buf[i];
            }
        } else {
            for i in 0..n {
                let x = state.x_buf[i];
                let y = state.y_buf[i];
                let z = state.z_buf[i];
                let d = (x * x + y * y + z * z).sqrt();
                if d > 1e-12 {
                    let gf = (g * state.masses[i]) / d;
                    state.fx[i] -= gf * x;
                    state.fy[i] -= gf * y;
                    state.fz[i] -= gf * z;
                }
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

    let jt = state.config.jitter_tolerance;
    let max_speed = 10.0 * jt.max(1.0);
    let target_speed = if total_swing > 1e-12 {
        (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
    } else {
        (state.global_speed * 1.5).min(max_speed)
    };
    if state.iter > 0 {
        let max_rise = 0.5 * state.global_speed;
        state.global_speed += (target_speed - state.global_speed).min(max_rise);
    } else {
        state.global_speed = target_speed;
    }
    state.global_speed = state.global_speed.clamp(MIN_FA2_SPEED, max_speed);

    let speed = state.global_speed;
    let speed_scale = if prevent_overlap { 0.1 } else { 1.0 };
    let max_disp = state.max_displacement;
    let (xs_mut, ys_mut, zs_mut) = layout.xyz_mut();
    for i in 0..n {
        let fxi = state.fx[i];
        let fyi = state.fy[i];
        let fzi = state.fz[i];
        let local_factor = speed_scale * (speed / (1.0 + (speed * state.swing_buf[i]).sqrt()));
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
                "ForceAtlas2Solver 3D coordinates became non-finite",
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

fn compute_3d_overlap_exact(state: &mut ForceAtlas2State) {
    let n = state.num_nodes;
    let sr = state.config.scaling_ratio;
    state.fx.fill(0.0);
    state.fy.fill(0.0);
    state.fz.fill(0.0);
    for i in 0..n {
        let xi = state.x_buf[i];
        let yi = state.y_buf[i];
        let zi = state.z_buf[i];
        let mi = state.masses[i];
        let ri = state.node_radii[i];
        for j in (i + 1)..n {
            let mj = state.masses[j];
            let r_sum = ri + state.node_radii[j];
            let dx = xi - state.x_buf[j];
            let dy = yi - state.y_buf[j];
            let dz = zi - state.z_buf[j];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let d_prime = dist - r_sum;
            if d_prime > 0.0 {
                let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
                let mult = (sr * mi * mj) / (dist * d_prime_safe);
                let fx = mult * dx;
                let fy = mult * dy;
                let fz = mult * dz;
                state.fx[i] += fx;
                state.fy[i] += fy;
                state.fz[i] += fz;
                state.fx[j] -= fx;
                state.fy[j] -= fy;
                state.fz[j] -= fz;
            } else if d_prime < 0.0 {
                let mag = OVERLAP_REPULSION_K * mi * mj;
                if dist > 1e-12 {
                    let scale = mag / dist;
                    let fx = scale * dx;
                    let fy = scale * dy;
                    let fz = scale * dz;
                    state.fx[i] += fx;
                    state.fy[i] += fy;
                    state.fz[i] += fz;
                    state.fx[j] -= fx;
                    state.fy[j] -= fy;
                    state.fz[j] -= fz;
                } else {
                    state.fx[i] += mag;
                    state.fx[j] -= mag;
                }
            }
        }
    }
}
