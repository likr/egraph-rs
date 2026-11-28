use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Scalar,
};

use super::repulsion::LinLogRepulsionBackend;
use super::solver::{COULOMB_EPS_SQ, LinLogState, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

#[derive(Debug, Clone)]
pub(super) struct LinLogBuf3d {
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    att_hess: Vec<f64>,
}

impl LinLogBuf3d {
    pub(super) fn new(num_nodes: usize) -> Self {
        let len = num_nodes * 3;
        Self {
            forces: vec![0.0; len],
            prev_forces: vec![0.0; len],
            att_hess: vec![0.0; num_nodes],
        }
    }
}

pub(super) fn step_3d<L: EuclideanLayout, B: LinLogRepulsionBackend<L>>(
    state: &mut LinLogState,
    buf: &mut LinLogBuf3d,
    backend: &B,
    rep_state: &mut B::State,
    layout: &mut L,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.iter == 0
        && let EuclideanStorageMut::Soa3d { x, y, z } = layout.as_euclidean_storage_mut()
    {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for i in 0..n {
            let xf = x[i].to_f64();
            let yf = y[i].to_f64();
            let zf = z[i].to_f64();
            min_x = min_x.min(xf);
            max_x = max_x.max(xf);
            min_y = min_y.min(yf);
            max_y = max_y.max(yf);
            min_z = min_z.min(zf);
            max_z = max_z.max(zf);
        }
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 && (max_z - min_z) < 1e-12 {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for i in 0..n {
                let nx = L::Scalar::from_f64(x[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                let ny = L::Scalar::from_f64(y[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                let nz = L::Scalar::from_f64(z[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "LinLogSolver 3D scatter coordinates became non-finite",
                    ));
                }
                x[i] = nx;
                y[i] = ny;
                z[i] = nz;
            }
        }
    }

    buf.forces.fill(0.0);
    backend.accumulate_repulsion(
        rep_state,
        layout,
        &state.masses,
        state.config.repulsion_exponent,
        COULOMB_EPS_SQ,
        &mut buf.forces,
    )?;

    let EuclideanStorageMut::Soa3d {
        x: xs_mut,
        y: ys_mut,
        z: zs_mut,
    } = layout.as_euclidean_storage_mut()
    else {
        return Err(CoreError::numerical_breakdown(
            "LinLogSolver 3D storage mismatch",
        ));
    };

    let a_exp = state.config.attraction_exponent;
    let a_is_zero = a_exp.abs() <= 1e-12;
    let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
    let a_hess_scale = a_exp.abs().max(1.0);
    buf.att_hess.fill(0.0);

    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = xs_mut[v].to_f64() - xs_mut[u].to_f64();
        let dy = ys_mut[v].to_f64() - ys_mut[u].to_f64();
        let dz = zs_mut[v].to_f64() - zs_mut[u].to_f64();
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
        buf.forces[3 * u] += ax;
        buf.forces[3 * u + 1] += ay;
        buf.forces[3 * u + 2] += az;
        buf.forces[3 * v] -= ax;
        buf.forces[3 * v + 1] -= ay;
        buf.forces[3 * v + 2] -= az;
        buf.att_hess[u] += hess;
        buf.att_hess[v] += hess;
    }

    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut bx = 0.0_f64;
        let mut by = 0.0_f64;
        let mut bz = 0.0_f64;
        for i in 0..n {
            let m = state.masses[i];
            bx += m * xs_mut[i].to_f64();
            by += m * ys_mut[i].to_f64();
            bz += m * zs_mut[i].to_f64();
        }
        bx /= state.total_mass;
        by /= state.total_mass;
        bz /= state.total_mass;

        let g = state.config.gravity;
        for i in 0..n {
            let dx = bx - xs_mut[i].to_f64();
            let dy = by - ys_mut[i].to_f64();
            let dz = bz - zs_mut[i].to_f64();
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 1e-12 {
                let factor = if a_is_zero {
                    (g * state.masses[i]) / d
                } else if a_is_one {
                    g * state.masses[i]
                } else {
                    g * state.masses[i] * d.powf(a_exp - 1.0)
                };
                buf.forces[3 * i] += factor * dx;
                buf.forces[3 * i + 1] += factor * dy;
                buf.forces[3 * i + 2] += factor * dz;
                buf.att_hess[i] += a_hess_scale * factor;
            }
        }
    }

    let mut total_swing = 0.0_f64;
    let mut total_traction = 0.0_f64;
    for i in 0..n {
        let fxi = if buf.forces[3 * i].is_finite() {
            buf.forces[3 * i]
        } else {
            0.0
        };
        let fyi = if buf.forces[3 * i + 1].is_finite() {
            buf.forces[3 * i + 1]
        } else {
            0.0
        };
        let fzi = if buf.forces[3 * i + 2].is_finite() {
            buf.forces[3 * i + 2]
        } else {
            0.0
        };
        buf.forces[3 * i] = fxi;
        buf.forces[3 * i + 1] = fyi;
        buf.forces[3 * i + 2] = fzi;

        let pfx = buf.prev_forces[3 * i];
        let pfy = buf.prev_forces[3 * i + 1];
        let pfz = buf.prev_forces[3 * i + 2];
        let dx_s = fxi - pfx;
        let dy_s = fyi - pfy;
        let dz_s = fzi - pfz;
        let swg = (dx_s * dx_s + dy_s * dy_s + dz_s * dz_s).sqrt();
        let dx_t = fxi + pfx;
        let dy_t = fyi + pfy;
        let dz_t = fzi + pfz;
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
    for i in 0..n {
        let fxi = buf.forces[3 * i];
        let fyi = buf.forces[3 * i + 1];
        let fzi = buf.forces[3 * i + 2];
        let local_factor =
            speed / (1.0 + (speed * state.swing_buf[i]).sqrt() + 0.5 * buf.att_hess[i]);
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
        let nx = L::Scalar::from_f64(xs_mut[i].to_f64() + dx);
        let ny = L::Scalar::from_f64(ys_mut[i].to_f64() + dy);
        let nz = L::Scalar::from_f64(zs_mut[i].to_f64() + dz);
        if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "LinLogSolver 3D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        zs_mut[i] = nz;
        buf.prev_forces[3 * i] = fxi;
        buf.prev_forces[3 * i + 1] = fyi;
        buf.prev_forces[3 * i + 2] = fzi;
    }

    Ok(())
}
