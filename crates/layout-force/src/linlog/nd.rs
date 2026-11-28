use egraph_core::{CoreError, CoreResult, DeterministicRng, LayoutNd, Scalar};

use super::solver::{COULOMB_EPS_SQ, LinLogState, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

/// Lazy exact N-D scratch buffers for [`LinLogSolver`](super::LinLogSolver) on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct LinLogNdScratch {
    dims: usize,
    coords: Vec<f64>,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    att_hess: Vec<f64>,
    barycenter: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
}

impl LinLogNdScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LinLog N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            coords: vec![0.0; total],
            forces: vec![0.0; total],
            prev_forces: vec![0.0; total],
            att_hess: vec![0.0; num_nodes],
            barycenter: vec![0.0; dims],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
        })
    }
}

fn step_nd_repulsion(
    state: &LinLogState,
    scratch: &mut LinLogNdScratch,
    r_exp: f64,
    is_coulomb: bool,
) {
    let n = state.num_nodes;
    let d = scratch.dims;
    scratch.forces.fill(0.0);
    let pow_exp = 0.5 * (r_exp - 1.0);
    for i in 0..n {
        let base_i = i * d;
        let mi = state.masses[i];
        for (j, &mj) in state.masses.iter().enumerate().take(n).skip(i + 1) {
            let base_j = j * d;
            let mut d_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                d_sq += delta * delta;
            }
            let d_sq_reg = d_sq + COULOMB_EPS_SQ;
            let mult = if is_coulomb {
                (mi * mj) / d_sq_reg
            } else {
                (mi * mj) * d_sq_reg.powf(pow_exp)
            };
            for axis in 0..d {
                let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                scratch.forces[base_i + axis] += f;
                scratch.forces[base_j + axis] -= f;
            }
        }
    }
}

fn step_nd_attraction_and_gravity(state: &LinLogState, scratch: &mut LinLogNdScratch) {
    let n = state.num_nodes;
    let d = scratch.dims;
    let a_exp = state.config.attraction_exponent;
    let a_is_zero = a_exp.abs() <= 1e-12;
    let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
    let a_hess_scale = a_exp.abs().max(1.0);
    scratch.att_hess.fill(0.0);

    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let base_u = u * d;
        let base_v = v * d;
        let mut d_sq = 0.0_f64;
        for axis in 0..d {
            let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
            d_sq += delta * delta;
        }
        let (factor, hess) = if a_is_zero {
            if d_sq > 1e-24 {
                let dist = d_sq.sqrt();
                (w / dist, w / dist)
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

        for axis in 0..d {
            let a = factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
            scratch.forces[base_u + axis] += a;
            scratch.forces[base_v + axis] -= a;
        }
        scratch.att_hess[u] += hess;
        scratch.att_hess[v] += hess;
    }

    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        scratch.barycenter.fill(0.0);
        for (i, &m) in state.masses.iter().enumerate().take(n) {
            let base = i * d;
            for axis in 0..d {
                scratch.barycenter[axis] += m * scratch.coords[base + axis];
            }
        }
        for b in &mut scratch.barycenter {
            *b /= state.total_mass;
        }

        let g = state.config.gravity;
        for i in 0..n {
            let base = i * d;
            let mut d_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.barycenter[axis] - scratch.coords[base + axis];
                d_sq += delta * delta;
            }
            let dist = d_sq.sqrt();
            if dist > 1e-12 {
                let factor = if a_is_zero {
                    (g * state.masses[i]) / dist
                } else if a_is_one {
                    g * state.masses[i]
                } else {
                    g * state.masses[i] * dist.powf(a_exp - 1.0)
                };
                for axis in 0..d {
                    scratch.forces[base + axis] +=
                        factor * (scratch.barycenter[axis] - scratch.coords[base + axis]);
                }
                scratch.att_hess[i] += a_hess_scale * factor;
            }
        }
    }
}

pub(super) fn step_nd<S: Scalar>(
    state: &mut LinLogState,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    let mut scratch = match state.nd_scratch.take() {
        Some(existing) if existing.dims == dims => existing,
        _ => LinLogNdScratch::new(n, dims)?,
    };

    let result = (|| -> CoreResult<()> {
        scratch.min_axis.fill(f64::INFINITY);
        scratch.max_axis.fill(f64::NEG_INFINITY);
        let src = layout.coords();
        for i in 0..n {
            let base = i * dims;
            for axis in 0..dims {
                let v = src[base + axis].to_f64();
                scratch.coords[base + axis] = v;
                scratch.min_axis[axis] = scratch.min_axis[axis].min(v);
                scratch.max_axis[axis] = scratch.max_axis[axis].max(v);
            }
        }

        if state.iter == 0 && (0..dims).all(|a| (scratch.max_axis[a] - scratch.min_axis[a]) < 1e-12)
        {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for v in &mut scratch.coords {
                *v += 1e-4 * (rng.next_f64() - 0.5);
            }
        }

        let r_exp = state.config.repulsion_exponent;
        let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;
        step_nd_repulsion(state, &mut scratch, r_exp, is_coulomb);
        step_nd_attraction_and_gravity(state, &mut scratch);

        let mut total_swing = 0.0_f64;
        let mut total_traction = 0.0_f64;
        for i in 0..n {
            let base = i * dims;
            let mut swg_sq = 0.0_f64;
            let mut tra_sq = 0.0_f64;
            for axis in 0..dims {
                let idx = base + axis;
                let fi = if scratch.forces[idx].is_finite() {
                    scratch.forces[idx]
                } else {
                    0.0
                };
                scratch.forces[idx] = fi;
                let df = fi - scratch.prev_forces[idx];
                let sf = fi + scratch.prev_forces[idx];
                swg_sq += df * df;
                tra_sq += sf * sf;
            }
            let swg = swg_sq.sqrt();
            let tra = 0.5 * tra_sq.sqrt();
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
        let coords_mut = layout.coords_mut();
        for i in 0..n {
            let base = i * dims;
            let local_factor =
                speed / (1.0 + (speed * state.swing_buf[i]).sqrt() + 0.5 * scratch.att_hess[i]);
            let mut disp_sq = 0.0_f64;
            for axis in 0..dims {
                let da = scratch.forces[base + axis] * local_factor;
                disp_sq += da * da;
            }
            let disp = disp_sq.sqrt();
            let cap = if disp > max_disp {
                max_disp / disp
            } else {
                1.0
            };
            for axis in 0..dims {
                let idx = base + axis;
                let fi = scratch.forces[idx];
                let next_c = S::from_f64(scratch.coords[idx] + fi * local_factor * cap);
                if !next_c.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "LinLogSolver N-D coordinates became non-finite",
                    ));
                }
                coords_mut[idx] = next_c;
                scratch.prev_forces[idx] = fi;
            }
        }
        Ok(())
    })();

    state.nd_scratch = Some(scratch);
    result
}
