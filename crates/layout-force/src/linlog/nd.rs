use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Scalar,
};

use super::repulsion::LinLogRepulsionBackend;
use super::solver::{COULOMB_EPS_SQ, LinLogState, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

#[derive(Debug, Clone)]
pub(super) struct LinLogNdScratch {
    pub(super) dims: usize,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    att_hess: Vec<f64>,
    barycenter: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
}

impl LinLogNdScratch {
    pub(super) fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LinLog N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            forces: vec![0.0; total],
            prev_forces: vec![0.0; total],
            att_hess: vec![0.0; num_nodes],
            barycenter: vec![0.0; dims],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
        })
    }
}

pub(super) fn step_nd<L: EuclideanLayout, B: LinLogRepulsionBackend<L>>(
    state: &mut LinLogState,
    scratch: &mut LinLogNdScratch,
    backend: &B,
    rep_state: &mut B::State,
    layout: &mut L,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = scratch.dims;

    if state.iter == 0
        && let EuclideanStorageMut::FlatNd { coords, .. } = layout.as_euclidean_storage_mut()
    {
        scratch.min_axis.fill(f64::INFINITY);
        scratch.max_axis.fill(f64::NEG_INFINITY);
        for i in 0..n {
            let base = i * dims;
            for axis in 0..dims {
                let v = coords[base + axis].to_f64();
                scratch.min_axis[axis] = scratch.min_axis[axis].min(v);
                scratch.max_axis[axis] = scratch.max_axis[axis].max(v);
            }
        }
        if (0..dims).all(|a| (scratch.max_axis[a] - scratch.min_axis[a]) < 1e-12) {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for slot in coords.iter_mut() {
                let next = L::Scalar::from_f64(slot.to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                if !next.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "LinLogSolver N-D scatter coordinates became non-finite",
                    ));
                }
                *slot = next;
            }
        }
    }

    scratch.forces.fill(0.0);
    backend.accumulate_repulsion(
        rep_state,
        layout,
        &state.masses,
        state.config.repulsion_exponent,
        COULOMB_EPS_SQ,
        &mut scratch.forces,
    )?;

    let EuclideanStorageMut::FlatNd { coords, .. } = layout.as_euclidean_storage_mut() else {
        return Err(CoreError::numerical_breakdown(
            "LinLogSolver N-D storage mismatch",
        ));
    };

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
        let base_u = u * dims;
        let base_v = v * dims;
        let mut d_sq = 0.0_f64;
        for axis in 0..dims {
            let delta = coords[base_v + axis].to_f64() - coords[base_u + axis].to_f64();
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

        for axis in 0..dims {
            let a = factor * (coords[base_v + axis].to_f64() - coords[base_u + axis].to_f64());
            scratch.forces[base_u + axis] += a;
            scratch.forces[base_v + axis] -= a;
        }
        scratch.att_hess[u] += hess;
        scratch.att_hess[v] += hess;
    }

    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        scratch.barycenter.fill(0.0);
        for (i, &m) in state.masses.iter().enumerate().take(n) {
            let base = i * dims;
            for axis in 0..dims {
                scratch.barycenter[axis] += m * coords[base + axis].to_f64();
            }
        }
        for b in &mut scratch.barycenter {
            *b /= state.total_mass;
        }

        let g = state.config.gravity;
        for i in 0..n {
            let base = i * dims;
            let mut d_sq = 0.0_f64;
            for axis in 0..dims {
                let delta = scratch.barycenter[axis] - coords[base + axis].to_f64();
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
                for axis in 0..dims {
                    scratch.forces[base + axis] +=
                        factor * (scratch.barycenter[axis] - coords[base + axis].to_f64());
                }
                scratch.att_hess[i] += a_hess_scale * factor;
            }
        }
    }

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
            let next_c = L::Scalar::from_f64(coords[idx].to_f64() + fi * local_factor * cap);
            if !next_c.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "LinLogSolver N-D coordinates became non-finite",
                ));
            }
            coords[idx] = next_c;
            scratch.prev_forces[idx] = fi;
        }
    }

    Ok(())
}
