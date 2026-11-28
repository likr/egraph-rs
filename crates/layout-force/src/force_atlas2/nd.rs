use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Scalar,
};

use super::repulsion::{ForceRepulsion, ForceRepulsionParams};
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

#[derive(Debug, Clone)]
pub(super) struct ForceAtlas2NdScratch {
    pub(super) dims: usize,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
}

impl ForceAtlas2NdScratch {
    pub(super) fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            forces: vec![0.0; total],
            prev_forces: vec![0.0; total],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
        })
    }
}

pub(super) fn step_nd<L: EuclideanLayout, B: ForceRepulsion<L>>(
    state: &mut ForceAtlas2State,
    scratch: &mut ForceAtlas2NdScratch,
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
                        "ForceAtlas2Solver N-D scatter coordinates became non-finite",
                    ));
                }
                *slot = next;
            }
        }
    }

    scratch.forces.fill(0.0);
    let params = ForceRepulsionParams::new(
        &state.masses,
        &state.node_radii,
        state.config.scaling_ratio,
        COULOMB_EPS_SQ,
        state.config.prevent_overlap,
        state.config.seed,
        state.iter,
    );
    backend.accumulate_repulsion(rep_state, layout, &params, &mut scratch.forces)?;

    let EuclideanStorageMut::FlatNd { coords, .. } = layout.as_euclidean_storage_mut() else {
        return Err(CoreError::numerical_breakdown(
            "ForceAtlas2Solver N-D storage mismatch",
        ));
    };

    let prevent_overlap = state.config.prevent_overlap;
    let linlog = state.config.linlog_mode;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let base_u = u * dims;
        let base_v = v * dims;
        let mut dist_sq = 0.0_f64;
        for axis in 0..dims {
            let delta = coords[base_v + axis].to_f64() - coords[base_u + axis].to_f64();
            dist_sq += delta * delta;
        }
        let factor = if prevent_overlap {
            let dist = dist_sq.sqrt();
            let d_prime = dist - (state.node_radii[u] + state.node_radii[v]);
            if d_prime > 0.0 && dist > 1e-12 {
                if linlog {
                    w * d_prime.ln_1p() / dist
                } else {
                    w * d_prime / dist
                }
            } else {
                0.0
            }
        } else if linlog {
            if dist_sq > 1e-24 {
                let dist = dist_sq.sqrt();
                w * dist.ln_1p() / dist
            } else {
                0.0
            }
        } else {
            w
        };
        if factor == 0.0 {
            continue;
        }
        for axis in 0..dims {
            let a = factor * (coords[base_v + axis].to_f64() - coords[base_u + axis].to_f64());
            scratch.forces[base_u + axis] += a;
            scratch.forces[base_v + axis] -= a;
        }
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for (i, &mi) in state.masses.iter().enumerate().take(n) {
                let gf = g * mi;
                let base = i * dims;
                for axis in 0..dims {
                    scratch.forces[base + axis] -= gf * coords[base + axis].to_f64();
                }
            }
        } else {
            for (i, &mi) in state.masses.iter().enumerate().take(n) {
                let base = i * dims;
                let mut dist_sq = 0.0_f64;
                for axis in 0..dims {
                    let c = coords[base + axis].to_f64();
                    dist_sq += c * c;
                }
                let dist = dist_sq.sqrt();
                if dist > 1e-12 {
                    let gf = (g * mi) / dist;
                    for axis in 0..dims {
                        scratch.forces[base + axis] -= gf * coords[base + axis].to_f64();
                    }
                }
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
    for (i, &swing) in state.swing_buf.iter().enumerate().take(n) {
        let base = i * dims;
        let local_factor = speed_scale * (speed / (1.0 + (speed * swing).sqrt()));
        let mut disp_sq = 0.0_f64;
        for axis in 0..dims {
            let d_axis = scratch.forces[base + axis] * local_factor;
            disp_sq += d_axis * d_axis;
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
            let next_val = L::Scalar::from_f64(coords[idx].to_f64() + fi * local_factor * cap);
            if !next_val.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "ForceAtlas2Solver N-D coordinates became non-finite",
                ));
            }
            coords[idx] = next_val;
            scratch.prev_forces[idx] = fi;
        }
    }

    Ok(())
}
