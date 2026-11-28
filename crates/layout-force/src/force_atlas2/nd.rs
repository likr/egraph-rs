use egraph_core::{CoreError, CoreResult, DeterministicRng, LayoutNd, Scalar};

use super::overlap_quadtree::{OVERLAP_DPRIME_MIN, OVERLAP_REPULSION_K};
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

/// Lazy exact N-D scratch buffers for [`ForceAtlas2Solver`](super::ForceAtlas2Solver) on
/// [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct ForceAtlas2NdScratch {
    dims: usize,
    coords: Vec<f64>,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
}

impl ForceAtlas2NdScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            coords: vec![0.0; total],
            forces: vec![0.0; total],
            prev_forces: vec![0.0; total],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
        })
    }
}

fn compute_nd_overlap_exact(state: &ForceAtlas2State, scratch: &mut ForceAtlas2NdScratch) {
    let n = state.num_nodes;
    let d = scratch.dims;
    let sr = state.config.scaling_ratio;
    scratch.forces.fill(0.0);
    for i in 0..n {
        let base_i = i * d;
        let mi = state.masses[i];
        let ri = state.node_radii[i];
        for j in (i + 1)..n {
            let base_j = j * d;
            let mj = state.masses[j];
            let r_sum = ri + state.node_radii[j];
            let mut dist_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                dist_sq += delta * delta;
            }
            let dist = dist_sq.sqrt();
            let d_prime = dist - r_sum;
            if d_prime > 0.0 {
                let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
                let mult = (sr * mi * mj) / (dist * d_prime_safe);
                for axis in 0..d {
                    let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                    scratch.forces[base_i + axis] += f;
                    scratch.forces[base_j + axis] -= f;
                }
            } else if d_prime < 0.0 {
                let mag = OVERLAP_REPULSION_K * mi * mj;
                if dist > 1e-12 {
                    let scale = mag / dist;
                    for axis in 0..d {
                        let f =
                            scale * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                        scratch.forces[base_i + axis] += f;
                        scratch.forces[base_j + axis] -= f;
                    }
                } else {
                    scratch.forces[base_i] += mag;
                    scratch.forces[base_j] -= mag;
                }
            }
        }
    }
}

fn compute_nd_coulomb_exact(state: &ForceAtlas2State, scratch: &mut ForceAtlas2NdScratch) {
    let n = state.num_nodes;
    let d = scratch.dims;
    let sr = state.config.scaling_ratio;
    scratch.forces.fill(0.0);
    for i in 0..n {
        let base_i = i * d;
        let mi_sr = sr * state.masses[i];
        for (j, &mj) in state.masses.iter().enumerate().take(n).skip(i + 1) {
            let base_j = j * d;
            let mut dist_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                dist_sq += delta * delta;
            }
            let mult = (mi_sr * mj) / (dist_sq + COULOMB_EPS_SQ);
            for axis in 0..d {
                let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                scratch.forces[base_i + axis] += f;
                scratch.forces[base_j + axis] -= f;
            }
        }
    }
}

fn step_nd_attraction_and_gravity(state: &ForceAtlas2State, scratch: &mut ForceAtlas2NdScratch) {
    let n = state.num_nodes;
    let d = scratch.dims;
    let prevent_overlap = state.config.prevent_overlap;
    let linlog = state.config.linlog_mode;

    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let base_u = u * d;
        let base_v = v * d;
        let mut dist_sq = 0.0_f64;
        for axis in 0..d {
            let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
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
        for axis in 0..d {
            let a = factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
            scratch.forces[base_u + axis] += a;
            scratch.forces[base_v + axis] -= a;
        }
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for (i, &mi) in state.masses.iter().enumerate().take(n) {
                let gf = g * mi;
                let base = i * d;
                for axis in 0..d {
                    scratch.forces[base + axis] -= gf * scratch.coords[base + axis];
                }
            }
        } else {
            for (i, &mi) in state.masses.iter().enumerate().take(n) {
                let base = i * d;
                let mut dist_sq = 0.0_f64;
                for axis in 0..d {
                    let c = scratch.coords[base + axis];
                    dist_sq += c * c;
                }
                let dist = dist_sq.sqrt();
                if dist > 1e-12 {
                    let gf = (g * mi) / dist;
                    for axis in 0..d {
                        scratch.forces[base + axis] -= gf * scratch.coords[base + axis];
                    }
                }
            }
        }
    }
}

pub(super) fn step_nd<S: Scalar>(
    state: &mut ForceAtlas2State,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    let mut scratch = match state.nd_scratch.take() {
        Some(existing) if existing.dims == dims => existing,
        _ => ForceAtlas2NdScratch::new(n, dims)?,
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

        if state.config.prevent_overlap {
            compute_nd_overlap_exact(state, &mut scratch);
        } else {
            compute_nd_coulomb_exact(state, &mut scratch);
        }
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
        let speed_scale = if state.config.prevent_overlap {
            0.1
        } else {
            1.0
        };
        let max_disp = state.max_displacement;
        let coords_mut = layout.coords_mut();
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
                let next_val = S::from_f64(scratch.coords[idx] + fi * local_factor * cap);
                if !next_val.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "ForceAtlas2Solver N-D coordinates became non-finite",
                    ));
                }
                coords_mut[idx] = next_val;
                scratch.prev_forces[idx] = fi;
            }
        }
        Ok(())
    })();

    state.nd_scratch = Some(scratch);
    result
}
