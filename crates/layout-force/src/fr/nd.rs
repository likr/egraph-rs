use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};

use super::solver::{COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST, FrState};

/// Lazy exact N-D scratch buffers for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver)
/// on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct FrNdScratch {
    dims: usize,
    coords: Vec<f64>,
    disp: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
    dir_buf: Vec<f64>,
}

impl FrNdScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FR N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            coords: vec![0.0; total],
            disp: vec![0.0; total],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
            dir_buf: vec![0.0; dims],
        })
    }
}

fn apply_repulsion_pair_nd(
    state: &mut FrState,
    scratch: &mut FrNdScratch,
    i: usize,
    j: usize,
    k_sq: f64,
) {
    let d = scratch.dims;
    let base_i = i * d;
    let base_j = j * d;
    let mut d_sq = 0.0_f64;
    for axis in 0..d {
        let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
        d_sq += delta * delta;
    }
    if d_sq > COINCIDENT_DIST_SQ {
        let mult = k_sq / d_sq;
        for axis in 0..d {
            let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
            scratch.disp[base_i + axis] += f;
            scratch.disp[base_j + axis] -= f;
        }
    } else {
        let mut norm_sq = 0.0_f64;
        for slot in scratch.dir_buf.iter_mut().take(d) {
            let comp = state.rng.next_f64() - 0.5;
            *slot = comp;
            norm_sq += comp * comp;
        }
        let norm = norm_sq.sqrt();
        let mag = k_sq / COINCIDENT_PERTURB_DIST;
        if norm > 1e-12 {
            let scale = mag / norm;
            for axis in 0..d {
                let f = scale * scratch.dir_buf[axis];
                scratch.disp[base_i + axis] += f;
                scratch.disp[base_j + axis] -= f;
            }
        } else {
            scratch.disp[base_i] += mag;
            scratch.disp[base_j] -= mag;
        }
    }
}

pub(super) fn step_nd<S: Scalar>(state: &mut FrState, layout: &mut LayoutNd<S>) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    let mut scratch = match state.nd_scratch.take() {
        Some(existing) if existing.dims == dims => existing,
        _ => FrNdScratch::new(n, dims)?,
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
            let spread = 0.5 * state.k;
            for i in 0..n {
                let base = i * dims;
                for axis in 0..dims {
                    scratch.coords[base + axis] += spread * (state.rng.next_f64() - 0.5);
                }
            }
        }

        scratch.disp.fill(0.0);
        let k_sq = state.k * state.k;

        for i in 0..n {
            for j in (i + 1)..n {
                apply_repulsion_pair_nd(state, &mut scratch, i, j, k_sq);
            }
        }

        let inv_k = 1.0 / state.k;
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
                let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
                d_sq += delta * delta;
            }
            let factor = w * d_sq.sqrt() * inv_k;
            for axis in 0..dims {
                let a = factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
                scratch.disp[base_u + axis] += a;
                scratch.disp[base_v + axis] -= a;
            }
        }

        if state.config.gravity > 0.0 {
            let g = state.config.gravity;
            for (d_val, &c_val) in scratch.disp.iter_mut().zip(scratch.coords.iter()) {
                *d_val -= g * c_val;
            }
        }

        let temp = state.current_temperature();
        let coords_mut = layout.coords_mut();
        for i in 0..n {
            let base = i * dims;
            let mut len_sq = 0.0_f64;
            for axis in 0..dims {
                let da = scratch.disp[base + axis];
                len_sq += da * da;
            }
            let len = len_sq.sqrt();
            if len > 1e-24 && temp > 0.0 {
                let scale = len.min(temp) / len;
                for axis in 0..dims {
                    let idx = base + axis;
                    let next_c = S::from_f64(scratch.coords[idx] + scratch.disp[idx] * scale);
                    if !next_c.is_finite() {
                        return Err(CoreError::numerical_breakdown(
                            "FruchtermanReingoldSolver N-D coordinates became non-finite",
                        ));
                    }
                    coords_mut[idx] = next_c;
                }
            }
        }
        Ok(())
    })();

    state.nd_scratch = Some(scratch);
    result
}
