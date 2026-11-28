use egraph_core::{CoreError, CoreResult, RiemannianLayout};

use super::solver::{COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST, FrState};

/// Lazy exact Riemannian tangent-space scratch buffers for generic [`RiemannianLayout`] targets.
#[derive(Debug, Clone)]
pub(super) struct FrRiemannianScratch {
    dims: usize,
    disp: Vec<f64>,
    log_ij: Vec<f64>,
    log_ji: Vec<f64>,
    delta_buf: Vec<f64>,
}

impl FrRiemannianScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FR Riemannian buffer size overflow"))?;
        Ok(Self {
            dims,
            disp: vec![0.0; total],
            log_ij: vec![0.0; dims],
            log_ji: vec![0.0; dims],
            delta_buf: vec![0.0; dims],
        })
    }
}

pub(super) fn step_riemannian<L: RiemannianLayout>(
    state: &mut FrState,
    layout: &mut L,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = layout.tangent_dim();

    let mut scratch = match state.riemannian_scratch.take() {
        Some(existing) if existing.dims == dims => existing,
        _ => FrRiemannianScratch::new(n, dims)?,
    };

    let result = (|| -> CoreResult<()> {
        scratch.disp.fill(0.0);
        let k_sq = state.k * state.k;

        // 1. Exact geodesic repulsion in tangent space.
        for i in 0..n {
            let base_i = i * dims;
            for j in (i + 1)..n {
                let base_j = j * dims;
                layout.log_map_into(i, j, &mut scratch.log_ij)?;
                layout.log_map_into(j, i, &mut scratch.log_ji)?;
                let mut d_sq = 0.0_f64;
                for &v in &scratch.log_ij {
                    d_sq += v * v;
                }
                if d_sq > COINCIDENT_DIST_SQ {
                    let mult = k_sq / d_sq;
                    for d in 0..dims {
                        scratch.disp[base_i + d] -= mult * scratch.log_ij[d];
                        scratch.disp[base_j + d] -= mult * scratch.log_ji[d];
                    }
                } else {
                    let mut norm_sq = 0.0_f64;
                    for slot in &mut scratch.delta_buf {
                        let c = state.rng.next_f64() - 0.5;
                        *slot = c;
                        norm_sq += c * c;
                    }
                    let norm = norm_sq.sqrt();
                    let mag = k_sq / COINCIDENT_PERTURB_DIST;
                    if norm > 1e-12 {
                        let scale = mag / norm;
                        for d in 0..dims {
                            let f = scale * scratch.delta_buf[d];
                            scratch.disp[base_i + d] += f;
                            scratch.disp[base_j + d] -= f;
                        }
                    } else {
                        scratch.disp[base_i] += mag;
                        scratch.disp[base_j] -= mag;
                    }
                }
            }
        }

        // 2. Geodesic edge attraction.
        let inv_k = 1.0 / state.k;
        for idx in 0..state.edges_u.len() {
            let u = state.edges_u[idx] as usize;
            let v = state.edges_v[idx] as usize;
            let w = state.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            layout.log_map_into(u, v, &mut scratch.log_ij)?;
            layout.log_map_into(v, u, &mut scratch.log_ji)?;
            let mut d_sq = 0.0_f64;
            for &val in &scratch.log_ij {
                d_sq += val * val;
            }
            let factor = w * d_sq.sqrt() * inv_k;
            let base_u = u * dims;
            let base_v = v * dims;
            for d in 0..dims {
                scratch.disp[base_u + d] += factor * scratch.log_ij[d];
                scratch.disp[base_v + d] += factor * scratch.log_ji[d];
            }
        }

        // 3. Temperature cooling and exponential map update.
        let temp = state.current_temperature();
        for i in 0..n {
            let base = i * dims;
            let mut len_sq = 0.0_f64;
            for d in 0..dims {
                let v = scratch.disp[base + d];
                len_sq += v * v;
            }
            let len = len_sq.sqrt();
            if len > 1e-24 && temp > 0.0 {
                let scale = len.min(temp) / len;
                for d in 0..dims {
                    scratch.delta_buf[d] = scratch.disp[base + d] * scale;
                }
                layout.exp_map(i, &scratch.delta_buf)?;
            }
        }
        Ok(())
    })();

    state.riemannian_scratch = Some(scratch);
    result
}
