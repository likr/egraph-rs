use egraph_core::{CoreResult, LayoutNd, Scalar};

use super::solver::GuMapState;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

pub(super) fn step_nd<S: Scalar>(
    state: &mut GuMapState<S>,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    let n = state.num_nodes;
    let coords = layout.coords_mut();

    if state.epoch == 0 && coords.iter().all(|&v| v == S::ZERO) {
        for val in coords.iter_mut() {
            *val = S::from_f64(state.rng.next_f64() * 20.0 - 10.0);
        }
    }

    let alpha = S::from_f64(
        state.config.learning_rate
            * (1.0 - (state.epoch as f64) / (state.config.iterations as f64)),
    );
    let a_s = S::from_f64(state.a);
    let b_s = S::from_f64(state.b);
    let two_b_s = S::from_f64(2.0 * state.b);
    let two_ab_s = S::from_f64(2.0 * state.a * state.b);
    let four_s = S::from_f64(GRAD_CLAMP);
    let neg_four_s = -four_s;
    let eps_sq_s = S::from_f64(MIN_DIST_SQ_CLAMP);
    let rep_eps_s = S::from_f64(REPULSIVE_DIST_SQ_EPS);

    let neg_rate = state.config.negative_sample_rate;
    let m = state.edge_src.len();

    for idx in 0..m {
        let p_edge = state.edge_weight[idx] as f64;
        if p_edge <= 0.0 || (p_edge < 1.0 && state.rng.next_f64() > p_edge) {
            continue;
        }

        let i = state.edge_src[idx] as usize;
        let j = state.edge_tgt[idx] as usize;
        let i_base = i * dims;
        let j_base = j * dims;

        let mut d2_raw = S::ZERO;
        for d in 0..dims {
            let diff = coords[i_base + d] - coords[j_base + d];
            d2_raw += diff * diff;
        }
        let d2 = d2_raw.max(eps_sq_s);
        let d_2b = d2.powf(b_s);
        let d_pow = d_2b / d2;
        let raw_grad = -two_ab_s * d_pow / (S::ONE + a_s * d_2b);
        if raw_grad.is_finite() {
            let grad_coeff = raw_grad.clamp(neg_four_s, four_s);
            let step_coeff = alpha * grad_coeff;
            for d in 0..dims {
                let diff = coords[i_base + d] - coords[j_base + d];
                let delta = step_coeff * diff;
                let new_i = coords[i_base + d] + delta;
                let new_j = coords[j_base + d] - delta;
                if new_i.is_finite() {
                    coords[i_base + d] = new_i;
                }
                if new_j.is_finite() {
                    coords[j_base + d] = new_j;
                }
            }
        }

        for _ in 0..neg_rate {
            let k = state.rng.gen_range_usize(n);
            if k == i {
                continue;
            }
            let k_base = k * dims;
            let mut r_d2_raw = S::ZERO;
            for d in 0..dims {
                let r_diff = coords[i_base + d] - coords[k_base + d];
                r_d2_raw += r_diff * r_diff;
            }
            let r_d2 = r_d2_raw.max(eps_sq_s);
            let r_d_2b = r_d2.powf(b_s);
            let denom = (r_d2_raw + rep_eps_s) * (S::ONE + a_s * r_d_2b);
            let rep_coeff = if denom > S::ZERO && denom.is_finite() {
                let raw_rep = two_b_s / denom;
                if raw_rep.is_finite() {
                    raw_rep.clamp(S::ZERO, four_s)
                } else {
                    four_s
                }
            } else {
                S::ZERO
            };
            let rep_step = alpha * rep_coeff;
            for d in 0..dims {
                let r_diff = coords[i_base + d] - coords[k_base + d];
                let r_delta = rep_step * r_diff;
                let new_i = coords[i_base + d] + r_delta;
                if new_i.is_finite() {
                    coords[i_base + d] = new_i;
                }
                if state.config.symmetric_negative {
                    let new_k = coords[k_base + d] - r_delta;
                    if new_k.is_finite() {
                        coords[k_base + d] = new_k;
                    }
                }
            }
        }
    }

    Ok(())
}
