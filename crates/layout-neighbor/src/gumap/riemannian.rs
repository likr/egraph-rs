use egraph_core::{CoreError, CoreResult, RiemannianLayout};

use super::solver::GuMapState;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

#[derive(Debug, Clone)]
pub(super) struct GuMapRiemannianScratch {
    dims: usize,
    log_ij: Vec<f64>,
    log_ji: Vec<f64>,
    delta: Vec<f64>,
}

pub(super) fn step_riemannian<L: RiemannianLayout>(
    state: &mut GuMapState<L::Scalar>,
    layout: &mut L,
) -> CoreResult<()> {
    let dims = layout.tangent_dim();
    if let Some(scratch) = &state.riemannian_scratch {
        if scratch.dims != dims {
            return Err(CoreError::dimension_mismatch(
                scratch.dims,
                dims,
                "GuMapSolver::riemannian::tangent_dim",
            ));
        }
    } else {
        state.riemannian_scratch = Some(GuMapRiemannianScratch {
            dims,
            log_ij: vec![0.0; dims],
            log_ji: vec![0.0; dims],
            delta: vec![0.0; dims],
        });
    }

    let n = state.num_nodes;
    let alpha = state.config.learning_rate
        * (1.0 - (state.epoch as f64) / (state.config.iterations as f64));
    let a = state.a;
    let b = state.b;
    let two_b = 2.0 * b;
    let two_ab = 2.0 * a * b;
    let neg_rate = state.config.negative_sample_rate;
    let m = state.edge_src.len();

    let scratch = state
        .riemannian_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapRiemannianScratch missing"))?;

    for idx in 0..m {
        let p_edge = state.edge_weight[idx] as f64;
        if p_edge <= 0.0 || (p_edge < 1.0 && state.rng.next_f64() > p_edge) {
            continue;
        }

        let i = state.edge_src[idx] as usize;
        let j = state.edge_tgt[idx] as usize;

        layout.log_map_into(i, j, &mut scratch.log_ij)?;
        layout.log_map_into(j, i, &mut scratch.log_ji)?;
        let mut d2_raw = 0.0_f64;
        for &v in &scratch.log_ij {
            d2_raw += v * v;
        }
        let d2 = d2_raw.max(MIN_DIST_SQ_CLAMP);
        let d_2b = d2.powf(b);
        let d_pow = d_2b / d2;
        let raw_grad = -two_ab * d_pow / (1.0 + a * d_2b);
        if raw_grad.is_finite() {
            let grad_coeff = raw_grad.clamp(-GRAD_CLAMP, GRAD_CLAMP);
            let step_coeff = -alpha * grad_coeff;
            for d in 0..dims {
                scratch.delta[d] = step_coeff * scratch.log_ij[d];
            }
            layout.exp_map(i, &scratch.delta)?;
            for d in 0..dims {
                scratch.delta[d] = step_coeff * scratch.log_ji[d];
            }
            layout.exp_map(j, &scratch.delta)?;
        }

        for _ in 0..neg_rate {
            let k = state.rng.gen_range_usize(n);
            if k == i {
                continue;
            }
            layout.log_map_into(i, k, &mut scratch.log_ij)?;
            let sym = state.config.symmetric_negative;
            if sym {
                layout.log_map_into(k, i, &mut scratch.log_ji)?;
            }
            let mut r_d2_raw = 0.0_f64;
            for &v in &scratch.log_ij {
                r_d2_raw += v * v;
            }
            let r_d2 = r_d2_raw.max(MIN_DIST_SQ_CLAMP);
            let r_d_2b = r_d2.powf(b);
            let denom = (r_d2_raw + REPULSIVE_DIST_SQ_EPS) * (1.0 + a * r_d_2b);
            let rep_coeff = if denom > 0.0 && denom.is_finite() {
                let raw_rep = two_b / denom;
                if raw_rep.is_finite() {
                    raw_rep.clamp(0.0, GRAD_CLAMP)
                } else {
                    GRAD_CLAMP
                }
            } else {
                0.0
            };
            let rep_step = -alpha * rep_coeff;
            for d in 0..dims {
                scratch.delta[d] = rep_step * scratch.log_ij[d];
            }
            layout.exp_map(i, &scratch.delta)?;
            if sym {
                for d in 0..dims {
                    scratch.delta[d] = rep_step * scratch.log_ji[d];
                }
                layout.exp_map(k, &scratch.delta)?;
            }
        }
    }

    Ok(())
}
