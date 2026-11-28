use egraph_core::{CoreError, CoreResult, Layout2d, LayoutStorage, Scalar};

use super::target::GuMapCore;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

pub(super) fn validate_basic_2d<S: Scalar>(
    core: &GuMapCore,
    target: &Layout2d<S>,
) -> CoreResult<()> {
    if target.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    LayoutStorage::validate_finite(target)
}

pub(super) fn step_2d<S: Scalar>(core: &mut GuMapCore, layout: &mut Layout2d<S>) -> CoreResult<()> {
    let n = core.num_nodes;
    let (xs, ys) = layout.xy_mut();

    if core.epoch == 0 && xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
        for idx in 0..n {
            xs[idx] = S::from_f64(core.rng.next_f64() * 20.0 - 10.0);
            ys[idx] = S::from_f64(core.rng.next_f64() * 20.0 - 10.0);
        }
    }

    let alpha = S::from_f64(
        core.config.learning_rate * (1.0 - (core.epoch as f64) / (core.config.iterations as f64)),
    );
    let a_s = S::from_f64(core.a);
    let b_s = S::from_f64(core.b);
    let two_b_s = S::from_f64(2.0 * core.b);
    let two_ab_s = S::from_f64(2.0 * core.a * core.b);
    let four_s = S::from_f64(GRAD_CLAMP);
    let neg_four_s = -four_s;
    let eps_sq_s = S::from_f64(MIN_DIST_SQ_CLAMP);
    let rep_eps_s = S::from_f64(REPULSIVE_DIST_SQ_EPS);

    let neg_rate = core.config.negative_sample_rate;
    let m = core.edge_src.len();

    for idx in 0..m {
        let p_edge = core.edge_weight[idx] as f64;
        if p_edge <= 0.0 || (p_edge < 1.0 && core.rng.next_f64() > p_edge) {
            continue;
        }

        let i = core.edge_src[idx] as usize;
        let j = core.edge_tgt[idx] as usize;

        let dx = xs[i] - xs[j];
        let dy = ys[i] - ys[j];
        let d2 = (dx * dx + dy * dy).max(eps_sq_s);
        let d_2b = d2.powf(b_s);
        let d_pow = d_2b / d2;
        let raw_grad = -two_ab_s * d_pow / (S::ONE + a_s * d_2b);
        if raw_grad.is_finite() {
            let grad_coeff = raw_grad.clamp(neg_four_s, four_s);
            let delta_x = alpha * grad_coeff * dx;
            let delta_y = alpha * grad_coeff * dy;
            let new_ix = xs[i] + delta_x;
            let new_iy = ys[i] + delta_y;
            let new_jx = xs[j] - delta_x;
            let new_jy = ys[j] - delta_y;
            if new_ix.is_finite() {
                xs[i] = new_ix;
            }
            if new_iy.is_finite() {
                ys[i] = new_iy;
            }
            if new_jx.is_finite() {
                xs[j] = new_jx;
            }
            if new_jy.is_finite() {
                ys[j] = new_jy;
            }
        }

        for _ in 0..neg_rate {
            let k = core.rng.gen_range_usize(n);
            if k == i {
                continue;
            }
            let r_dx = xs[i] - xs[k];
            let r_dy = ys[i] - ys[k];
            let r_d2_raw = r_dx * r_dx + r_dy * r_dy;
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
            let r_delta_x = alpha * rep_coeff * r_dx;
            let r_delta_y = alpha * rep_coeff * r_dy;
            let new_ix = xs[i] + r_delta_x;
            let new_iy = ys[i] + r_delta_y;
            if new_ix.is_finite() {
                xs[i] = new_ix;
            }
            if new_iy.is_finite() {
                ys[i] = new_iy;
            }
            if core.config.symmetric_negative {
                let new_kx = xs[k] - r_delta_x;
                let new_ky = ys[k] - r_delta_y;
                if new_kx.is_finite() {
                    xs[k] = new_kx;
                }
                if new_ky.is_finite() {
                    ys[k] = new_ky;
                }
            }
        }
    }

    Ok(())
}
