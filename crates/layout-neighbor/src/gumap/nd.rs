use egraph_core::{CoreError, CoreResult, LayoutNd, LayoutStorage, Scalar};

use super::target::{GuMapCore, GuMapSpaceKind, GuMapWorkspace};

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

pub(super) fn validate_nd<S: Scalar>(
    core: &GuMapCore,
    workspace: &GuMapWorkspace,
    target: &LayoutNd<S>,
) -> CoreResult<()> {
    if target.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    let dims = target.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let GuMapSpaceKind::EuclideanNd { locked_dims } = &workspace.0
        && let Some(prev) = *locked_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "GuMapSolver::validate_target::dims",
        ));
    }
    LayoutStorage::validate_finite(target)
}

pub(super) fn step_nd<S: Scalar>(
    core: &mut GuMapCore,
    workspace: &mut GuMapWorkspace,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let dims = layout.dims();
    if let GuMapSpaceKind::EuclideanNd { locked_dims } = &mut workspace.0 {
        if locked_dims.is_none() {
            *locked_dims = Some(dims);
        }
    } else {
        return Err(CoreError::numerical_breakdown(
            "GuMapNdWorkspace variant mismatch",
        ));
    }

    let n = core.num_nodes;
    let coords = layout.coords_mut();

    if core.epoch == 0 && coords.iter().all(|&v| v == S::ZERO) {
        for val in coords.iter_mut() {
            *val = S::from_f64(core.rng.next_f64() * 20.0 - 10.0);
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
            let k = core.rng.gen_range_usize(n);
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
                if core.config.symmetric_negative {
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
