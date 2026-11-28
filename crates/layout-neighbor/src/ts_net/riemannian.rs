use egraph_core::{CoreError, CoreResult, RiemannianLayout, Scalar};

use super::config::{MIN_LAMBDA_SUM, compute_stage_lr};
use super::solver::TsNetState;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

#[derive(Debug, Clone)]
pub(super) struct TsNetRiemannianScratch {
    dims: usize,
    attr: Vec<f64>,
    vel: Vec<f64>,
    rep: Vec<f64>,
    ent: Vec<f64>,
    log_buf: Vec<f64>,
    rf_buf: Vec<f64>,
    ef_buf: Vec<f64>,
    delta_buf: Vec<f64>,
    vel_buf: Vec<f64>,
}

fn ensure_riemannian_scratch<S: Scalar>(state: &mut TsNetState<S>, dims: usize) -> CoreResult<()> {
    if let Some(scratch) = &state.riemannian_scratch {
        if scratch.dims != dims {
            return Err(CoreError::dimension_mismatch(
                scratch.dims,
                dims,
                "TsNetSolver::riemannian::tangent_dim",
            ));
        }
        return Ok(());
    }
    let n = state.num_nodes;
    let total = n
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("RiemannianLayout scratch size overflow"))?;
    let any_entropy = state
        .config
        .stages
        .iter()
        .any(|s| s.iterations > 0 && s.lambda_r > 0.0);

    state.riemannian_scratch = Some(TsNetRiemannianScratch {
        dims,
        attr: vec![0.0; total],
        vel: vec![0.0; total],
        rep: vec![0.0; total],
        ent: vec![0.0; if any_entropy { total } else { 0 }],
        log_buf: vec![0.0; dims],
        rf_buf: vec![0.0; dims],
        ef_buf: vec![0.0; if any_entropy { dims } else { 0 }],
        delta_buf: vec![0.0; dims],
        vel_buf: vec![0.0; dims],
    });
    Ok(())
}

fn enter_current_stage_riemannian<S: Scalar>(state: &mut TsNetState<S>) {
    let stage = &state.config.stages[state.stage_idx];
    let base_lr = state.config.resolve_learning_rate(state.num_nodes);
    let stage_lr = compute_stage_lr(stage, state.num_nodes, base_lr, state.config.momentum);
    if stage_lr < state.prev_stage_lr {
        let vel_scale = stage_lr / state.prev_stage_lr;
        if let Some(scratch) = &mut state.riemannian_scratch {
            for v in &mut scratch.vel {
                *v *= vel_scale;
            }
        }
    }
    state.prev_stage_lr = stage_lr;
    state.active_stage_lr = stage_lr;
    state.stage_entered = true;
}

pub(super) fn step_riemannian<L: RiemannianLayout>(
    state: &mut TsNetState<L::Scalar>,
    layout: &mut L,
) -> CoreResult<()> {
    let dims = layout.tangent_dim();
    ensure_riemannian_scratch(state, dims)?;
    if !state.stage_entered {
        enter_current_stage_riemannian(state);
    }

    let stage = state.config.stages[state.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;
    let n = state.num_nodes;
    let eps_r = state.config.epsilon_r;

    {
        let scratch = state.riemannian_scratch.as_mut().ok_or_else(|| {
            CoreError::numerical_breakdown("TsNetRiemannianScratch workspace missing")
        })?;
        for i in 0..n {
            let i_base = i * dims;
            scratch.attr[i_base..i_base + dims].fill(0.0);
            let start = state.indptr[i] as usize;
            let end = state.indptr[i + 1] as usize;
            for pos in start..end {
                let j = state.indices[pos] as usize;
                if j == i {
                    continue;
                }
                layout.log_map_into(i, j, &mut scratch.log_buf)?;
                let mut dist_sq = 0.0_f64;
                for &v in &scratch.log_buf {
                    dist_sq += v * v;
                }
                let q_unnorm = 1.0 / (1.0 + dist_sq);
                let mult = (state.p_vals[pos] * stage.exaggeration) * q_unnorm;
                for d in 0..dims {
                    scratch.attr[i_base + d] += mult * (-scratch.log_buf[d]);
                }
            }
        }

        let mut total_z = 0.0_f64;
        for i in 0..n {
            let i_base = i * dims;
            let mut z_i = 0.0_f64;
            scratch.rf_buf.fill(0.0);
            if compute_entropy {
                scratch.ef_buf.fill(0.0);
            }

            for j in 0..n {
                if j == i {
                    continue;
                }
                layout.log_map_into(i, j, &mut scratch.log_buf)?;
                let mut dist_sq = 0.0_f64;
                for &v in &scratch.log_buf {
                    dist_sq += v * v;
                }
                let w = 1.0 / (1.0 + dist_sq);
                z_i += w;
                let w2 = w * w;
                let ent_mult = if compute_entropy {
                    let dist = dist_sq.sqrt();
                    if dist > EXACT_ENTROPY_MIN_DIST {
                        Some(1.0 / (dist * (dist + eps_r)))
                    } else {
                        None
                    }
                } else {
                    None
                };
                for d in 0..dims {
                    let dx = -scratch.log_buf[d];
                    scratch.rf_buf[d] += w2 * dx;
                    if let Some(mult) = ent_mult {
                        scratch.ef_buf[d] += mult * dx;
                    }
                }
            }

            total_z += z_i;
            for d in 0..dims {
                scratch.rep[i_base + d] = scratch.rf_buf[d];
                if compute_entropy {
                    scratch.ent[i_base + d] = scratch.ef_buf[d];
                }
            }
        }

        let denom = total_z.max(MIN_TOTAL_Z);
        for v in &mut scratch.rep {
            *v /= denom;
        }
    }

    let n_sq_f64 = (n as f64) * (n as f64);
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = state.active_stage_lr;
    let momentum = state.config.momentum;
    let use_jacobi = state.config.jacobi_precondition;

    let scratch = state.riemannian_scratch.as_mut().ok_or_else(|| {
        CoreError::numerical_breakdown("TsNetRiemannianScratch workspace missing")
    })?;

    for (i, &inv_j) in state.inv_jacobi.iter().enumerate().take(n) {
        let i_base = i * dims;
        let lr_i = if use_jacobi {
            stage_lr * inv_j
        } else {
            stage_lr
        };
        let mut any_nonzero = false;
        for d in 0..dims {
            let idx = i_base + d;
            let grad_kl = 4.0 * (scratch.attr[idx] - scratch.rep[idx]);
            let grad_r = if compute_entropy {
                -scratch.ent[idx] / n_sq_f64
            } else {
                0.0
            };
            let raw_g = (stage.lambda_kl * grad_kl + stage.lambda_r * grad_r) / lambda_sum;
            let g = if raw_g.is_finite() { raw_g } else { 0.0 };
            let mut v = momentum * scratch.vel[idx] - lr_i * g;
            if !v.is_finite() {
                v = 0.0;
            }
            scratch.vel[idx] = v;
            scratch.delta_buf[d] = v;
            scratch.vel_buf[d] = v;
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        if any_nonzero {
            layout.exp_map_and_transport(i, &scratch.delta_buf, &mut scratch.vel_buf)?;
            for d in 0..dims {
                let transported = scratch.vel_buf[d];
                scratch.vel[i_base + d] = if transported.is_finite() {
                    transported
                } else {
                    0.0
                };
            }
        }
    }
    Ok(())
}
