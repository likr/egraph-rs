use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};

use super::config::{MIN_LAMBDA_SUM, TsNetStage, compute_stage_lr};
use super::solver::TsNetState;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

/// Lazily allocated exact N-D scratch buffers for [`TsNetSolver`](super::TsNetSolver) on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct TsNetNdScratch {
    attr: Vec<f64>,
    vel: Vec<f64>,
    rep: Vec<f64>,
    ent: Vec<f64>,
    mean: Vec<f64>,
    rf_buf: Vec<f64>,
    ef_buf: Vec<f64>,
}

fn ensure_nd_scratch<S: Scalar>(state: &mut TsNetState<S>, dims: usize) -> CoreResult<()> {
    if state.nd_scratch.is_some() {
        return Ok(());
    }
    let n = state.num_nodes;
    let total = n
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
    let any_entropy = state
        .config
        .stages
        .iter()
        .any(|s| s.iterations > 0 && s.lambda_r > 0.0);

    state.nd_scratch = Some(TsNetNdScratch {
        attr: vec![0.0; total],
        vel: vec![0.0; total],
        rep: vec![0.0; total],
        ent: vec![0.0; if any_entropy { total } else { 0 }],
        mean: vec![0.0; dims],
        rf_buf: vec![0.0; dims],
        ef_buf: vec![0.0; if any_entropy { dims } else { 0 }],
    });
    Ok(())
}

fn enter_current_stage_nd<S: Scalar>(state: &mut TsNetState<S>) {
    let stage = &state.config.stages[state.stage_idx];
    let base_lr = state.config.resolve_learning_rate(state.num_nodes);
    let stage_lr = compute_stage_lr(stage, state.num_nodes, base_lr, state.config.momentum);
    if stage_lr < state.prev_stage_lr {
        let vel_scale = stage_lr / state.prev_stage_lr;
        if let Some(scratch) = &mut state.nd_scratch {
            for v in &mut scratch.vel {
                *v *= vel_scale;
            }
        }
    }
    state.prev_stage_lr = stage_lr;
    state.active_stage_lr = stage_lr;
    state.stage_entered = true;
}

fn compute_attractions_nd<S: Scalar>(
    state: &mut TsNetState<S>,
    coords: &[S],
    dims: usize,
    exaggeration: f64,
) {
    let Some(scratch) = &mut state.nd_scratch else {
        return;
    };
    let n = state.num_nodes;
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
            let j_base = j * dims;
            let mut dist_sq = 0.0_f64;
            for d in 0..dims {
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                dist_sq += dx * dx;
            }
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (state.p_vals[pos] * exaggeration) * q_unnorm;
            for d in 0..dims {
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                scratch.attr[i_base + d] += mult * dx;
            }
        }
    }
}

fn compute_exact_repulsion_nd<S: Scalar>(
    state: &mut TsNetState<S>,
    coords: &[S],
    dims: usize,
    compute_entropy: bool,
) {
    let Some(scratch) = &mut state.nd_scratch else {
        return;
    };
    let n = state.num_nodes;
    let eps_r = state.config.epsilon_r;
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
            let j_base = j * dims;
            let mut dist_sq = 0.0_f64;
            for d in 0..dims {
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                dist_sq += dx * dx;
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
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
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

fn apply_nd_update<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &mut LayoutNd<S>,
    dims: usize,
    stage: &TsNetStage,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let n_f64 = n as f64;
    let n_sq_f64 = n_f64 * n_f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = state.active_stage_lr;
    let momentum = state.config.momentum;
    let use_jacobi = state.config.jacobi_precondition;
    let compute_entropy = stage.lambda_r > 0.0;

    let scratch = state
        .nd_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetNdScratch workspace missing"))?;
    scratch.mean.fill(0.0);

    let coords = layout.coords_mut();
    for (i, &inv_j) in state.inv_jacobi.iter().enumerate().take(n) {
        let lr_i = if use_jacobi {
            stage_lr * inv_j
        } else {
            stage_lr
        };
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let x_d = coords[idx].to_f64();
            let grad_kl = 4.0 * (scratch.attr[idx] - scratch.rep[idx]);
            let grad_c = x_d / n_f64;
            let grad_r = if compute_entropy {
                -scratch.ent[idx] / n_sq_f64
            } else {
                0.0
            };

            let raw_g =
                (stage.lambda_kl * grad_kl + stage.lambda_c * grad_c + stage.lambda_r * grad_r)
                    / lambda_sum;
            let g = if raw_g.is_finite() { raw_g } else { 0.0 };

            let mut v = momentum * scratch.vel[idx] - lr_i * g;
            if !v.is_finite() {
                v = 0.0;
            }
            scratch.vel[idx] = v;

            let cand = x_d + v;
            let new_val = if cand.is_finite() { cand } else { x_d };
            scratch.attr[idx] = new_val;
            scratch.mean[d] += new_val;
        }
    }

    for mean in &mut scratch.mean {
        let m = *mean / n_f64;
        *mean = if m.is_finite() { m } else { 0.0 };
    }

    for i in 0..n {
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let centered = S::from_f64(scratch.attr[idx] - scratch.mean[d]);
            if centered.is_finite() {
                coords[idx] = centered;
            }
        }
    }
    Ok(())
}

pub(super) fn step_nd<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    ensure_nd_scratch(state, dims)?;

    if !state.stage_entered {
        enter_current_stage_nd(state);
    }

    let stage = state.config.stages[state.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    compute_attractions_nd(state, layout.coords(), dims, stage.exaggeration);
    compute_exact_repulsion_nd(state, layout.coords(), dims, compute_entropy);
    apply_nd_update(state, layout, dims, &stage)
}
