use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_spatial_fft::{FitField2d, FitFieldConfig};

use super::config::{MIN_LAMBDA_SUM, TsNetBackend, compute_stage_lr};
use super::solver::TsNetState;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

fn enter_current_stage_2d<S: Scalar>(state: &mut TsNetState<S>) -> CoreResult<()> {
    let stage = &state.config.stages[state.stage_idx];
    let base_lr = state.config.resolve_learning_rate(state.num_nodes);
    let stage_lr = compute_stage_lr(stage, state.num_nodes, base_lr, state.config.momentum);
    if stage_lr < state.prev_stage_lr {
        let vel_scale = stage_lr / state.prev_stage_lr;
        for v in &mut state.vel_x {
            *v *= vel_scale;
        }
        for v in &mut state.vel_y {
            *v *= vel_scale;
        }
    }
    state.prev_stage_lr = stage_lr;
    state.active_stage_lr = stage_lr;

    let need_entropy = stage.lambda_r > 0.0;
    if matches!(
        state.resolved_backend,
        TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
    ) {
        let needs_rebuild = match &state.fit_field {
            Some(ff) => ff.config().compute_entropy != need_entropy,
            None => true,
        };
        if needs_rebuild {
            state.fit_field = Some(FitField2d::new(
                FitFieldConfig::default()
                    .with_ibfft(state.config.effective_ibfft_for(state.resolved_backend))
                    .with_epsilon_r(state.config.epsilon_r)
                    .with_compute_entropy(need_entropy),
            )?);
        }
    }

    state.stage_entered = true;
    Ok(())
}

fn compute_attractions_2d<S: Scalar>(
    state: &mut TsNetState<S>,
    xs: &[S],
    ys: &[S],
    exaggeration: f64,
) {
    let n = state.num_nodes;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let start = state.indptr[i] as usize;
        let end = state.indptr[i + 1] as usize;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        for pos in start..end {
            let j = state.indices[pos] as usize;
            if j == i {
                continue;
            }
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dist_sq = dx * dx + dy * dy;
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (state.p_vals[pos] * exaggeration) * q_unnorm;
            ax += mult * dx;
            ay += mult * dy;
        }
        state.attr_x[i] = ax;
        state.attr_y[i] = ay;
    }
}

fn compute_exact_repulsion_2d<S: Scalar>(
    state: &mut TsNetState<S>,
    xs: &[S],
    ys: &[S],
    compute_entropy: bool,
) {
    let n = state.num_nodes;
    let eps_r = state.config.epsilon_r;
    let mut total_z = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let mut z_i = 0.0_f64;
        let mut rfx = 0.0_f64;
        let mut rfy = 0.0_f64;
        let mut efx = 0.0_f64;
        let mut efy = 0.0_f64;

        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dist_sq = dx * dx + dy * dy;
            let w = 1.0 / (1.0 + dist_sq);
            z_i += w;
            let w2 = w * w;
            rfx += w2 * dx;
            rfy += w2 * dy;
            if compute_entropy {
                let dist = dist_sq.sqrt();
                if dist > EXACT_ENTROPY_MIN_DIST {
                    let mult = 1.0 / (dist * (dist + eps_r));
                    efx += mult * dx;
                    efy += mult * dy;
                }
            }
        }
        total_z += z_i;
        state.rep_x[i] = S::from_f64(rfx);
        state.rep_y[i] = S::from_f64(rfy);
        if compute_entropy {
            state.ent_x[i] = S::from_f64(efx);
            state.ent_y[i] = S::from_f64(efy);
        }
    }

    let denom = total_z.max(MIN_TOTAL_Z);
    for i in 0..n {
        state.rep_x[i] = S::from_f64(state.rep_x[i].to_f64() / denom);
        state.rep_y[i] = S::from_f64(state.rep_y[i].to_f64() / denom);
    }
}

fn compute_barnes_hut_repulsion_2d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &Layout2d<S>,
    compute_entropy: bool,
) -> CoreResult<()> {
    state.quadtree.build(layout)?;
    let theta_s = S::from_f64(state.config.theta);
    state
        .quadtree
        .compute_kl_repulsion(theta_s, &mut state.rep_x, &mut state.rep_y)?;
    if compute_entropy {
        let eps_s = S::from_f64(state.config.epsilon_r).max(S::MIN_POSITIVE);
        state.quadtree.compute_entropy_repulsion(
            theta_s,
            eps_s,
            &mut state.ent_x,
            &mut state.ent_y,
        )?;
    }
    Ok(())
}

pub(super) fn step_2d<S: Scalar>(
    state: &mut TsNetState<S>,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    if !state.stage_entered {
        enter_current_stage_2d(state)?;
    }

    let stage = state.config.stages[state.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    compute_attractions_2d(state, layout.x(), layout.y(), stage.exaggeration);

    match state.resolved_backend {
        TsNetBackend::Exact => {
            compute_exact_repulsion_2d(state, layout.x(), layout.y(), compute_entropy);
        }
        TsNetBackend::BarnesHut => {
            compute_barnes_hut_repulsion_2d(state, layout, compute_entropy)?;
        }
        TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
            let fit_field = state
                .fit_field
                .as_mut()
                .ok_or_else(|| CoreError::numerical_breakdown("FitField2d workspace missing"))?;
            fit_field.evaluate(layout, &mut state.fit_out)?;
            state.last_fft_stats = Some(state.fit_out.stats.clone());
            state.total_fft_2d_calls += state.fit_out.stats.total_fft_2d_calls;
        }
        TsNetBackend::Auto => {
            return Err(CoreError::numerical_breakdown(
                "TsNetBackend::Auto was not resolved at construction",
            ));
        }
    }

    let n = state.num_nodes;
    let n_f64 = n as f64;
    let n_sq_f64 = n_f64 * n_f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = state.active_stage_lr;
    let momentum = state.config.momentum;
    let use_jacobi = state.config.jacobi_precondition;

    let use_fft_buffers = matches!(
        state.resolved_backend,
        TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
    );

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        let (rep_x_slice, rep_y_slice) = if use_fft_buffers {
            (
                state.fit_out.kl_repulsion_x.as_slice(),
                state.fit_out.kl_repulsion_y.as_slice(),
            )
        } else {
            (state.rep_x.as_slice(), state.rep_y.as_slice())
        };
        let (ent_x_slice, ent_y_slice) = if compute_entropy {
            if use_fft_buffers {
                (
                    state.fit_out.entropy_repulsion_x.as_slice(),
                    state.fit_out.entropy_repulsion_y.as_slice(),
                )
            } else {
                (state.ent_x.as_slice(), state.ent_y.as_slice())
            }
        } else {
            (&[][..], &[][..])
        };

        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();

            let grad_kl_x = 4.0 * (state.attr_x[i] - rep_x_slice[i].to_f64());
            let grad_kl_y = 4.0 * (state.attr_y[i] - rep_y_slice[i].to_f64());

            let grad_c_x = xi / n_f64;
            let grad_c_y = yi / n_f64;

            let (grad_r_x, grad_r_y) = if compute_entropy {
                (
                    -ent_x_slice[i].to_f64() / n_sq_f64,
                    -ent_y_slice[i].to_f64() / n_sq_f64,
                )
            } else {
                (0.0, 0.0)
            };

            let raw_gx = (stage.lambda_kl * grad_kl_x
                + stage.lambda_c * grad_c_x
                + stage.lambda_r * grad_r_x)
                / lambda_sum;
            let raw_gy = (stage.lambda_kl * grad_kl_y
                + stage.lambda_c * grad_c_y
                + stage.lambda_r * grad_r_y)
                / lambda_sum;
            let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
            let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };

            let lr_i = if use_jacobi {
                stage_lr * state.inv_jacobi[i]
            } else {
                stage_lr
            };
            let mut vx = momentum * state.vel_x[i] - lr_i * gx;
            let mut vy = momentum * state.vel_y[i] - lr_i * gy;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            state.vel_x[i] = vx;
            state.vel_y[i] = vy;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let new_x = if cand_x.is_finite() { cand_x } else { xi };
            let new_y = if cand_y.is_finite() { cand_y } else { yi };

            // Reuse `attr_x` / `attr_y` as temporary `f64` coordinate buffers before recentering.
            state.attr_x[i] = new_x;
            state.attr_y[i] = new_y;
            mean_x += new_x;
            mean_y += new_y;
        }
    }

    mean_x /= n_f64;
    mean_y /= n_f64;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let cx = S::from_f64(state.attr_x[i] - shift_x);
        let cy = S::from_f64(state.attr_y[i] - shift_y);
        if cx.is_finite() {
            xs_mut[i] = cx;
        }
        if cy.is_finite() {
            ys_mut[i] = cy;
        }
    }

    Ok(())
}
