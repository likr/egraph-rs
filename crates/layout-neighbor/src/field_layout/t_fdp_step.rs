use egraph_core::{CoreError, CoreResult, Layout2d, LayoutNd, Scalar};

use super::config::{FieldFftStats, FieldRepulsionBackend};
use super::t_fdp::TFdpState;

#[derive(Debug, Clone)]
pub(super) struct TFdpNdScratch<S: Scalar> {
    attr: Vec<f64>,
    vel: Vec<f64>,
    next: Vec<f64>,
    rep: Vec<S>,
    mean: Vec<f64>,
}

pub(super) fn step_2d<S: Scalar>(
    state: &mut TFdpState<S>,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.iteration == 0 {
        let (xs, ys) = layout.xy_mut();
        if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
                ys[idx] = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let inv_alpha_sq = 1.0 / (state.config.alpha * state.config.alpha);
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let start = state.indptr[i] as usize;
            let end = state.indptr[i + 1] as usize;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            for pos in start..end {
                let j = state.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let d2 = dx * dx + dy * dy;
                let mult = (state.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                fx -= mult * dx;
                fy -= mult * dy;
            }
            state.attr_x[i] = fx;
            state.attr_y[i] = fy;
        }
    }

    let gamma = state.config.gamma;
    match state.config.backend {
        FieldRepulsionBackend::IbFft => {
            let ibfft = state.ibfft.as_mut().ok_or_else(|| {
                CoreError::numerical_breakdown("IbFft2d workspace missing for TFdpSolver")
            })?;
            let xs = layout.x();
            let ys = layout.y();
            let mut min_x = xs[0].to_f64();
            let mut max_x = min_x;
            let mut min_y = ys[0].to_f64();
            let mut max_y = min_y;
            for i in 1..n {
                let x = xs[i].to_f64();
                let y = ys[i].to_f64();
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            let span_cap = (ibfft.config().max_grid_dim as f64) * 0.4;
            let max_span = (max_x - min_x).max(max_y - min_y);
            let before_calls = state.fft_ws.fft_2d_call_count();
            let particle_fwd;
            if max_span > span_cap && max_span.is_finite() && span_cap > 0.0 {
                let cx = 0.5 * (min_x + max_x);
                let cy = 0.5 * (min_y + max_y);
                let scale = span_cap / max_span;
                let inv_scale = max_span / span_cap;
                let inv_scale_sq = inv_scale * inv_scale;
                for i in 0..n {
                    state.scaled_x[i] = S::from_f64((xs[i].to_f64() - cx) * scale);
                    state.scaled_y[i] = S::from_f64((ys[i].to_f64() - cy) * scale);
                }
                particle_fwd =
                    ibfft.prepare_xy(&mut state.fft_ws, &state.scaled_x, &state.scaled_y)?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                ibfft.evaluate_isotropic_kernel(
                    &mut state.fft_ws,
                    |d2| 1.0 / (1.0 + soft + d2 * inv_scale_sq).powf(gamma),
                    &mut state.phi_1,
                    &mut state.rep_x,
                    &mut state.rep_y,
                )?;
                for i in 0..n {
                    state.rep_x[i] = S::from_f64(state.rep_x[i].to_f64() * inv_scale);
                    state.rep_y[i] = S::from_f64(state.rep_y[i].to_f64() * inv_scale);
                }
            } else {
                particle_fwd = ibfft.prepare_xy(&mut state.fft_ws, xs, ys)?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                ibfft.evaluate_isotropic_kernel(
                    &mut state.fft_ws,
                    |d2| 1.0 / (1.0 + soft + d2).powf(gamma),
                    &mut state.phi_1,
                    &mut state.rep_x,
                    &mut state.rep_y,
                )?;
            }
            let total_calls = state
                .fft_ws
                .fft_2d_call_count()
                .saturating_sub(before_calls);
            let spec = ibfft.grid_spec();
            state.last_fft_stats = Some(FieldFftStats {
                particle_forward_ffts: particle_fwd,
                kernel_forward_ffts: u64::from(total_calls > particle_fwd),
                inverse_ffts: total_calls.saturating_sub(particle_fwd + 1),
                total_fft_2d_calls: total_calls,
                nx: spec.map_or(0, |s| s.nx),
                ny: spec.map_or(0, |s| s.ny),
            });
        }
        FieldRepulsionBackend::BarnesHut => {
            state.quadtree.build(layout)?;
            let theta_s = S::from_f64(state.config.theta);
            if (gamma - 1.0).abs() <= 1e-12 {
                state.quadtree.compute_coulomb_repulsion(
                    theta_s,
                    S::ONE,
                    &mut state.rep_x,
                    &mut state.rep_y,
                )?;
            } else if (gamma - 2.0).abs() <= 1e-12 {
                let total_z = state.quadtree.compute_kl_repulsion(
                    theta_s,
                    &mut state.rep_x,
                    &mut state.rep_y,
                )?;
                let z_f64 = total_z.to_f64();
                for i in 0..n {
                    state.rep_x[i] = S::from_f64(state.rep_x[i].to_f64() * z_f64);
                    state.rep_y[i] = S::from_f64(state.rep_y[i].to_f64() * z_f64);
                }
            } else {
                let q = 2.0 * gamma - 2.0;
                state.quadtree.compute_maxent_repulsion(
                    theta_s,
                    q,
                    S::ONE,
                    &mut state.rep_x,
                    &mut state.rep_y,
                )?;
            }
        }
        FieldRepulsionBackend::Exact => {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let d2 = dx * dx + dy * dy;
                    let k = 1.0 / (1.0 + d2).powf(gamma);
                    fx += k * dx;
                    fy += k * dy;
                }
                state.rep_x[i] = S::from_f64(fx);
                state.rep_y[i] = S::from_f64(fy);
            }
        }
    }

    let rho = state.current_rho() * state.rep_scale;
    let step_lr = state.config.learning_rate
        * (1.0 - (state.iteration as f64) / (state.total_iterations as f64));
    let momentum = state.config.momentum;
    let max_step = 5.0 * state.config.alpha.max(1.0) * state.config.learning_rate.max(1.0);

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let raw_fx = state.attr_x[i] + rho * state.rep_x[i].to_f64();
            let raw_fy = state.attr_y[i] + rho * state.rep_y[i].to_f64();
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

            let mut vx = momentum * state.vel_x[i] + step_lr * fx;
            let mut vy = momentum * state.vel_y[i] + step_lr * fy;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            let v_norm = vx.hypot(vy);
            if v_norm > max_step && v_norm > 0.0 {
                let scale = max_step / v_norm;
                vx *= scale;
                vy *= scale;
            }
            state.vel_x[i] = vx;
            state.vel_y[i] = vy;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            state.next_x[i] = nx_val;
            state.next_y[i] = ny_val;
            mean_x += nx_val;
            mean_y += ny_val;
        }
    }

    let inv_n = 1.0 / (n as f64);
    mean_x *= inv_n;
    mean_y *= inv_n;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let cx = S::from_f64(state.next_x[i] - shift_x);
        let cy = S::from_f64(state.next_y[i] - shift_y);
        if cx.is_finite() {
            xs_mut[i] = cx;
        }
        if cy.is_finite() {
            ys_mut[i] = cy;
        }
    }
    Ok(())
}

pub(super) fn step_nd<S: Scalar>(
    state: &mut TFdpState<S>,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let dims = layout.dims();
    if state.locked_nd_dims.is_none() {
        state.locked_nd_dims = Some(dims);
    }

    let total = n
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
    if state.nd_scratch.is_none() {
        state.nd_scratch = Some(TFdpNdScratch {
            attr: vec![0.0; total],
            vel: vec![0.0; total],
            next: vec![0.0; total],
            rep: vec![S::ZERO; total],
            mean: vec![0.0; dims],
        });
    }

    if state.iteration == 0 {
        let coords = layout.coords_mut();
        if coords.iter().all(|&v| v == S::ZERO) {
            for v in coords.iter_mut() {
                *v = S::from_f64(state.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let rho = state.current_rho() * state.rep_scale;
    let step_lr = state.config.learning_rate
        * (1.0 - (state.iteration as f64) / (state.total_iterations as f64));
    let momentum = state.config.momentum;
    let max_step = 5.0 * state.config.alpha.max(1.0) * state.config.learning_rate.max(1.0);
    let inv_alpha_sq = 1.0 / (state.config.alpha * state.config.alpha);
    let gamma = state.config.gamma;

    let scratch = state
        .nd_scratch
        .as_mut()
        .ok_or_else(|| CoreError::numerical_breakdown("TFdpNdScratch workspace missing"))?;

    {
        let coords = layout.coords();
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
                let mut d2 = 0.0_f64;
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    d2 += diff * diff;
                }
                let mult = (state.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    scratch.attr[i_base + d] -= mult * diff;
                }
            }

            scratch.rep[i_base..i_base + dims].fill(S::ZERO);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let j_base = j * dims;
                let mut d2 = 0.0_f64;
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    d2 += diff * diff;
                }
                let k = 1.0 / (1.0 + d2).powf(gamma);
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    let cur = scratch.rep[i_base + d].to_f64();
                    scratch.rep[i_base + d] = S::from_f64(cur + k * diff);
                }
            }
        }
    }

    scratch.mean.fill(0.0);
    {
        let coords = layout.coords();
        for i in 0..n {
            let i_base = i * dims;
            let mut v_sq = 0.0_f64;
            for d in 0..dims {
                let idx = i_base + d;
                let raw_f = scratch.attr[idx] + rho * scratch.rep[idx].to_f64();
                let f = if raw_f.is_finite() { raw_f } else { 0.0 };
                let mut v = momentum * scratch.vel[idx] + step_lr * f;
                if !v.is_finite() {
                    v = 0.0;
                }
                scratch.vel[idx] = v;
                v_sq += v * v;
            }
            let v_norm = v_sq.sqrt();
            if v_norm > max_step && v_norm > 0.0 {
                let scale = max_step / v_norm;
                for d in 0..dims {
                    scratch.vel[i_base + d] *= scale;
                }
            }
            for d in 0..dims {
                let idx = i_base + d;
                let xi = coords[idx].to_f64();
                let cand = xi + scratch.vel[idx];
                let nx = if cand.is_finite() { cand } else { xi };
                scratch.next[idx] = nx;
                scratch.mean[d] += nx;
            }
        }
    }

    let inv_n = 1.0 / (n as f64);
    for mean in &mut scratch.mean {
        let m = *mean * inv_n;
        *mean = if m.is_finite() { m } else { 0.0 };
    }

    let coords_mut = layout.coords_mut();
    for i in 0..n {
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let centered = S::from_f64(scratch.next[idx] - scratch.mean[d]);
            if centered.is_finite() {
                coords_mut[idx] = centered;
            }
        }
    }
    Ok(())
}
