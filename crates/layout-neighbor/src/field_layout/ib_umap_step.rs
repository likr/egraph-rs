use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::config::{FieldFftStats, FieldRepulsionBackend};
use super::ib_umap::IbUmapSolver;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 0.05;

pub(super) fn step_inner<S: Scalar>(
    solver: &mut IbUmapSolver<S>,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    let n = solver.num_nodes;
    if solver.iteration == 0 {
        let (xs, ys) = layout.xy_mut();
        if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(solver.rng.next_f64() * 20.0 - 10.0);
                ys[idx] = S::from_f64(solver.rng.next_f64() * 20.0 - 10.0);
            }
        }
    }

    let two_ab = 2.0 * solver.a * solver.b;
    let a = solver.a;
    let b = solver.b;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let start = solver.indptr[i] as usize;
            let end = solver.indptr[i + 1] as usize;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            for pos in start..end {
                let j = solver.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let p_ij = solver.edge_weights[pos] as f64;
                if p_ij <= 0.0 {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let d2 = (dx * dx + dy * dy).max(MIN_DIST_SQ_CLAMP);
                let d_2b = d2.powf(b);
                let d_pow = d_2b / d2;
                let coeff = -two_ab * p_ij * d_pow / (1.0 + a * d_2b);
                if coeff.is_finite() {
                    fx += coeff * dx;
                    fy += coeff * dy;
                }
            }
            solver.attr_x[i] = fx;
            solver.attr_y[i] = fy;
        }
    }

    let two_b_ck = 2.0 * b * solver.config.c_k;
    match solver.config.backend {
        FieldRepulsionBackend::IbFft => {
            let ibfft = solver.ibfft.as_mut().ok_or_else(|| {
                CoreError::numerical_breakdown("IbFft2d workspace missing for IbUmapSolver")
            })?;
            let before_calls = solver.fft_ws.fft_2d_call_count();
            let particle_fwd = ibfft.prepare_xy(&mut solver.fft_ws, layout.x(), layout.y())?;
            let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
            let eps_eff = REPULSIVE_DIST_SQ_EPS + soft;
            ibfft.evaluate_isotropic_kernel(
                &mut solver.fft_ws,
                |d2| {
                    let d2_pos = d2.max(0.0);
                    two_b_ck / ((eps_eff + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                },
                &mut solver.phi_1,
                &mut solver.rep_x,
                &mut solver.rep_y,
            )?;
            let total_calls = solver
                .fft_ws
                .fft_2d_call_count()
                .saturating_sub(before_calls);
            let spec = ibfft.grid_spec();
            solver.last_fft_stats = Some(FieldFftStats {
                particle_forward_ffts: particle_fwd,
                kernel_forward_ffts: u64::from(total_calls > particle_fwd),
                inverse_ffts: total_calls.saturating_sub(particle_fwd + 1),
                total_fft_2d_calls: total_calls,
                nx: spec.map_or(0, |s| s.nx),
                ny: spec.map_or(0, |s| s.ny),
            });
        }
        FieldRepulsionBackend::BarnesHut => {
            solver.field_quadtree.build_xy(layout.x(), layout.y())?;
            solver.field_quadtree.compute_isotropic_repulsion(
                solver.config.theta,
                |d2| {
                    let d2_pos = d2.max(0.0);
                    two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                },
                &mut solver.rep_x,
                &mut solver.rep_y,
            );
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
                    let k = two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2) * (1.0 + a * d2.powf(b)));
                    fx += k * dx;
                    fy += k * dy;
                }
                solver.rep_x[i] = S::from_f64(fx);
                solver.rep_y[i] = S::from_f64(fy);
            }
        }
    }

    let alpha_t = solver.config.learning_rate
        * (1.0 - (solver.iteration as f64) / (solver.total_iterations as f64));
    let tau_t = solver.config.tau_scale * alpha_t;
    let inv_n = 1.0 / (n as f64);

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let raw_fx = solver.attr_x[i] + solver.rep_x[i].to_f64() * inv_n;
            let raw_fy = solver.attr_y[i] + solver.rep_y[i].to_f64() * inv_n;
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

            let step_scale = alpha_t * solver.hub_eta[i];
            let mut dx = step_scale * fx;
            let mut dy = step_scale * fy;
            if !dx.is_finite() {
                dx = 0.0;
            }
            if !dy.is_finite() {
                dy = 0.0;
            }
            let norm = dx.hypot(dy);
            if norm > tau_t && norm > 0.0 {
                let clip = tau_t / norm;
                dx *= clip;
                dy *= clip;
            }

            let cand_x = xi + dx;
            let cand_y = yi + dy;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            solver.next_x[i] = nx_val;
            solver.next_y[i] = ny_val;
            mean_x += nx_val;
            mean_y += ny_val;
        }
    }

    mean_x *= inv_n;
    mean_y *= inv_n;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let cx = S::from_f64(solver.next_x[i] - shift_x);
        let cy = S::from_f64(solver.next_y[i] - shift_y);
        if cx.is_finite() {
            xs_mut[i] = cx;
        }
        if cy.is_finite() {
            ys_mut[i] = cy;
        }
    }
    Ok(())
}
