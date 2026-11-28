use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, Scalar};

use super::solver::{COULOMB_EPS_SQ, LinLogState, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

pub(super) fn step_2d<S: Scalar>(
    state: &mut LinLogState,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    {
        let xs = layout.x();
        let ys = layout.y();
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        if state.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for i in 0..n {
                state.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                state.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
            }
        }
    }

    let r_exp = state.config.repulsion_exponent;
    let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;

    if state.config.use_barnes_hut {
        if is_coulomb {
            state
                .quadtree
                .build_weighted_xy(&state.x_buf, &state.y_buf, &state.masses)?;
            state.quadtree.compute_coulomb_repulsion(
                state.config.theta,
                COULOMB_EPS_SQ,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
        } else {
            state
                .gen_quadtree
                .build(&state.x_buf, &state.y_buf, &state.masses);
            state.gen_quadtree.compute_general_repulsion(
                &state.x_buf,
                &state.y_buf,
                &state.masses,
                state.config.theta,
                r_exp,
                COULOMB_EPS_SQ,
                &mut state.rep_x,
                &mut state.rep_y,
            );
        }
        for i in 0..n {
            let mi = state.masses[i];
            state.fx[i] = mi * state.rep_x[i];
            state.fy[i] = mi * state.rep_y[i];
        }
    } else {
        state.fx.fill(0.0);
        state.fy.fill(0.0);
        let pow_exp = 0.5 * (r_exp - 1.0);
        for i in 0..n {
            let xi = state.x_buf[i];
            let yi = state.y_buf[i];
            let mi = state.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - state.x_buf[j];
                let dy = yi - state.y_buf[j];
                let d_sq_reg = dx * dx + dy * dy + COULOMB_EPS_SQ;
                let mult = if is_coulomb {
                    (mi * state.masses[j]) / d_sq_reg
                } else {
                    (mi * state.masses[j]) * d_sq_reg.powf(pow_exp)
                };
                let fx_ij = mult * dx;
                let fy_ij = mult * dy;
                acc_x += fx_ij;
                acc_y += fy_ij;
                state.fx[j] -= fx_ij;
                state.fy[j] -= fy_ij;
            }
            state.fx[i] += acc_x;
            state.fy[i] += acc_y;
        }
    }

    let a_exp = state.config.attraction_exponent;
    let a_is_zero = a_exp.abs() <= 1e-12;
    let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
    let a_hess_scale = a_exp.abs().max(1.0);
    state.att_hess.fill(0.0);

    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = state.x_buf[v] - state.x_buf[u];
        let dy = state.y_buf[v] - state.y_buf[u];
        let d_sq = dx * dx + dy * dy;
        let (factor, hess) = if a_is_zero {
            if d_sq > 1e-24 {
                let d = d_sq.sqrt();
                (w / d, w / d)
            } else {
                (0.0, 0.0)
            }
        } else if a_is_one {
            (w, w)
        } else if a_exp > 0.0 && d_sq <= 1e-24 {
            (0.0, 0.0)
        } else {
            let d_reg = (d_sq + 1e-24).sqrt();
            let f = w * d_reg.powf(a_exp - 1.0);
            (f, a_hess_scale * f)
        };

        let ax = factor * dx;
        let ay = factor * dy;
        state.fx[u] += ax;
        state.fy[u] += ay;
        state.fx[v] -= ax;
        state.fy[v] -= ay;
        state.att_hess[u] += hess;
        state.att_hess[v] += hess;
    }

    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut bx = 0.0_f64;
        let mut by = 0.0_f64;
        for i in 0..n {
            bx += state.masses[i] * state.x_buf[i];
            by += state.masses[i] * state.y_buf[i];
        }
        bx /= state.total_mass;
        by /= state.total_mass;

        let g = state.config.gravity;
        for i in 0..n {
            let dx = bx - state.x_buf[i];
            let dy = by - state.y_buf[i];
            let d = dx.hypot(dy);
            if d > 1e-12 {
                let factor = if a_is_zero {
                    (g * state.masses[i]) / d
                } else if a_is_one {
                    g * state.masses[i]
                } else {
                    g * state.masses[i] * d.powf(a_exp - 1.0)
                };
                state.fx[i] += factor * dx;
                state.fy[i] += factor * dy;
                state.att_hess[i] += a_hess_scale * factor;
            }
        }
    }

    let mut total_swing = 0.0_f64;
    let mut total_traction = 0.0_f64;
    for i in 0..n {
        let fxi = if state.fx[i].is_finite() {
            state.fx[i]
        } else {
            0.0
        };
        let fyi = if state.fy[i].is_finite() {
            state.fy[i]
        } else {
            0.0
        };
        state.fx[i] = fxi;
        state.fy[i] = fyi;

        let swg = (fxi - state.prev_fx[i]).hypot(fyi - state.prev_fy[i]);
        let tra = 0.5 * (fxi + state.prev_fx[i]).hypot(fyi + state.prev_fy[i]);
        state.swing_buf[i] = swg;
        let m = state.masses[i];
        total_swing += m * swg;
        total_traction += m * tra;
    }

    let target_speed = if total_swing > 1e-12 {
        (total_traction / total_swing).clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED)
    } else {
        (state.global_speed * 1.5).min(MAX_LINLOG_SPEED)
    };
    if state.iter > 0 {
        let max_rise = 0.5 * state.global_speed;
        state.global_speed += (target_speed - state.global_speed).min(max_rise);
    } else {
        state.global_speed = target_speed;
    }
    state.global_speed = state.global_speed.clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED);

    let speed = state.global_speed;
    let max_disp = state.max_displacement;
    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let fxi = state.fx[i];
        let fyi = state.fy[i];
        let local_factor =
            speed / (1.0 + (speed * state.swing_buf[i]).sqrt() + 0.5 * state.att_hess[i]);
        let mut dx = fxi * local_factor;
        let mut dy = fyi * local_factor;
        let disp = dx.hypot(dy);
        if disp > max_disp {
            let cap = max_disp / disp;
            dx *= cap;
            dy *= cap;
        }
        let nx = S::from_f64(state.x_buf[i] + dx);
        let ny = S::from_f64(state.y_buf[i] + dy);
        if !nx.is_finite() || !ny.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "LinLogSolver 2D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        state.prev_fx[i] = fxi;
        state.prev_fy[i] = fyi;
    }

    Ok(())
}
