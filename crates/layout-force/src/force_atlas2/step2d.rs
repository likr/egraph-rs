use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, Scalar};

use super::overlap_quadtree::overlap_repulsion_pair;
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

pub(super) fn step_2d<S: Scalar>(
    state: &mut ForceAtlas2State,
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

        // Break completely collapsed initial layouts deterministically so forces are non-zero.
        if state.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for i in 0..n {
                state.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                state.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
            }
        }
    }

    // 1. Repulsion: Jacomy (2014) border-to-border `d' = d - r1 - r2` when `prevent_overlap`,
    //    otherwise standard O(N log N) weighted Barnes-Hut or O(N^2) exact.
    let prevent_overlap = state.config.prevent_overlap;
    if prevent_overlap {
        if state.config.use_barnes_hut {
            state
                .overlap_tree
                .build(&state.x_buf, &state.y_buf, &state.masses, &state.node_radii);
            state.overlap_tree.compute_repulsion(
                &state.x_buf,
                &state.y_buf,
                &state.masses,
                &state.node_radii,
                state.config.theta,
                state.config.scaling_ratio,
                state.config.seed,
                state.iter,
                &mut state.fx,
                &mut state.fy,
            );
        } else {
            state.fx.fill(0.0);
            state.fy.fill(0.0);
            let sr = state.config.scaling_ratio;
            let seed = state.config.seed;
            let iter = state.iter;
            for i in 0..n {
                let xi = state.x_buf[i];
                let yi = state.y_buf[i];
                let mi = state.masses[i];
                let ri = state.node_radii[i];
                let mut acc_x = 0.0_f64;
                let mut acc_y = 0.0_f64;
                for j in (i + 1)..n {
                    let (fx_ij, fy_ij) = overlap_repulsion_pair(
                        xi - state.x_buf[j],
                        yi - state.y_buf[j],
                        mi,
                        state.masses[j],
                        ri + state.node_radii[j],
                        sr,
                        seed,
                        iter,
                        i,
                        j,
                    );
                    acc_x += fx_ij;
                    acc_y += fy_ij;
                    state.fx[j] -= fx_ij;
                    state.fy[j] -= fy_ij;
                }
                state.fx[i] += acc_x;
                state.fy[i] += acc_y;
            }
        }
    } else if state.config.use_barnes_hut {
        state
            .quadtree
            .build_weighted_xy(&state.x_buf, &state.y_buf, &state.masses)?;
        state.quadtree.compute_coulomb_repulsion(
            state.config.theta,
            COULOMB_EPS_SQ,
            &mut state.rep_x,
            &mut state.rep_y,
        )?;
        let sr = state.config.scaling_ratio;
        for i in 0..n {
            let scale = sr * state.masses[i];
            state.fx[i] = scale * state.rep_x[i];
            state.fy[i] = scale * state.rep_y[i];
        }
    } else {
        state.fx.fill(0.0);
        state.fy.fill(0.0);
        let sr = state.config.scaling_ratio;
        for i in 0..n {
            let xi = state.x_buf[i];
            let yi = state.y_buf[i];
            let mi_sr = sr * state.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - state.x_buf[j];
                let dy = yi - state.y_buf[j];
                let denom = dx * dx + dy * dy + COULOMB_EPS_SQ;
                let mult = (mi_sr * state.masses[j]) / denom;
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

    // 2. Attraction: linear or LinLog along canonical undirected edges.
    let linlog = state.config.linlog_mode;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = state.x_buf[v] - state.x_buf[u];
        let dy = state.y_buf[v] - state.y_buf[u];
        let factor = if prevent_overlap {
            let d = dx.hypot(dy);
            let d_prime = d - (state.node_radii[u] + state.node_radii[v]);
            if d_prime > 0.0 && d > 1e-12 {
                if linlog {
                    w * d_prime.ln_1p() / d
                } else {
                    w * d_prime / d
                }
            } else {
                0.0
            }
        } else if linlog {
            let d_sq = dx * dx + dy * dy;
            if d_sq > 1e-24 {
                let d = d_sq.sqrt();
                w * d.ln_1p() / d
            } else {
                0.0
            }
        } else {
            w
        };
        if factor == 0.0 {
            continue;
        }
        let ax = factor * dx;
        let ay = factor * dy;
        state.fx[u] += ax;
        state.fy[u] += ay;
        state.fx[v] -= ax;
        state.fy[v] -= ay;
    }

    // 3. Gravity: standard (constant pull m_i * g) or strong (m_i * g * d).
    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for i in 0..n {
                let gf = g * state.masses[i];
                state.fx[i] -= gf * state.x_buf[i];
                state.fy[i] -= gf * state.y_buf[i];
            }
        } else {
            for i in 0..n {
                let d = state.x_buf[i].hypot(state.y_buf[i]);
                if d > 1e-12 {
                    let gf = (g * state.masses[i]) / d;
                    state.fx[i] -= gf * state.x_buf[i];
                    state.fy[i] -= gf * state.y_buf[i];
                }
            }
        }
    }

    // 4. Adaptive speed from per-node swing and traction.
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

    let jt = state.config.jitter_tolerance;
    let max_speed = 10.0 * jt.max(1.0);
    let target_speed = if total_swing > 1e-12 {
        (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
    } else {
        (state.global_speed * 1.5).min(max_speed)
    };
    if state.iter > 0 {
        let max_rise = 0.5 * state.global_speed;
        state.global_speed += (target_speed - state.global_speed).min(max_rise);
    } else {
        state.global_speed = target_speed;
    }
    state.global_speed = state.global_speed.clamp(MIN_FA2_SPEED, max_speed);

    // 5. Apply adaptive displacements back into layout.
    let speed = state.global_speed;
    let speed_scale = if prevent_overlap { 0.1 } else { 1.0 };
    let max_disp = state.max_displacement;
    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let fxi = state.fx[i];
        let fyi = state.fy[i];
        let local_factor = speed_scale * (speed / (1.0 + (speed * state.swing_buf[i]).sqrt()));
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
                "ForceAtlas2Solver 2D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        state.prev_fx[i] = fxi;
        state.prev_fy[i] = fyi;
    }

    Ok(())
}
