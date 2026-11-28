use egraph_core::{CoreError, CoreResult, Layout2d, Scalar, SolverStatus};

use super::solver::{PROLONGATION_JITTER_FACTOR, YifanHuState};

fn initialize_coarsest_coords_2d<S: Scalar>(state: &mut YifanHuState, layout: &Layout2d<S>) {
    let n = state.num_nodes;
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        state.active_x[i] = x;
        state.active_y[i] = y;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let coarsest = state.levels.len() - 1;
    let coarse_n = state.levels[coarsest].graph.num_nodes();
    if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
        let k_coarse = state.levels[coarsest].optimal_distance;
        let spread = k_coarse * (coarse_n as f64).sqrt().max(1.0);
        for i in 0..coarse_n {
            state.active_x[i] = spread * (state.rng.next_f64() - 0.5);
            state.active_y[i] = spread * (state.rng.next_f64() - 0.5);
        }
    } else if coarsest > 0 {
        let map = &state.fine_to_level[coarsest];
        state.fx[..coarse_n].fill(0.0);
        state.fy[..coarse_n].fill(0.0);
        state.rep_x[..coarse_n].fill(0.0);
        for u in 0..n {
            let c = map[u] as usize;
            state.fx[c] += state.active_x[u];
            state.fy[c] += state.active_y[u];
            state.rep_x[c] += 1.0;
        }
        for c in 0..coarse_n {
            let cnt = state.rep_x[c].max(1.0);
            state.active_x[c] = state.fx[c] / cnt;
            state.active_y[c] = state.fy[c] / cnt;
        }
    }
    state.initialized_coords = true;
}

fn prolongate_to_finer_level_2d(state: &mut YifanHuState) {
    let finer = state.current_level - 1;
    let fine_n = state.levels[finer].graph.num_nodes();
    let k_fine = state.levels[finer].optimal_distance;
    let Some(coarse_of) = state.levels[finer].coarse_of_finer.as_ref() else {
        return;
    };
    let Some(counts) = state.levels[finer].cluster_counts_of_finer.as_ref() else {
        return;
    };

    for u in 0..fine_n {
        let c = coarse_of[u] as usize;
        let cx = state.active_x[c];
        let cy = state.active_y[c];
        if counts[c] <= 1 {
            state.fx[u] = cx;
            state.fy[u] = cy;
        } else {
            let angle = std::f64::consts::TAU * state.rng.next_f64();
            let radius = PROLONGATION_JITTER_FACTOR * k_fine * (0.25 + 0.75 * state.rng.next_f64());
            let (s, cs) = angle.sin_cos();
            state.fx[u] = cx + radius * cs;
            state.fy[u] = cy + radius * s;
        }
    }
    state.active_x[..fine_n].copy_from_slice(&state.fx[..fine_n]);
    state.active_y[..fine_n].copy_from_slice(&state.fy[..fine_n]);

    state.current_level = finer;
    state.level_iter = 0;
    state.level_step_len = state.config.step_size * k_fine;
    state.level_progress = 0;
    state.prev_energy = f64::INFINITY;
}

fn write_active_coords_to_layout_2d<S: Scalar>(
    state: &YifanHuState,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    let (xs_mut, ys_mut) = layout.xy_mut();
    if state.current_level == 0 {
        for i in 0..state.num_nodes {
            let nx = S::from_f64(state.active_x[i]);
            let ny = S::from_f64(state.active_y[i]);
            if !nx.is_finite() || !ny.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "YifanHuSolver 2D coordinates became non-finite",
                ));
            }
            xs_mut[i] = nx;
            ys_mut[i] = ny;
        }
    } else {
        let map = &state.fine_to_level[state.current_level];
        for (u, &c) in map.iter().enumerate() {
            let nx = S::from_f64(state.active_x[c as usize]);
            let ny = S::from_f64(state.active_y[c as usize]);
            if !nx.is_finite() || !ny.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "YifanHuSolver 2D coarse coordinates became non-finite",
                ));
            }
            xs_mut[u] = nx;
            ys_mut[u] = ny;
        }
    }
    Ok(())
}

pub(super) fn step_2d<S: Scalar>(
    state: &mut YifanHuState,
    layout: &mut Layout2d<S>,
) -> CoreResult<SolverStatus> {
    let n = state.num_nodes;
    if !state.initialized_coords {
        initialize_coarsest_coords_2d(state, layout);
    } else if state.current_level == 0 {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            state.active_x[i] = xs[i].to_f64();
            state.active_y[i] = ys[i].to_f64();
        }
    }

    let level = &state.levels[state.current_level];
    let ln = level.graph.num_nodes();
    let k_level = level.optimal_distance;

    if ln <= 1 {
        state.level_iter += 1;
        state.total_iter_executed += 1;
        if state.current_level > 0 {
            prolongate_to_finer_level_2d(state);
            write_active_coords_to_layout_2d(state, layout)?;
            state.status = SolverStatus::Running;
        } else {
            write_active_coords_to_layout_2d(state, layout)?;
            state.status = SolverStatus::Converged;
        }
        return Ok(state.status);
    }

    // 1. Repulsion at active level via unit-mass ArenaQuadtree2d Coulomb evaluation.
    state.quadtree.build_weighted_xy(
        &state.active_x[..ln],
        &state.active_y[..ln],
        &state.unit_masses[..ln],
    )?;
    let eps_sq = (1e-8 * k_level * k_level).max(1e-24);
    state.quadtree.compute_coulomb_repulsion(
        state.config.theta,
        eps_sq,
        &mut state.rep_x[..ln],
        &mut state.rep_y[..ln],
    )?;

    let rep_scale = state.config.relative_strength * k_level * k_level;
    for i in 0..ln {
        state.fx[i] = rep_scale * state.rep_x[i];
        state.fy[i] = rep_scale * state.rep_y[i];
    }

    // 2. Quadratic spring attraction.
    let level_graph = &state.levels[state.current_level].graph;
    let use_weights = state.current_level == 0 && state.has_explicit_weights;
    for u in 0..ln {
        let ux = state.active_x[u];
        let uy = state.active_y[u];
        let nbrs = level_graph.neighbors(u);
        let weights = level_graph.neighbor_weights(u);
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            let w = if use_weights {
                weights.map_or(1.0, |ws| ws[pos])
            } else {
                1.0
            };
            if w == 0.0 {
                continue;
            }
            let dx = state.active_x[v] - ux;
            let dy = state.active_y[v] - uy;
            let d = dx.hypot(dy);
            let mult = w * (d / k_level);
            ax += mult * dx;
            ay += mult * dy;
        }
        state.fx[u] += ax;
        state.fy[u] += ay;
    }

    // 3. Compute total squared force energy and adapt step length.
    let mut energy = 0.0_f64;
    for i in 0..ln {
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
        energy += fxi * fxi + fyi * fyi;
    }
    if !energy.is_finite() {
        energy = f64::MAX;
    }

    if state.level_iter > 0 {
        if energy < state.prev_energy {
            state.level_progress += 1;
            if state.level_progress >= 5 {
                state.level_progress = 0;
                state.level_step_len =
                    (state.level_step_len / state.config.step_ratio).min(4.0 * k_level);
            }
        } else {
            state.level_progress = 0;
            state.level_step_len *= state.config.step_ratio;
        }
    }
    state.prev_energy = energy;

    // 4. Displace vertices along normalized force direction by adaptive step length.
    let mut max_disp = 0.0_f64;
    let mut max_force = 0.0_f64;
    let step_len = state.level_step_len;
    for i in 0..ln {
        let fxi = state.fx[i];
        let fyi = state.fy[i];
        let f_norm = fxi.hypot(fyi);
        if f_norm > max_force {
            max_force = f_norm;
        }
        if f_norm > 1e-12 {
            let scale = step_len / f_norm;
            let nx = state.active_x[i] + fxi * scale;
            let ny = state.active_y[i] + fyi * scale;
            if nx.is_finite() && ny.is_finite() {
                state.active_x[i] = nx;
                state.active_y[i] = ny;
                if step_len > max_disp {
                    max_disp = step_len;
                }
            }
        }
    }

    state.level_iter += 1;
    state.total_iter_executed += 1;

    let level_converged =
        max_disp <= state.config.convergence_tolerance * k_level || max_force <= 1e-12;
    let level_exhausted = state.level_iter >= state.config.max_iterations_per_level;

    if level_converged || level_exhausted {
        if state.current_level > 0 {
            prolongate_to_finer_level_2d(state);
            write_active_coords_to_layout_2d(state, layout)?;
            state.status = SolverStatus::Running;
        } else {
            write_active_coords_to_layout_2d(state, layout)?;
            state.status = if level_converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
        }
    } else {
        write_active_coords_to_layout_2d(state, layout)?;
        state.status = SolverStatus::Running;
    }

    Ok(state.status)
}
