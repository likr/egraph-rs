use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageMut, Scalar, SolverStatus,
};

use super::repulsion::{YifanHuLevelCoords, YifanHuRepulsion};
use super::solver::{PROLONGATION_JITTER_FACTOR, YifanHuState};

#[derive(Debug, Clone)]
pub(super) struct YhBuf3d {
    active_x: Vec<f64>,
    active_y: Vec<f64>,
    active_z: Vec<f64>,
    tmp_x: Vec<f64>,
    tmp_y: Vec<f64>,
    tmp_z: Vec<f64>,
    counts: Vec<f64>,
    forces: Vec<f64>,
}

impl YhBuf3d {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            active_x: vec![0.0; num_nodes],
            active_y: vec![0.0; num_nodes],
            active_z: vec![0.0; num_nodes],
            tmp_x: vec![0.0; num_nodes],
            tmp_y: vec![0.0; num_nodes],
            tmp_z: vec![0.0; num_nodes],
            counts: vec![0.0; num_nodes],
            forces: vec![0.0; num_nodes * 3],
        }
    }
}

fn initialize_coarsest_coords_3d<S: Scalar>(
    state: &mut YifanHuState,
    buf: &mut YhBuf3d,
    xs: &[S],
    ys: &[S],
    zs: &[S],
) {
    let n = state.num_nodes;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let z = zs[i].to_f64();
        buf.active_x[i] = x;
        buf.active_y[i] = y;
        buf.active_z[i] = z;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }

    let coarsest = state.levels.len() - 1;
    let coarse_n = state.levels[coarsest].graph.num_nodes();
    if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 && (max_z - min_z) < 1e-12 {
        let k_coarse = state.levels[coarsest].optimal_distance;
        let spread = k_coarse * (coarse_n as f64).sqrt().max(1.0);
        for i in 0..coarse_n {
            buf.active_x[i] = spread * (state.rng.next_f64() - 0.5);
            buf.active_y[i] = spread * (state.rng.next_f64() - 0.5);
            buf.active_z[i] = spread * (state.rng.next_f64() - 0.5);
        }
    } else if coarsest > 0 {
        let map = &state.fine_to_level[coarsest];
        buf.tmp_x[..coarse_n].fill(0.0);
        buf.tmp_y[..coarse_n].fill(0.0);
        buf.tmp_z[..coarse_n].fill(0.0);
        buf.counts[..coarse_n].fill(0.0);
        for u in 0..n {
            let c = map[u] as usize;
            buf.tmp_x[c] += buf.active_x[u];
            buf.tmp_y[c] += buf.active_y[u];
            buf.tmp_z[c] += buf.active_z[u];
            buf.counts[c] += 1.0;
        }
        for c in 0..coarse_n {
            let cnt = buf.counts[c].max(1.0);
            buf.active_x[c] = buf.tmp_x[c] / cnt;
            buf.active_y[c] = buf.tmp_y[c] / cnt;
            buf.active_z[c] = buf.tmp_z[c] / cnt;
        }
    }
    state.initialized_coords = true;
}

fn prolongate_to_finer_level_3d(state: &mut YifanHuState, buf: &mut YhBuf3d) {
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
        let cx = buf.active_x[c];
        let cy = buf.active_y[c];
        let cz = buf.active_z[c];
        if counts[c] <= 1 {
            buf.tmp_x[u] = cx;
            buf.tmp_y[u] = cy;
            buf.tmp_z[u] = cz;
        } else {
            let dx = state.rng.next_f64() - 0.5;
            let dy = state.rng.next_f64() - 0.5;
            let dz = state.rng.next_f64() - 0.5;
            let norm = (dx * dx + dy * dy + dz * dz).sqrt();
            let radius = PROLONGATION_JITTER_FACTOR * k_fine * (0.25 + 0.75 * state.rng.next_f64());
            if norm > 1e-12 {
                let scale = radius / norm;
                buf.tmp_x[u] = cx + scale * dx;
                buf.tmp_y[u] = cy + scale * dy;
                buf.tmp_z[u] = cz + scale * dz;
            } else {
                buf.tmp_x[u] = cx + radius;
                buf.tmp_y[u] = cy;
                buf.tmp_z[u] = cz;
            }
        }
    }
    buf.active_x[..fine_n].copy_from_slice(&buf.tmp_x[..fine_n]);
    buf.active_y[..fine_n].copy_from_slice(&buf.tmp_y[..fine_n]);
    buf.active_z[..fine_n].copy_from_slice(&buf.tmp_z[..fine_n]);

    state.current_level = finer;
    state.level_iter = 0;
    state.level_step_len = state.config.step_size * k_fine;
    state.level_progress = 0;
    state.prev_energy = f64::INFINITY;
}

fn write_active_coords_to_layout_3d<S: Scalar>(
    state: &YifanHuState,
    buf: &YhBuf3d,
    xs_mut: &mut [S],
    ys_mut: &mut [S],
    zs_mut: &mut [S],
) -> CoreResult<()> {
    if state.current_level == 0 {
        for i in 0..state.num_nodes {
            let nx = S::from_f64(buf.active_x[i]);
            let ny = S::from_f64(buf.active_y[i]);
            let nz = S::from_f64(buf.active_z[i]);
            if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "YifanHuSolver 3D coordinates became non-finite",
                ));
            }
            xs_mut[i] = nx;
            ys_mut[i] = ny;
            zs_mut[i] = nz;
        }
    } else {
        let map = &state.fine_to_level[state.current_level];
        for (u, &c) in map.iter().enumerate() {
            let ci = c as usize;
            let nx = S::from_f64(buf.active_x[ci]);
            let ny = S::from_f64(buf.active_y[ci]);
            let nz = S::from_f64(buf.active_z[ci]);
            if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "YifanHuSolver 3D coarse coordinates became non-finite",
                ));
            }
            xs_mut[u] = nx;
            ys_mut[u] = ny;
            zs_mut[u] = nz;
        }
    }
    Ok(())
}

pub(super) fn step_3d<L: EuclideanLayout, B: YifanHuRepulsion<L>>(
    state: &mut YifanHuState,
    buf: &mut YhBuf3d,
    backend: &B,
    rep_state: &mut B::State,
    layout: &mut L,
) -> CoreResult<SolverStatus> {
    let n = state.num_nodes;
    {
        let EuclideanStorageMut::Soa3d {
            x: xs,
            y: ys,
            z: zs,
        } = layout.as_euclidean_storage_mut()
        else {
            return Err(CoreError::numerical_breakdown(
                "YifanHuSolver 3D storage mismatch",
            ));
        };
        if !state.initialized_coords {
            initialize_coarsest_coords_3d(state, buf, xs, ys, zs);
        } else if state.current_level == 0 {
            for i in 0..n {
                buf.active_x[i] = xs[i].to_f64();
                buf.active_y[i] = ys[i].to_f64();
                buf.active_z[i] = zs[i].to_f64();
            }
        }
    }

    let ln = state.levels[state.current_level].graph.num_nodes();
    let k_level = state.levels[state.current_level].optimal_distance;

    if ln <= 1 {
        state.level_iter += 1;
        state.total_iter_executed += 1;
        if state.current_level > 0 {
            prolongate_to_finer_level_3d(state, buf);
            state.status = SolverStatus::Running;
        } else {
            state.status = SolverStatus::Converged;
        }
        let EuclideanStorageMut::Soa3d { x, y, z } = layout.as_euclidean_storage_mut() else {
            return Err(CoreError::numerical_breakdown(
                "YifanHuSolver 3D storage mismatch",
            ));
        };
        write_active_coords_to_layout_3d(state, buf, x, y, z)?;
        return Ok(state.status);
    }

    let eps_sq = (1e-8 * k_level * k_level).max(1e-24);
    let rep_scale = state.config.relative_strength * k_level * k_level;
    buf.forces[..(ln * 3)].fill(0.0);
    let coords_view = YifanHuLevelCoords::Soa3d {
        x: &buf.active_x[..ln],
        y: &buf.active_y[..ln],
        z: &buf.active_z[..ln],
    };
    backend.accumulate_level_repulsion(
        rep_state,
        &coords_view,
        eps_sq,
        rep_scale,
        &mut buf.forces[..(ln * 3)],
    )?;

    let level_graph = &state.levels[state.current_level].graph;
    let use_weights = state.current_level == 0 && state.has_explicit_weights;
    for u in 0..ln {
        let ux = buf.active_x[u];
        let uy = buf.active_y[u];
        let uz = buf.active_z[u];
        let nbrs = level_graph.neighbors(u);
        let weights = level_graph.neighbor_weights(u);
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;
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
            let dx = buf.active_x[v] - ux;
            let dy = buf.active_y[v] - uy;
            let dz = buf.active_z[v] - uz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let mult = w * (d / k_level);
            ax += mult * dx;
            ay += mult * dy;
            az += mult * dz;
        }
        buf.forces[3 * u] += ax;
        buf.forces[3 * u + 1] += ay;
        buf.forces[3 * u + 2] += az;
    }

    let mut energy = 0.0_f64;
    for i in 0..ln {
        let fxi = if buf.forces[3 * i].is_finite() {
            buf.forces[3 * i]
        } else {
            0.0
        };
        let fyi = if buf.forces[3 * i + 1].is_finite() {
            buf.forces[3 * i + 1]
        } else {
            0.0
        };
        let fzi = if buf.forces[3 * i + 2].is_finite() {
            buf.forces[3 * i + 2]
        } else {
            0.0
        };
        buf.forces[3 * i] = fxi;
        buf.forces[3 * i + 1] = fyi;
        buf.forces[3 * i + 2] = fzi;
        energy += fxi * fxi + fyi * fyi + fzi * fzi;
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

    let mut max_disp = 0.0_f64;
    let mut max_force = 0.0_f64;
    let step_len = state.level_step_len;
    for i in 0..ln {
        let fxi = buf.forces[3 * i];
        let fyi = buf.forces[3 * i + 1];
        let fzi = buf.forces[3 * i + 2];
        let f_norm = (fxi * fxi + fyi * fyi + fzi * fzi).sqrt();
        if f_norm > max_force {
            max_force = f_norm;
        }
        if f_norm > 1e-12 {
            let scale = step_len / f_norm;
            let nx = buf.active_x[i] + fxi * scale;
            let ny = buf.active_y[i] + fyi * scale;
            let nz = buf.active_z[i] + fzi * scale;
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                buf.active_x[i] = nx;
                buf.active_y[i] = ny;
                buf.active_z[i] = nz;
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
            prolongate_to_finer_level_3d(state, buf);
            state.status = SolverStatus::Running;
        } else {
            state.status = if level_converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
        }
    } else {
        state.status = SolverStatus::Running;
    }

    let EuclideanStorageMut::Soa3d { x, y, z } = layout.as_euclidean_storage_mut() else {
        return Err(CoreError::numerical_breakdown(
            "YifanHuSolver 3D storage mismatch",
        ));
    };
    write_active_coords_to_layout_3d(state, buf, x, y, z)?;
    Ok(state.status)
}
