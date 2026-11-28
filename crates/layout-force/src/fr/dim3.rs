use egraph_core::{CoreError, CoreResult, Layout3d, Scalar};

use super::config::FrBackend;
use super::helpers::{Grid3Cell, Grid3Entry, clamp_cell_coord};
use super::solver::{BH_EPS_SQ, COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST, FrState};

const HALF_NEIGHBOR_OFFSETS_3D: [(i64, i64, i64); 13] = [
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

fn apply_repulsion_pair_3d(
    state: &mut FrState,
    i: usize,
    j: usize,
    k_sq: f64,
    cutoff_sq: Option<f64>,
) {
    let dx = state.x_buf[i] - state.x_buf[j];
    let dy = state.y_buf[i] - state.y_buf[j];
    let dz = state.z_buf[i] - state.z_buf[j];
    let d_sq = dx * dx + dy * dy + dz * dz;
    if let Some(limit_sq) = cutoff_sq
        && d_sq >= limit_sq
    {
        return;
    }
    let (fx, fy, fz) = if d_sq > COINCIDENT_DIST_SQ {
        let mult = k_sq / d_sq;
        (mult * dx, mult * dy, mult * dz)
    } else {
        let rx = state.rng.next_f64() - 0.5;
        let ry = state.rng.next_f64() - 0.5;
        let rz = state.rng.next_f64() - 0.5;
        let norm = (rx * rx + ry * ry + rz * rz).sqrt();
        let mag = k_sq / COINCIDENT_PERTURB_DIST;
        if norm > 1e-12 {
            let s = mag / norm;
            (s * rx, s * ry, s * rz)
        } else {
            (mag, 0.0, 0.0)
        }
    };
    state.disp_x[i] += fx;
    state.disp_y[i] += fy;
    state.disp_z[i] += fz;
    state.disp_x[j] -= fx;
    state.disp_y[j] -= fy;
    state.disp_z[j] -= fz;
}

fn compute_grid_repulsion_3d(state: &mut FrState, min_x: f64, min_y: f64, min_z: f64, k_sq: f64) {
    let n = state.num_nodes;
    let cell_size = 2.0 * state.k;
    let inv_cell = 1.0 / cell_size;
    let cutoff_sq = cell_size * cell_size;

    state.grid3_entries.clear();
    state.grid3_entries.reserve(n);
    for i in 0..n {
        let cx = clamp_cell_coord(((state.x_buf[i] - min_x) * inv_cell).floor());
        let cy = clamp_cell_coord(((state.y_buf[i] - min_y) * inv_cell).floor());
        let cz = clamp_cell_coord(((state.z_buf[i] - min_z) * inv_cell).floor());
        state.grid3_entries.push(Grid3Entry {
            cz,
            cy,
            cx,
            node: i,
        });
    }
    state
        .grid3_entries
        .sort_unstable_by_key(|e| (e.cz, e.cy, e.cx, e.node));

    state.grid3_cells.clear();
    let mut idx = 0usize;
    while idx < n {
        let cz = state.grid3_entries[idx].cz;
        let cy = state.grid3_entries[idx].cy;
        let cx = state.grid3_entries[idx].cx;
        let start = idx;
        idx += 1;
        while idx < n
            && state.grid3_entries[idx].cz == cz
            && state.grid3_entries[idx].cy == cy
            && state.grid3_entries[idx].cx == cx
        {
            idx += 1;
        }
        state.grid3_cells.push(Grid3Cell {
            cz,
            cy,
            cx,
            start,
            end: idx,
        });
    }

    let num_cells = state.grid3_cells.len();
    for c_idx in 0..num_cells {
        let cell = state.grid3_cells[c_idx];
        for p in cell.start..cell.end {
            let u = state.grid3_entries[p].node;
            for q in (p + 1)..cell.end {
                let v = state.grid3_entries[q].node;
                apply_repulsion_pair_3d(state, u, v, k_sq, Some(cutoff_sq));
            }
        }

        for &(dz, dy, dx) in &HALF_NEIGHBOR_OFFSETS_3D {
            let target_key = (cell.cz + dz, cell.cy + dy, cell.cx + dx);
            if let Ok(nb_idx) = state
                .grid3_cells
                .binary_search_by_key(&target_key, |c| (c.cz, c.cy, c.cx))
            {
                let nb = state.grid3_cells[nb_idx];
                for p in cell.start..cell.end {
                    let u = state.grid3_entries[p].node;
                    for q in nb.start..nb.end {
                        let v = state.grid3_entries[q].node;
                        apply_repulsion_pair_3d(state, u, v, k_sq, Some(cutoff_sq));
                    }
                }
            }
        }
    }
}

pub(super) fn step_3d<S: Scalar>(state: &mut FrState, layout: &mut Layout3d<S>) -> CoreResult<()> {
    let n = state.num_nodes;
    let (min_x, min_y, min_z) = {
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
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
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            state.z_buf[i] = z;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        if state.iter == 0
            && (max_x - min_x) < 1e-12
            && (max_y - min_y) < 1e-12
            && (max_z - min_z) < 1e-12
        {
            let spread = 0.5 * state.k;
            min_x = f64::INFINITY;
            min_y = f64::INFINITY;
            min_z = f64::INFINITY;
            for i in 0..n {
                state.x_buf[i] += spread * (state.rng.next_f64() - 0.5);
                state.y_buf[i] += spread * (state.rng.next_f64() - 0.5);
                state.z_buf[i] += spread * (state.rng.next_f64() - 0.5);
                min_x = min_x.min(state.x_buf[i]);
                min_y = min_y.min(state.y_buf[i]);
                min_z = min_z.min(state.z_buf[i]);
            }
        }
        (min_x, min_y, min_z)
    };

    state.disp_x.fill(0.0);
    state.disp_y.fill(0.0);
    state.disp_z.fill(0.0);
    let k_sq = state.k * state.k;

    match state.config.backend {
        FrBackend::Exact => {
            for i in 0..n {
                for j in (i + 1)..n {
                    apply_repulsion_pair_3d(state, i, j, k_sq, None);
                }
            }
        }
        FrBackend::BarnesHut => {
            state
                .octree
                .build_xyz(&state.x_buf, &state.y_buf, &state.z_buf)?;
            state.octree.compute_coulomb_repulsion(
                state.config.theta,
                BH_EPS_SQ,
                &mut state.rep_x,
                &mut state.rep_y,
                &mut state.rep_z,
            )?;
            for i in 0..n {
                state.disp_x[i] = k_sq * state.rep_x[i];
                state.disp_y[i] = k_sq * state.rep_y[i];
                state.disp_z[i] = k_sq * state.rep_z[i];
            }
        }
        FrBackend::Grid => {
            compute_grid_repulsion_3d(state, min_x, min_y, min_z, k_sq);
        }
    }

    let inv_k = 1.0 / state.k;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = state.x_buf[v] - state.x_buf[u];
        let dy = state.y_buf[v] - state.y_buf[u];
        let dz = state.z_buf[v] - state.z_buf[u];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        let factor = w * d * inv_k;
        let ax = factor * dx;
        let ay = factor * dy;
        let az = factor * dz;
        state.disp_x[u] += ax;
        state.disp_y[u] += ay;
        state.disp_z[u] += az;
        state.disp_x[v] -= ax;
        state.disp_y[v] -= ay;
        state.disp_z[v] -= az;
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        for i in 0..n {
            state.disp_x[i] -= g * state.x_buf[i];
            state.disp_y[i] -= g * state.y_buf[i];
            state.disp_z[i] -= g * state.z_buf[i];
        }
    }

    let temp = state.current_temperature();
    let (xs_mut, ys_mut, zs_mut) = layout.xyz_mut();
    for i in 0..n {
        let dx = state.disp_x[i];
        let dy = state.disp_y[i];
        let dz = state.disp_z[i];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len > 1e-24 && temp > 0.0 {
            let scale = len.min(temp) / len;
            let nx = S::from_f64(state.x_buf[i] + dx * scale);
            let ny = S::from_f64(state.y_buf[i] + dy * scale);
            let nz = S::from_f64(state.z_buf[i] + dz * scale);
            if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "FruchtermanReingoldSolver 3D coordinates became non-finite",
                ));
            }
            xs_mut[i] = nx;
            ys_mut[i] = ny;
            zs_mut[i] = nz;
        }
    }

    Ok(())
}
