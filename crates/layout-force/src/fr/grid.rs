use egraph_core::DeterministicRng;

use super::helpers::{Grid3Cell, Grid3Entry, GridCell, GridEntry, clamp_cell_coord};
use super::solver::{COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST};

#[derive(Debug, Clone)]
pub struct FrGrid2dState {
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    entries: Vec<GridEntry>,
    cells: Vec<GridCell>,
}

impl FrGrid2dState {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            x_buf: vec![0.0; num_nodes],
            y_buf: vec![0.0; num_nodes],
            entries: Vec::new(),
            cells: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrGrid3dState {
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) z_buf: Vec<f64>,
    entries: Vec<Grid3Entry>,
    cells: Vec<Grid3Cell>,
}

impl FrGrid3dState {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            x_buf: vec![0.0; num_nodes],
            y_buf: vec![0.0; num_nodes],
            z_buf: vec![0.0; num_nodes],
            entries: Vec::new(),
            cells: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_grid_pair_2d(
    x_buf: &[f64],
    y_buf: &[f64],
    i: usize,
    j: usize,
    k_sq: f64,
    cutoff_sq: f64,
    rng: &mut DeterministicRng,
    disp: &mut [f64],
) {
    let dx = x_buf[i] - x_buf[j];
    let dy = y_buf[i] - y_buf[j];
    let d_sq = dx * dx + dy * dy;
    if d_sq >= cutoff_sq {
        return;
    }
    let (fx, fy) = if d_sq > COINCIDENT_DIST_SQ {
        let mult = k_sq / d_sq;
        (mult * dx, mult * dy)
    } else {
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let mag = k_sq / COINCIDENT_PERTURB_DIST;
        (mag * angle.cos(), mag * angle.sin())
    };
    disp[2 * i] += fx;
    disp[2 * i + 1] += fy;
    disp[2 * j] -= fx;
    disp[2 * j + 1] -= fy;
}

pub(super) fn compute_grid_repulsion_2d(
    state: &mut FrGrid2dState,
    min_x: f64,
    min_y: f64,
    k: f64,
    rng: &mut DeterministicRng,
    disp: &mut [f64],
) {
    let n = state.x_buf.len();
    let cell_size = 2.0 * k;
    let inv_cell = 1.0 / cell_size;
    let cutoff_sq = cell_size * cell_size;
    let k_sq = k * k;

    state.entries.clear();
    state.entries.reserve(n);
    for i in 0..n {
        let cx = clamp_cell_coord(((state.x_buf[i] - min_x) * inv_cell).floor());
        let cy = clamp_cell_coord(((state.y_buf[i] - min_y) * inv_cell).floor());
        state.entries.push(GridEntry { cy, cx, node: i });
    }
    state.entries.sort_unstable_by_key(|e| (e.cy, e.cx, e.node));

    state.cells.clear();
    let mut idx = 0usize;
    while idx < n {
        let cy = state.entries[idx].cy;
        let cx = state.entries[idx].cx;
        let start = idx;
        idx += 1;
        while idx < n && state.entries[idx].cy == cy && state.entries[idx].cx == cx {
            idx += 1;
        }
        state.cells.push(GridCell {
            cy,
            cx,
            start,
            end: idx,
        });
    }

    const HALF_NEIGHBOR_OFFSETS: [(i64, i64); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
    let num_cells = state.cells.len();
    for c_idx in 0..num_cells {
        let cell = state.cells[c_idx];
        for p in cell.start..cell.end {
            let u = state.entries[p].node;
            for q in (p + 1)..cell.end {
                let v = state.entries[q].node;
                apply_grid_pair_2d(&state.x_buf, &state.y_buf, u, v, k_sq, cutoff_sq, rng, disp);
            }
        }

        for &(dy, dx) in &HALF_NEIGHBOR_OFFSETS {
            let target_key = (cell.cy + dy, cell.cx + dx);
            if let Ok(nb_idx) = state
                .cells
                .binary_search_by_key(&target_key, |c| (c.cy, c.cx))
            {
                let nb = state.cells[nb_idx];
                for p in cell.start..cell.end {
                    let u = state.entries[p].node;
                    for q in nb.start..nb.end {
                        let v = state.entries[q].node;
                        apply_grid_pair_2d(
                            &state.x_buf,
                            &state.y_buf,
                            u,
                            v,
                            k_sq,
                            cutoff_sq,
                            rng,
                            disp,
                        );
                    }
                }
            }
        }
    }
}

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

#[allow(clippy::too_many_arguments)]
fn apply_grid_pair_3d(
    x_buf: &[f64],
    y_buf: &[f64],
    z_buf: &[f64],
    i: usize,
    j: usize,
    k_sq: f64,
    cutoff_sq: f64,
    rng: &mut DeterministicRng,
    disp: &mut [f64],
) {
    let dx = x_buf[i] - x_buf[j];
    let dy = y_buf[i] - y_buf[j];
    let dz = z_buf[i] - z_buf[j];
    let d_sq = dx * dx + dy * dy + dz * dz;
    if d_sq >= cutoff_sq {
        return;
    }
    let (fx, fy, fz) = if d_sq > COINCIDENT_DIST_SQ {
        let mult = k_sq / d_sq;
        (mult * dx, mult * dy, mult * dz)
    } else {
        let rx = rng.next_f64() - 0.5;
        let ry = rng.next_f64() - 0.5;
        let rz = rng.next_f64() - 0.5;
        let norm = (rx * rx + ry * ry + rz * rz).sqrt();
        let mag = k_sq / COINCIDENT_PERTURB_DIST;
        if norm > 1e-12 {
            let s = mag / norm;
            (s * rx, s * ry, s * rz)
        } else {
            (mag, 0.0, 0.0)
        }
    };
    disp[3 * i] += fx;
    disp[3 * i + 1] += fy;
    disp[3 * i + 2] += fz;
    disp[3 * j] -= fx;
    disp[3 * j + 1] -= fy;
    disp[3 * j + 2] -= fz;
}

pub(super) fn compute_grid_repulsion_3d(
    state: &mut FrGrid3dState,
    min_x: f64,
    min_y: f64,
    min_z: f64,
    k: f64,
    rng: &mut DeterministicRng,
    disp: &mut [f64],
) {
    let n = state.x_buf.len();
    let cell_size = 2.0 * k;
    let inv_cell = 1.0 / cell_size;
    let cutoff_sq = cell_size * cell_size;
    let k_sq = k * k;

    state.entries.clear();
    state.entries.reserve(n);
    for i in 0..n {
        let cx = clamp_cell_coord(((state.x_buf[i] - min_x) * inv_cell).floor());
        let cy = clamp_cell_coord(((state.y_buf[i] - min_y) * inv_cell).floor());
        let cz = clamp_cell_coord(((state.z_buf[i] - min_z) * inv_cell).floor());
        state.entries.push(Grid3Entry {
            cz,
            cy,
            cx,
            node: i,
        });
    }
    state
        .entries
        .sort_unstable_by_key(|e| (e.cz, e.cy, e.cx, e.node));

    state.cells.clear();
    let mut idx = 0usize;
    while idx < n {
        let cz = state.entries[idx].cz;
        let cy = state.entries[idx].cy;
        let cx = state.entries[idx].cx;
        let start = idx;
        idx += 1;
        while idx < n
            && state.entries[idx].cz == cz
            && state.entries[idx].cy == cy
            && state.entries[idx].cx == cx
        {
            idx += 1;
        }
        state.cells.push(Grid3Cell {
            cz,
            cy,
            cx,
            start,
            end: idx,
        });
    }

    let num_cells = state.cells.len();
    for c_idx in 0..num_cells {
        let cell = state.cells[c_idx];
        for p in cell.start..cell.end {
            let u = state.entries[p].node;
            for q in (p + 1)..cell.end {
                let v = state.entries[q].node;
                apply_grid_pair_3d(
                    &state.x_buf,
                    &state.y_buf,
                    &state.z_buf,
                    u,
                    v,
                    k_sq,
                    cutoff_sq,
                    rng,
                    disp,
                );
            }
        }

        for &(dz, dy, dx) in &HALF_NEIGHBOR_OFFSETS_3D {
            let target_key = (cell.cz + dz, cell.cy + dy, cell.cx + dx);
            if let Ok(nb_idx) = state
                .cells
                .binary_search_by_key(&target_key, |c| (c.cz, c.cy, c.cx))
            {
                let nb = state.cells[nb_idx];
                for p in cell.start..cell.end {
                    let u = state.entries[p].node;
                    for q in nb.start..nb.end {
                        let v = state.entries[q].node;
                        apply_grid_pair_3d(
                            &state.x_buf,
                            &state.y_buf,
                            &state.z_buf,
                            u,
                            v,
                            k_sq,
                            cutoff_sq,
                            rng,
                            disp,
                        );
                    }
                }
            }
        }
    }
}
