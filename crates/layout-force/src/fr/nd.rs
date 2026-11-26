use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};
use egraph_spatial_trees::ArenaOctree3d;

use super::config::FrBackend;
use super::helpers::clamp_cell_coord;
use super::solver::{
    BH_EPS_SQ, COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST, FruchtermanReingoldSolver,
};

#[derive(Debug, Clone, Copy)]
struct Grid3Entry {
    cz: i64,
    cy: i64,
    cx: i64,
    node: usize,
}

#[derive(Debug, Clone, Copy)]
struct Grid3Cell {
    cz: i64,
    cy: i64,
    cx: i64,
    start: usize,
    end: usize,
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

/// Lazy N-D scratch buffers for [`FruchtermanReingoldSolver`] on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct FrNdScratch {
    dims: usize,
    coords: Vec<f64>,
    disp: Vec<f64>,
    rep: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
    dir_buf: Vec<f64>,
    octree: ArenaOctree3d<f64>,
    grid3_entries: Vec<Grid3Entry>,
    grid3_cells: Vec<Grid3Cell>,
}

impl FrNdScratch {
    fn new(num_nodes: usize, dims: usize, backend: FrBackend) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FR N-D buffer size overflow"))?;
        let rep_len = if backend == FrBackend::BarnesHut && dims == 3 {
            total
        } else {
            0
        };
        Ok(Self {
            dims,
            coords: vec![0.0; total],
            disp: vec![0.0; total],
            rep: vec![0.0; rep_len],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
            dir_buf: vec![0.0; dims],
            octree: ArenaOctree3d::new(),
            grid3_entries: Vec::new(),
            grid3_cells: Vec::new(),
        })
    }
}

impl<S: Scalar> FruchtermanReingoldSolver<S> {
    fn apply_repulsion_pair_nd(
        &mut self,
        scratch: &mut FrNdScratch,
        i: usize,
        j: usize,
        k_sq: f64,
        cutoff_sq: Option<f64>,
    ) {
        let d = scratch.dims;
        let base_i = i * d;
        let base_j = j * d;
        let mut d_sq = 0.0_f64;
        for axis in 0..d {
            let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
            d_sq += delta * delta;
        }
        if let Some(limit_sq) = cutoff_sq
            && d_sq >= limit_sq
        {
            return;
        }
        if d_sq > COINCIDENT_DIST_SQ {
            let mult = k_sq / d_sq;
            for axis in 0..d {
                let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                scratch.disp[base_i + axis] += f;
                scratch.disp[base_j + axis] -= f;
            }
        } else {
            let mut norm_sq = 0.0_f64;
            for slot in scratch.dir_buf.iter_mut().take(d) {
                let comp = self.rng.next_f64() - 0.5;
                *slot = comp;
                norm_sq += comp * comp;
            }
            let norm = norm_sq.sqrt();
            let mag = k_sq / COINCIDENT_PERTURB_DIST;
            if norm > 1e-12 {
                let scale = mag / norm;
                for axis in 0..d {
                    let f = scale * scratch.dir_buf[axis];
                    scratch.disp[base_i + axis] += f;
                    scratch.disp[base_j + axis] -= f;
                }
            } else {
                scratch.disp[base_i] += mag;
                scratch.disp[base_j] -= mag;
            }
        }
    }

    fn compute_grid_repulsion_3d(&mut self, scratch: &mut FrNdScratch, k_sq: f64) {
        let n = self.num_nodes;
        let min_x = scratch.min_axis[0];
        let min_y = scratch.min_axis[1];
        let min_z = scratch.min_axis[2];
        let cell_size = 2.0 * self.k;
        let inv_cell = 1.0 / cell_size;
        let cutoff_sq = cell_size * cell_size;

        scratch.grid3_entries.clear();
        scratch.grid3_entries.reserve(n);
        for i in 0..n {
            let base = i * 3;
            let cx = clamp_cell_coord(((scratch.coords[base] - min_x) * inv_cell).floor());
            let cy = clamp_cell_coord(((scratch.coords[base + 1] - min_y) * inv_cell).floor());
            let cz = clamp_cell_coord(((scratch.coords[base + 2] - min_z) * inv_cell).floor());
            scratch.grid3_entries.push(Grid3Entry {
                cz,
                cy,
                cx,
                node: i,
            });
        }
        scratch
            .grid3_entries
            .sort_unstable_by_key(|e| (e.cz, e.cy, e.cx, e.node));

        scratch.grid3_cells.clear();
        let mut idx = 0usize;
        while idx < n {
            let cz = scratch.grid3_entries[idx].cz;
            let cy = scratch.grid3_entries[idx].cy;
            let cx = scratch.grid3_entries[idx].cx;
            let start = idx;
            idx += 1;
            while idx < n
                && scratch.grid3_entries[idx].cz == cz
                && scratch.grid3_entries[idx].cy == cy
                && scratch.grid3_entries[idx].cx == cx
            {
                idx += 1;
            }
            scratch.grid3_cells.push(Grid3Cell {
                cz,
                cy,
                cx,
                start,
                end: idx,
            });
        }

        let num_cells = scratch.grid3_cells.len();
        for c_idx in 0..num_cells {
            let cell = scratch.grid3_cells[c_idx];
            for p in cell.start..cell.end {
                let u = scratch.grid3_entries[p].node;
                for q in (p + 1)..cell.end {
                    let v = scratch.grid3_entries[q].node;
                    self.apply_repulsion_pair_nd(scratch, u, v, k_sq, Some(cutoff_sq));
                }
            }

            for &(dz, dy, dx) in &HALF_NEIGHBOR_OFFSETS_3D {
                let target_key = (cell.cz + dz, cell.cy + dy, cell.cx + dx);
                if let Ok(nb_idx) = scratch
                    .grid3_cells
                    .binary_search_by_key(&target_key, |c| (c.cz, c.cy, c.cx))
                {
                    let nb = scratch.grid3_cells[nb_idx];
                    for p in cell.start..cell.end {
                        let u = scratch.grid3_entries[p].node;
                        for q in nb.start..nb.end {
                            let v = scratch.grid3_entries[q].node;
                            self.apply_repulsion_pair_nd(scratch, u, v, k_sq, Some(cutoff_sq));
                        }
                    }
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for FruchtermanReingoldSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FruchtermanReingoldSolver::step",
            ));
        }
        let dims = layout.dims();
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        layout.validate_finite()?;
        if self.active_dims == 0 {
            self.active_dims = dims;
        } else if self.active_dims != dims {
            return Err(CoreError::dimension_mismatch(
                self.active_dims,
                dims,
                "FruchtermanReingoldSolver::step::dims",
            ));
        }
        if matches!(self.config.backend, FrBackend::BarnesHut | FrBackend::Grid)
            && dims != 2
            && dims != 3
        {
            return Err(CoreError::unsupported(format!(
                "Fruchterman-Reingold {:?} backend only supports 2D and 3D layouts, got {dims}D",
                self.config.backend
            )));
        }

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let mut scratch = match self.nd_scratch.take() {
            Some(existing) if existing.dims == dims => existing,
            _ => FrNdScratch::new(n, dims, self.config.backend)?,
        };

        let result = (|| -> CoreResult<SolverStatus> {
            scratch.min_axis.fill(f64::INFINITY);
            scratch.max_axis.fill(f64::NEG_INFINITY);
            let src = layout.coords();
            for i in 0..n {
                let base = i * dims;
                for axis in 0..dims {
                    let v = src[base + axis].to_f64();
                    scratch.coords[base + axis] = v;
                    scratch.min_axis[axis] = scratch.min_axis[axis].min(v);
                    scratch.max_axis[axis] = scratch.max_axis[axis].max(v);
                }
            }

            if self.iter == 0
                && (0..dims).all(|a| (scratch.max_axis[a] - scratch.min_axis[a]) < 1e-12)
            {
                let spread = 0.5 * self.k;
                scratch.min_axis.fill(f64::INFINITY);
                for i in 0..n {
                    let base = i * dims;
                    for axis in 0..dims {
                        let v = scratch.coords[base + axis] + spread * (self.rng.next_f64() - 0.5);
                        scratch.coords[base + axis] = v;
                        scratch.min_axis[axis] = scratch.min_axis[axis].min(v);
                    }
                }
            }

            scratch.disp.fill(0.0);
            let k_sq = self.k * self.k;

            match self.config.backend {
                FrBackend::Exact => {
                    for i in 0..n {
                        for j in (i + 1)..n {
                            self.apply_repulsion_pair_nd(&mut scratch, i, j, k_sq, None);
                        }
                    }
                }
                FrBackend::BarnesHut => {
                    if dims == 3 {
                        scratch.octree.build_flat(&scratch.coords, n)?;
                        scratch.octree.compute_coulomb_repulsion_flat(
                            self.config.theta,
                            BH_EPS_SQ,
                            &mut scratch.rep,
                        )?;
                        for (dst, &r) in scratch.disp.iter_mut().zip(scratch.rep.iter()) {
                            *dst = k_sq * r;
                        }
                    } else {
                        for i in 0..n {
                            self.x_buf[i] = scratch.coords[i * 2];
                            self.y_buf[i] = scratch.coords[i * 2 + 1];
                        }
                        self.quadtree.build_xy(&self.x_buf, &self.y_buf)?;
                        self.quadtree.compute_coulomb_repulsion(
                            self.config.theta,
                            BH_EPS_SQ,
                            &mut self.rep_x,
                            &mut self.rep_y,
                        )?;
                        for i in 0..n {
                            scratch.disp[i * 2] = k_sq * self.rep_x[i];
                            scratch.disp[i * 2 + 1] = k_sq * self.rep_y[i];
                        }
                    }
                }
                FrBackend::Grid => {
                    if dims == 3 {
                        self.compute_grid_repulsion_3d(&mut scratch, k_sq);
                    } else {
                        for i in 0..n {
                            self.x_buf[i] = scratch.coords[i * 2];
                            self.y_buf[i] = scratch.coords[i * 2 + 1];
                        }
                        self.disp_x.fill(0.0);
                        self.disp_y.fill(0.0);
                        self.compute_grid_repulsion(scratch.min_axis[0], scratch.min_axis[1], k_sq);
                        for i in 0..n {
                            scratch.disp[i * 2] = self.disp_x[i];
                            scratch.disp[i * 2 + 1] = self.disp_y[i];
                        }
                    }
                }
            }

            let inv_k = 1.0 / self.k;
            for idx in 0..self.edges_u.len() {
                let u = self.edges_u[idx] as usize;
                let v = self.edges_v[idx] as usize;
                let w = self.edges_w[idx];
                if w == 0.0 {
                    continue;
                }
                let base_u = u * dims;
                let base_v = v * dims;
                let mut d_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
                    d_sq += delta * delta;
                }
                let factor = w * d_sq.sqrt() * inv_k;
                for axis in 0..dims {
                    let a =
                        factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
                    scratch.disp[base_u + axis] += a;
                    scratch.disp[base_v + axis] -= a;
                }
            }

            if self.config.gravity > 0.0 {
                let g = self.config.gravity;
                for (d_val, &c_val) in scratch.disp.iter_mut().zip(scratch.coords.iter()) {
                    *d_val -= g * c_val;
                }
            }

            let temp = self.current_temperature();
            let coords_mut = layout.coords_mut();
            for i in 0..n {
                let base = i * dims;
                let mut len_sq = 0.0_f64;
                for axis in 0..dims {
                    let da = scratch.disp[base + axis];
                    len_sq += da * da;
                }
                let len = len_sq.sqrt();
                if len > 1e-24 && temp > 0.0 {
                    let scale = len.min(temp) / len;
                    for axis in 0..dims {
                        let idx = base + axis;
                        let next_c = S::from_f64(scratch.coords[idx] + scratch.disp[idx] * scale);
                        if next_c.is_finite() {
                            coords_mut[idx] = next_c;
                        }
                    }
                }
            }

            self.iter += 1;
            self.status = if self.iter >= self.config.iterations {
                SolverStatus::Completed
            } else {
                SolverStatus::Running
            };
            Ok(self.status)
        })();

        self.nd_scratch = Some(scratch);
        result
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.iterations
    }
}
