use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};
use egraph_spatial_trees::ArenaOctree3d;

use super::yifan_hu::YifanHuSolver;
use super::yifan_hu_step::PROLONGATION_JITTER_FACTOR;

/// Lazy N-D scratch buffers for [`YifanHuSolver`] on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct YifanHuNdScratch {
    dims: usize,
    active: Vec<f64>,
    forces: Vec<f64>,
    rep: Vec<f64>,
    counts: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
    dir_buf: Vec<f64>,
    octree: ArenaOctree3d<f64>,
}

impl YifanHuNdScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("YifanHu N-D buffer size overflow"))?;
        Ok(Self {
            dims,
            active: vec![0.0; total],
            forces: vec![0.0; total],
            rep: vec![0.0; if dims == 3 { total } else { 0 }],
            counts: vec![0.0; num_nodes],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
            dir_buf: vec![0.0; dims],
            octree: ArenaOctree3d::new(),
        })
    }
}

impl<S: Scalar> YifanHuSolver<S> {
    fn initialize_coarsest_coords_nd(
        &mut self,
        layout: &LayoutNd<S>,
        scratch: &mut YifanHuNdScratch,
    ) {
        let n = self.num_nodes;
        let d = scratch.dims;
        let src = layout.coords();
        scratch.min_axis.fill(f64::INFINITY);
        scratch.max_axis.fill(f64::NEG_INFINITY);

        for i in 0..n {
            let base = i * d;
            for axis in 0..d {
                let v = src[base + axis].to_f64();
                scratch.active[base + axis] = v;
                scratch.min_axis[axis] = scratch.min_axis[axis].min(v);
                scratch.max_axis[axis] = scratch.max_axis[axis].max(v);
            }
        }

        let coarsest = self.levels.len() - 1;
        let coarse_n = self.levels[coarsest].graph.num_nodes();
        if (0..d).all(|axis| (scratch.max_axis[axis] - scratch.min_axis[axis]) < 1e-12) {
            let k_coarse = self.levels[coarsest].optimal_distance;
            let spread = k_coarse * (coarse_n as f64).sqrt().max(1.0);
            for v in &mut scratch.active[..(coarse_n * d)] {
                *v = spread * (self.rng.next_f64() - 0.5);
            }
        } else if coarsest > 0 {
            let map = &self.fine_to_level[coarsest];
            scratch.forces[..(coarse_n * d)].fill(0.0);
            scratch.counts[..coarse_n].fill(0.0);
            for (u, &c_u32) in map.iter().enumerate().take(n) {
                let c = c_u32 as usize;
                let dst_base = c * d;
                let src_base = u * d;
                for axis in 0..d {
                    scratch.forces[dst_base + axis] += scratch.active[src_base + axis];
                }
                scratch.counts[c] += 1.0;
            }
            for c in 0..coarse_n {
                let cnt = scratch.counts[c].max(1.0);
                let base = c * d;
                for axis in 0..d {
                    scratch.active[base + axis] = scratch.forces[base + axis] / cnt;
                }
            }
        }
        self.initialized_coords = true;
    }

    fn prolongate_to_finer_level_nd(&mut self, scratch: &mut YifanHuNdScratch) {
        let finer = self.current_level - 1;
        let fine_n = self.levels[finer].graph.num_nodes();
        let k_fine = self.levels[finer].optimal_distance;
        let d = scratch.dims;
        let Some(coarse_of) = self.levels[finer].coarse_of_finer.as_ref() else {
            return;
        };
        let Some(counts) = self.levels[finer].cluster_counts_of_finer.as_ref() else {
            return;
        };

        for (u, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
            let c = c_u32 as usize;
            let dst_base = u * d;
            let src_base = c * d;
            if counts[c] <= 1 {
                scratch.forces[dst_base..dst_base + d]
                    .copy_from_slice(&scratch.active[src_base..src_base + d]);
            } else {
                let mut norm_sq = 0.0_f64;
                for slot in scratch.dir_buf.iter_mut().take(d) {
                    let comp = self.rng.next_f64() - 0.5;
                    *slot = comp;
                    norm_sq += comp * comp;
                }
                let norm = norm_sq.sqrt();
                let radius =
                    PROLONGATION_JITTER_FACTOR * k_fine * (0.25 + 0.75 * self.rng.next_f64());
                if norm > 1e-12 {
                    let scale = radius / norm;
                    for axis in 0..d {
                        scratch.forces[dst_base + axis] =
                            scratch.active[src_base + axis] + scale * scratch.dir_buf[axis];
                    }
                } else {
                    for axis in 0..d {
                        let jitter = if axis == 0 { radius } else { 0.0 };
                        scratch.forces[dst_base + axis] = scratch.active[src_base + axis] + jitter;
                    }
                }
            }
        }
        let len = fine_n * d;
        scratch.active[..len].copy_from_slice(&scratch.forces[..len]);

        self.current_level = finer;
        self.level_iter = 0;
        self.level_step_len = self.config.step_size * k_fine;
        self.level_progress = 0;
        self.prev_energy = f64::INFINITY;
    }

    fn write_active_coords_to_layout_nd(
        &self,
        layout: &mut LayoutNd<S>,
        scratch: &YifanHuNdScratch,
    ) {
        let d = scratch.dims;
        let coords_mut = layout.coords_mut();
        if self.current_level == 0 {
            for (dst, &val) in coords_mut.iter_mut().zip(scratch.active.iter()) {
                let converted = S::from_f64(val);
                if converted.is_finite() {
                    *dst = converted;
                }
            }
        } else {
            let map = &self.fine_to_level[self.current_level];
            for (u, &c) in map.iter().enumerate() {
                let src_base = (c as usize) * d;
                let dst_base = u * d;
                for axis in 0..d {
                    let converted = S::from_f64(scratch.active[src_base + axis]);
                    if converted.is_finite() {
                        coords_mut[dst_base + axis] = converted;
                    }
                }
            }
        }
    }

    fn compute_nd_repulsion(
        &mut self,
        scratch: &mut YifanHuNdScratch,
        ln: usize,
        k_level: f64,
    ) -> CoreResult<()> {
        let d = scratch.dims;
        let eps_sq = (1e-8 * k_level * k_level).max(1e-24);
        let rep_scale = self.config.relative_strength * k_level * k_level;

        if d == 3 {
            let len = ln * 3;
            scratch.octree.build_weighted_flat(
                &scratch.active[..len],
                ln,
                &self.unit_masses[..ln],
            )?;
            scratch.octree.compute_coulomb_repulsion_flat(
                self.config.theta,
                eps_sq,
                &mut scratch.rep[..len],
            )?;
            for idx in 0..len {
                scratch.forces[idx] = rep_scale * scratch.rep[idx];
            }
        } else if d == 2 {
            for i in 0..ln {
                self.active_x[i] = scratch.active[i * 2];
                self.active_y[i] = scratch.active[i * 2 + 1];
            }
            self.quadtree.build_weighted_xy(
                &self.active_x[..ln],
                &self.active_y[..ln],
                &self.unit_masses[..ln],
            )?;
            self.quadtree.compute_coulomb_repulsion(
                self.config.theta,
                eps_sq,
                &mut self.rep_x[..ln],
                &mut self.rep_y[..ln],
            )?;
            for i in 0..ln {
                scratch.forces[i * 2] = rep_scale * self.rep_x[i];
                scratch.forces[i * 2 + 1] = rep_scale * self.rep_y[i];
            }
        } else {
            scratch.forces[..(ln * d)].fill(0.0);
            for i in 0..ln {
                let base_i = i * d;
                for j in (i + 1)..ln {
                    let base_j = j * d;
                    let mut dist_sq = 0.0_f64;
                    for axis in 0..d {
                        let delta = scratch.active[base_i + axis] - scratch.active[base_j + axis];
                        dist_sq += delta * delta;
                    }
                    let mult = rep_scale / (dist_sq + eps_sq);
                    for axis in 0..d {
                        let f =
                            mult * (scratch.active[base_i + axis] - scratch.active[base_j + axis]);
                        scratch.forces[base_i + axis] += f;
                        scratch.forces[base_j + axis] -= f;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for YifanHuSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "YifanHuSolver::step",
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
                "YifanHuSolver::step::dims",
            ));
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
            _ => YifanHuNdScratch::new(n, dims)?,
        };

        let result = (|| -> CoreResult<SolverStatus> {
            if !self.initialized_coords {
                self.initialize_coarsest_coords_nd(layout, &mut scratch);
            } else if self.current_level == 0 {
                let src = layout.coords();
                for (dst, &v) in scratch.active.iter_mut().zip(src.iter()) {
                    *dst = v.to_f64();
                }
            }

            let ln = self.levels[self.current_level].graph.num_nodes();
            let k_level = self.levels[self.current_level].optimal_distance;

            if ln <= 1 {
                self.level_iter += 1;
                self.total_iter_executed += 1;
                if self.current_level > 0 {
                    self.prolongate_to_finer_level_nd(&mut scratch);
                    self.write_active_coords_to_layout_nd(layout, &scratch);
                    self.status = SolverStatus::Running;
                } else {
                    self.write_active_coords_to_layout_nd(layout, &scratch);
                    self.status = SolverStatus::Converged;
                }
                return Ok(self.status);
            }

            self.compute_nd_repulsion(&mut scratch, ln, k_level)?;

            let level_graph = &self.levels[self.current_level].graph;
            let use_weights = self.current_level == 0 && self.has_explicit_weights;
            for u in 0..ln {
                let base_u = u * dims;
                let nbrs = level_graph.neighbors(u);
                let weights = level_graph.neighbor_weights(u);
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
                    let base_v = v * dims;
                    let mut dist_sq = 0.0_f64;
                    for axis in 0..dims {
                        let delta = scratch.active[base_v + axis] - scratch.active[base_u + axis];
                        dist_sq += delta * delta;
                    }
                    let mult = w * (dist_sq.sqrt() / k_level);
                    for axis in 0..dims {
                        let delta = scratch.active[base_v + axis] - scratch.active[base_u + axis];
                        scratch.forces[base_u + axis] += mult * delta;
                    }
                }
            }

            let mut energy = 0.0_f64;
            for f_slot in scratch.forces.iter_mut().take(ln * dims) {
                let fi = if f_slot.is_finite() { *f_slot } else { 0.0 };
                *f_slot = fi;
                energy += fi * fi;
            }
            if !energy.is_finite() {
                energy = f64::MAX;
            }

            if self.level_iter > 0 {
                if energy < self.prev_energy {
                    self.level_progress += 1;
                    if self.level_progress >= 5 {
                        self.level_progress = 0;
                        self.level_step_len =
                            (self.level_step_len / self.config.step_ratio).min(4.0 * k_level);
                    }
                } else {
                    self.level_progress = 0;
                    self.level_step_len *= self.config.step_ratio;
                }
            }
            self.prev_energy = energy;

            let mut max_disp = 0.0_f64;
            let mut max_force = 0.0_f64;
            let step_len = self.level_step_len;
            for i in 0..ln {
                let base = i * dims;
                let mut f_norm_sq = 0.0_f64;
                for axis in 0..dims {
                    let fi = scratch.forces[base + axis];
                    f_norm_sq += fi * fi;
                }
                let f_norm = f_norm_sq.sqrt();
                if f_norm > max_force {
                    max_force = f_norm;
                }
                if f_norm > 1e-12 {
                    let scale = step_len / f_norm;
                    let mut all_finite = true;
                    for axis in 0..dims {
                        let next_c =
                            scratch.active[base + axis] + scratch.forces[base + axis] * scale;
                        scratch.dir_buf[axis] = next_c;
                        if !next_c.is_finite() {
                            all_finite = false;
                        }
                    }
                    if all_finite {
                        scratch.active[base..base + dims].copy_from_slice(&scratch.dir_buf[..dims]);
                        if step_len > max_disp {
                            max_disp = step_len;
                        }
                    }
                }
            }

            self.level_iter += 1;
            self.total_iter_executed += 1;

            let level_converged =
                max_disp <= self.config.convergence_tolerance * k_level || max_force <= 1e-12;
            let level_exhausted = self.level_iter >= self.config.max_iterations_per_level;

            if level_converged || level_exhausted {
                if self.current_level > 0 {
                    self.prolongate_to_finer_level_nd(&mut scratch);
                    self.write_active_coords_to_layout_nd(layout, &scratch);
                    self.status = SolverStatus::Running;
                } else {
                    self.write_active_coords_to_layout_nd(layout, &scratch);
                    self.status = if level_converged {
                        SolverStatus::Converged
                    } else {
                        SolverStatus::Completed
                    };
                }
            } else {
                self.write_active_coords_to_layout_nd(layout, &scratch);
                self.status = SolverStatus::Running;
            }

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
        self.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.max_total_iterations
    }
}
