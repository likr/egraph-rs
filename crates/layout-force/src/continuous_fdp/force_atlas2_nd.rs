use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, LayoutNd, Scalar, SolverStatus,
};
use egraph_spatial_trees::ArenaOctree3d;

use super::force_atlas2::ForceAtlas2Solver;
use super::force_atlas2_step::{COULOMB_EPS_SQ, MIN_FA2_SPEED};
use super::overlap_quadtree::{OVERLAP_DPRIME_MIN, OVERLAP_REPULSION_K};

/// Lazy N-D scratch buffers for [`ForceAtlas2Solver`] on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct ForceAtlas2NdScratch {
    dims: usize,
    coords: Vec<f64>,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    rep: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
    octree: ArenaOctree3d<f64>,
}

impl ForceAtlas2NdScratch {
    fn new(
        num_nodes: usize,
        dims: usize,
        use_barnes_hut: bool,
        prevent_overlap: bool,
    ) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 N-D buffer size overflow"))?;
        let rep_len = if use_barnes_hut && !prevent_overlap && dims == 3 {
            total
        } else {
            0
        };
        Ok(Self {
            dims,
            coords: vec![0.0; total],
            forces: vec![0.0; total],
            prev_forces: vec![0.0; total],
            rep: vec![0.0; rep_len],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
            octree: ArenaOctree3d::new(),
        })
    }
}

impl<S: Scalar> ForceAtlas2Solver<S> {
    fn compute_nd_overlap_exact(&self, scratch: &mut ForceAtlas2NdScratch) {
        let n = self.num_nodes;
        let d = scratch.dims;
        let sr = self.config.scaling_ratio;
        scratch.forces.fill(0.0);
        for i in 0..n {
            let base_i = i * d;
            let mi = self.masses[i];
            let ri = self.node_radii[i];
            for j in (i + 1)..n {
                let base_j = j * d;
                let mj = self.masses[j];
                let r_sum = ri + self.node_radii[j];
                let mut dist_sq = 0.0_f64;
                for axis in 0..d {
                    let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                    dist_sq += delta * delta;
                }
                let dist = dist_sq.sqrt();
                let d_prime = dist - r_sum;
                if d_prime > 0.0 {
                    let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
                    let mult = (sr * mi * mj) / (dist * d_prime_safe);
                    for axis in 0..d {
                        let f =
                            mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                        scratch.forces[base_i + axis] += f;
                        scratch.forces[base_j + axis] -= f;
                    }
                } else if d_prime < 0.0 {
                    let mag = OVERLAP_REPULSION_K * mi * mj;
                    if dist > 1e-12 {
                        let scale = mag / dist;
                        for axis in 0..d {
                            let f = scale
                                * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                            scratch.forces[base_i + axis] += f;
                            scratch.forces[base_j + axis] -= f;
                        }
                    } else {
                        // Deterministic axis-0 separation when vertices are coincident.
                        scratch.forces[base_i] += mag;
                        scratch.forces[base_j] -= mag;
                    }
                }
            }
        }
    }

    fn compute_nd_coulomb_exact(&self, scratch: &mut ForceAtlas2NdScratch) {
        let n = self.num_nodes;
        let d = scratch.dims;
        let sr = self.config.scaling_ratio;
        scratch.forces.fill(0.0);
        for i in 0..n {
            let base_i = i * d;
            let mi_sr = sr * self.masses[i];
            for (j, &mj) in self.masses.iter().enumerate().take(n).skip(i + 1) {
                let base_j = j * d;
                let mut dist_sq = 0.0_f64;
                for axis in 0..d {
                    let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                    dist_sq += delta * delta;
                }
                let mult = (mi_sr * mj) / (dist_sq + COULOMB_EPS_SQ);
                for axis in 0..d {
                    let f = mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                    scratch.forces[base_i + axis] += f;
                    scratch.forces[base_j + axis] -= f;
                }
            }
        }
    }

    fn step_nd_repulsion(&mut self, scratch: &mut ForceAtlas2NdScratch) -> CoreResult<()> {
        let n = self.num_nodes;
        let d = scratch.dims;
        let prevent_overlap = self.config.prevent_overlap;

        if prevent_overlap {
            if self.config.use_barnes_hut && d == 2 {
                for i in 0..n {
                    self.x_buf[i] = scratch.coords[i * 2];
                    self.y_buf[i] = scratch.coords[i * 2 + 1];
                }
                self.overlap_tree
                    .build(&self.x_buf, &self.y_buf, &self.masses, &self.node_radii);
                self.overlap_tree.compute_repulsion(
                    &self.x_buf,
                    &self.y_buf,
                    &self.masses,
                    &self.node_radii,
                    self.config.theta,
                    self.config.scaling_ratio,
                    self.config.seed,
                    self.iter,
                    &mut self.fx,
                    &mut self.fy,
                );
                for i in 0..n {
                    scratch.forces[i * 2] = self.fx[i];
                    scratch.forces[i * 2 + 1] = self.fy[i];
                }
            } else {
                self.compute_nd_overlap_exact(scratch);
            }
        } else if self.config.use_barnes_hut {
            let sr = self.config.scaling_ratio;
            if d == 3 {
                scratch
                    .octree
                    .build_weighted_flat(&scratch.coords, n, &self.masses)?;
                scratch.octree.compute_coulomb_repulsion_flat(
                    self.config.theta,
                    COULOMB_EPS_SQ,
                    &mut scratch.rep,
                )?;
                for (i, &mi) in self.masses.iter().enumerate().take(n) {
                    let scale = sr * mi;
                    let base = i * 3;
                    scratch.forces[base] = scale * scratch.rep[base];
                    scratch.forces[base + 1] = scale * scratch.rep[base + 1];
                    scratch.forces[base + 2] = scale * scratch.rep[base + 2];
                }
            } else {
                for i in 0..n {
                    self.x_buf[i] = scratch.coords[i * 2];
                    self.y_buf[i] = scratch.coords[i * 2 + 1];
                }
                self.quadtree
                    .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
                self.quadtree.compute_coulomb_repulsion(
                    self.config.theta,
                    COULOMB_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
                for i in 0..n {
                    let scale = sr * self.masses[i];
                    scratch.forces[i * 2] = scale * self.rep_x[i];
                    scratch.forces[i * 2 + 1] = scale * self.rep_y[i];
                }
            }
        } else {
            self.compute_nd_coulomb_exact(scratch);
        }
        Ok(())
    }

    fn step_nd_attraction_and_gravity(&self, scratch: &mut ForceAtlas2NdScratch) {
        let n = self.num_nodes;
        let d = scratch.dims;
        let prevent_overlap = self.config.prevent_overlap;
        let linlog = self.config.linlog_mode;

        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let base_u = u * d;
            let base_v = v * d;
            let mut dist_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
                dist_sq += delta * delta;
            }
            let factor = if prevent_overlap {
                let dist = dist_sq.sqrt();
                let d_prime = dist - (self.node_radii[u] + self.node_radii[v]);
                if d_prime > 0.0 && dist > 1e-12 {
                    if linlog {
                        w * d_prime.ln_1p() / dist
                    } else {
                        w * d_prime / dist
                    }
                } else {
                    0.0
                }
            } else if linlog {
                if dist_sq > 1e-24 {
                    let dist = dist_sq.sqrt();
                    w * dist.ln_1p() / dist
                } else {
                    0.0
                }
            } else {
                w
            };
            if factor == 0.0 {
                continue;
            }
            for axis in 0..d {
                let a = factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
                scratch.forces[base_u + axis] += a;
                scratch.forces[base_v + axis] -= a;
            }
        }

        if self.config.gravity > 0.0 {
            let g = self.config.gravity;
            if self.config.strong_gravity {
                for (i, &mi) in self.masses.iter().enumerate().take(n) {
                    let gf = g * mi;
                    let base = i * d;
                    for axis in 0..d {
                        scratch.forces[base + axis] -= gf * scratch.coords[base + axis];
                    }
                }
            } else {
                for (i, &mi) in self.masses.iter().enumerate().take(n) {
                    let base = i * d;
                    let mut dist_sq = 0.0_f64;
                    for axis in 0..d {
                        let c = scratch.coords[base + axis];
                        dist_sq += c * c;
                    }
                    let dist = dist_sq.sqrt();
                    if dist > 1e-12 {
                        let gf = (g * mi) / dist;
                        for axis in 0..d {
                            scratch.forces[base + axis] -= gf * scratch.coords[base + axis];
                        }
                    }
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for ForceAtlas2Solver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ForceAtlas2Solver::step",
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
                "ForceAtlas2Solver::step::dims",
            ));
        }
        if self.config.use_barnes_hut && dims != 2 && dims != 3 {
            return Err(CoreError::unsupported(format!(
                "ForceAtlas2 Barnes-Hut repulsion only supports 2D and 3D layouts, got {dims}D"
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
            _ => ForceAtlas2NdScratch::new(
                n,
                dims,
                self.config.use_barnes_hut,
                self.config.prevent_overlap,
            )?,
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
                let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
                for v in &mut scratch.coords {
                    *v += 1e-4 * (rng.next_f64() - 0.5);
                }
            }

            self.step_nd_repulsion(&mut scratch)?;
            self.step_nd_attraction_and_gravity(&mut scratch);

            let mut total_swing = 0.0_f64;
            let mut total_traction = 0.0_f64;
            for i in 0..n {
                let base = i * dims;
                let mut swg_sq = 0.0_f64;
                let mut tra_sq = 0.0_f64;
                for axis in 0..dims {
                    let idx = base + axis;
                    let fi = if scratch.forces[idx].is_finite() {
                        scratch.forces[idx]
                    } else {
                        0.0
                    };
                    scratch.forces[idx] = fi;
                    let df = fi - scratch.prev_forces[idx];
                    let sf = fi + scratch.prev_forces[idx];
                    swg_sq += df * df;
                    tra_sq += sf * sf;
                }
                let swg = swg_sq.sqrt();
                let tra = 0.5 * tra_sq.sqrt();
                self.swing_buf[i] = swg;
                let m = self.masses[i];
                total_swing += m * swg;
                total_traction += m * tra;
            }

            let jt = self.config.jitter_tolerance;
            let max_speed = 10.0 * jt.max(1.0);
            let target_speed = if total_swing > 1e-12 {
                (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
            } else {
                (self.global_speed * 1.5).min(max_speed)
            };
            if self.iter > 0 {
                let max_rise = 0.5 * self.global_speed;
                self.global_speed += (target_speed - self.global_speed).min(max_rise);
            } else {
                self.global_speed = target_speed;
            }
            self.global_speed = self.global_speed.clamp(MIN_FA2_SPEED, max_speed);

            let speed = self.global_speed;
            let speed_scale = if self.config.prevent_overlap {
                0.1
            } else {
                1.0
            };
            let max_disp = self.max_displacement;
            let coords_mut = layout.coords_mut();
            for (i, &swing) in self.swing_buf.iter().enumerate().take(n) {
                let base = i * dims;
                let local_factor = speed_scale * (speed / (1.0 + (speed * swing).sqrt()));
                let mut disp_sq = 0.0_f64;
                for axis in 0..dims {
                    let d_axis = scratch.forces[base + axis] * local_factor;
                    disp_sq += d_axis * d_axis;
                }
                let disp = disp_sq.sqrt();
                let cap = if disp > max_disp {
                    max_disp / disp
                } else {
                    1.0
                };
                for axis in 0..dims {
                    let idx = base + axis;
                    let fi = scratch.forces[idx];
                    let next_val = S::from_f64(scratch.coords[idx] + fi * local_factor * cap);
                    if next_val.is_finite() {
                        coords_mut[idx] = next_val;
                    }
                    scratch.prev_forces[idx] = fi;
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
