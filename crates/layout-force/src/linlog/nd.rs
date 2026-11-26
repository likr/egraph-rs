use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, LayoutNd, Scalar, SolverStatus,
};
use egraph_spatial_trees::ArenaOctree3d;

use super::solver::LinLogSolver;
use super::step::{COULOMB_EPS_SQ, MAX_LINLOG_SPEED, MIN_LINLOG_SPEED};

/// Lazy N-D scratch buffers for [`LinLogSolver`] on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct LinLogNdScratch {
    dims: usize,
    coords: Vec<f64>,
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
    rep: Vec<f64>,
    barycenter: Vec<f64>,
    min_axis: Vec<f64>,
    max_axis: Vec<f64>,
    octree: ArenaOctree3d<f64>,
}

impl LinLogNdScratch {
    fn new(num_nodes: usize, dims: usize, use_barnes_hut: bool) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LinLog N-D buffer size overflow"))?;
        let rep_len = if use_barnes_hut && dims == 3 {
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
            barycenter: vec![0.0; dims],
            min_axis: vec![0.0; dims],
            max_axis: vec![0.0; dims],
            octree: ArenaOctree3d::new(),
        })
    }
}

impl<S: Scalar> LinLogSolver<S> {
    fn step_nd_repulsion(
        &mut self,
        scratch: &mut LinLogNdScratch,
        r_exp: f64,
        is_coulomb: bool,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        let d = scratch.dims;

        if self.config.use_barnes_hut {
            if d == 3 {
                scratch
                    .octree
                    .build_weighted_flat(&scratch.coords, n, &self.masses)?;
                scratch.octree.compute_power_repulsion_flat(
                    self.config.theta,
                    r_exp,
                    COULOMB_EPS_SQ,
                    &mut scratch.rep,
                )?;
                for (i, &mi) in self.masses.iter().enumerate().take(n) {
                    let base = i * 3;
                    scratch.forces[base] = mi * scratch.rep[base];
                    scratch.forces[base + 1] = mi * scratch.rep[base + 1];
                    scratch.forces[base + 2] = mi * scratch.rep[base + 2];
                }
            } else {
                for i in 0..n {
                    self.x_buf[i] = scratch.coords[i * 2];
                    self.y_buf[i] = scratch.coords[i * 2 + 1];
                }
                if is_coulomb {
                    self.quadtree
                        .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
                    self.quadtree.compute_coulomb_repulsion(
                        self.config.theta,
                        COULOMB_EPS_SQ,
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                } else {
                    self.gen_quadtree
                        .build(&self.x_buf, &self.y_buf, &self.masses);
                    self.gen_quadtree.compute_general_repulsion(
                        &self.x_buf,
                        &self.y_buf,
                        &self.masses,
                        self.config.theta,
                        r_exp,
                        COULOMB_EPS_SQ,
                        &mut self.rep_x,
                        &mut self.rep_y,
                    );
                }
                for i in 0..n {
                    let mi = self.masses[i];
                    scratch.forces[i * 2] = mi * self.rep_x[i];
                    scratch.forces[i * 2 + 1] = mi * self.rep_y[i];
                }
            }
        } else {
            scratch.forces.fill(0.0);
            let pow_exp = 0.5 * (r_exp - 1.0);
            for i in 0..n {
                let base_i = i * d;
                let mi = self.masses[i];
                for (j, &mj) in self.masses.iter().enumerate().take(n).skip(i + 1) {
                    let base_j = j * d;
                    let mut d_sq = 0.0_f64;
                    for axis in 0..d {
                        let delta = scratch.coords[base_i + axis] - scratch.coords[base_j + axis];
                        d_sq += delta * delta;
                    }
                    let d_sq_reg = d_sq + COULOMB_EPS_SQ;
                    let mult = if is_coulomb {
                        (mi * mj) / d_sq_reg
                    } else {
                        (mi * mj) * d_sq_reg.powf(pow_exp)
                    };
                    for axis in 0..d {
                        let f =
                            mult * (scratch.coords[base_i + axis] - scratch.coords[base_j + axis]);
                        scratch.forces[base_i + axis] += f;
                        scratch.forces[base_j + axis] -= f;
                    }
                }
            }
        }
        Ok(())
    }

    fn step_nd_attraction_and_gravity(&mut self, scratch: &mut LinLogNdScratch) {
        let n = self.num_nodes;
        let d = scratch.dims;
        let a_exp = self.config.attraction_exponent;
        let a_is_zero = a_exp.abs() <= 1e-12;
        let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
        let a_hess_scale = a_exp.abs().max(1.0);
        self.att_hess.fill(0.0);

        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let base_u = u * d;
            let base_v = v * d;
            let mut d_sq = 0.0_f64;
            for axis in 0..d {
                let delta = scratch.coords[base_v + axis] - scratch.coords[base_u + axis];
                d_sq += delta * delta;
            }
            let (factor, hess) = if a_is_zero {
                if d_sq > 1e-24 {
                    let dist = d_sq.sqrt();
                    (w / dist, w / dist)
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

            for axis in 0..d {
                let a = factor * (scratch.coords[base_v + axis] - scratch.coords[base_u + axis]);
                scratch.forces[base_u + axis] += a;
                scratch.forces[base_v + axis] -= a;
            }
            self.att_hess[u] += hess;
            self.att_hess[v] += hess;
        }

        if self.config.gravity > 0.0 && self.total_mass > 0.0 {
            scratch.barycenter.fill(0.0);
            for (i, &m) in self.masses.iter().enumerate().take(n) {
                let base = i * d;
                for axis in 0..d {
                    scratch.barycenter[axis] += m * scratch.coords[base + axis];
                }
            }
            for b in &mut scratch.barycenter {
                *b /= self.total_mass;
            }

            let g = self.config.gravity;
            for i in 0..n {
                let base = i * d;
                let mut d_sq = 0.0_f64;
                for axis in 0..d {
                    let delta = scratch.barycenter[axis] - scratch.coords[base + axis];
                    d_sq += delta * delta;
                }
                let dist = d_sq.sqrt();
                if dist > 1e-12 {
                    let factor = if a_is_zero {
                        (g * self.masses[i]) / dist
                    } else if a_is_one {
                        g * self.masses[i]
                    } else {
                        g * self.masses[i] * dist.powf(a_exp - 1.0)
                    };
                    for axis in 0..d {
                        scratch.forces[base + axis] +=
                            factor * (scratch.barycenter[axis] - scratch.coords[base + axis]);
                    }
                    self.att_hess[i] += a_hess_scale * factor;
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for LinLogSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "LinLogSolver::step",
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
                "LinLogSolver::step::dims",
            ));
        }
        if self.config.use_barnes_hut && dims != 2 && dims != 3 {
            return Err(CoreError::unsupported(format!(
                "LinLog Barnes-Hut repulsion only supports 2D and 3D layouts, got {dims}D"
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
            _ => LinLogNdScratch::new(n, dims, self.config.use_barnes_hut)?,
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

            let r_exp = self.config.repulsion_exponent;
            let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;
            self.step_nd_repulsion(&mut scratch, r_exp, is_coulomb)?;
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

            let target_speed = if total_swing > 1e-12 {
                (total_traction / total_swing).clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED)
            } else {
                (self.global_speed * 1.5).min(MAX_LINLOG_SPEED)
            };
            if self.iter > 0 {
                let max_rise = 0.5 * self.global_speed;
                self.global_speed += (target_speed - self.global_speed).min(max_rise);
            } else {
                self.global_speed = target_speed;
            }
            self.global_speed = self.global_speed.clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED);

            let speed = self.global_speed;
            let max_disp = self.max_displacement;
            let coords_mut = layout.coords_mut();
            for i in 0..n {
                let base = i * dims;
                let local_factor =
                    speed / (1.0 + (speed * self.swing_buf[i]).sqrt() + 0.5 * self.att_hess[i]);
                let mut disp_sq = 0.0_f64;
                for axis in 0..dims {
                    let da = scratch.forces[base + axis] * local_factor;
                    disp_sq += da * da;
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
                    let next_c = S::from_f64(scratch.coords[idx] + fi * local_factor * cap);
                    if next_c.is_finite() {
                        coords_mut[idx] = next_c;
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
