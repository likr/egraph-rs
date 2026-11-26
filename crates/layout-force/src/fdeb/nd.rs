use std::mem;

use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};

use super::MIN_EDGE_LEN;
use super::compatibility::edge_compatibility_nd;
use super::solver::{EdgePair, FdebSolver};
use super::step::{MAX_SPRING_ALPHA_KP, MIN_SEG_DIST};

impl<S: Scalar> FdebSolver<S> {
    /// Returns the active coordinate dimensionality of bundled edge polylines (`0` before first `step`).
    #[inline]
    pub fn point_dims(&self) -> usize {
        self.active_dims
    }

    /// Returns the flat row-major N-D coordinates of all edge control points (`[total_pts * point_dims]`).
    #[inline]
    pub fn points_nd(&self) -> &[S] {
        &self.points_nd
    }

    /// Extracts a single coordinate `axis` (`0..point_dims()`) across all bundled edge control points.
    pub fn points_axis(&self, axis: usize) -> CoreResult<Vec<S>> {
        if self.active_dims == 0 {
            return Err(CoreError::index_out_of_bounds(
                axis,
                0,
                "FdebSolver::points_axis",
            ));
        }
        if axis >= self.active_dims {
            return Err(CoreError::index_out_of_bounds(
                axis,
                self.active_dims,
                "FdebSolver::points_axis",
            ));
        }
        if self.active_dims == 2 && self.points_nd.is_empty() {
            return Ok(if axis == 0 {
                self.points_x.clone()
            } else {
                self.points_y.clone()
            });
        }
        let d = self.active_dims;
        let total_pts = self.points_nd.len() / d;
        let mut out = Vec::with_capacity(total_pts);
        for p in 0..total_pts {
            out.push(self.points_nd[p * d + axis]);
        }
        Ok(out)
    }

    /// Extracts the `z` coordinate (`axis == 2`) across all bundled edge control points.
    #[inline]
    pub fn points_z(&self) -> CoreResult<Vec<S>> {
        self.points_axis(2)
    }

    /// Returns all edge polyline points packed as row-major interleaved coordinates.
    pub fn to_interleaved_points_nd(&self) -> Vec<S> {
        if self.points_nd.is_empty() && !self.points_x.is_empty() {
            return self.to_interleaved_points();
        }
        self.points_nd.clone()
    }

    fn initialize_polylines_nd(&mut self, layout: &LayoutNd<S>) -> CoreResult<()> {
        let dims = layout.dims();
        let coords = layout.coords();
        let num_edges = self.num_edges;
        let max_flat = self
            .max_total_pts
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FDEB N-D buffer size overflow"))?;

        let mut p1_buf = vec![0.0_f64; dims];
        let mut p2_buf = vec![0.0_f64; dims];
        let mut q1_buf = vec![0.0_f64; dims];
        let mut q2_buf = vec![0.0_f64; dims];

        for ep in &mut self.edges {
            let u_base = (ep.source as usize) * dims;
            let v_base = (ep.target as usize) * dims;
            let mut len_sq = 0.0_f64;
            for axis in 0..dims {
                let delta = coords[v_base + axis].to_f64() - coords[u_base + axis].to_f64();
                len_sq += delta * delta;
            }
            ep.length = len_sq.sqrt();
        }

        self.edge_pairs.clear();
        for p in 0..num_edges {
            let ep = self.edges[p];
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let p1_base = (ep.source as usize) * dims;
            let p2_base = (ep.target as usize) * dims;
            for axis in 0..dims {
                p1_buf[axis] = coords[p1_base + axis].to_f64();
                p2_buf[axis] = coords[p2_base + axis].to_f64();
            }

            for (q, &eq) in self.edges.iter().enumerate().skip(p + 1) {
                if eq.length <= MIN_EDGE_LEN {
                    continue;
                }
                let q1_base = (eq.source as usize) * dims;
                let q2_base = (eq.target as usize) * dims;
                for axis in 0..dims {
                    q1_buf[axis] = coords[q1_base + axis].to_f64();
                    q2_buf[axis] = coords[q2_base + axis].to_f64();
                }

                let (c_e, cos_theta) =
                    edge_compatibility_nd(&p1_buf, &p2_buf, ep.length, &q1_buf, &q2_buf, eq.length);
                if c_e >= self.config.minimum_edge_compatibility {
                    let theta = cos_theta.clamp(-1.0, 1.0).acos();
                    self.edge_pairs.push(EdgePair {
                        p: p as u32,
                        q: q as u32,
                        compatibility: c_e,
                        same_dir: theta < std::f64::consts::FRAC_PI_2,
                    });
                }
            }
        }

        let init_flat = num_edges * 2 * dims;
        self.edge_offsets.clear();
        self.edge_offsets.reserve(num_edges + 1);
        self.nd_work.clear();
        self.nd_work.reserve(max_flat);
        self.points_nd.clear();
        self.points_nd.reserve(max_flat);
        self.nd_scratch = Vec::with_capacity(max_flat);

        self.edge_offsets.push(0u32);
        for (e_idx, ep) in self.edges.iter().enumerate() {
            let s_base = (ep.source as usize) * dims;
            let t_base = (ep.target as usize) * dims;
            for axis in 0..dims {
                let sv = coords[s_base + axis];
                self.nd_work.push(sv.to_f64());
                self.points_nd.push(sv);
            }
            for axis in 0..dims {
                let tv = coords[t_base + axis];
                self.nd_work.push(tv.to_f64());
                self.points_nd.push(tv);
            }
            self.edge_offsets.push(((e_idx + 1) * 2) as u32);
        }

        self.nd_vel.resize(init_flat, 0.0);
        self.initialized_polylines = true;
        Ok(())
    }

    fn subdivide_edges_nd(&mut self, dims: usize) {
        let old_pts = self.pts_per_edge;
        let new_pts = 2 * old_pts - 1;
        let total_new = self.num_edges * new_pts;
        let total_flat = total_new * dims;

        self.nd_scratch.clear();
        self.nd_scratch.reserve(total_flat);

        for e in 0..self.num_edges {
            let base = e * old_pts;
            for j in 0..(old_pts - 1) {
                let b0 = (base + j) * dims;
                let b1 = (base + j + 1) * dims;
                for axis in 0..dims {
                    self.nd_scratch.push(self.nd_work[b0 + axis]);
                }
                for axis in 0..dims {
                    self.nd_scratch
                        .push(0.5 * (self.nd_work[b0 + axis] + self.nd_work[b1 + axis]));
                }
            }
            let last_b = (base + old_pts - 1) * dims;
            for axis in 0..dims {
                self.nd_scratch.push(self.nd_work[last_b + axis]);
            }
        }

        mem::swap(&mut self.nd_work, &mut self.nd_scratch);
        self.pts_per_edge = new_pts;

        for (e, offset) in self.edge_offsets.iter_mut().enumerate() {
            *offset = (e * new_pts) as u32;
        }
        self.nd_vel.resize(total_flat, 0.0);
        self.points_nd.resize(total_flat, S::ZERO);
    }

    fn apply_forces_and_integrate_nd(&mut self, dims: usize) {
        let pts = self.pts_per_edge;
        let num_mid = pts - 2;
        if num_mid == 0 {
            return;
        }

        self.nd_vel.fill(0.0);

        for pair in &self.edge_pairs {
            let base_p = (pair.p as usize) * pts;
            let base_q = (pair.q as usize) * pts;
            let c_e = pair.compatibility;
            for i in 0..num_mid {
                let ip = (base_p + 1 + i) * dims;
                let iq = (if pair.same_dir {
                    base_q + 1 + i
                } else {
                    base_q + num_mid - i
                }) * dims;
                let mut dist_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = self.nd_work[iq + axis] - self.nd_work[ip + axis];
                    dist_sq += delta * delta;
                }
                let dist = dist_sq.sqrt();
                if dist > MIN_SEG_DIST {
                    let w = c_e / dist;
                    for axis in 0..dims {
                        let f = (self.nd_work[iq + axis] - self.nd_work[ip + axis]) * w;
                        self.nd_vel[ip + axis] += f;
                        self.nd_vel[iq + axis] -= f;
                    }
                }
            }
        }

        let alpha = self.alpha;
        for v in &mut self.nd_vel {
            *v *= alpha;
        }

        let k = self.config.spring_k;
        let num_mid_f = num_mid as f64;
        for (e, ep) in self.edges.iter().enumerate() {
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let kp = k / (num_mid_f * ep.length);
            let alpha_kp = (alpha * kp).min(MAX_SPRING_ALPHA_KP);
            let base = e * pts;
            for i in 1..=num_mid {
                let p0 = (base + i - 1) * dims;
                let p1 = (base + i) * dims;
                let p2 = (base + i + 1) * dims;
                for axis in 0..dims {
                    self.nd_vel[p1 + axis] += alpha_kp
                        * (self.nd_work[p0 + axis] - self.nd_work[p1 + axis]
                            + self.nd_work[p2 + axis]
                            - self.nd_work[p1 + axis]);
                }
            }
        }

        for (e, ep) in self.edges.iter().enumerate() {
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let max_disp = ep.length / (num_mid_f + 1.0);
            let base = e * pts;
            for i in 1..=num_mid {
                let p_base = (base + i) * dims;
                let mut norm_sq = 0.0_f64;
                for axis in 0..dims {
                    let v = if self.nd_vel[p_base + axis].is_finite() {
                        self.nd_vel[p_base + axis]
                    } else {
                        0.0
                    };
                    self.nd_vel[p_base + axis] = v;
                    norm_sq += v * v;
                }
                let norm = norm_sq.sqrt();
                if !norm.is_finite() {
                    continue;
                }
                let scale = if norm > max_disp && norm > 0.0 {
                    max_disp / norm
                } else {
                    1.0
                };
                for axis in 0..dims {
                    let idx = p_base + axis;
                    let next_c = self.nd_work[idx] + self.nd_vel[idx] * scale;
                    if next_c.is_finite() {
                        self.nd_work[idx] = next_c;
                    }
                }
            }
        }

        for (dst, &val) in self.points_nd.iter_mut().zip(self.nd_work.iter()) {
            *dst = S::from_f64(val);
        }
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for FdebSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FdebSolver::step",
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
                "FdebSolver::step::dims",
            ));
        }

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if !self.initialized_polylines {
            self.initialize_polylines_nd(layout)?;
        }

        if self.num_edges == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        while self.current_cycle < self.cycle_iters.len()
            && self.cycle_iters[self.current_cycle] == 0
        {
            self.current_cycle += 1;
            self.alpha *= self.config.s_step;
        }

        if self.current_cycle >= self.cycle_iters.len() || self.iter >= self.total_iterations {
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }

        if self.cycle_iter == 0 {
            self.subdivide_edges_nd(dims);
        }

        self.apply_forces_and_integrate_nd(dims);

        self.iter += 1;
        self.cycle_iter += 1;
        while self.current_cycle < self.cycle_iters.len()
            && self.cycle_iter >= self.cycle_iters[self.current_cycle]
        {
            self.cycle_iter = 0;
            self.current_cycle += 1;
            self.alpha *= self.config.s_step;
        }

        self.status =
            if self.iter >= self.total_iterations || self.current_cycle >= self.cycle_iters.len() {
                SolverStatus::Completed
            } else {
                SolverStatus::Running
            };

        Ok(self.status)
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
        self.total_iterations
    }
}
