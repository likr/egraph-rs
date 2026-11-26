use std::mem;

use egraph_core::{CoreError, CoreResult, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::MIN_EDGE_LEN;
use super::compatibility::edge_compatibility;
use super::solver::{EdgePair, FdebSolver};

const MIN_SEG_DIST: f64 = 1e-12;
const MAX_SPRING_ALPHA_KP: f64 = 0.45;

impl<S: Scalar> FdebSolver<S> {
    fn initialize_polylines(&mut self, layout: &Layout2d<S>) {
        let xs = layout.x();
        let ys = layout.y();

        for ep in &mut self.edges {
            let ux = xs[ep.source as usize].to_f64();
            let uy = ys[ep.source as usize].to_f64();
            let vx = xs[ep.target as usize].to_f64();
            let vy = ys[ep.target as usize].to_f64();
            ep.length = (vx - ux).hypot(vy - uy);
        }

        // Pre-compute compatible edge pairs.
        self.edge_pairs.clear();
        let num_edges = self.num_edges;
        for p in 0..num_edges {
            let ep = self.edges[p];
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let p1x = xs[ep.source as usize].to_f64();
            let p1y = ys[ep.source as usize].to_f64();
            let p2x = xs[ep.target as usize].to_f64();
            let p2y = ys[ep.target as usize].to_f64();

            for (q, &eq) in self.edges.iter().enumerate().skip(p + 1) {
                if eq.length <= MIN_EDGE_LEN {
                    continue;
                }
                let q1x = xs[eq.source as usize].to_f64();
                let q1y = ys[eq.source as usize].to_f64();
                let q2x = xs[eq.target as usize].to_f64();
                let q2y = ys[eq.target as usize].to_f64();

                let (c_e, cos_theta) = edge_compatibility(
                    p1x, p1y, p2x, p2y, ep.length, q1x, q1y, q2x, q2y, eq.length,
                );
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

        // Initialize unsubdivided 2-point polylines per edge (source, target).
        let init_pts = num_edges * 2;
        self.edge_offsets.clear();
        self.edge_offsets.reserve(num_edges + 1);
        self.work_x.clear();
        self.work_x.reserve(self.max_total_pts);
        self.work_y.clear();
        self.work_y.reserve(self.max_total_pts);
        self.points_x.clear();
        self.points_x.reserve(self.max_total_pts);
        self.points_y.clear();
        self.points_y.reserve(self.max_total_pts);
        self.scratch_x = Vec::with_capacity(self.max_total_pts);
        self.scratch_y = Vec::with_capacity(self.max_total_pts);

        self.edge_offsets.push(0u32);
        for (e_idx, ep) in self.edges.iter().enumerate() {
            let sx = xs[ep.source as usize];
            let sy = ys[ep.source as usize];
            let tx = xs[ep.target as usize];
            let ty = ys[ep.target as usize];
            self.work_x.push(sx.to_f64());
            self.work_y.push(sy.to_f64());
            self.work_x.push(tx.to_f64());
            self.work_y.push(ty.to_f64());
            self.points_x.push(sx);
            self.points_y.push(sy);
            self.points_x.push(tx);
            self.points_y.push(ty);
            self.edge_offsets.push(((e_idx + 1) * 2) as u32);
        }

        self.vx.resize(init_pts, 0.0);
        self.vy.resize(init_pts, 0.0);
        self.initialized_polylines = true;
    }

    fn subdivide_edges(&mut self) {
        let old_pts = self.pts_per_edge;
        let new_pts = 2 * old_pts - 1;
        let total_new = self.num_edges * new_pts;

        self.scratch_x.clear();
        self.scratch_y.clear();
        self.scratch_x.reserve(total_new);
        self.scratch_y.reserve(total_new);

        for e in 0..self.num_edges {
            let base = e * old_pts;
            for j in 0..(old_pts - 1) {
                let x0 = self.work_x[base + j];
                let y0 = self.work_y[base + j];
                let x1 = self.work_x[base + j + 1];
                let y1 = self.work_y[base + j + 1];
                self.scratch_x.push(x0);
                self.scratch_y.push(y0);
                self.scratch_x.push(0.5 * (x0 + x1));
                self.scratch_y.push(0.5 * (y0 + y1));
            }
            let last_x = self.work_x[base + old_pts - 1];
            let last_y = self.work_y[base + old_pts - 1];
            self.scratch_x.push(last_x);
            self.scratch_y.push(last_y);
        }

        mem::swap(&mut self.work_x, &mut self.scratch_x);
        mem::swap(&mut self.work_y, &mut self.scratch_y);
        self.pts_per_edge = new_pts;

        for (e, offset) in self.edge_offsets.iter_mut().enumerate() {
            *offset = (e * new_pts) as u32;
        }
        self.vx.resize(total_new, 0.0);
        self.vy.resize(total_new, 0.0);
        self.points_x.resize(total_new, S::ZERO);
        self.points_y.resize(total_new, S::ZERO);
    }

    fn apply_forces_and_integrate(&mut self) {
        let pts = self.pts_per_edge;
        let num_mid = pts - 2;
        if num_mid == 0 {
            return;
        }

        self.vx.fill(0.0);
        self.vy.fill(0.0);

        // 1. Electrostatic attraction between compatible edge pairs (force accumulation).
        for pair in &self.edge_pairs {
            let base_p = (pair.p as usize) * pts;
            let base_q = (pair.q as usize) * pts;
            let c_e = pair.compatibility;
            for i in 0..num_mid {
                let ip = base_p + 1 + i;
                let iq = if pair.same_dir {
                    base_q + 1 + i
                } else {
                    base_q + num_mid - i
                };
                let dx = self.work_x[iq] - self.work_x[ip];
                let dy = self.work_y[iq] - self.work_y[ip];
                let dist = dx.hypot(dy);
                if dist > MIN_SEG_DIST {
                    let w = c_e / dist;
                    let fx = dx * w;
                    let fy = dy * w;
                    self.vx[ip] += fx;
                    self.vy[ip] += fy;
                    self.vx[iq] -= fx;
                    self.vy[iq] -= fy;
                }
            }
        }

        // Scale electrostatic forces by step size `alpha` before adding clamped spring steps.
        let alpha = self.alpha;
        for v in &mut self.vx {
            *v *= alpha;
        }
        for v in &mut self.vy {
            *v *= alpha;
        }

        // 2. Clamped spring displacements along each subdivided edge: (alpha * kp).min(0.45).
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
                let p0x = self.work_x[base + i - 1];
                let p0y = self.work_y[base + i - 1];
                let p1x = self.work_x[base + i];
                let p1y = self.work_y[base + i];
                let p2x = self.work_x[base + i + 1];
                let p2y = self.work_y[base + i + 1];
                self.vx[base + i] += alpha_kp * (p0x - p1x + p2x - p1x);
                self.vy[base + i] += alpha_kp * (p0y - p1y + p2y - p1y);
            }
        }

        // 3. Per-point max displacement cap & explicit Euler update (endpoints remain fixed).
        for (e, ep) in self.edges.iter().enumerate() {
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let max_disp = ep.length / (num_mid_f + 1.0);
            let base = e * pts;
            for i in 1..=num_mid {
                let idx = base + i;
                let mut dx = if self.vx[idx].is_finite() {
                    self.vx[idx]
                } else {
                    0.0
                };
                let mut dy = if self.vy[idx].is_finite() {
                    self.vy[idx]
                } else {
                    0.0
                };
                let norm = dx.hypot(dy);
                if !norm.is_finite() {
                    continue;
                }
                if norm > max_disp && norm > 0.0 {
                    let scale = max_disp / norm;
                    dx *= scale;
                    dy *= scale;
                }
                let nx = self.work_x[idx] + dx;
                let ny = self.work_y[idx] + dy;
                if nx.is_finite() {
                    self.work_x[idx] = nx;
                }
                if ny.is_finite() {
                    self.work_y[idx] = ny;
                }
            }
        }

        for i in 0..self.work_x.len() {
            self.points_x[i] = S::from_f64(self.work_x[i]);
            self.points_y[i] = S::from_f64(self.work_y[i]);
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for FdebSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FdebSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if !self.initialized_polylines {
            self.initialize_polylines(layout);
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
            self.subdivide_edges();
        }

        self.apply_forces_and_integrate();

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
