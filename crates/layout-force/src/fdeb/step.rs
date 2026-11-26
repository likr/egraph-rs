use std::mem;

use egraph_core::{CoreResult, Scalar, SolverStatus};

use super::MIN_EDGE_LEN;
use super::solver::FdebSolver;

const MIN_SEG_DIST: f64 = 1e-12;
const MAX_SPRING_ALPHA_KP: f64 = 0.45;

impl<S: Scalar> FdebSolver<S> {
    /// Executes a single FDEB force iteration (subdividing edges when entering a new cycle)
    /// and returns the updated [`SolverStatus`].
    pub fn step(&mut self) -> CoreResult<SolverStatus> {
        if self.status.is_finished() {
            return Ok(self.status);
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

    /// Executes up to `steps` iterations or stops early upon completion.
    pub fn run_steps(&mut self, steps: usize) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if self.status.is_finished() {
                break;
            }
            let st = self.step()?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    /// Runs the solver to completion.
    pub fn run(&mut self) -> CoreResult<SolverStatus> {
        while !self.status.is_finished() {
            let st = self.step()?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
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
