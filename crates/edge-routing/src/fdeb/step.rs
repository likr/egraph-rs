use std::mem;

use egraph_core::{
    CoreError, CoreResult, EdgePolylines, EuclideanLayout, IterativeSolver, SolverStatus,
};

use super::MIN_EDGE_LEN;
use super::compatibility::{norm_diff, norm_vec};
use super::solver::{FdebSolver, validate_work_for_scalar, write_work_to_storage};

pub(super) const MIN_SEG_DIST: f64 = 1e-12;
pub(super) const MAX_SPRING_ALPHA_KP: f64 = 0.45;

impl<L: EuclideanLayout> FdebSolver<L> {
    fn subdivide_work(&mut self) -> CoreResult<(usize, usize, Vec<u32>)> {
        let dims = self.dims;
        let old_pts = self.pts_per_edge;
        let new_pts = 2 * old_pts - 1;
        let total_new = self
            .num_edges
            .checked_mul(new_pts)
            .ok_or_else(|| CoreError::invalid_input("FDEB point buffer size overflow"))?;
        let total_flat = total_new
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FDEB flat buffer size overflow"))?;

        self.scratch.clear();
        self.scratch.reserve(total_flat);

        for e in 0..self.num_edges {
            let base = e * old_pts;
            for j in 0..(old_pts - 1) {
                let b0 = (base + j) * dims;
                let b1 = (base + j + 1) * dims;
                for axis in 0..dims {
                    self.scratch.push(self.work[b0 + axis]);
                }
                for axis in 0..dims {
                    let mid = 0.5 * (self.work[b0 + axis] + self.work[b1 + axis]);
                    if !mid.is_finite() {
                        return Err(CoreError::numerical_breakdown(
                            "FDEB midpoint subdivision became non-finite",
                        ));
                    }
                    self.scratch.push(mid);
                }
            }
            let last_b = (base + old_pts - 1) * dims;
            for axis in 0..dims {
                self.scratch.push(self.work[last_b + axis]);
            }
        }

        mem::swap(&mut self.work, &mut self.scratch);
        self.pts_per_edge = new_pts;
        self.vel.resize(total_flat, 0.0);

        let mut new_offsets = Vec::with_capacity(self.num_edges + 1);
        new_offsets.push(0u32);
        for e in 0..self.num_edges {
            new_offsets.push(((e + 1) * new_pts) as u32);
        }

        Ok((new_pts, total_new, new_offsets))
    }

    fn apply_forces_and_integrate(&mut self) -> CoreResult<()> {
        let dims = self.dims;
        let pts = self.pts_per_edge;
        let num_mid = pts - 2;
        if num_mid == 0 {
            return Ok(());
        }

        self.vel.fill(0.0);

        // 1. Electrostatic attraction between compatible edge pairs.
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
                let dist = norm_diff(&self.work[iq..iq + dims], &self.work[ip..ip + dims]);
                if !dist.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "FDEB pairwise distance became non-finite",
                    ));
                }
                if dist > MIN_SEG_DIST {
                    let w = c_e / dist;
                    for axis in 0..dims {
                        let f = (self.work[iq + axis] - self.work[ip + axis]) * w;
                        self.vel[ip + axis] += f;
                        self.vel[iq + axis] -= f;
                    }
                }
            }
        }

        // Scale electrostatic forces by step size `alpha` before adding clamped spring steps.
        let alpha = self.alpha;
        for v in &mut self.vel {
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
                let p0 = (base + i - 1) * dims;
                let p1 = (base + i) * dims;
                let p2 = (base + i + 1) * dims;
                for axis in 0..dims {
                    self.vel[p1 + axis] += alpha_kp
                        * (self.work[p0 + axis] - self.work[p1 + axis] + self.work[p2 + axis]
                            - self.work[p1 + axis]);
                }
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
                let p_base = (base + i) * dims;
                let norm = norm_vec(&self.vel[p_base..p_base + dims]);
                if !norm.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "FDEB velocity norm became non-finite",
                    ));
                }
                let scale = if norm > max_disp && norm > 0.0 {
                    max_disp / norm
                } else {
                    1.0
                };
                for axis in 0..dims {
                    let idx = p_base + axis;
                    let next_c = self.work[idx] + self.vel[idx] * scale;
                    if !next_c.is_finite() {
                        return Err(CoreError::numerical_breakdown(
                            "FDEB coordinate update became non-finite",
                        ));
                    }
                    self.work[idx] = next_c;
                }
            }
        }

        Ok(())
    }

    fn fail_with(&mut self, err: CoreError) -> CoreResult<SolverStatus> {
        self.status = SolverStatus::Failed;
        Err(err)
    }
}

impl<L: EuclideanLayout> IterativeSolver for FdebSolver<L> {
    type Target = EdgePolylines<L>;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        let actual_dims = target.points().dims();
        if actual_dims != self.dims {
            return Err(CoreError::dimension_mismatch(
                self.dims,
                actual_dims,
                "FdebSolver::validate_target::dims",
            ));
        }
        if target.num_edges() != self.num_edges {
            return Err(CoreError::dimension_mismatch(
                self.num_edges,
                target.num_edges(),
                "FdebSolver::validate_target::num_edges",
            ));
        }
        if target.total_points() != self.expected_total_points {
            return Err(CoreError::dimension_mismatch(
                self.expected_total_points,
                target.total_points(),
                "FdebSolver::validate_target::total_points",
            ));
        }
        if target.edge_offsets() != self.expected_offsets.as_slice() {
            return Err(CoreError::invalid_input(
                "FdebSolver target edge_offsets do not match solver state",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

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
            if !self.alpha.is_finite() {
                return self.fail_with(CoreError::numerical_breakdown(
                    "FDEB step size alpha became non-finite",
                ));
            }
        }

        if self.current_cycle >= self.cycle_iters.len() || self.iter >= self.total_iterations {
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }

        if self.cycle_iter == 0 {
            let (_new_pts, total_new, new_offsets) = match self.subdivide_work() {
                Ok(res) => res,
                Err(err) => return self.fail_with(err),
            };
            if let Err(err) = self.apply_forces_and_integrate() {
                return self.fail_with(err);
            }
            if let Err(err) = validate_work_for_scalar::<L::Scalar>(&self.work) {
                return self.fail_with(err);
            }
            let mut new_points = match target.points().zeros_like(total_new) {
                Ok(p) => p,
                Err(err) => return self.fail_with(err),
            };
            if let Err(err) =
                write_work_to_storage(&self.work, self.dims, new_points.as_euclidean_storage_mut())
            {
                return self.fail_with(err);
            }
            if let Err(err) = target.replace_parts(new_offsets.clone(), new_points) {
                return self.fail_with(err);
            }
            self.expected_total_points = total_new;
            self.expected_offsets = new_offsets;
        } else {
            if let Err(err) = self.apply_forces_and_integrate() {
                return self.fail_with(err);
            }
            if let Err(err) = validate_work_for_scalar::<L::Scalar>(&self.work) {
                return self.fail_with(err);
            }
            if let Err(err) = write_work_to_storage(
                &self.work,
                self.dims,
                target.points_mut().as_euclidean_storage_mut(),
            ) {
                return self.fail_with(err);
            }
        }

        self.iter += 1;
        self.cycle_iter += 1;
        while self.current_cycle < self.cycle_iters.len()
            && self.cycle_iter >= self.cycle_iters[self.current_cycle]
        {
            self.cycle_iter = 0;
            self.current_cycle += 1;
            self.alpha *= self.config.s_step;
            if !self.alpha.is_finite() {
                return self.fail_with(CoreError::numerical_breakdown(
                    "FDEB step size alpha became non-finite",
                ));
            }
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
    fn total_iterations(&self) -> Option<usize> {
        Some(self.total_iterations)
    }
}
