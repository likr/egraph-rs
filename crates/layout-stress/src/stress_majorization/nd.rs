use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};

use super::StressMajorizationSolver;

#[inline]
fn euclid_norm_diff_flat<S: Scalar>(coords: &[S], d: usize, i: usize, j: usize) -> f64 {
    let base_i = i * d;
    let base_j = j * d;
    let mut sum_sq = 0.0_f64;
    for a in 0..d {
        let diff = coords[base_i + a].to_f64() - coords[base_j + a].to_f64();
        sum_sq += diff * diff;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let abs_d = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()).abs();
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let r = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

#[inline]
fn euclid_norm_diff_axis_major(sol: &[f64], n: usize, d: usize, i: usize, j: usize) -> f64 {
    let mut sum_sq = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let diff = sol[off + i] - sol[off + j];
        sum_sq += diff * diff;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let abs_d = (sol[off + i] - sol[off + j]).abs();
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let r = (sol[off + i] - sol[off + j]) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

impl<S: Scalar> StressMajorizationSolver<S> {
    /// Evaluates total weighted stress $\sum_{i < j} w_{ij} (\|x_i - x_j\|_2 - d_{ij})^2$ for an
    /// $N$-dimensional [`LayoutNd`].
    pub fn compute_stress_nd(&self, layout: &LayoutNd<S>) -> CoreResult<f64> {
        let d = layout.dims();
        if d == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        if layout.num_nodes() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.num_nodes(),
                "StressMajorizationSolver::compute_stress_nd",
            ));
        }
        layout.validate_finite()?;
        Ok(self.compute_stress_nd_slice(layout.coords(), d))
    }

    fn compute_stress_nd_slice(&self, coords: &[S], d: usize) -> f64 {
        let n = self.num_nodes;
        let mut total = 0.0_f64;
        for i in 0..n {
            let row_off = i * n;
            for j in (i + 1)..n {
                let w = self.weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                let d_target = self.dists[row_off + j];
                let norm = euclid_norm_diff_flat(coords, d, i, j);
                let diff = norm - d_target;
                total += w * diff * diff;
            }
        }
        total
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for StressMajorizationSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        let d = layout.dims();
        if d == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        if layout.num_nodes() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.num_nodes(),
                "StressMajorizationSolver::step",
            ));
        }
        layout.validate_finite()?;
        if let Some(expected_dims) = self.nd_dims
            && d != expected_dims
        {
            return Err(CoreError::dimension_mismatch(
                expected_dims,
                d,
                "StressMajorizationSolver::step::dims",
            ));
        }
        self.nd_dims = Some(d);

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.prev_stress = Some(0.0);
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let s_old = match self.prev_stress {
            Some(s) => s,
            None => self.compute_stress_nd_slice(layout.coords(), d),
        };

        let needed = n
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd buffer size overflow"))?;
        if self.nd_rhs.len() != needed {
            self.nd_rhs.resize(needed, 0.0);
            self.nd_sol.resize(needed, 0.0);
        }
        self.nd_rhs.fill(0.0);

        {
            let coords = layout.coords();
            for i in 0..n {
                let base = i * d;
                for a in 0..d {
                    self.nd_sol[a * n + i] = coords[base + a].to_f64();
                }
            }
        }

        let min_norm = self.config.min_norm;
        for i in 0..n {
            let row_off = i * n;
            for j in (i + 1)..n {
                let w = self.weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                let d_target = self.dists[row_off + j];
                let norm = euclid_norm_diff_axis_major(&self.nd_sol, n, d, i, j);
                if norm >= min_norm && norm.is_finite() {
                    let f = w * d_target / norm;
                    for a in 0..d {
                        let off = a * n;
                        let fa = f * (self.nd_sol[off + i] - self.nd_sol[off + j]);
                        self.nd_rhs[off + i] += fa;
                        self.nd_rhs[off + j] -= fa;
                    }
                }
            }
        }

        let pcg_max = self.config.pcg_max_iterations.unwrap_or((2 * n).max(16));
        for a in 0..d {
            let off = a * n;
            self.pcg.solve_dense_symmetric(
                &self.l_w,
                &self.nd_rhs[off..off + n],
                &mut self.nd_sol[off..off + n],
                self.config.pcg_tolerance,
                pcg_max,
            )?;
        }

        {
            let coords_mut = layout.coords_mut();
            for i in 0..n {
                let base = i * d;
                for a in 0..d {
                    let nv = S::from_f64(self.nd_sol[a * n + i]);
                    if nv.is_finite() {
                        coords_mut[base + a] = nv;
                    }
                }
            }
        }

        let new_stress = self.compute_stress_nd_slice(layout.coords(), d);
        self.prev_stress = Some(new_stress);
        self.iter += 1;

        let converged = if s_old > 1e-30 {
            (s_old - new_stress).abs() / s_old < self.config.epsilon
        } else {
            new_stress <= 1e-30
        };

        self.status = if converged {
            SolverStatus::Converged
        } else if self.iter >= self.config.max_iterations {
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
        self.config.max_iterations
    }
}
