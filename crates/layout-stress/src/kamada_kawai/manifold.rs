use egraph_core::{CoreError, CoreResult, IterativeSolver, ManifoldLayout2d, Scalar, SolverStatus};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::{HESSIAN_DET_TOL, KamadaKawaiSolver, MAX_NEWTON_STEP, MIN_SEPARATION};

impl<S: Scalar> KamadaKawaiSolver<S> {
    fn select_node_manifold(
        &self,
        layout: &ManifoldLayout2d<S>,
        min_sep: f64,
    ) -> CoreResult<Option<usize>> {
        let n = self.num_nodes;
        let eps_sq = self.config.eps * self.config.eps;
        let mut best_m = 0usize;
        let mut best_delta2 = 0.0_f64;

        for m in 0..n {
            let row = m * n;
            let mut dedx = 0.0_f64;
            let mut dedy = 0.0_f64;
            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let (tx_s, ty_s) = layout.tangent_delta(m, i)?;
                let dx = -tx_s.to_f64();
                let dy = -ty_s.to_f64();
                let d = dx.hypot(dy);
                if d > min_sep && d.is_finite() {
                    let f = k_mi * (1.0 - self.l[row + i] / d);
                    dedx += f * dx;
                    dedy += f * dy;
                }
            }
            let delta2 = dedx * dedx + dedy * dedy;
            if delta2.is_finite() && delta2 > best_delta2 {
                best_delta2 = delta2;
                best_m = m;
            }
        }

        if best_delta2 < eps_sq {
            Ok(None)
        } else {
            Ok(Some(best_m))
        }
    }
}

impl<S: Scalar> IterativeSolver<S, ManifoldLayout2d<S>> for KamadaKawaiSolver<S> {
    fn step(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "KamadaKawaiSolver::step",
            ));
        }
        layout.validate_finite()?;
        self.active_layout
            .lock_manifold(layout.kind(), "KamadaKawaiSolver")?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let scale = layout.scale();
        let min_sep = MIN_SEPARATION * scale.min(1.0);
        let Some(m) = self.select_node_manifold(layout, min_sep)? else {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        };

        let eps_sq = self.config.eps * self.config.eps;
        let row = m * n;
        let max_newton = MAX_NEWTON_STEP * scale;

        for _ in 0..self.config.max_inner_iters {
            let mut hxx = 0.0_f64;
            let mut hyy = 0.0_f64;
            let mut hxy = 0.0_f64;
            let mut dedx = 0.0_f64;
            let mut dedy = 0.0_f64;

            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let l_mi = self.l[row + i];
                let (tx_s, ty_s) = layout.tangent_delta(m, i)?;
                let dx = -tx_s.to_f64();
                let dy = -ty_s.to_f64();
                let d = dx.hypot(dy);
                if d <= min_sep || !d.is_finite() {
                    continue;
                }
                let d3 = d * d * d;
                if d3 > 0.0 && d3.is_finite() {
                    hxx += k_mi * (1.0 - l_mi * dy * dy / d3);
                    hyy += k_mi * (1.0 - l_mi * dx * dx / d3);
                    hxy += k_mi * l_mi * dx * dy / d3;
                }
                let f = k_mi * (1.0 - l_mi / d);
                dedx += f * dx;
                dedy += f * dy;
            }

            let delta2 = dedx * dedx + dedy * dedy;
            if !delta2.is_finite() || delta2 < eps_sq {
                break;
            }

            let det = hxx * hyy - hxy * hxy;
            if !det.is_finite() || det.abs() <= HESSIAN_DET_TOL || (det > 0.0 && (hxx + hyy) <= 0.0)
            {
                // Singular or negative-definite tangent Hessian: take one bounded steepest-descent
                // fallback step and exit the inner loop cleanly.
                let g_norm = delta2.sqrt();
                if g_norm > 0.0 && g_norm.is_finite() {
                    let step_len = (0.1 / g_norm).min(0.25) * scale;
                    let step_x = -step_len * dedx;
                    let step_y = -step_len * dedy;
                    if step_x.is_finite() && step_y.is_finite() {
                        apply_clamped_manifold_step(layout, m, step_x, step_y)?;
                    }
                }
                break;
            }

            let mut delta_x = (hyy * dedx - hxy * dedy) / det;
            let mut delta_y = (hxx * dedy - hxy * dedx) / det;
            if !delta_x.is_finite() || !delta_y.is_finite() {
                break;
            }
            let step_norm = delta_x.hypot(delta_y);
            if step_norm > max_newton {
                let s = max_newton / step_norm;
                delta_x *= s;
                delta_y *= s;
            }

            apply_clamped_manifold_step(layout, m, -delta_x, -delta_y)?;

            if step_norm <= self.config.eps * 1e-2 {
                break;
            }
        }

        self.iter += 1;
        self.status = if self.iter >= self.max_iterations {
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
        self.max_iterations
    }
}
