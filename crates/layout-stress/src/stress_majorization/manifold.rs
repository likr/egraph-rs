use egraph_core::{CoreError, CoreResult, IterativeSolver, ManifoldLayout2d, Scalar, SolverStatus};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::StressMajorizationSolver;

const BACKTRACK_ALPHAS: [f64; 6] = [0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125];

impl<S: Scalar> StressMajorizationSolver<S> {
    /// Evaluates total weighted geodesic stress $\sum_{i < j} w_{ij} (d_{\mathcal{M}}(x_i, x_j) - d_{ij})^2$
    /// for a [`ManifoldLayout2d`].
    pub fn compute_stress_manifold(&self, layout: &ManifoldLayout2d<S>) -> CoreResult<f64> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "StressMajorizationSolver::compute_stress_manifold",
            ));
        }
        layout.validate_finite()?;
        self.compute_stress_manifold_unchecked(layout)
    }

    fn compute_stress_manifold_unchecked(&self, layout: &ManifoldLayout2d<S>) -> CoreResult<f64> {
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
                let norm = layout.geodesic_distance(i, j)?.to_f64();
                let diff = norm - d_target;
                total += w * diff * diff;
            }
        }
        Ok(total)
    }
}

impl<S: Scalar> IterativeSolver<S, ManifoldLayout2d<S>> for StressMajorizationSolver<S> {
    fn step(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "StressMajorizationSolver::step",
            ));
        }
        layout.validate_finite()?;
        self.active_layout
            .lock_manifold(layout.kind(), "StressMajorizationSolver")?;

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
            None => self.compute_stress_manifold_unchecked(layout)?,
        };

        let eff_min_norm = self.config.min_norm * layout.scale().min(1.0);
        for i in 0..n {
            self.sol_x[i] = layout.x()[i].to_f64();
            self.sol_y[i] = layout.y()[i].to_f64();

            let row_off = i * n;
            let mut gx = 0.0_f64;
            let mut gy = 0.0_f64;
            let mut diag_w = 0.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let w = self.weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                diag_w += w;
                let (tx_s, ty_s) = layout.tangent_delta(i, j)?;
                let tx = tx_s.to_f64();
                let ty = ty_s.to_f64();
                let norm = tx.hypot(ty);
                if norm >= eff_min_norm && norm.is_finite() {
                    let d_target = self.dists[row_off + j];
                    let coeff = w * (1.0 - d_target / norm);
                    if coeff.is_finite() {
                        gx += coeff * tx;
                        gy += coeff * ty;
                    }
                }
            }
            if diag_w > 0.0 && gx.is_finite() && gy.is_finite() {
                self.rhs_x[i] = gx / diag_w;
                self.rhs_y[i] = gy / diag_w;
            } else {
                self.rhs_x[i] = 0.0;
                self.rhs_y[i] = 0.0;
            }
        }

        // Backtrack over damped Jacobi SMACOF steps so curved manifolds never increase stress.
        let mut new_stress = s_old;
        for &alpha in &BACKTRACK_ALPHAS {
            for i in 0..n {
                let vx = alpha * self.rhs_x[i];
                let vy = alpha * self.rhs_y[i];
                if vx != 0.0 || vy != 0.0 {
                    apply_clamped_manifold_step(layout, i, vx, vy)?;
                }
            }
            let cand_stress = self.compute_stress_manifold_unchecked(layout)?;
            if cand_stress.is_finite() && cand_stress <= s_old {
                new_stress = cand_stress;
                break;
            }
            for i in 0..n {
                layout.set_point(i, S::from_f64(self.sol_x[i]), S::from_f64(self.sol_y[i]))?;
            }
        }

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
