use egraph_core::{CoreError, CoreResult, SolverStatus};
use egraph_linalg::LinearOperator;

use super::RESIDUAL_RECOMPUTE_INTERVAL;
use super::builder::AmgPcgSolver;
use super::smoother::{dot_slice, norm2, scaled_dot};
use super::workspace::AmgPcgWorkspace;

/// Applies one AMG V-cycle preconditioner $y = M^{-1} x$ (with zero-row masking).
///
/// Starting from zero initial guess with symmetric Gauss-Seidel pre/post sweeps and a
/// linear coarsest-grid solve, each invocation acts as a deterministic linear preconditioner
/// operator on $\mathbb{R}^n$ using an internal lazily-allocated scratch workspace.
impl LinearOperator<f64> for AmgPcgSolver {
    #[inline]
    fn dim(&self) -> usize {
        self.num_nodes
    }

    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> CoreResult<()> {
        let n = self.num_nodes;
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "AmgPcgSolver::LinearOperator::apply::x",
            ));
        }
        if y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                y.len(),
                "AmgPcgSolver::LinearOperator::apply::y",
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::LinearOperator::apply::x",
            ));
        }
        if n == 0 {
            return Ok(());
        }

        let mut ws = self
            .operator_ws
            .take()
            .unwrap_or_else(|| Box::new(AmgPcgWorkspace::for_solver(self)));
        ws.prepare_for_solver(self);
        ws.r.copy_from_slice(x);
        self.apply_preconditioner(&mut ws);
        y.copy_from_slice(&ws.z);
        self.operator_ws = Some(ws);
        Ok(())
    }
}

impl AmgPcgSolver {
    /// Initializes a cooperative matrix-free PCG solve against `op` for RHS `b` and optional
    /// initial guess `x_init` inside `ws` without per-step heap allocations.
    pub fn init_solve_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        b: &[f64],
        x_init: Option<&[f64]>,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::init_solve_with_operator::op",
            ));
        }
        if b.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                b.len(),
                "AmgPcgSolver::init_solve_with_operator::b",
            ));
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::init_solve_with_operator::b",
            ));
        }
        if let Some(x0) = x_init {
            if x0.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    x0.len(),
                    "AmgPcgSolver::init_solve_with_operator::x_init",
                ));
            }
            if x0.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite(
                    "AmgPcgSolver::init_solve_with_operator::x_init",
                ));
            }
        }

        ws.prepare_for_solver(self);
        if n > 0 {
            ws.b.copy_from_slice(b);
            match x_init {
                Some(x0) => ws.x.copy_from_slice(x0),
                None => ws.x.fill(0.0),
            }
        }
        self.init_solve_from_ws_buffers_with_operator(op, ws)?;
        Ok(ws.status)
    }

    /// Executes a single matrix-free PCG iteration using system operator `op` on the active
    /// solve inside `ws`.
    pub fn step_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::step_with_operator::op",
            ));
        }
        if ws.b.len() != n || ws.x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                ws.x.len(),
                "AmgPcgSolver::step_with_operator::workspace",
            ));
        }
        if ws.status.is_finished() {
            return Ok(ws.status);
        }
        if ws.iter >= ws.max_iterations {
            self.finalize_true_residual_with_operator(op, ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        op.apply(&ws.p, &mut ws.q)?;
        let (sp, sq, pq) = scaled_dot(&ws.p, &ws.q);
        let p_scaled_sq = if sp == 1.0 {
            dot_slice(&ws.p, &ws.p)
        } else {
            ws.p.iter()
                .map(|&x| {
                    let s = x / sp;
                    s * s
                })
                .sum()
        };
        let pq_thresh = if sp == 1.0 && sq == 1.0 {
            1e-30 * p_scaled_sq * self.max_diag0.max(1e-300)
        } else if sq == 0.0 {
            0.0
        } else {
            let sp_over_sq = sp / sq;
            if sp_over_sq.is_finite() {
                1e-30 * p_scaled_sq * sp_over_sq * self.max_diag0.max(1e-300)
            } else {
                1e-30 * p_scaled_sq * sp * (self.max_diag0.max(1e-300) / sq)
            }
        };
        if !pq.is_finite() || pq <= pq_thresh {
            self.finalize_true_residual_with_operator(op, ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        let alpha = if ws.rz_scale_r == 1.0 && ws.rz_scale_z == 1.0 && sp == 1.0 && sq == 1.0 {
            ws.rz_old / pq
        } else {
            (ws.rz_old / pq) * (ws.rz_scale_r / sq) * (ws.rz_scale_z / sp)
        };
        if !alpha.is_finite() {
            self.finalize_true_residual_with_operator(op, ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        for i in 0..n {
            ws.x[i] += alpha * ws.p[i];
        }
        ws.iter += 1;

        if ws.iter.is_multiple_of(RESIDUAL_RECOMPUTE_INTERVAL) {
            op.apply(&ws.x, &mut ws.q)?;
            for i in 0..n {
                ws.r[i] = ws.b[i] - ws.q[i];
            }
        } else {
            for i in 0..n {
                ws.r[i] -= alpha * ws.q[i];
            }
        }

        let r_norm = norm2(&ws.r);
        if !r_norm.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::step::residual"));
        }
        ws.residual_norm = r_norm;
        ws.relative_residual = if ws.b_norm > 0.0 {
            r_norm / ws.b_norm
        } else {
            0.0
        };

        if r_norm <= ws.stop_tol {
            // Verify true residual before declaring convergence to guard against drift.
            self.finalize_true_residual_with_operator(op, ws)?;
            if ws.converged {
                ws.status = SolverStatus::Converged;
                return Ok(ws.status);
            }
            if ws.iter >= ws.max_iterations {
                ws.status = SolverStatus::Completed;
                return Ok(ws.status);
            }
            // Residual drifted: restart direction from exact residual.
            self.apply_preconditioner_with_fallback(ws);
            if ws.status.is_finished() {
                return Ok(ws.status);
            }
            ws.p.copy_from_slice(&ws.z);
            ws.status = SolverStatus::Running;
            return Ok(ws.status);
        }

        if ws.iter >= ws.max_iterations {
            self.finalize_true_residual_with_operator(op, ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        let prev_jacobi = ws.using_jacobi_fallback;
        if prev_jacobi {
            self.apply_jacobi_fallback(ws);
        } else {
            self.apply_preconditioner(ws);
        }
        let (mut sr_new, mut sz_new, mut rz_new) = scaled_dot(&ws.r, &ws.z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            if !prev_jacobi {
                self.apply_jacobi_fallback(ws);
                ws.using_jacobi_fallback = true;
                (sr_new, sz_new, rz_new) = scaled_dot(&ws.r, &ws.z);
                if !rz_new.is_finite() || rz_new <= 0.0 {
                    self.finalize_true_residual_with_operator(op, ws)?;
                    ws.status = if ws.converged {
                        SolverStatus::Converged
                    } else {
                        SolverStatus::Completed
                    };
                    return Ok(ws.status);
                }
                // Restart search direction since the preconditioner changed.
                ws.p.copy_from_slice(&ws.z);
                ws.rz_old = rz_new;
                ws.rz_scale_r = sr_new;
                ws.rz_scale_z = sz_new;
                ws.status = SolverStatus::Running;
                return Ok(ws.status);
            }
            self.finalize_true_residual_with_operator(op, ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        let beta = if sr_new == 1.0 && sz_new == 1.0 && ws.rz_scale_r == 1.0 && ws.rz_scale_z == 1.0
        {
            rz_new / ws.rz_old
        } else {
            (rz_new / ws.rz_old) * (sr_new / ws.rz_scale_r) * (sz_new / ws.rz_scale_z)
        };
        if !beta.is_finite() {
            ws.p.copy_from_slice(&ws.z);
        } else {
            for i in 0..n {
                ws.p[i] = ws.z[i] + beta * ws.p[i];
            }
        }
        ws.rz_old = rz_new;
        ws.rz_scale_r = sr_new;
        ws.rz_scale_z = sz_new;
        ws.status = SolverStatus::Running;
        Ok(ws.status)
    }

    /// Executes up to `steps` matrix-free PCG iterations using `op` on the active solve inside `ws`.
    pub fn run_steps_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        ws: &mut AmgPcgWorkspace,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::run_steps_with_operator::op",
            ));
        }
        if ws.b.len() != n || ws.x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                ws.x.len(),
                "AmgPcgSolver::run_steps_with_operator::workspace",
            ));
        }
        for _ in 0..steps {
            if ws.status.is_finished() {
                break;
            }
            let status = self.step_with_operator(op, ws)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(ws.status)
    }

    /// Runs the active matrix-free solve inside `ws` using `op` to convergence or `max_iterations`.
    pub fn run_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::run_with_operator::op",
            ));
        }
        if ws.b.len() != n || ws.x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                ws.x.len(),
                "AmgPcgSolver::run_with_operator::workspace",
            ));
        }
        while !ws.status.is_finished() {
            let status = self.step_with_operator(op, ws)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(ws.status)
    }
}
