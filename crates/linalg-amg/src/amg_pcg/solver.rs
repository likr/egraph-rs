use egraph_core::{CoreError, CoreResult, SolverStatus};

use super::RESIDUAL_RECOMPUTE_INTERVAL;
use super::builder::AmgPcgSolver;
use super::config::{AmgPcgBatchResult, AmgPcgSolveResult};
use super::smoother::{dot_slice, norm2, scaled_dot};
use super::workspace::AmgPcgWorkspace;

impl AmgPcgSolver {
    /// Initializes a cooperative PCG solve for RHS `b` and optional initial guess `x_init`
    /// inside `ws` without per-step heap allocations.
    pub fn init_solve(
        &self,
        b: &[f64],
        x_init: Option<&[f64]>,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if b.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                b.len(),
                "AmgPcgSolver::init_solve::b",
            ));
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("AmgPcgSolver::init_solve::b"));
        }
        if let Some(x0) = x_init {
            if x0.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    x0.len(),
                    "AmgPcgSolver::init_solve::x_init",
                ));
            }
            if x0.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("AmgPcgSolver::init_solve::x_init"));
            }
        }

        ws.prepare_for_solver(self);
        ws.iter = 0;
        ws.max_iterations = self.config.max_iterations;
        ws.using_jacobi_fallback = false;
        ws.converged = false;

        if n == 0 {
            ws.b_norm = 0.0;
            ws.stop_tol = 0.0;
            ws.rz_old = 0.0;
            ws.rz_scale_r = 1.0;
            ws.rz_scale_z = 1.0;
            ws.residual_norm = 0.0;
            ws.relative_residual = 0.0;
            ws.converged = true;
            ws.status = SolverStatus::Converged;
            return Ok(ws.status);
        }

        ws.b.copy_from_slice(b);
        match x_init {
            Some(x0) => ws.x.copy_from_slice(x0),
            None => ws.x.fill(0.0),
        }

        let b_norm = norm2(&ws.b);
        if !b_norm.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::b_norm"));
        }
        ws.b_norm = b_norm;
        ws.stop_tol = self.config.abs_tol.max(self.config.rel_tol * b_norm);

        if b_norm == 0.0 {
            ws.x.fill(0.0);
            ws.r.fill(0.0);
            ws.z.fill(0.0);
            ws.p.fill(0.0);
            ws.q.fill(0.0);
            ws.rz_old = 0.0;
            ws.rz_scale_r = 1.0;
            ws.rz_scale_z = 1.0;
            ws.residual_norm = 0.0;
            ws.relative_residual = 0.0;
            ws.converged = true;
            ws.status = SolverStatus::Converged;
            return Ok(ws.status);
        }

        // Initial residual r = b - A x.
        self.finest_matrix().spmv(&ws.x, &mut ws.q)?;
        for i in 0..n {
            ws.r[i] = ws.b[i] - ws.q[i];
        }
        let r_norm = norm2(&ws.r);
        if !r_norm.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::initial_residual"));
        }
        ws.residual_norm = r_norm;
        ws.relative_residual = r_norm / b_norm;

        if r_norm <= ws.stop_tol {
            ws.converged = true;
            ws.status = SolverStatus::Converged;
            return Ok(ws.status);
        }

        // Initial preconditioning z = M^{-1} r.
        self.apply_preconditioner(ws);
        let (mut sr, mut sz, mut rz) = scaled_dot(&ws.r, &ws.z);
        if !rz.is_finite() || rz <= 0.0 {
            self.apply_jacobi_fallback(ws);
            ws.using_jacobi_fallback = true;
            (sr, sz, rz) = scaled_dot(&ws.r, &ws.z);
            if !rz.is_finite() || rz <= 0.0 {
                ws.rz_old = 0.0;
                ws.rz_scale_r = 1.0;
                ws.rz_scale_z = 1.0;
                ws.status = SolverStatus::Completed;
                return Ok(ws.status);
            }
        }

        ws.p.copy_from_slice(&ws.z);
        ws.rz_old = rz;
        ws.rz_scale_r = sr;
        ws.rz_scale_z = sz;
        ws.status = SolverStatus::Initialized;
        Ok(ws.status)
    }

    /// Executes a single PCG iteration on the active solve inside `ws`.
    pub fn step(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<SolverStatus> {
        if ws.status.is_finished() {
            return Ok(ws.status);
        }
        let n = self.num_nodes;
        if ws.b.len() != n || ws.x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                ws.x.len(),
                "AmgPcgSolver::step::workspace",
            ));
        }
        if ws.iter >= ws.max_iterations {
            self.finalize_true_residual(ws)?;
            ws.status = if ws.converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
            return Ok(ws.status);
        }

        let a = self.finest_matrix();
        a.spmv(&ws.p, &mut ws.q)?;
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
            self.finalize_true_residual(ws)?;
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
            self.finalize_true_residual(ws)?;
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
            a.spmv(&ws.x, &mut ws.q)?;
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
            self.finalize_true_residual(ws)?;
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
            self.finalize_true_residual(ws)?;
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
                    self.finalize_true_residual(ws)?;
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
            self.finalize_true_residual(ws)?;
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

    /// Executes up to `steps` PCG iterations on the active solve inside `ws`.
    pub fn run_steps(&self, ws: &mut AmgPcgWorkspace, steps: usize) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if ws.status.is_finished() {
                break;
            }
            let status = self.step(ws)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(ws.status)
    }

    /// Runs the active solve inside `ws` to convergence or `max_iterations`.
    pub fn run(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<SolverStatus> {
        while !ws.status.is_finished() {
            let status = self.step(ws)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(ws.status)
    }

    /// Solves $A x = b$ synchronously using `x` as initial guess (zero-initialized by caller if desired)
    /// and reusable `ws`, writing the solution back to `x`.
    pub fn solve(
        &self,
        b: &[f64],
        x: &mut [f64],
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<AmgPcgSolveResult> {
        let n = self.num_nodes;
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "AmgPcgSolver::solve::x",
            ));
        }
        self.init_solve(b, Some(x), ws)?;
        self.run(ws)?;
        x.copy_from_slice(&ws.x);
        Ok(ws.result())
    }

    /// Convenience synchronous solver returning `(x, result)` with zero initial guess.
    pub fn solve_vec(&self, b: &[f64]) -> CoreResult<(Vec<f64>, AmgPcgSolveResult)> {
        let mut ws = self.new_workspace();
        let mut x = vec![0.0_f64; self.num_nodes];
        let res = self.solve(b, &mut x, &mut ws)?;
        Ok((x, res))
    }

    /// Solves $A X = B$ for row-major `(num_nodes, ncols)` matrices `b_row_major` and
    /// `x_row_major` using a single reusable `ws`.
    pub fn solve_batch(
        &self,
        b_row_major: &[f64],
        x_row_major: &mut [f64],
        ncols: usize,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<AmgPcgBatchResult> {
        let n = self.num_nodes;
        let expected = n
            .checked_mul(ncols)
            .ok_or_else(|| CoreError::invalid_input("AmgPcgSolver::solve_batch size overflow"))?;
        if b_row_major.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                b_row_major.len(),
                "AmgPcgSolver::solve_batch::b",
            ));
        }
        if x_row_major.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                x_row_major.len(),
                "AmgPcgSolver::solve_batch::x",
            ));
        }
        if b_row_major.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("AmgPcgSolver::solve_batch::b"));
        }
        if x_row_major.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("AmgPcgSolver::solve_batch::x"));
        }
        if ncols == 0 {
            return Ok(AmgPcgBatchResult {
                total_iterations: 0,
                max_iterations_used: 0,
                min_iterations_used: 0,
                max_residual_norm: 0.0,
                max_relative_residual: 0.0,
                all_converged: true,
                iterations_per_col: Vec::new(),
                residual_norms: Vec::new(),
                relative_residuals: Vec::new(),
                converged_per_col: Vec::new(),
            });
        }

        ws.prepare_for_solver(self);

        let mut total_iterations = 0usize;
        let mut max_iterations_used = 0usize;
        let mut min_iterations_used = usize::MAX;
        let mut max_residual_norm = 0.0_f64;
        let mut max_relative_residual = 0.0_f64;
        let mut all_converged = true;
        let mut iterations_per_col = Vec::with_capacity(ncols);
        let mut residual_norms = Vec::with_capacity(ncols);
        let mut relative_residuals = Vec::with_capacity(ncols);
        let mut converged_per_col = Vec::with_capacity(ncols);

        for c in 0..ncols {
            for r in 0..n {
                let idx = r * ncols + c;
                ws.col_b[r] = b_row_major[idx];
                ws.col_x[r] = x_row_major[idx];
            }
            // Copy column vectors to b and x via init_solve without extra allocation.
            ws.b.copy_from_slice(&ws.col_b);
            ws.x.copy_from_slice(&ws.col_x);
            self.init_solve_from_ws_buffers(ws)?;
            self.run(ws)?;
            for r in 0..n {
                x_row_major[r * ncols + c] = ws.x[r];
            }

            let res = ws.result();
            total_iterations += res.iterations;
            max_iterations_used = max_iterations_used.max(res.iterations);
            min_iterations_used = min_iterations_used.min(res.iterations);
            max_residual_norm = max_residual_norm.max(res.residual_norm);
            max_relative_residual = max_relative_residual.max(res.relative_residual);
            all_converged &= res.converged;
            iterations_per_col.push(res.iterations);
            residual_norms.push(res.residual_norm);
            relative_residuals.push(res.relative_residual);
            converged_per_col.push(res.converged);
        }

        Ok(AmgPcgBatchResult {
            total_iterations,
            max_iterations_used,
            min_iterations_used: if min_iterations_used == usize::MAX {
                0
            } else {
                min_iterations_used
            },
            max_residual_norm,
            max_relative_residual,
            all_converged,
            iterations_per_col,
            residual_norms,
            relative_residuals,
            converged_per_col,
        })
    }
}
