use egraph_core::{CoreError, CoreResult, SolverStatus};

use super::builder::AmgPcgSolver;
use super::smoother::{backward_gs_sweep, forward_gs_sweep, norm2, scaled_dot, solve_coarsest};
use super::workspace::AmgPcgWorkspace;

impl AmgPcgSolver {
    pub(super) fn init_solve_from_ws_buffers(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<()> {
        let n = self.num_nodes;
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
            return Ok(());
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
            return Ok(());
        }

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
            return Ok(());
        }

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
                return Ok(());
            }
        }

        ws.p.copy_from_slice(&ws.z);
        ws.rz_old = rz;
        ws.rz_scale_r = sr;
        ws.rz_scale_z = sz;
        ws.status = SolverStatus::Initialized;
        Ok(())
    }

    pub(super) fn apply_preconditioner_with_fallback(&self, ws: &mut AmgPcgWorkspace) {
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
                return;
            }
        }
        ws.rz_old = rz;
        ws.rz_scale_r = sr;
        ws.rz_scale_z = sz;
    }

    pub(super) fn finalize_true_residual(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<()> {
        let n = self.num_nodes;
        if n == 0 {
            ws.residual_norm = 0.0;
            ws.relative_residual = 0.0;
            ws.converged = true;
            return Ok(());
        }
        self.finest_matrix().spmv(&ws.x, &mut ws.q)?;
        for i in 0..n {
            ws.r[i] = ws.b[i] - ws.q[i];
        }
        let r_norm = norm2(&ws.r);
        if !r_norm.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::true_residual"));
        }
        ws.residual_norm = r_norm;
        ws.relative_residual = if ws.b_norm > 0.0 {
            r_norm / ws.b_norm
        } else {
            0.0
        };
        ws.converged = r_norm <= ws.stop_tol;
        Ok(())
    }

    pub(super) fn apply_jacobi_fallback(&self, ws: &mut AmgPcgWorkspace) {
        let n = self.num_nodes;
        for i in 0..n {
            ws.z[i] = if self.zero_row_mask[i] {
                0.0
            } else {
                self.inv_diag0[i] * ws.r[i]
            };
        }
    }

    pub(super) fn apply_preconditioner(&self, ws: &mut AmgPcgWorkspace) {
        let n = self.num_nodes;
        if n == 0 {
            return;
        }
        if self.levels.is_empty() {
            ws.coarsest_rhs.copy_from_slice(&ws.r);
            ws.coarsest_sol.fill(0.0);
            solve_coarsest(
                &self.coarsest_a,
                &self.coarsest_solver,
                &ws.coarsest_rhs,
                &mut ws.coarsest_sol,
            );
            ws.z.copy_from_slice(&ws.coarsest_sol);
            for (i, &is_zero) in self.zero_row_mask.iter().enumerate() {
                if is_zero {
                    ws.z[i] = 0.0;
                }
            }
            return;
        }

        let num_levels = self.levels.len();
        ws.level_rhs[0].copy_from_slice(&ws.r);

        // Downward V-cycle pass: pre-smooth, compute residual, restrict to coarse level.
        for lvl_idx in 0..num_levels {
            let lvl = &self.levels[lvl_idx];
            let ln = lvl.a.num_nodes();
            ws.level_sol[lvl_idx].fill(0.0);

            for _ in 0..self.config.pre_sweeps {
                forward_gs_sweep(
                    &lvl.a,
                    &lvl.inv_diag,
                    &ws.level_rhs[lvl_idx],
                    &mut ws.level_sol[lvl_idx],
                );
            }

            // Compute fine residual: res = rhs - A * sol.
            let _ = lvl
                .a
                .spmv(&ws.level_sol[lvl_idx], &mut ws.level_res[lvl_idx]);
            for i in 0..ln {
                ws.level_res[lvl_idx][i] = ws.level_rhs[lvl_idx][i] - ws.level_res[lvl_idx][i];
            }

            // Restrict residual: rhs_coarse = P^T * res.
            let next_rhs = if lvl_idx + 1 < num_levels {
                &mut ws.level_rhs[lvl_idx + 1]
            } else {
                &mut ws.coarsest_rhs
            };
            next_rhs.fill(0.0);
            for i in 0..ln {
                let agg = lvl.agg_id[i] as usize;
                next_rhs[agg] += lvl.p_val[i] * ws.level_res[lvl_idx][i];
            }
        }

        // Coarsest-grid solve.
        ws.coarsest_sol.fill(0.0);
        solve_coarsest(
            &self.coarsest_a,
            &self.coarsest_solver,
            &ws.coarsest_rhs,
            &mut ws.coarsest_sol,
        );

        // Upward V-cycle pass: prolongate coarse correction, post-smooth backward GS.
        for lvl_idx in (0..num_levels).rev() {
            let lvl = &self.levels[lvl_idx];
            let ln = lvl.a.num_nodes();
            if lvl_idx + 1 < num_levels {
                let (left, right) = ws.level_sol.split_at_mut(lvl_idx + 1);
                let fine_sol = &mut left[lvl_idx];
                let coarse_sol = &right[0];
                for i in 0..ln {
                    let agg = lvl.agg_id[i] as usize;
                    fine_sol[i] += lvl.p_val[i] * coarse_sol[agg];
                }
            } else {
                let fine_sol = &mut ws.level_sol[lvl_idx];
                for i in 0..ln {
                    let agg = lvl.agg_id[i] as usize;
                    fine_sol[i] += lvl.p_val[i] * ws.coarsest_sol[agg];
                }
            }
            for _ in 0..self.config.post_sweeps {
                backward_gs_sweep(
                    &lvl.a,
                    &lvl.inv_diag,
                    &ws.level_rhs[lvl_idx],
                    &mut ws.level_sol[lvl_idx],
                );
            }
        }

        ws.z.copy_from_slice(&ws.level_sol[0]);
        for (i, &is_zero) in self.zero_row_mask.iter().enumerate() {
            if is_zero {
                ws.z[i] = 0.0;
            }
        }
    }
}
