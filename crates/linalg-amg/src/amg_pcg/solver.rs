use egraph_core::{CoreError, CoreResult, SolverStatus};
use egraph_linalg::LinearOperator;

use super::builder::AmgPcgSolver;
use super::config::{AmgPcgBatchResult, AmgPcgSolveResult};
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
        let mut a = self.finest_matrix();
        self.init_solve_with_operator(&mut a, b, x_init, ws)
    }

    /// Executes a single PCG iteration on the active solve inside `ws`.
    pub fn step(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<SolverStatus> {
        let mut a = self.finest_matrix();
        self.step_with_operator(&mut a, ws)
    }

    /// Executes up to `steps` PCG iterations on the active solve inside `ws`.
    pub fn run_steps(&self, ws: &mut AmgPcgWorkspace, steps: usize) -> CoreResult<SolverStatus> {
        let mut a = self.finest_matrix();
        self.run_steps_with_operator(&mut a, ws, steps)
    }

    /// Runs the active solve inside `ws` to convergence or `max_iterations`.
    pub fn run(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<SolverStatus> {
        let mut a = self.finest_matrix();
        self.run_with_operator(&mut a, ws)
    }

    /// Solves $A x = b$ synchronously using `x` as initial guess (zero-initialized by caller if desired)
    /// and reusable `ws`, writing the solution back to `x`.
    pub fn solve(
        &self,
        b: &[f64],
        x: &mut [f64],
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<AmgPcgSolveResult> {
        let mut a = self.finest_matrix();
        self.solve_with_operator(&mut a, b, x, ws)
    }

    /// Convenience synchronous solver returning `(x, result)` with zero initial guess.
    pub fn solve_vec(&self, b: &[f64]) -> CoreResult<(Vec<f64>, AmgPcgSolveResult)> {
        let mut a = self.finest_matrix();
        self.solve_vec_with_operator(&mut a, b)
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
        let mut a = self.finest_matrix();
        self.solve_batch_with_operator(&mut a, b_row_major, x_row_major, ncols, ws)
    }

    /// Solves `op(x) = b` synchronously using matrix-free operator `op`, `x` as initial guess,
    /// and reusable `ws`, writing the solution back to `x`.
    pub fn solve_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        b: &[f64],
        x: &mut [f64],
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<AmgPcgSolveResult> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::solve_with_operator::op",
            ));
        }
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "AmgPcgSolver::solve_with_operator::x",
            ));
        }
        self.init_solve_with_operator(op, b, Some(x), ws)?;
        self.run_with_operator(op, ws)?;
        x.copy_from_slice(&ws.x);
        Ok(ws.result())
    }

    /// Convenience synchronous matrix-free solver returning `(x, result)` with zero initial guess.
    pub fn solve_vec_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        b: &[f64],
    ) -> CoreResult<(Vec<f64>, AmgPcgSolveResult)> {
        let mut ws = self.new_workspace();
        let mut x = vec![0.0_f64; self.num_nodes];
        let res = self.solve_with_operator(op, b, &mut x, &mut ws)?;
        Ok((x, res))
    }

    /// Solves `op(X) = B` for row-major `(num_nodes, ncols)` matrices `b_row_major` and
    /// `x_row_major` using matrix-free operator `op` and a single reusable `ws`.
    pub fn solve_batch_with_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        b_row_major: &[f64],
        x_row_major: &mut [f64],
        ncols: usize,
        ws: &mut AmgPcgWorkspace,
    ) -> CoreResult<AmgPcgBatchResult> {
        let n = self.num_nodes;
        if op.dim() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                op.dim(),
                "AmgPcgSolver::solve_batch_with_operator::op",
            ));
        }
        let expected = n.checked_mul(ncols).ok_or_else(|| {
            CoreError::invalid_input("AmgPcgSolver::solve_batch_with_operator size overflow")
        })?;
        if b_row_major.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                b_row_major.len(),
                "AmgPcgSolver::solve_batch_with_operator::b",
            ));
        }
        if x_row_major.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                x_row_major.len(),
                "AmgPcgSolver::solve_batch_with_operator::x",
            ));
        }
        if b_row_major.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::solve_batch_with_operator::b",
            ));
        }
        if x_row_major.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::solve_batch_with_operator::x",
            ));
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
            ws.b.copy_from_slice(&ws.col_b);
            ws.x.copy_from_slice(&ws.col_x);
            self.init_solve_from_ws_buffers_with_operator(op, ws)?;
            self.run_with_operator(op, ws)?;
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
