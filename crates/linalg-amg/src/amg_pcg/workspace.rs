use egraph_core::SolverStatus;

use super::builder::AmgPcgSolver;
use super::config::AmgPcgSolveResult;

/// Reusable, zero-per-iteration-allocation workspace for [`AmgPcgSolver`].
///
/// Holds both the PCG vectors (`r`, `z`, `p`, `q`, `b`, `x`) and per-level V-cycle
/// scratch buffers (`level_rhs`, `level_sol`, `level_res`), enabling bit-identical
/// synchronous `solve()` and cooperative `step()` / `run_steps()` execution.
#[derive(Debug, Clone)]
pub struct AmgPcgWorkspace {
    pub(super) b: Vec<f64>,
    pub(super) x: Vec<f64>,
    pub(super) r: Vec<f64>,
    pub(super) z: Vec<f64>,
    pub(super) p: Vec<f64>,
    pub(super) q: Vec<f64>,
    pub(super) col_b: Vec<f64>,
    pub(super) col_x: Vec<f64>,
    pub(super) level_rhs: Vec<Vec<f64>>,
    pub(super) level_sol: Vec<Vec<f64>>,
    pub(super) level_res: Vec<Vec<f64>>,
    pub(super) coarsest_rhs: Vec<f64>,
    pub(super) coarsest_sol: Vec<f64>,
    pub(super) b_norm: f64,
    pub(super) stop_tol: f64,
    pub(super) rz_old: f64,
    pub(super) rz_scale_r: f64,
    pub(super) rz_scale_z: f64,
    pub(super) iter: usize,
    pub(super) max_iterations: usize,
    pub(super) residual_norm: f64,
    pub(super) relative_residual: f64,
    pub(super) status: SolverStatus,
    pub(super) converged: bool,
    pub(super) using_jacobi_fallback: bool,
}

impl Default for AmgPcgWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl AmgPcgWorkspace {
    /// Creates an empty [`AmgPcgWorkspace`] that sizes itself lazily on first use.
    pub fn new() -> Self {
        Self {
            b: Vec::new(),
            x: Vec::new(),
            r: Vec::new(),
            z: Vec::new(),
            p: Vec::new(),
            q: Vec::new(),
            col_b: Vec::new(),
            col_x: Vec::new(),
            level_rhs: Vec::new(),
            level_sol: Vec::new(),
            level_res: Vec::new(),
            coarsest_rhs: Vec::new(),
            coarsest_sol: Vec::new(),
            b_norm: 0.0,
            stop_tol: 0.0,
            rz_old: 0.0,
            rz_scale_r: 1.0,
            rz_scale_z: 1.0,
            iter: 0,
            max_iterations: 0,
            residual_norm: 0.0,
            relative_residual: 0.0,
            status: SolverStatus::Initialized,
            converged: false,
            using_jacobi_fallback: false,
        }
    }

    /// Creates a pre-sized [`AmgPcgWorkspace`] matching `solver`'s hierarchy dimensions.
    pub fn for_solver(solver: &AmgPcgSolver) -> Self {
        let mut ws = Self::new();
        ws.prepare_for_solver(solver);
        ws
    }

    pub(super) fn prepare_for_solver(&mut self, solver: &AmgPcgSolver) {
        let n = solver.num_nodes;
        if self.b.len() != n {
            self.b.resize(n, 0.0);
            self.x.resize(n, 0.0);
            self.r.resize(n, 0.0);
            self.z.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.q.resize(n, 0.0);
            self.col_b.resize(n, 0.0);
            self.col_x.resize(n, 0.0);
        }

        let num_fine_levels = solver.levels.len();
        if self.level_rhs.len() != num_fine_levels {
            self.level_rhs.resize_with(num_fine_levels, Vec::new);
            self.level_sol.resize_with(num_fine_levels, Vec::new);
            self.level_res.resize_with(num_fine_levels, Vec::new);
        }
        for (idx, level) in solver.levels.iter().enumerate() {
            let ln = level.a.num_nodes();
            if self.level_rhs[idx].len() != ln {
                self.level_rhs[idx].resize(ln, 0.0);
                self.level_sol[idx].resize(ln, 0.0);
                self.level_res[idx].resize(ln, 0.0);
            }
        }

        let nc = solver.coarsest_a.num_nodes();
        if self.coarsest_rhs.len() != nc {
            self.coarsest_rhs.resize(nc, 0.0);
            self.coarsest_sol.resize(nc, 0.0);
        }
    }

    /// Returns the current solver lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of PCG iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the maximum scheduled iterations for the active solve.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns `true` if the active solve has reached a terminal state.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// Returns `true` if the active solve converged within tolerance.
    #[inline]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the most recent Euclidean residual norm $\|r\|_2$.
    #[inline]
    pub fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Returns the most recent relative Euclidean residual $\|r\|_2 / \|b\|_2$.
    #[inline]
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Returns a slice view of the current solution vector `x`.
    #[inline]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Returns the summary [`AmgPcgSolveResult`] of the active solve.
    #[inline]
    pub fn result(&self) -> AmgPcgSolveResult {
        AmgPcgSolveResult {
            iterations: self.iter,
            residual_norm: self.residual_norm,
            relative_residual: self.relative_residual,
            converged: self.converged,
            status: self.status,
        }
    }
}
