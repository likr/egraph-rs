use egraph_core::{CoreError, CoreResult, SolverStatus};

/// Configuration hyperparameters for [`AmgPcgSolver`](super::AmgPcgSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmgPcgConfig {
    /// Relative Euclidean residual stopping tolerance (`default: 1e-8`).
    pub rel_tol: f64,
    /// Absolute Euclidean residual stopping tolerance (`default: 0.0`).
    pub abs_tol: f64,
    /// Maximum PCG iterations (`default: 500`).
    pub max_iterations: usize,
    /// Strength-of-connection threshold $\theta \in [0, 1)$ (`default: 0.25`).
    pub theta: f64,
    /// Maximum number of AMG hierarchy levels (`default: 32`).
    pub max_levels: usize,
    /// Maximum dimension for the dense Moore-Penrose coarsest-grid solver (`default: 64`).
    pub max_coarse_size: usize,
    /// Number of forward Gauss-Seidel pre-smoothing sweeps per level (`default: 1`).
    pub pre_sweeps: usize,
    /// Number of backward Gauss-Seidel post-smoothing sweeps per level (`default: 1`).
    pub post_sweeps: usize,
    /// Maximum aggregate size cap per coarsening level (`default: 8`).
    pub max_agg_size: usize,
    /// Diagonal negativity / zero-row tolerance relative to `max_diag` (`default: 1e-14`).
    pub diag_eps: f64,
    /// Relative eigenvalue threshold for coarsest Moore-Penrose pseudoinverse (`default: 1e-12`).
    pub eig_eps: f64,
    /// Symmetry validation tolerance for input matrices (`default: 1e-10`).
    pub sym_tol: f64,
    /// Deterministic PRNG seed for optional tie-breaking (`default: 42`).
    pub seed: u64,
    /// Whether to break equal-degree ties deterministically via `DeterministicRng(seed)` (`default: false`).
    pub randomize_ties: bool,
}

impl Default for AmgPcgConfig {
    fn default() -> Self {
        Self {
            rel_tol: 1e-8,
            abs_tol: 0.0,
            max_iterations: 500,
            theta: 0.25,
            max_levels: 32,
            max_coarse_size: 64,
            pre_sweeps: 1,
            post_sweeps: 1,
            max_agg_size: 8,
            diag_eps: 1e-14,
            eig_eps: 1e-12,
            sym_tol: 1e-10,
            seed: 42,
            randomize_ties: false,
        }
    }
}

impl AmgPcgConfig {
    /// Validates all AMG-PCG configuration hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.rel_tol.is_finite() {
            return Err(CoreError::non_finite("AmgPcgConfig::rel_tol"));
        }
        if self.rel_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "rel_tol must be >= 0.0, got {}",
                self.rel_tol
            )));
        }
        if !self.abs_tol.is_finite() {
            return Err(CoreError::non_finite("AmgPcgConfig::abs_tol"));
        }
        if self.abs_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "abs_tol must be >= 0.0, got {}",
                self.abs_tol
            )));
        }
        if self.max_iterations == 0 {
            return Err(CoreError::invalid_input("max_iterations must be >= 1"));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("AmgPcgConfig::theta"));
        }
        if !(0.0..1.0).contains(&self.theta) {
            return Err(CoreError::invalid_input(format!(
                "theta must be in [0.0, 1.0), got {}",
                self.theta
            )));
        }
        if self.max_levels == 0 {
            return Err(CoreError::invalid_input("max_levels must be >= 1"));
        }
        if self.max_coarse_size == 0 {
            return Err(CoreError::invalid_input("max_coarse_size must be >= 1"));
        }
        if self.max_agg_size < 2 {
            return Err(CoreError::invalid_input(format!(
                "max_agg_size must be >= 2, got {}",
                self.max_agg_size
            )));
        }
        if !self.diag_eps.is_finite() || self.diag_eps < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "diag_eps must be finite and >= 0.0, got {}",
                self.diag_eps
            )));
        }
        if !self.eig_eps.is_finite() || self.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eig_eps must be finite and >= 0.0, got {}",
                self.eig_eps
            )));
        }
        if !self.sym_tol.is_finite() || self.sym_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sym_tol must be finite and >= 0.0, got {}",
                self.sym_tol
            )));
        }
        Ok(())
    }
}

/// Diagnostic result returned by synchronous single-RHS [`AmgPcgSolver::solve`](super::AmgPcgSolver::solve).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmgPcgSolveResult {
    /// Number of PCG iterations executed.
    pub iterations: usize,
    /// True Euclidean residual norm $\|b - A x\|_2$.
    pub residual_norm: f64,
    /// Relative residual norm $\|b - A x\|_2 / \|b\|_2$ (`0.0` when $\|b\|_2 = 0$).
    pub relative_residual: f64,
    /// Whether the stopping criterion $\|r\|_2 \le \max(\mathrm{abs\_tol}, \mathrm{rel\_tol}\cdot \|b\|_2)$ was met.
    pub converged: bool,
    /// Final solver lifecycle status.
    pub status: SolverStatus,
}

/// Diagnostic summary returned by batched [`AmgPcgSolver::solve_batch`](super::AmgPcgSolver::solve_batch).
#[derive(Debug, Clone, PartialEq)]
pub struct AmgPcgBatchResult {
    /// Sum of PCG iterations across all columns.
    pub total_iterations: usize,
    /// Maximum PCG iterations executed across all columns.
    pub max_iterations_used: usize,
    /// Minimum PCG iterations executed across all columns (`0` if `ncols == 0`).
    pub min_iterations_used: usize,
    /// Maximum true residual norm across all columns.
    pub max_residual_norm: f64,
    /// Maximum relative residual norm across all columns.
    pub max_relative_residual: f64,
    /// `true` if every column converged within tolerance.
    pub all_converged: bool,
    /// Iteration count per RHS column.
    pub iterations_per_col: Vec<usize>,
    /// True residual norm $\|b_c - A x_c\|_2$ per RHS column.
    pub residual_norms: Vec<f64>,
    /// Relative residual norm per RHS column.
    pub relative_residuals: Vec<f64>,
    /// Convergence flag per RHS column.
    pub converged_per_col: Vec<bool>,
}
