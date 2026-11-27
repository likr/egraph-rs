use egraph_core::{CoreError, CoreResult, SolverStatus};

/// Configuration hyperparameters for [`AmgPcgSolver`](super::AmgPcgSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new [`AmgPcgConfig`] initialized with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the relative Euclidean residual stopping tolerance `rel_tol`.
    pub fn with_rel_tol(mut self, rel_tol: f64) -> Self {
        self.rel_tol = rel_tol;
        self
    }

    /// Sets the absolute Euclidean residual stopping tolerance `abs_tol`.
    pub fn with_abs_tol(mut self, abs_tol: f64) -> Self {
        self.abs_tol = abs_tol;
        self
    }

    /// Sets the maximum PCG iterations `max_iterations`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the strength-of-connection threshold `theta`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the maximum number of AMG hierarchy levels `max_levels`.
    pub fn with_max_levels(mut self, max_levels: usize) -> Self {
        self.max_levels = max_levels;
        self
    }

    /// Sets the maximum dimension for the coarsest-grid solver `max_coarse_size`.
    pub fn with_max_coarse_size(mut self, max_coarse_size: usize) -> Self {
        self.max_coarse_size = max_coarse_size;
        self
    }

    /// Sets the number of forward Gauss-Seidel pre-smoothing sweeps `pre_sweeps`.
    pub fn with_pre_sweeps(mut self, pre_sweeps: usize) -> Self {
        self.pre_sweeps = pre_sweeps;
        self
    }

    /// Sets the number of backward Gauss-Seidel post-smoothing sweeps `post_sweeps`.
    pub fn with_post_sweeps(mut self, post_sweeps: usize) -> Self {
        self.post_sweeps = post_sweeps;
        self
    }

    /// Sets the maximum aggregate size cap per coarsening level `max_agg_size`.
    pub fn with_max_agg_size(mut self, max_agg_size: usize) -> Self {
        self.max_agg_size = max_agg_size;
        self
    }

    /// Sets the relative diagonal negativity tolerance `diag_eps`.
    pub fn with_diag_eps(mut self, diag_eps: f64) -> Self {
        self.diag_eps = diag_eps;
        self
    }

    /// Sets the relative eigenvalue threshold for the coarsest pseudoinverse `eig_eps`.
    pub fn with_eig_eps(mut self, eig_eps: f64) -> Self {
        self.eig_eps = eig_eps;
        self
    }

    /// Sets the input matrix symmetry validation tolerance `sym_tol`.
    pub fn with_sym_tol(mut self, sym_tol: f64) -> Self {
        self.sym_tol = sym_tol;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets whether equal-degree ties are broken deterministically via the seeded PRNG.
    pub fn with_randomize_ties(mut self, randomize_ties: bool) -> Self {
        self.randomize_ties = randomize_ties;
        self
    }

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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
#[non_exhaustive]
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
