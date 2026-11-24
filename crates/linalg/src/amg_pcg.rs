//! Hardened, strictly single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient
//! (`AmgPcgSolver`) for symmetric positive (semi-)definite sparse linear systems.
//!
//! Remediates all six legacy AMG/PCG defect classes:
//! 1. **Fill-in explosion on power-law hubs (`web-Stanford`)**: Uses unsmoothed
//!    double-pairwise + size-capped leaf-absorption aggregation with min-degree ordering,
//!    yielding a 1-nonzero-per-row orthonormal prolongator ($P^T P = I$) and strictly
//!    non-increasing coarse sparsity ($\mathrm{nnz}(A_{l+1}) \le \mathrm{nnz}(A_l)$) via
//!    $O(\mathrm{nnz})$ stamp-based Galerkin assembly $A_c = P^T A P$.
//! 2. **Smoother instability and allocations**: Uses zero-allocation Symmetric Gauss-Seidel
//!    (forward GS pre-sweep + backward GS post-sweep) with cached diagonals and a
//!    positive-diagonal guard, eliminating $\omega = 2/3$ Jacobi spectral divergence.
//! 3. **Coarsest-grid solver & Cholesky breakdown**: Precomputes a dense Moore-Penrose
//!    pseudoinverse $A_c^\dagger = V \Lambda^\dagger V^T$ via [`symmetric_eigen_jacobi`]
//!    for $n_c \le \mathrm{max\_coarse\_size}$, zeroing nullspace modes cleanly, and falls
//!    back to bounded symmetric Gauss-Seidel sweeps if coarsening stalls.
//! 4. **Zero diagonals, isolated vertices, and disconnected components**: Validates symmetry,
//!    finiteness, and non-negativity of diagonals, tracks `zero_row_mask` (mapping `z[i] = 0.0`
//!    and handling inconsistent RHS cleanly), and supports $n = 0$, $n = 1$, edgeless, and
//!    multi-component singular Laplacians without `NaN` or panics.
//! 5. **Galerkin asymmetry, strength sign, and near-nullspace**: Symmetrizes coarse operators
//!    $A_c \leftarrow \frac{1}{2}(A_c + A_c^T)$ with zero pruning, filters strength strictly
//!    on negative off-diagonals, and propagates near-nullspace vectors ($D^{1/2}\mathbf{1}$ for
//!    $L_{\mathrm{sym}}$, $\mathbf{1}$ for combinatorial $L$).
//! 6. **PCG stopping, breakdown guards, drift, and determinism**: Enforces relative + absolute
//!    Euclidean residual stopping $\|r\|_2 \le \max(\mathrm{abs\_tol}, \mathrm{rel\_tol}\cdot \|b\|_2)$,
//!    $b = 0$ fast-path, `rz <= 0` Jacobi fallback, `pq <= 0` breakdown guard, periodic exact
//!    residual recomputation every 50 iterations, and zero-per-solve-allocation [`AmgPcgWorkspace`].

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, SolverStatus};

use crate::csr_matrix::{
    CsrMatrix, LaplacianKind, combinatorial_laplacian, symmetric_normalized_laplacian,
};
use crate::eigen::symmetric_eigen_jacobi;

const RESIDUAL_RECOMPUTE_INTERVAL: usize = 50;
const STALLED_COARSEST_SGS_SWEEPS: usize = 10;
const SMOOTHER_DIAG_EPS: f64 = 1e-30;
const JACOBI_COARSEST_MAX_SWEEPS: usize = 120;
const JACOBI_COARSEST_TOL: f64 = 1e-12;

/// Configuration hyperparameters for [`AmgPcgSolver`].
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

/// Diagnostic result returned by synchronous single-RHS [`AmgPcgSolver::solve`].
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

/// Diagnostic summary returned by batched [`AmgPcgSolver::solve_batch`].
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

#[derive(Debug, Clone)]
struct AmgLevel {
    a: CsrMatrix,
    inv_diag: Vec<f64>,
    agg_id: Vec<u32>,
    p_val: Vec<f64>,
}

struct BuiltLevel {
    agg_id: Vec<u32>,
    p_val: Vec<f64>,
    next_ns: Vec<f64>,
    next_a: CsrMatrix,
}

#[derive(Debug, Clone)]
enum CoarsestSolver {
    DensePinverse { n: usize, pinv: Vec<f64> },
    BoundedSgs { sweeps: usize, inv_diag: Vec<f64> },
}

/// Reusable, zero-per-iteration-allocation workspace for [`AmgPcgSolver`].
///
/// Holds both the PCG vectors (`r`, `z`, `p`, `q`, `b`, `x`) and per-level V-cycle
/// scratch buffers (`level_rhs`, `level_sol`, `level_res`), enabling bit-identical
/// synchronous `solve()` and cooperative `step()` / `run_steps()` execution.
#[derive(Debug, Clone)]
pub struct AmgPcgWorkspace {
    b: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
    col_b: Vec<f64>,
    col_x: Vec<f64>,
    level_rhs: Vec<Vec<f64>>,
    level_sol: Vec<Vec<f64>>,
    level_res: Vec<Vec<f64>>,
    coarsest_rhs: Vec<f64>,
    coarsest_sol: Vec<f64>,
    b_norm: f64,
    stop_tol: f64,
    rz_old: f64,
    rz_scale_r: f64,
    rz_scale_z: f64,
    iter: usize,
    max_iterations: usize,
    residual_norm: f64,
    relative_residual: f64,
    status: SolverStatus,
    converged: bool,
    using_jacobi_fallback: bool,
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

    fn prepare_for_solver(&mut self, solver: &AmgPcgSolver) {
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

/// Hardened, strictly single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct AmgPcgSolver {
    num_nodes: usize,
    config: AmgPcgConfig,
    levels: Vec<AmgLevel>,
    coarsest_a: CsrMatrix,
    coarsest_solver: CoarsestSolver,
    zero_row_mask: Vec<bool>,
    inv_diag0: Vec<f64>,
    max_diag0: f64,
    input_nnz: usize,
}

impl AmgPcgSolver {
    /// Builds an [`AmgPcgSolver`] for a symmetric positive (semi-)definite [`CsrMatrix`]
    /// using uniform $\mathbf{1}$ near-nullspace initialization.
    pub fn new(matrix: &CsrMatrix, config: AmgPcgConfig) -> CoreResult<Self> {
        Self::with_near_nullspace(matrix, None, config)
    }

    /// Builds an [`AmgPcgSolver`] for `matrix` with an optional custom `near_nullspace` vector.
    pub fn with_near_nullspace(
        matrix: &CsrMatrix,
        near_nullspace: Option<&[f64]>,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = matrix.num_nodes();
        if let Some(ns) = near_nullspace {
            if ns.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    ns.len(),
                    "AmgPcgSolver::near_nullspace",
                ));
            }
            if ns.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("AmgPcgSolver::near_nullspace"));
            }
        }

        if !matrix.is_symmetric(config.sym_tol) {
            return Err(CoreError::invalid_input(
                "AmgPcgSolver requires a symmetric matrix within config.sym_tol",
            ));
        }

        let a0 = if matrix.is_symmetric(0.0) {
            matrix.clone()
        } else {
            matrix.symmetrize()?
        };

        if n == 0 {
            return Ok(Self {
                num_nodes: 0,
                config,
                levels: Vec::new(),
                coarsest_a: a0,
                coarsest_solver: CoarsestSolver::DensePinverse {
                    n: 0,
                    pinv: Vec::new(),
                },
                zero_row_mask: Vec::new(),
                inv_diag0: Vec::new(),
                max_diag0: 0.0,
                input_nnz: 0,
            });
        }

        let diag0 = a0.diagonal();
        let mut max_diag0 = 0.0_f64;
        for &d in diag0 {
            if !d.is_finite() {
                return Err(CoreError::non_finite("AmgPcgSolver::diagonal"));
            }
            if d > max_diag0 {
                max_diag0 = d;
            }
        }
        let neg_thresh = if max_diag0 > 0.0 {
            -config.diag_eps * max_diag0
        } else {
            0.0
        };

        let mut zero_row_mask = vec![false; n];
        for (row, &d) in diag0.iter().enumerate() {
            if d < neg_thresh {
                return Err(CoreError::invalid_input(format!(
                    "negative diagonal entry at row {row}: {d}"
                )));
            }
            if d <= 0.0 {
                let start = a0.indptr()[row] as usize;
                let end = a0.indptr()[row + 1] as usize;
                let row_u32 = row as u32;
                let mut has_nonzero_off = false;
                for pos in start..end {
                    if a0.indices()[pos] != row_u32 && a0.values()[pos] != 0.0 {
                        has_nonzero_off = true;
                        break;
                    }
                }
                if has_nonzero_off {
                    return Err(CoreError::invalid_input(format!(
                        "zero diagonal entry on non-zero row {row}"
                    )));
                }
                zero_row_mask[row] = true;
            }
        }

        let inv_diag0 = compute_safe_inv_diag(diag0, max_diag0, Some(&zero_row_mask));
        let input_nnz = a0.nnz();

        let mut current_a = a0;
        let mut current_ns: Vec<f64> = match near_nullspace {
            Some(ns) => ns.to_vec(),
            None => vec![1.0_f64; n],
        };

        let mut levels = Vec::new();
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let mut gal_scratch = GalerkinScratch::default();

        for _level_idx in 0..config.max_levels {
            let nc_curr = current_a.num_nodes();
            if nc_curr <= config.max_coarse_size {
                break;
            }

            let Some(built) = build_double_pairwise_level(
                &current_a,
                &current_ns,
                &config,
                &mut rng,
                &mut gal_scratch,
            )?
            else {
                break;
            };

            let curr_inv_diag = if levels.is_empty() {
                inv_diag0.clone()
            } else {
                let d = current_a.diagonal();
                let m = d.iter().copied().fold(0.0_f64, f64::max);
                compute_safe_inv_diag(d, m, None)
            };

            levels.push(AmgLevel {
                a: current_a,
                inv_diag: curr_inv_diag,
                agg_id: built.agg_id,
                p_val: built.p_val,
            });
            current_a = built.next_a;
            current_ns = built.next_ns;
        }

        let coarsest_solver = build_coarsest_solver(&current_a, &config)?;

        Ok(Self {
            num_nodes: n,
            config,
            levels,
            coarsest_a: current_a,
            coarsest_solver,
            zero_row_mask,
            inv_diag0,
            max_diag0,
            input_nnz,
        })
    }

    /// Builds an [`AmgPcgSolver`] for the regularized symmetric normalized Laplacian
    /// $A = I + \gamma L_{\mathrm{sym}}$ ($\gamma \ge 0$) with exact $D^{1/2}\mathbf{1}$ near-nullspace.
    pub fn from_regularized_sym_laplacian<W: Scalar>(
        graph: &CsrGraph<W>,
        gamma: f64,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        if !gamma.is_finite() {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::from_regularized_sym_laplacian::gamma",
            ));
        }
        if gamma < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gamma must be >= 0.0, got {gamma}"
            )));
        }
        let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
        let a = l_sym.scale_and_shift(gamma, 1.0)?;
        let near_nullspace: Vec<f64> = inv_sqrt_deg
            .into_iter()
            .map(|inv| if inv > 0.0 { 1.0 / inv } else { 1.0 })
            .collect();
        Self::with_near_nullspace(&a, Some(&near_nullspace), config)
    }

    /// Builds an [`AmgPcgSolver`] for the shifted graph Laplacian $L + \mathrm{shift}\cdot I$
    /// (`shift >= 0.0`) with matching near-nullspace ($D^{1/2}\mathbf{1}$ for `SymmetricNormalized`,
    /// $\mathbf{1}$ for `Combinatorial`).
    pub fn from_laplacian<W: Scalar>(
        graph: &CsrGraph<W>,
        kind: LaplacianKind,
        shift: f64,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        if !shift.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::from_laplacian::shift"));
        }
        if shift < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "shift must be >= 0.0, got {shift}"
            )));
        }
        match kind {
            LaplacianKind::Combinatorial => {
                let l = combinatorial_laplacian(graph);
                let a = if shift == 0.0 {
                    l
                } else {
                    l.scale_and_shift(1.0, shift)?
                };
                Self::with_near_nullspace(&a, None, config)
            }
            LaplacianKind::SymmetricNormalized => {
                let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
                let a = if shift == 0.0 {
                    l_sym
                } else {
                    l_sym.scale_and_shift(1.0, shift)?
                };
                let near_nullspace: Vec<f64> = inv_sqrt_deg
                    .into_iter()
                    .map(|inv| if inv > 0.0 { 1.0 / inv } else { 1.0 })
                    .collect();
                Self::with_near_nullspace(&a, Some(&near_nullspace), config)
            }
            LaplacianKind::RandomWalk => Err(CoreError::unsupported(
                "AmgPcgSolver does not support asymmetric RandomWalk Laplacian; use Combinatorial or SymmetricNormalized",
            )),
        }
    }

    /// Returns the dimension $n$ of the linear system.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of levels in the AMG hierarchy (including the coarsest level).
    #[inline]
    pub fn num_levels(&self) -> usize {
        if self.num_nodes == 0 {
            0
        } else {
            self.levels.len() + 1
        }
    }

    /// Returns the dimension of the coarsest AMG level.
    #[inline]
    pub fn coarsest_size(&self) -> usize {
        self.coarsest_a.num_nodes()
    }

    /// Returns the AMG operator complexity $\sum_{l=0}^L \mathrm{nnz}(A_l) / \mathrm{nnz}(A_0)$.
    pub fn operator_complexity(&self) -> f64 {
        if self.input_nnz == 0 {
            return 1.0;
        }
        let mut total_nnz = 0usize;
        for lvl in &self.levels {
            total_nnz += lvl.a.nnz();
        }
        total_nnz += self.coarsest_a.nnz();
        (total_nnz as f64) / (self.input_nnz as f64)
    }

    /// Returns the maximum CSR nonzero count across all levels in the AMG hierarchy.
    pub fn max_level_nnz(&self) -> usize {
        let mut max_nnz = self.coarsest_a.nnz();
        for lvl in &self.levels {
            max_nnz = max_nnz.max(lvl.a.nnz());
        }
        max_nnz
    }

    /// Returns the active [`AmgPcgConfig`].
    #[inline]
    pub fn config(&self) -> AmgPcgConfig {
        self.config
    }

    /// Allocates a workspace pre-sized for this solver.
    pub fn new_workspace(&self) -> AmgPcgWorkspace {
        AmgPcgWorkspace::for_solver(self)
    }

    #[inline]
    fn finest_matrix(&self) -> &CsrMatrix {
        if let Some(lvl0) = self.levels.first() {
            &lvl0.a
        } else {
            &self.coarsest_a
        }
    }

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

    fn init_solve_from_ws_buffers(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<()> {
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

    fn apply_preconditioner_with_fallback(&self, ws: &mut AmgPcgWorkspace) {
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

    fn finalize_true_residual(&self, ws: &mut AmgPcgWorkspace) -> CoreResult<()> {
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

    fn apply_jacobi_fallback(&self, ws: &mut AmgPcgWorkspace) {
        let n = self.num_nodes;
        for i in 0..n {
            ws.z[i] = if self.zero_row_mask[i] {
                0.0
            } else {
                self.inv_diag0[i] * ws.r[i]
            };
        }
    }

    fn apply_preconditioner(&self, ws: &mut AmgPcgWorkspace) {
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

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn max_abs_slice(a: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for &x in a {
        let ax = x.abs();
        if !ax.is_finite() {
            return f64::NAN;
        }
        if ax > max_abs {
            max_abs = ax;
        }
    }
    max_abs
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    let max_abs = max_abs_slice(a);
    if !max_abs.is_finite() {
        return f64::NAN;
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    if (1e-100..=1e100).contains(&max_abs) {
        return dot_slice(a, a).sqrt();
    }
    let sum_sq: f64 = a
        .iter()
        .map(|&x| {
            let s = x / max_abs;
            s * s
        })
        .sum();
    max_abs * sum_sq.sqrt()
}

/// Computes `(sa, sb, sum_i (a[i]/sa)*(b[i]/sb))` so `a^T b = sa * sb * dot` without
/// overflowing or underflowing across `1e-290..1e290`, while remaining bitwise identical
/// to `dot_slice` on `[1e-100, 1e100]`.
#[inline]
fn scaled_dot(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let ma = max_abs_slice(a);
    let mb = max_abs_slice(b);
    if !ma.is_finite() || !mb.is_finite() {
        return (1.0, 1.0, f64::NAN);
    }
    if ma == 0.0 || mb == 0.0 {
        return (1.0, 1.0, 0.0);
    }
    if (1e-100..=1e100).contains(&ma) && (1e-100..=1e100).contains(&mb) {
        return (1.0, 1.0, dot_slice(a, b));
    }
    let dot_scaled: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x / ma) * (y / mb))
        .sum();
    (ma, mb, dot_scaled)
}

fn compute_safe_inv_diag(diag: &[f64], max_diag: f64, zero_mask: Option<&[bool]>) -> Vec<f64> {
    let thresh = SMOOTHER_DIAG_EPS * max_diag;
    let mut inv_diag = Vec::with_capacity(diag.len());
    for (i, &d) in diag.iter().enumerate() {
        let masked = zero_mask.is_some_and(|m| m[i]);
        if !masked && d > 0.0 && d > thresh {
            let inv = 1.0 / d;
            inv_diag.push(if inv.is_finite() { inv } else { 0.0 });
        } else {
            inv_diag.push(0.0);
        }
    }
    inv_diag
}

fn forward_gs_sweep(a: &CsrMatrix, inv_diag: &[f64], rhs: &[f64], x: &mut [f64]) {
    let n = a.num_nodes();
    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();
    for row in 0..n {
        let inv_d = inv_diag[row];
        if inv_d == 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut sum = 0.0_f64;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row {
                sum += values[pos] * x[col];
            }
        }
        x[row] = (rhs[row] - sum) * inv_d;
    }
}

fn backward_gs_sweep(a: &CsrMatrix, inv_diag: &[f64], rhs: &[f64], x: &mut [f64]) {
    let n = a.num_nodes();
    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();
    for row in (0..n).rev() {
        let inv_d = inv_diag[row];
        if inv_d == 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut sum = 0.0_f64;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row {
                sum += values[pos] * x[col];
            }
        }
        x[row] = (rhs[row] - sum) * inv_d;
    }
}

fn build_coarsest_solver(
    coarsest_a: &CsrMatrix,
    config: &AmgPcgConfig,
) -> CoreResult<CoarsestSolver> {
    let nc = coarsest_a.num_nodes();
    if nc == 0 {
        return Ok(CoarsestSolver::DensePinverse {
            n: 0,
            pinv: Vec::new(),
        });
    }
    if nc <= config.max_coarse_size {
        if nc == 1 {
            let d = coarsest_a.diagonal()[0];
            let pinv_val = if d > 0.0 && (1.0 / d).is_finite() {
                1.0 / d
            } else {
                0.0
            };
            return Ok(CoarsestSolver::DensePinverse {
                n: 1,
                pinv: vec![pinv_val],
            });
        }
        let dense = coarsest_a.to_dense()?;
        let (lambdas, vectors) =
            symmetric_eigen_jacobi(&dense, nc, JACOBI_COARSEST_MAX_SWEEPS, JACOBI_COARSEST_TOL)?;
        let lambda_max = lambdas.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = if lambda_max > 0.0 {
            config.eig_eps * lambda_max
        } else {
            0.0
        };
        let mut inv_lambdas = vec![0.0_f64; nc];
        for k in 0..nc {
            let lam = lambdas[k];
            if lambda_max > 0.0 && lam > cutoff {
                let inv = 1.0 / lam;
                if inv.is_finite() {
                    inv_lambdas[k] = inv;
                }
            }
        }

        let mut pinv = vec![0.0_f64; nc * nc];
        for i in 0..nc {
            for j in i..nc {
                let mut sum = 0.0_f64;
                let row_i = i * nc;
                let row_j = j * nc;
                for (k, &inv_l) in inv_lambdas.iter().enumerate() {
                    if inv_l != 0.0 {
                        sum += vectors[row_i + k] * inv_l * vectors[row_j + k];
                    }
                }
                pinv[row_i + j] = sum;
                pinv[row_j + i] = sum;
            }
        }
        return Ok(CoarsestSolver::DensePinverse { n: nc, pinv });
    }

    let diag = coarsest_a.diagonal();
    let max_diag = diag.iter().copied().fold(0.0_f64, f64::max);
    let inv_diag = compute_safe_inv_diag(diag, max_diag, None);
    Ok(CoarsestSolver::BoundedSgs {
        sweeps: STALLED_COARSEST_SGS_SWEEPS,
        inv_diag,
    })
}

fn solve_coarsest(a: &CsrMatrix, solver: &CoarsestSolver, rhs: &[f64], sol: &mut [f64]) {
    match solver {
        CoarsestSolver::DensePinverse { n, pinv } => {
            let nc = *n;
            for i in 0..nc {
                let row = &pinv[i * nc..(i + 1) * nc];
                let mut sum = 0.0_f64;
                for j in 0..nc {
                    sum += row[j] * rhs[j];
                }
                sol[i] = sum;
            }
        }
        CoarsestSolver::BoundedSgs { sweeps, inv_diag } => {
            for _ in 0..*sweeps {
                forward_gs_sweep(a, inv_diag, rhs, sol);
                backward_gs_sweep(a, inv_diag, rhs, sol);
            }
        }
    }
}

#[derive(Default)]
struct GalerkinScratch {
    stamp: Vec<u32>,
    epoch: u32,
    col_pos: Vec<u32>,
    row_pairs: Vec<(u32, f64)>,
}

fn build_double_pairwise_level(
    a: &CsrMatrix,
    near_nullspace: &[f64],
    config: &AmgPcgConfig,
    rng: &mut DeterministicRng,
    scratch: &mut GalerkinScratch,
) -> CoreResult<Option<BuiltLevel>> {
    let n = a.num_nodes();
    let pass1_cap = (config.max_agg_size / 2).max(2);

    let (agg1, agg1_sizes) = pairwise_aggregate(
        a,
        &vec![1usize; n],
        config.theta,
        pass1_cap,
        true,
        config.randomize_ties,
        rng,
    );
    let n1 = agg1_sizes.len();
    if n1 == 0 || n1 >= n {
        return Ok(None);
    }

    let (p1_val, ns1) = build_prolongator_values(&agg1, n1, near_nullspace);
    let a1 = galerkin_ptap(a, &agg1, &p1_val, n1, scratch)?;

    // Stop after single pairwise pass if already within coarsest threshold.
    if n1 <= config.max_coarse_size {
        return Ok(Some(BuiltLevel {
            agg_id: agg1,
            p_val: p1_val,
            next_ns: ns1,
            next_a: a1,
        }));
    }

    let (agg2, agg2_sizes) = pairwise_aggregate(
        &a1,
        &agg1_sizes,
        config.theta,
        config.max_agg_size,
        true,
        config.randomize_ties,
        rng,
    );
    let n2 = agg2_sizes.len();

    let stall_limit = ((0.95 * (n as f64)).floor() as usize).max(1);
    if n2 == 0 || n2 >= n1 {
        if n1 <= stall_limit {
            return Ok(Some(BuiltLevel {
                agg_id: agg1,
                p_val: p1_val,
                next_ns: ns1,
                next_a: a1,
            }));
        }
        return Ok(None);
    }
    if n2 > stall_limit {
        return Ok(None);
    }

    let (p2_val, ns2) = build_prolongator_values(&agg2, n2, &ns1);
    let a2 = galerkin_ptap(&a1, &agg2, &p2_val, n2, scratch)?;

    // Compose P = P1 * P2 (still 1 nonzero per fine row, P^T P = I).
    let mut agg = Vec::with_capacity(n);
    let mut p_val = Vec::with_capacity(n);
    for i in 0..n {
        let a_mid = agg1[i] as usize;
        agg.push(agg2[a_mid]);
        p_val.push(p1_val[i] * p2_val[a_mid]);
    }

    Ok(Some(BuiltLevel {
        agg_id: agg,
        p_val,
        next_ns: ns2,
        next_a: a2,
    }))
}

fn pairwise_aggregate(
    a: &CsrMatrix,
    node_sizes: &[usize],
    theta: f64,
    max_agg_size: usize,
    allow_leaf_bypass: bool,
    randomize_ties: bool,
    rng: &mut DeterministicRng,
) -> (Vec<u32>, Vec<usize>) {
    let n = a.num_nodes();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();
    let diag = a.diagonal();

    // 1. Symmetric negative-off-diagonal strength of connection (Defect 5).
    let mut max_neg = vec![0.0_f64; n];
    for row in 0..n {
        if diag[row] <= 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut best = 0.0_f64;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row {
                let neg = -values[pos];
                if neg > best {
                    best = neg;
                }
            }
        }
        max_neg[row] = best;
    }

    let mut s_deg = vec![0usize; n];
    let mut max_s_deg = 0usize;
    for row in 0..n {
        let mn_row = max_neg[row];
        if mn_row <= 0.0 {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut cnt = 0usize;
        for pos in start..end {
            let col = indices[pos] as usize;
            if col != row && values[pos] < 0.0 {
                let mn_col = max_neg[col];
                if mn_col > 0.0 && (-values[pos]) >= theta * mn_row.min(mn_col) {
                    cnt += 1;
                }
            }
        }
        s_deg[row] = cnt;
        if cnt > max_s_deg {
            max_s_deg = cnt;
        }
    }

    // 2. O(n) counting sort in ascending strong-degree order.
    let mut counts = vec![0usize; max_s_deg + 1];
    for &d in &s_deg {
        counts[d] += 1;
    }
    let mut offsets = vec![0usize; max_s_deg + 1];
    let mut running = 0usize;
    for (d, &c) in counts.iter().enumerate() {
        offsets[d] = running;
        running += c;
    }
    let mut bucket_cursor = offsets.clone();
    let mut order = vec![0usize; n];
    for (i, &d) in s_deg.iter().enumerate() {
        order[bucket_cursor[d]] = i;
        bucket_cursor[d] += 1;
    }
    if randomize_ties {
        for d in 0..=max_s_deg {
            let start = offsets[d];
            let end = start + counts[d];
            if end - start > 1 {
                rng.shuffle(&mut order[start..end]);
            }
        }
    }

    // 3. Pairwise matching + size-capped leaf absorption.
    let mut agg_id = vec![u32::MAX; n];
    let mut agg_sizes: Vec<usize> = Vec::with_capacity(n / 2 + 1);

    for &u in &order {
        if agg_id[u] != u32::MAX {
            continue;
        }
        let mn_u = max_neg[u];
        let u_size = node_sizes[u];
        let u_is_small_leaf = allow_leaf_bypass && s_deg[u] <= 2 && u_size <= 2;

        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;

        let mut best_unagg: Option<(usize, f64, usize)> = None;
        let mut best_agg: Option<(usize, f64, usize)> = None;

        if mn_u > 0.0 {
            for pos in start..end {
                let v = indices[pos] as usize;
                if v == u || values[pos] >= 0.0 {
                    continue;
                }
                let mn_v = max_neg[v];
                if mn_v <= 0.0 {
                    continue;
                }
                let strength = -values[pos];
                if strength < theta * mn_u.min(mn_v) {
                    continue;
                }

                let v_agg = agg_id[v];
                if v_agg == u32::MAX {
                    let combined = u_size + node_sizes[v];
                    if combined <= max_agg_size || (u_is_small_leaf && node_sizes[v] <= 2) {
                        let replace = match best_unagg {
                            None => true,
                            Some((bv, bs, bdeg)) => {
                                strength > bs
                                    || (strength == bs
                                        && (s_deg[v] < bdeg || (s_deg[v] == bdeg && v < bv)))
                            }
                        };
                        if replace {
                            best_unagg = Some((v, strength, s_deg[v]));
                        }
                    }
                } else {
                    let a_idx = v_agg as usize;
                    let a_size = agg_sizes[a_idx];
                    if a_size + u_size <= max_agg_size || u_is_small_leaf {
                        let replace = match best_agg {
                            None => true,
                            Some((ba, bs, bas)) => {
                                strength > bs
                                    || (strength == bs
                                        && (a_size < bas || (a_size == bas && a_idx < ba)))
                            }
                        };
                        if replace {
                            best_agg = Some((a_idx, strength, a_size));
                        }
                    }
                }
            }
        }

        if let Some((v, _, _)) = best_unagg {
            let new_agg = agg_sizes.len() as u32;
            agg_id[u] = new_agg;
            agg_id[v] = new_agg;
            agg_sizes.push(u_size + node_sizes[v]);
        } else if let Some((a_idx, _, _)) = best_agg {
            agg_id[u] = a_idx as u32;
            agg_sizes[a_idx] += u_size;
        } else {
            let new_agg = agg_sizes.len() as u32;
            agg_id[u] = new_agg;
            agg_sizes.push(u_size);
        }
    }

    (agg_id, agg_sizes)
}

fn build_prolongator_values(
    agg_id: &[u32],
    num_coarse: usize,
    near_nullspace: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = agg_id.len();
    let mut norm_sq = vec![0.0_f64; num_coarse];
    let mut counts = vec![0usize; num_coarse];
    for (i, &a) in agg_id.iter().enumerate() {
        let a_idx = a as usize;
        let v = near_nullspace[i];
        norm_sq[a_idx] += v * v;
        counts[a_idx] += 1;
    }

    let mut ns_coarse = vec![0.0_f64; num_coarse];
    for a in 0..num_coarse {
        ns_coarse[a] = norm_sq[a].sqrt();
    }

    let mut p_val = Vec::with_capacity(n);
    for (i, &a) in agg_id.iter().enumerate() {
        let a_idx = a as usize;
        let s_a = ns_coarse[a_idx];
        if s_a > 1e-30 {
            p_val.push(near_nullspace[i] / s_a);
        } else {
            p_val.push(1.0 / (counts[a_idx].max(1) as f64).sqrt());
        }
    }

    (p_val, ns_coarse)
}

fn galerkin_ptap(
    a: &CsrMatrix,
    agg_id: &[u32],
    p_val: &[f64],
    num_coarse: usize,
    scratch: &mut GalerkinScratch,
) -> CoreResult<CsrMatrix> {
    let n_fine = a.num_nodes();
    let mut group_counts = vec![0u32; num_coarse];
    for &g in agg_id {
        group_counts[g as usize] += 1;
    }
    let mut group_ptr = Vec::with_capacity(num_coarse + 1);
    group_ptr.push(0u32);
    let mut running = 0u32;
    for &c in &group_counts {
        running += c;
        group_ptr.push(running);
    }
    let mut cursor = group_ptr[..num_coarse].to_vec();
    let mut group_nodes = vec![0u32; n_fine];
    for (i, &g) in agg_id.iter().enumerate() {
        let g_idx = g as usize;
        group_nodes[cursor[g_idx] as usize] = i as u32;
        cursor[g_idx] += 1;
    }

    if scratch.stamp.len() < num_coarse {
        scratch.stamp.resize(num_coarse, 0);
        scratch.col_pos.resize(num_coarse, 0);
    }

    let indptr = a.indptr();
    let indices = a.indices();
    let values = a.values();

    let mut out_indptr = Vec::with_capacity(num_coarse + 1);
    out_indptr.push(0u32);
    let mut out_indices = Vec::new();
    let mut out_values = Vec::new();

    for row_c in 0..num_coarse {
        if scratch.epoch == u32::MAX {
            scratch.stamp.fill(0);
            scratch.epoch = 1;
        } else {
            scratch.epoch += 1;
        }
        let epoch = scratch.epoch;
        scratch.row_pairs.clear();

        let g_start = group_ptr[row_c] as usize;
        let g_end = group_ptr[row_c + 1] as usize;
        for &fine_u32 in &group_nodes[g_start..g_end] {
            let i = fine_u32 as usize;
            let pi = p_val[i];
            let a_start = indptr[i] as usize;
            let a_end = indptr[i + 1] as usize;
            for pos in a_start..a_end {
                let j = indices[pos] as usize;
                let col_c = agg_id[j];
                let contrib = pi * values[pos] * p_val[j];
                let col_idx = col_c as usize;
                if scratch.stamp[col_idx] != epoch {
                    scratch.stamp[col_idx] = epoch;
                    scratch.col_pos[col_idx] = scratch.row_pairs.len() as u32;
                    scratch.row_pairs.push((col_c, contrib));
                } else {
                    scratch.row_pairs[scratch.col_pos[col_idx] as usize].1 += contrib;
                }
            }
        }

        scratch.row_pairs.sort_unstable_by_key(|&(col, _)| col);
        for &(col, val) in &scratch.row_pairs {
            if val != 0.0 {
                out_indices.push(col);
                out_values.push(val);
            }
        }
        let ptr = u32::try_from(out_indices.len())
            .map_err(|_| CoreError::invalid_csr("Galerkin coarse nnz exceeds u32::MAX"))?;
        out_indptr.push(ptr);
    }

    let raw = CsrMatrix::from_csr(num_coarse, out_indptr, out_indices, out_values)?;
    raw.symmetrize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + (cols as u32)));
                }
            }
        }
        CsrGraph::from_edges(rows * cols, &edges, None, false, true).unwrap()
    }

    fn make_power_law_hub_graph(num_hubs: usize, hub_deg: usize) -> CsrGraph<f64> {
        // Construct a synthetic web-Stanford-like multi-hub graph with thousands of degree-1
        // leaves and interconnected high-degree hubs.
        let mut edges = Vec::new();
        for h in 0..num_hubs {
            for h2 in (h + 1)..num_hubs {
                edges.push((h as u32, h2 as u32));
            }
        }
        let mut next_node = num_hubs;
        for h in 0..num_hubs {
            for _ in 0..hub_deg {
                let leaf = next_node as u32;
                edges.push((h as u32, leaf));
                next_node += 1;
            }
            // Add a chain bridge between hubs through degree-2 nodes.
            if h + 1 < num_hubs {
                let b1 = next_node as u32;
                let b2 = (next_node + 1) as u32;
                edges.push((h as u32, b1));
                edges.push((b1, b2));
                edges.push((b2, (h + 1) as u32));
                next_node += 2;
            }
        }
        CsrGraph::from_edges(next_node, &edges, None, false, true).unwrap()
    }

    #[test]
    fn test_spd_regularized_laplacians_and_shifted_combinatorial() {
        let grid = make_grid_graph(12, 12);
        let n = grid.num_nodes();
        let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) as f64).sin()).collect();

        for &gamma in &[0.01, 1.0, 100.0, 10_000.0] {
            let solver =
                AmgPcgSolver::from_regularized_sym_laplacian(&grid, gamma, AmgPcgConfig::default())
                    .unwrap();
            assert!(solver.num_levels() >= 2);
            assert!(solver.coarsest_size() <= solver.config().max_coarse_size);
            assert!(solver.max_level_nnz() <= solver.finest_matrix().nnz());

            let (x, res) = solver.solve_vec(&b).unwrap();
            assert!(res.converged, "gamma={gamma} failed: {res:?}");
            assert!(res.relative_residual <= 1e-8);
            assert!(x.iter().all(|v| v.is_finite()));
        }

        let comb_solver = AmgPcgSolver::from_laplacian(
            &grid,
            LaplacianKind::Combinatorial,
            1e-3,
            AmgPcgConfig::default(),
        )
        .unwrap();
        let (x_c, res_c) = comb_solver.solve_vec(&b).unwrap();
        assert!(res_c.converged, "shifted combinatorial failed: {res_c:?}");
        assert!(res_c.relative_residual <= 1e-8);
        assert!(x_c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_singular_disconnected_isolated_empty_singleton_and_zero_rhs() {
        // 1. Empty graph (n = 0)
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let s0 = AmgPcgSolver::from_laplacian(
            &g0,
            LaplacianKind::Combinatorial,
            0.0,
            AmgPcgConfig::default(),
        )
        .unwrap();
        assert_eq!(s0.num_nodes(), 0);
        let (x0, r0) = s0.solve_vec(&[]).unwrap();
        assert!(x0.is_empty());
        assert!(r0.converged);
        assert_eq!(r0.iterations, 0);

        // 2. Singleton graph (n = 1) unshifted L = [0]
        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let s1 = AmgPcgSolver::from_laplacian(
            &g1,
            LaplacianKind::Combinatorial,
            0.0,
            AmgPcgConfig::default(),
        )
        .unwrap();
        let (x1_zero, r1_zero) = s1.solve_vec(&[0.0]).unwrap();
        assert_eq!(x1_zero, vec![0.0]);
        assert!(r1_zero.converged);
        // Inconsistent RHS on zero row must not NaN or panic.
        let (x1_bad, r1_bad) = s1.solve_vec(&[3.0]).unwrap();
        assert!(x1_bad.iter().all(|v| v.is_finite()));
        assert!(!r1_bad.converged);
        assert!((r1_bad.residual_norm - 3.0).abs() < 1e-12);

        // 3. Edgeless graph (n = 32) unshifted L = 0
        let g_edgeless = CsrGraph::<f64>::from_edges(32, &[], None, false, false).unwrap();
        let s_edgeless = AmgPcgSolver::from_laplacian(
            &g_edgeless,
            LaplacianKind::Combinatorial,
            0.0,
            AmgPcgConfig::default(),
        )
        .unwrap();
        let (xe, re) = s_edgeless.solve_vec(&vec![0.0; 32]).unwrap();
        assert_eq!(xe, vec![0.0; 32]);
        assert!(re.converged);
        let (xe_inc, re_inc) = s_edgeless.solve_vec(&vec![1.0; 32]).unwrap();
        assert!(xe_inc.iter().all(|v| v.is_finite()));
        assert!(!re_inc.converged);

        // 4. Multi-component graph + isolated zero-row vertices with unshifted combinatorial L
        let mut edges = Vec::new();
        // Component 1: 0..30 cycle
        for i in 0..30u32 {
            edges.push((i, (i + 1) % 30));
        }
        // Component 2: 30..60 cycle
        for i in 30..60u32 {
            edges.push((i, 30 + (i - 30 + 1) % 30));
        }
        // Component 3: 60..90 path
        for i in 60..89u32 {
            edges.push((i, i + 1));
        }
        // Nodes 90, 91, 92 are isolated (zero rows in combinatorial L)
        let g_multi = CsrGraph::<f64>::from_edges(93, &edges, None, false, true).unwrap();
        let s_multi = AmgPcgSolver::from_laplacian(
            &g_multi,
            LaplacianKind::Combinatorial,
            0.0,
            AmgPcgConfig::default(),
        )
        .unwrap();
        // Construct consistent RHS orthogonal to each component's constant vector and 0 on isolated nodes.
        let mut b_cons = vec![0.0_f64; 93];
        for (i, v) in b_cons.iter_mut().enumerate().take(30) {
            *v = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        for (i, v) in b_cons.iter_mut().enumerate().take(60).skip(30) {
            *v = if i % 2 == 0 { 2.0 } else { -2.0 };
        }
        for (i, v) in b_cons.iter_mut().enumerate().take(90).skip(60) {
            *v = if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        let (x_m, r_m) = s_multi.solve_vec(&b_cons).unwrap();
        assert!(r_m.converged, "multi-CC singular L failed: {r_m:?}");
        assert!(r_m.relative_residual <= 1e-8);
        assert!(x_m.iter().all(|v| v.is_finite()));
        assert_eq!(x_m[90], 0.0);
        assert_eq!(x_m[91], 0.0);
        assert_eq!(x_m[92], 0.0);

        // Zero RHS fast path
        let (x_z, r_z) = s_multi.solve_vec(&vec![0.0; 93]).unwrap();
        assert_eq!(x_z, vec![0.0; 93]);
        assert!(r_z.converged);
        assert_eq!(r_z.iterations, 0);
    }

    #[test]
    fn test_power_law_hub_web_stanford_like_no_fill_explosion() {
        let g = make_power_law_hub_graph(3, 2_000);
        let n = g.num_nodes();
        let cfg = AmgPcgConfig::default();
        let solver = AmgPcgSolver::from_laplacian(&g, LaplacianKind::Combinatorial, 1e-3, cfg)
            .expect("hub graph solver construction");
        let input_nnz = solver.finest_matrix().nnz();
        assert!(
            solver.max_level_nnz() <= input_nnz,
            "fill-in explosion: max_level_nnz={} > input_nnz={}",
            solver.max_level_nnz(),
            input_nnz
        );
        assert!(
            solver.operator_complexity() < 2.0,
            "operator_complexity={} >= 2.0",
            solver.operator_complexity()
        );

        let b: Vec<f64> = (0..n).map(|i| ((i * 13 + 5) as f64).cos()).collect();
        let (x, res) = solver.solve_vec(&b).unwrap();
        assert!(res.converged, "hub graph solve failed: {res:?}");
        assert!(
            res.iterations < 40,
            "too many iterations: {}",
            res.iterations
        );
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_step_run_equivalence_batch_determinism_and_scale_invariance() {
        let g = make_grid_graph(10, 10);
        let n = g.num_nodes();
        let cfg = AmgPcgConfig {
            randomize_ties: true,
            seed: 123,
            ..AmgPcgConfig::default()
        };
        let solver1 = AmgPcgSolver::from_regularized_sym_laplacian(&g, 10.0, cfg).unwrap();
        let solver2 = AmgPcgSolver::from_regularized_sym_laplacian(&g, 10.0, cfg).unwrap();

        let b: Vec<f64> = (0..n).map(|i| ((i * 11 + 1) as f64).sin()).collect();
        let mut ws_sync = solver1.new_workspace();
        let mut x_sync = vec![0.0; n];
        let res_sync = solver1.solve(&b, &mut x_sync, &mut ws_sync).unwrap();

        let mut ws_step = AmgPcgWorkspace::new();
        solver2.init_solve(&b, None, &mut ws_step).unwrap();
        assert_eq!(ws_step.status(), SolverStatus::Initialized);
        solver2.run_steps(&mut ws_step, 2).unwrap();
        while !ws_step.is_finished() {
            solver2.step(&mut ws_step).unwrap();
        }
        assert_eq!(ws_step.x(), x_sync.as_slice());
        assert_eq!(ws_step.iteration(), res_sync.iterations);
        assert_eq!(ws_step.converged(), res_sync.converged);

        // Batched solve equivalence with single-RHS solves
        let ncols = 3;
        let mut b_batch = vec![0.0; n * ncols];
        let mut x_batch = vec![0.0; n * ncols];
        let mut cols = vec![vec![0.0; n]; ncols];
        for c in 0..ncols {
            for r in 0..n {
                let v = (((r * 17 + c * 31) % 97) as f64) / 50.0 - 1.0;
                b_batch[r * ncols + c] = v;
                cols[c][r] = v;
            }
        }
        let batch_res = solver1
            .solve_batch(&b_batch, &mut x_batch, ncols, &mut ws_sync)
            .unwrap();
        assert!(batch_res.all_converged);
        for c in 0..ncols {
            let (x_col, _) = solver1.solve_vec(&cols[c]).unwrap();
            for r in 0..n {
                assert_eq!(x_batch[r * ncols + c], x_col[r]);
            }
        }

        // Scale invariance: b * 1e-12 vs b * 1e10 must NOT exit at iter 0 for tiny b
        let b_tiny: Vec<f64> = b.iter().map(|&v| v * 1e-12).collect();
        let b_huge: Vec<f64> = b.iter().map(|&v| v * 1e10).collect();
        let (x_tiny, r_tiny) = solver1.solve_vec(&b_tiny).unwrap();
        let (x_huge, r_huge) = solver1.solve_vec(&b_huge).unwrap();
        assert!(r_tiny.converged && r_tiny.iterations > 0);
        assert!(r_huge.converged && r_huge.iterations > 0);
        assert_eq!(r_tiny.iterations, r_huge.iterations);
        for i in 0..n {
            assert!((x_tiny[i] * 1e22 - x_huge[i]).abs() <= 1e-6 * x_huge[i].abs().max(1.0));
        }

        // Extreme scale invariance across 1e-290 .. 1e+290 (no overflow/underflow)
        let b_unit: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).cos()).collect();
        let (x_ref, r_ref) = solver1.solve_vec(&b_unit).unwrap();
        assert!(r_ref.converged);
        for &scale in &[
            1e-155_f64, 1e155_f64, 1e-250_f64, 1e250_f64, 1e-290_f64, 1e290_f64,
        ] {
            let b_scaled: Vec<f64> = b_unit.iter().map(|&v| v * scale).collect();
            let (x_scaled, r_scaled) = solver1.solve_vec(&b_scaled).unwrap();
            assert!(
                r_scaled.converged,
                "failed to converge at scale={scale:.1e}: rel_res={:.3e}",
                r_scaled.relative_residual
            );
            assert_eq!(r_scaled.iterations, r_ref.iterations);
            for i in 0..n {
                let rel_err = (x_scaled[i] / scale - x_ref[i]).abs();
                assert!(
                    rel_err <= 1e-7 * x_ref[i].abs().max(1.0),
                    "scale={scale:.1e} mismatch at {i}: rel_err={rel_err}"
                );
            }
        }
    }

    #[test]
    fn test_adversarial_validation_rejects_bad_inputs() {
        // Asymmetric matrix
        let asym = CsrMatrix::from_csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -2.0, 2.0],
        )
        .unwrap();
        assert!(AmgPcgSolver::new(&asym, AmgPcgConfig::default()).is_err());

        // Negative diagonal
        let neg_diag = CsrMatrix::from_csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![-1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert!(AmgPcgSolver::new(&neg_diag, AmgPcgConfig::default()).is_err());

        // Zero diagonal with non-zero off-diagonal
        let zero_diag_off = CsrMatrix::from_csr(
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![0.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        assert!(AmgPcgSolver::new(&zero_diag_off, AmgPcgConfig::default()).is_err());

        // Non-finite RHS / dimension mismatch
        let g = make_grid_graph(3, 3);
        let solver =
            AmgPcgSolver::from_regularized_sym_laplacian(&g, 1.0, AmgPcgConfig::default()).unwrap();
        assert!(solver.solve_vec(&[f64::NAN; 9]).is_err());
        assert!(solver.solve_vec(&[1.0; 8]).is_err());
        assert!(
            AmgPcgSolver::from_laplacian(
                &g,
                LaplacianKind::RandomWalk,
                0.0,
                AmgPcgConfig::default()
            )
            .is_err()
        );
    }
}
