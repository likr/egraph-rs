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
//!    pseudoinverse $A_c^\dagger = V \Lambda^\dagger V^T$ via [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi)
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

mod builder;
mod config;
mod hierarchy;
mod operator;
mod smoother;
mod solver;
mod vcycle;
mod workspace;

pub use builder::AmgPcgSolver;
pub use config::{AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult};
pub use workspace::AmgPcgWorkspace;

pub(super) const RESIDUAL_RECOMPUTE_INTERVAL: usize = 50;
pub(super) const STALLED_COARSEST_SGS_SWEEPS: usize = 10;
pub(super) const SMOOTHER_DIAG_EPS: f64 = 1e-30;
pub(super) const JACOBI_COARSEST_MAX_SWEEPS: usize = 120;
pub(super) const JACOBI_COARSEST_TOL: f64 = 1e-12;

#[cfg(test)]
mod tests;
