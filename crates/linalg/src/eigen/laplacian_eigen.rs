use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight};

use crate::csr_matrix::symmetric_normalized_laplacian;
use crate::operator::LinearOperator;

use super::jacobi::symmetric_eigen_jacobi;
use super::lanczos::LanczosEigen;

/// Result of computing the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct EigenResult {
    /// Smallest nontrivial eigenvalues sorted in ascending order.
    pub lambdas: Vec<f64>,
    /// Row-major orthonormal eigenvectors of shape `(num_nodes, num_modes)`.
    pub vectors: Vec<f64>,
    /// Inverse square-root degree vector $D^{-1/2}$ (`0.0` for isolated vertices).
    pub inv_sqrt_deg: Vec<f64>,
    /// Number of vertices in the graph.
    pub num_nodes: usize,
    /// Number of returned nontrivial modes (`min(n_modes, num_nodes - 1)` when `num_nodes >= 2`).
    pub num_modes: usize,
    /// Whether the dense Jacobi fallback path was used instead of iterative Lanczos.
    pub used_dense_fallback: bool,
    /// Number of Lanczos thick-restart iterations executed (`0` when using dense Jacobi fallback).
    pub iterations: usize,
}

fn validate_eigenpair_config(n_modes: usize, config: &LanczosEigen) -> CoreResult<()> {
    if n_modes == 0 {
        return Err(CoreError::invalid_input("n_modes must be >= 1"));
    }
    if !config.tol.is_finite() {
        return Err(CoreError::non_finite("LanczosEigen::tol"));
    }
    if config.tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if config.max_iters == 0 {
        return Err(CoreError::invalid_input("max_iters must be >= 1"));
    }
    Ok(())
}

/// Computes the smallest `n_modes` nontrivial eigenpairs of $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$
/// using an explicit [`LanczosEigen`] configuration (including optional `subspace_dim`),
/// skipping the trivial $\lambda_0 \approx 0$ mode cleanly.
///
/// Uses dense [`symmetric_eigen_jacobi`] fallback when `n <= max(48, 2 * actual_modes + 4)`,
/// and SpMV-only block thick-restart [`LanczosEigen`] on the shifted operator
/// $M v = 2.5 v - L_{\mathrm{sym}} v$ for larger graphs.
pub fn smallest_sym_laplacian_eigenpairs_with_config<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    n_modes: usize,
    config: &LanczosEigen,
) -> CoreResult<EigenResult> {
    validate_eigenpair_config(n_modes, config)?;

    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
    let n = graph.num_nodes();
    if n <= 1 {
        return Ok(EigenResult {
            lambdas: Vec::new(),
            vectors: Vec::new(),
            inv_sqrt_deg,
            num_nodes: n,
            num_modes: 0,
            used_dense_fallback: true,
            iterations: 0,
        });
    }

    let actual_modes = n_modes.min(n - 1);
    let use_dense = n <= 48usize.max(2 * actual_modes + 4);

    if use_dense {
        let dense = l_sym.to_dense()?;
        let (evals, evecs) = symmetric_eigen_jacobi(&dense, n, 120, config.tol.min(1e-12))?;
        let lambdas: Vec<f64> = evals[1..=actual_modes]
            .iter()
            .map(|&v| v.max(0.0))
            .collect();
        let mut vectors = vec![0.0_f64; n * actual_modes];
        for row in 0..n {
            for m in 0..actual_modes {
                vectors[row * actual_modes + m] = evecs[row * n + (1 + m)];
            }
        }
        return Ok(EigenResult {
            lambdas,
            vectors,
            inv_sqrt_deg,
            num_nodes: n,
            num_modes: actual_modes,
            used_dense_fallback: true,
            iterations: 0,
        });
    }

    const SHIFT: f64 = 2.5;
    let k_total = actual_modes + 1;
    let mut l_sym_ref = &l_sym;
    let (raw_lambdas, mu_vecs, iterations) =
        config.smallest_eigenpairs_operator_with_iters(&mut l_sym_ref, k_total, Some(SHIFT))?;

    let lambdas: Vec<f64> = raw_lambdas[1..=actual_modes]
        .iter()
        .map(|&lam| lam.max(0.0))
        .collect();
    let mut vectors = vec![0.0_f64; n * actual_modes];
    for row in 0..n {
        for m in 0..actual_modes {
            vectors[row * actual_modes + m] = mu_vecs[row * k_total + (1 + m)];
        }
    }

    Ok(EigenResult {
        lambdas,
        vectors,
        inv_sqrt_deg,
        num_nodes: n,
        num_modes: actual_modes,
        used_dense_fallback: false,
        iterations,
    })
}

/// Computes the smallest `n_modes` nontrivial eigenpairs of $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$,
/// skipping the trivial $\lambda_0 \approx 0$ mode cleanly.
///
/// Convenience wrapper around [`smallest_sym_laplacian_eigenpairs_with_config`] using
/// `LanczosEigen::new(tol, max_iters, seed)`.
pub fn smallest_sym_laplacian_eigenpairs<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    n_modes: usize,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<EigenResult> {
    let config = LanczosEigen::new(tol, max_iters, seed);
    smallest_sym_laplacian_eigenpairs_with_config(graph, n_modes, &config)
}

/// Computes the smallest `n_modes` nontrivial eigenpairs of $L_{\mathrm{sym}}$ via shift-invert
/// block thick-restart Lanczos using an externally supplied inverse shifted operator
/// `inv_shifted_op` representing $(L_{\mathrm{sym}} + \sigma I)^{-1}$ for `shift = sigma > 0`.
///
/// Largest eigenvalues $\mu_i$ of $(L_{\mathrm{sym}} + \sigma I)^{-1}$ map back to the smallest
/// eigenvalues $\lambda_i = 1 / \mu_i - \sigma$ of $L_{\mathrm{sym}}$, skipping the trivial
/// $\lambda_0 \approx 0$ mode ($\mu_0 \approx 1/\sigma$). Supplying a preconditioned solver such as
/// `AmgPcgSolver` via a [`LinearOperator<f64>`] adapter avoids coupling `egraph-linalg` or
/// `egraph-linalg-mds` to specific linear-solver crates.
pub fn smallest_eigenpairs_from_inverse_operator<O: LinearOperator<f64> + ?Sized>(
    inv_shifted_op: &mut O,
    inv_sqrt_deg: Vec<f64>,
    n_modes: usize,
    shift: f64,
    lanczos: &LanczosEigen,
) -> CoreResult<EigenResult> {
    validate_eigenpair_config(n_modes, lanczos)?;
    if !shift.is_finite() {
        return Err(CoreError::non_finite(
            "smallest_eigenpairs_from_inverse_operator::shift",
        ));
    }
    if shift <= 0.0 {
        return Err(CoreError::invalid_input(
            "shift must be strictly positive for shift-invert Lanczos",
        ));
    }
    let n = inv_shifted_op.dim();
    if inv_sqrt_deg.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            inv_sqrt_deg.len(),
            "smallest_eigenpairs_from_inverse_operator::inv_sqrt_deg",
        ));
    }
    if inv_sqrt_deg.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite(
            "smallest_eigenpairs_from_inverse_operator::inv_sqrt_deg",
        ));
    }
    if n <= 1 {
        return Ok(EigenResult {
            lambdas: Vec::new(),
            vectors: Vec::new(),
            inv_sqrt_deg,
            num_nodes: n,
            num_modes: 0,
            used_dense_fallback: false,
            iterations: 0,
        });
    }

    let actual_modes = n_modes.min(n - 1);
    let k_total = actual_modes + 1;
    let (mu_desc, mu_vecs, iterations) =
        lanczos.largest_eigenpairs_operator_with_iters(inv_shifted_op, k_total)?;

    let mut lambdas = Vec::with_capacity(actual_modes);
    for &mu in &mu_desc[1..=actual_modes] {
        if !mu.is_finite() || mu <= 0.0 {
            return Err(CoreError::numerical_breakdown(
                "inverse operator produced non-positive or non-finite Ritz value",
            ));
        }
        let lam = (1.0 / mu - shift).max(0.0);
        if !lam.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "shift-invert eigenvalue mapping overflowed f64",
            ));
        }
        lambdas.push(lam);
    }

    let mut vectors = vec![0.0_f64; n * actual_modes];
    for row in 0..n {
        for m in 0..actual_modes {
            vectors[row * actual_modes + m] = mu_vecs[row * k_total + (1 + m)];
        }
    }

    Ok(EigenResult {
        lambdas,
        vectors,
        inv_sqrt_deg,
        num_nodes: n,
        num_modes: actual_modes,
        used_dense_fallback: false,
        iterations,
    })
}
