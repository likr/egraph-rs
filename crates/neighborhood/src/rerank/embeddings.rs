//! Spectral resistance and heat-diffusion coordinate embeddings over $L_{\mathrm{sym}}$.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;

/// Computes effective-resistance spectral embedding $X_{\mathrm{res}} = D^{-1/2} U \Lambda^{-1/2}$
/// from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$, filtering eigenvalues
/// $\lambda \le \mathrm{eig\_eps}$.
///
/// Returns `(coords_row_major, lambdas_kept)` where `coords_row_major` has shape
/// `(graph.num_nodes(), lambdas_kept.len())`.
pub fn compute_resistance_embedding<W: Scalar>(
    graph: &CsrGraph<W>,
    dim: usize,
    tol: f64,
    max_iters: usize,
    seed: u64,
    eig_eps: f64,
) -> CoreResult<(Vec<f64>, Vec<f64>)> {
    if dim == 0 {
        return Err(CoreError::invalid_input("dim must be >= 1"));
    }
    if !eig_eps.is_finite() {
        return Err(CoreError::non_finite(
            "compute_resistance_embedding::eig_eps",
        ));
    }
    if eig_eps < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eig_eps must be >= 0, got {eig_eps}"
        )));
    }

    let n = graph.num_nodes();
    if n <= 1 {
        return Ok((Vec::new(), Vec::new()));
    }

    let eig = smallest_sym_laplacian_eigenpairs(graph, dim, tol, max_iters, seed)?;
    let kept_modes: Vec<usize> = eig
        .lambdas
        .iter()
        .enumerate()
        .filter_map(|(m, &lam)| (lam > eig_eps).then_some(m))
        .collect();
    let k_kept = kept_modes.len();
    if k_kept == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let lambdas_kept: Vec<f64> = kept_modes.iter().map(|&m| eig.lambdas[m]).collect();
    let inv_sqrt_lam: Vec<f64> = lambdas_kept.iter().map(|&lam| lam.sqrt().recip()).collect();

    let mut coords = vec![0.0_f64; n * k_kept];
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let out_row = &mut coords[i * k_kept..(i + 1) * k_kept];
        for (col, &m) in kept_modes.iter().enumerate() {
            out_row[col] = d_inv * eig.vectors[i * eig.num_modes + m] * inv_sqrt_lam[col];
        }
    }
    Ok((coords, lambdas_kept))
}

/// Computes heat-diffusion spectral embedding $X_{\mathrm{diff}} = D^{-1/2} U \exp(-0.5 t \Lambda)$
/// from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$.
///
/// When `t` is `None`, automatically sets $t = -\ln(p) / \lambda_k$ (falling back to $-\ln(p)$
/// when $\lambda_k \le 10^{-12}$). Returns `(coords_row_major, lambdas, t_used)`.
pub fn compute_diffusion_embedding<W: Scalar>(
    graph: &CsrGraph<W>,
    dim: usize,
    p: f64,
    t: Option<f64>,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<(Vec<f64>, Vec<f64>, f64)> {
    if dim == 0 {
        return Err(CoreError::invalid_input("dim must be >= 1"));
    }
    if !p.is_finite() {
        return Err(CoreError::non_finite("compute_diffusion_embedding::p"));
    }
    if !(0.0 < p && p < 1.0) {
        return Err(CoreError::invalid_input(format!(
            "diffusion parameter p must satisfy 0 < p < 1, got {p}"
        )));
    }
    if let Some(t_val) = t {
        if !t_val.is_finite() {
            return Err(CoreError::non_finite("compute_diffusion_embedding::t"));
        }
        if t_val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "diffusion time t must be >= 0, got {t_val}"
            )));
        }
    }

    let n = graph.num_nodes();
    if n <= 1 {
        let t_used = t.unwrap_or_else(|| -p.ln());
        return Ok((Vec::new(), Vec::new(), t_used));
    }

    let eig = smallest_sym_laplacian_eigenpairs(graph, dim, tol, max_iters, seed)?;
    let lambdas: Vec<f64> = eig.lambdas.iter().map(|&lam| lam.max(0.0)).collect();
    let t_used = match t {
        Some(tv) => tv,
        None => match lambdas.last() {
            Some(&lk) if lk > 1e-12 => -p.ln() / lk,
            _ => -p.ln(),
        },
    };
    let weights: Vec<f64> = lambdas
        .iter()
        .map(|&lam| (-0.5 * t_used * lam).exp())
        .collect();

    let k_modes = eig.num_modes;
    let mut coords = vec![0.0_f64; n * k_modes];
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let out_row = &mut coords[i * k_modes..(i + 1) * k_modes];
        for m in 0..k_modes {
            out_row[m] = d_inv * eig.vectors[i * k_modes + m] * weights[m];
        }
    }
    Ok((coords, lambdas, t_used))
}
