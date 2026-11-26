use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use crate::csr_matrix::symmetric_normalized_laplacian;

use super::jacobi::symmetric_eigen_jacobi;
use super::subspace::{SubspaceEigen, fill_random_unit_orthogonal_buf, norm_slice};

/// Result of computing the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$.
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
}

/// Pure-Rust SpMV-only block thick-restart Lanczos eigensolver with two-pass MGS
/// reorthogonalization.
///
/// Uses block width $b = 4$ (when `k >= 2` and `n > 64`) so degenerate eigenvalues of
/// multiplicity up to 4 (e.g., 2D grids and periodic tori) are resolved deterministically
/// without relying on floating-point rounding drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanczosEigen {
    /// Relative convergence tolerance (weighted by spectral gap for tight meshes).
    pub tol: f64,
    /// Maximum number of thick-restart cycles.
    pub max_iters: usize,
    /// Optional explicit Krylov subspace size override.
    pub subspace_dim: Option<usize>,
    /// Deterministic PRNG seed for initial block and invariant-subspace recovery.
    pub seed: u64,
}

impl Default for LanczosEigen {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iters: 500,
            subspace_dim: None,
            seed: 42,
        }
    }
}

impl LanczosEigen {
    /// Creates a new [`LanczosEigen`] solver configuration.
    pub fn new(tol: f64, max_iters: usize, seed: u64) -> Self {
        Self {
            tol,
            max_iters,
            subspace_dim: None,
            seed,
        }
    }

    /// Computes the `k` largest eigenpairs of an $n \times n$ symmetric linear operator `matvec`
    /// returning `(eigenvalues_desc, row_major_vectors)` with `vectors` shape `(n, k)`.
    pub fn largest_eigenpairs<F>(
        &self,
        n: usize,
        k: usize,
        mut matvec: F,
    ) -> CoreResult<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut(&[f64], &mut [f64]) -> CoreResult<()>,
    {
        if k == 0 {
            return Err(CoreError::invalid_input("k must be >= 1"));
        }
        if k > n {
            return Err(CoreError::invalid_input(format!(
                "requested k={k} exceeds matrix dimension n={n}"
            )));
        }
        if !self.tol.is_finite() {
            return Err(CoreError::non_finite("LanczosEigen::tol"));
        }
        if self.tol <= 0.0 {
            return Err(CoreError::invalid_input("tol must be strictly positive"));
        }
        if self.max_iters == 0 {
            return Err(CoreError::invalid_input("max_iters must be >= 1"));
        }

        let b = if k == 1 || n <= 64 {
            1
        } else {
            4.min(n / 8).max(1)
        };
        let default_m = if b > 1 {
            (4 * k).max(64)
        } else {
            (3 * k).max(48)
        };
        let mut m_sub = self.subspace_dim.unwrap_or(default_m).max(k + 2 * b).min(n);
        if m_sub < n && b > 1 {
            m_sub = (m_sub / b).max(4) * b;
            m_sub = m_sub.min(n);
        }
        let eff_b = if m_sub == n { 1 } else { b };

        let mut ws = SubspaceEigen::new(n, m_sub, eff_b);
        let mut rng = DeterministicRng::from_seed_u64(self.seed);

        let mut tmp_vec = vec![0.0_f64; n];
        for c in 0..eff_b {
            if !fill_random_unit_orthogonal_buf(&ws.q_vecs, c, &[], 0, n, &mut tmp_vec, &mut rng) {
                return Err(CoreError::numerical_breakdown(
                    "failed to initialize orthogonal Lanczos starting block",
                ));
            }
            ws.q_vecs[c * n..(c + 1) * n].copy_from_slice(&tmp_vec);
        }

        let mut start_j = 0usize;
        let k_keep = if m_sub < n {
            (k + ((m_sub - k) / 3).max(2 * eff_b)).min(m_sub.saturating_sub(2 * eff_b))
        } else {
            k
        };

        let mut final_evals = Vec::new();
        let mut final_evecs = Vec::new();
        let mut final_m = m_sub;
        let mut w_orig_norms = vec![0.0_f64; eff_b];

        for iter in 0..self.max_iters {
            let mut j = start_j;
            let mut effective_m = m_sub;

            while j + eff_b <= m_sub {
                for c in 0..eff_b {
                    let q_jc = &ws.q_vecs[(j + c) * n..(j + c + 1) * n];
                    let w_dst = &mut ws.w_block[c * n..(c + 1) * n];
                    matvec(q_jc, w_dst)?;
                    w_orig_norms[c] = norm_slice(w_dst);
                }

                ws.orthogonalize_w_against_q(j, start_j);
                let dst_slot = j + eff_b;
                let rank_ok = ws.qr_next_block(j + eff_b, &w_orig_norms, dst_slot, &mut rng);

                if j + 2 * eff_b <= m_sub {
                    for r in 0..eff_b {
                        for c in 0..eff_b {
                            let val = ws.r_block[r * eff_b + c];
                            ws.b_mat[(j + eff_b + r) * m_sub + (j + c)] = val;
                            ws.b_mat[(j + c) * m_sub + (j + eff_b + r)] = val;
                        }
                    }
                    if rank_ok < eff_b {
                        effective_m = j + eff_b + rank_ok;
                        ws.r_block.fill(0.0);
                        break;
                    }
                } else {
                    effective_m = dst_slot;
                    break;
                }
                j += eff_b;
            }

            let mut b_sub = Vec::with_capacity(effective_m * effective_m);
            for r in 0..effective_m {
                b_sub.extend_from_slice(&ws.b_mat[r * m_sub..r * m_sub + effective_m]);
            }
            let (evals, evecs) = symmetric_eigen_jacobi(&b_sub, effective_m, 120, 1e-14)?;

            let gap_factor = if effective_m > k {
                (evals[effective_m - k] - evals[effective_m - k - 1])
                    .abs()
                    .clamp(1e-3, 1.0)
            } else {
                1.0
            };
            let eff_tol = (self.tol * gap_factor).max(1e-13);

            let bot_start = effective_m.saturating_sub(eff_b);
            let mut all_converged = true;
            for p in 0..k {
                let eig_idx = effective_m - 1 - p;
                let mut res_sq = 0.0_f64;
                for r in 0..(effective_m - bot_start) {
                    let mut rho_r = 0.0_f64;
                    for c in r..(effective_m - bot_start) {
                        rho_r += ws.r_block[r * eff_b + c]
                            * evecs[(bot_start + c) * effective_m + eig_idx];
                    }
                    res_sq += rho_r * rho_r;
                }
                let res_norm = res_sq.sqrt();
                let scale = evals[eig_idx].abs().max(1.0);
                if res_norm > eff_tol * scale {
                    all_converged = false;
                    break;
                }
            }

            if all_converged
                || effective_m == n
                || effective_m < k_keep + eff_b
                || iter + 1 == self.max_iters
            {
                final_evals = evals;
                final_evecs = evecs;
                final_m = effective_m;
                break;
            }

            ws.restart_with_ritz(effective_m, k_keep, &evals, &evecs, &mut rng);
            start_j = k_keep;
        }

        let mut lambdas_desc = Vec::with_capacity(k);
        let mut vectors = vec![0.0_f64; n * k];
        for r in 0..k {
            let eig_idx = final_m - 1 - r;
            lambdas_desc.push(final_evals[eig_idx]);
            for p in 0..final_m {
                let coeff = final_evecs[p * final_m + eig_idx];
                let q_p = &ws.q_vecs[p * n..(p + 1) * n];
                for row in 0..n {
                    vectors[row * k + r] += coeff * q_p[row];
                }
            }
            let mut nrm_sq = 0.0_f64;
            for row in 0..n {
                let val = vectors[row * k + r];
                nrm_sq += val * val;
            }
            let nrm = nrm_sq.sqrt();
            if nrm > 1e-14 {
                let inv = 1.0 / nrm;
                for row in 0..n {
                    vectors[row * k + r] *= inv;
                }
            }
        }

        Ok((lambdas_desc, vectors))
    }
}

/// Computes the smallest `n_modes` nontrivial eigenpairs of $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$,
/// skipping the trivial $\lambda_0 \approx 0$ mode cleanly.
///
/// Uses dense [`symmetric_eigen_jacobi`] fallback when `n <= max(48, 2 * actual_modes + 4)`,
/// and SpMV-only block thick-restart [`LanczosEigen`] on the shifted operator
/// $M v = 2.5 v - L_{\mathrm{sym}} v$ for larger graphs.
pub fn smallest_sym_laplacian_eigenpairs<W: Scalar>(
    graph: &CsrGraph<W>,
    n_modes: usize,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<EigenResult> {
    if n_modes == 0 {
        return Err(CoreError::invalid_input("n_modes must be >= 1"));
    }
    if !tol.is_finite() {
        return Err(CoreError::non_finite(
            "smallest_sym_laplacian_eigenpairs::tol",
        ));
    }
    if tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input("max_iters must be >= 1"));
    }

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
        });
    }

    let actual_modes = n_modes.min(n - 1);
    let use_dense = n <= 48usize.max(2 * actual_modes + 4);

    if use_dense {
        let dense = l_sym.to_dense()?;
        let (evals, evecs) = symmetric_eigen_jacobi(&dense, n, 120, tol.min(1e-12))?;
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
        });
    }

    const SHIFT: f64 = 2.5;
    let k_total = actual_modes + 1;
    let lanczos = LanczosEigen::new(tol, max_iters, seed);
    let mut tmp = vec![0.0_f64; n];
    let (mu_desc, mu_vecs) = lanczos.largest_eigenpairs(n, k_total, |x, y| {
        l_sym.spmv(x, &mut tmp)?;
        for i in 0..n {
            y[i] = SHIFT * x[i] - tmp[i];
        }
        Ok(())
    })?;

    let lambdas: Vec<f64> = mu_desc[1..=actual_modes]
        .iter()
        .map(|&mu| (SHIFT - mu).max(0.0))
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
    })
}
