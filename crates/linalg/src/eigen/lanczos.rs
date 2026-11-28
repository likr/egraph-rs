use egraph_core::{CoreError, CoreResult, DeterministicRng};

use crate::csr_matrix::estimate_lambda_max_operator;
use crate::operator::{FnOperator, LinearOperator, ShiftedOperator};

use super::jacobi::symmetric_eigen_jacobi;
use super::subspace::{SubspaceEigen, fill_random_unit_orthogonal_buf, norm_slice};

/// Pure-Rust SpMV-only block thick-restart Lanczos eigensolver with two-pass MGS
/// reorthogonalization.
///
/// Uses block width $b = 4$ (when `k >= 2` and `n > 64`) so degenerate eigenvalues of
/// multiplicity up to 4 (e.g., 2D grids and periodic tori) are resolved deterministically
/// without relying on floating-point rounding drift.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

    /// Sets the relative convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum number of thick-restart cycles `max_iters`.
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the optional explicit Krylov subspace size override `subspace_dim`.
    pub fn with_subspace_dim(mut self, subspace_dim: Option<usize>) -> Self {
        self.subspace_dim = subspace_dim;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn validate_params(&self, n: usize, k: usize) -> CoreResult<()> {
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
        Ok(())
    }

    /// Computes the `k` largest eigenpairs of a symmetric [`LinearOperator<f64>`],
    /// returning `(eigenvalues_desc, row_major_vectors)` with `vectors` shape `(op.dim(), k)`.
    pub fn largest_eigenpairs_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        k: usize,
    ) -> CoreResult<(Vec<f64>, Vec<f64>)> {
        let (lambdas, vectors, _) = self.largest_eigenpairs_operator_with_iters(op, k)?;
        Ok((lambdas, vectors))
    }

    /// Computes the `k` smallest eigenpairs of a symmetric [`LinearOperator<f64>`] via the
    /// shifted operator $M v = \sigma v - A v$, returning `(eigenvalues_asc, row_major_vectors)`.
    ///
    /// When `shift` is `Some(s)`, uses $\sigma = s$ directly (requiring $\sigma > \lambda_{\max}(A)$
    /// for optimal convergence). When `shift` is `None`, estimates an upper spectral bound via
    /// [`estimate_lambda_max_operator`].
    pub fn smallest_eigenpairs_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        k: usize,
        shift: Option<f64>,
    ) -> CoreResult<(Vec<f64>, Vec<f64>)> {
        let (lambdas, vectors, _) = self.smallest_eigenpairs_operator_with_iters(op, k, shift)?;
        Ok((lambdas, vectors))
    }

    pub(crate) fn smallest_eigenpairs_operator_with_iters<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        k: usize,
        shift: Option<f64>,
    ) -> CoreResult<(Vec<f64>, Vec<f64>, usize)> {
        let n = op.dim();
        self.validate_params(n, k)?;
        let eff_shift = match shift {
            Some(s) => {
                if !s.is_finite() {
                    return Err(CoreError::non_finite(
                        "LanczosEigen::smallest_eigenpairs_operator::shift",
                    ));
                }
                s
            }
            None => {
                let bound = estimate_lambda_max_operator(op, 16.min(n).max(1), 1.25, self.seed)?;
                if bound == 0.0 { 1.0 } else { bound + 0.5 }
            }
        };
        let mut shifted = ShiftedOperator::new(op, eff_shift);
        let (mu_desc, vectors, iters) =
            self.largest_eigenpairs_operator_with_iters(&mut shifted, k)?;
        let lambdas_asc: Vec<f64> = mu_desc.into_iter().map(|mu| eff_shift - mu).collect();
        Ok((lambdas_asc, vectors, iters))
    }

    /// Computes the `k` largest eigenpairs of an $n \times n$ symmetric linear operator `matvec`
    /// returning `(eigenvalues_desc, row_major_vectors)` with `vectors` shape `(n, k)`.
    pub fn largest_eigenpairs<F>(
        &self,
        n: usize,
        k: usize,
        matvec: F,
    ) -> CoreResult<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut(&[f64], &mut [f64]) -> CoreResult<()>,
    {
        let mut op = FnOperator::new(n, matvec);
        self.largest_eigenpairs_operator(&mut op, k)
    }

    pub(crate) fn largest_eigenpairs_operator_with_iters<O: LinearOperator<f64> + ?Sized>(
        &self,
        op: &mut O,
        k: usize,
    ) -> CoreResult<(Vec<f64>, Vec<f64>, usize)> {
        let n = op.dim();
        self.validate_params(n, k)?;

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
        let mut executed_iters = 0usize;
        let mut w_orig_norms = vec![0.0_f64; eff_b];

        for iter in 0..self.max_iters {
            let mut j = start_j;
            let mut effective_m = m_sub;

            while j + eff_b <= m_sub {
                for c in 0..eff_b {
                    let q_jc = &ws.q_vecs[(j + c) * n..(j + c + 1) * n];
                    let w_dst = &mut ws.w_block[c * n..(c + 1) * n];
                    op.apply(q_jc, w_dst)?;
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
                executed_iters = iter + 1;
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

        Ok((lambdas_desc, vectors, executed_iters))
    }
}
