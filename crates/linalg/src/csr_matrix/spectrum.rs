use egraph_core::{CoreError, CoreResult, DeterministicRng};

use crate::eigen::symmetric_eigen_jacobi;

use super::matrix::CsrMatrix;

/// Default number of deterministic power-iteration steps in [`CsrMatrix::estimate_lambda_max`].
pub const DEFAULT_SPECTRUM_POWER_ITERS: usize = 10;
/// Default multiplicative safety margin in [`CsrMatrix::estimate_lambda_max`].
pub const DEFAULT_SPECTRUM_SAFETY_FACTOR: f64 = 1.02;
/// Default number of Lanczos steps in [`CsrMatrix::estimate_lambda_max_lanczos`].
pub const DEFAULT_SPECTRUM_LANCZOS_STEPS: usize = 16;
/// Default multiplicative safety margin in [`CsrMatrix::estimate_lambda_max_lanczos`].
pub const DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR: f64 = 1.05;

impl CsrMatrix {
    /// Returns the symmetric Gershgorin spectral upper bound
    /// $\max_i \left(|A_{ii}| + \sum_{j \neq i} |A_{ij}|\right)$.
    ///
    /// Returns `0.0` for empty (`num_nodes == 0`) or zero matrices.
    pub fn gershgorin_spectral_bound(&self) -> f64 {
        let n = self.num_nodes;
        if n == 0 {
            return 0.0;
        }
        let mut max_bound = 0.0_f64;
        for row in 0..n {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_u32 = row as u32;
            let mut row_abs_sum = 0.0_f64;
            let mut saw_diag = false;
            for pos in start..end {
                let abs_val = self.values[pos].abs();
                if !abs_val.is_finite() {
                    return f64::NAN;
                }
                if self.indices[pos] == row_u32 {
                    saw_diag = true;
                }
                row_abs_sum += abs_val;
            }
            if !saw_diag {
                let d_abs = self.diagonal[row].abs();
                if !d_abs.is_finite() {
                    return f64::NAN;
                }
                row_abs_sum += d_abs;
            }
            if row_abs_sum > max_bound {
                max_bound = row_abs_sum;
            }
        }
        max_bound
    }

    /// Estimates $\lambda_{\max}(A)$ for a symmetric [`CsrMatrix`] using deterministic
    /// Rademacher power iteration clamped by [`CsrMatrix::gershgorin_spectral_bound`],
    /// scaled by `safety_factor >= 1.0`.
    ///
    /// Returns `0.0` for empty or zero matrices, and `gershgorin_spectral_bound()` when `num_iters == 0`.
    pub fn estimate_lambda_max(
        &self,
        num_iters: usize,
        safety_factor: f64,
        seed: u64,
    ) -> CoreResult<f64> {
        if !safety_factor.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max::safety_factor",
            ));
        }
        if safety_factor < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "safety_factor must be >= 1.0, got {safety_factor}"
            )));
        }
        let gersh = self.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max::matrix",
            ));
        }
        let n = self.num_nodes;
        if n == 0 || gersh == 0.0 {
            return Ok(0.0);
        }
        if num_iters == 0 {
            return Ok(gersh);
        }
        if n == 1 {
            return Ok(self.diagonal[0].abs().min(gersh));
        }

        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut v = vec![0.0_f64; n];
        let mut w = vec![0.0_f64; n];

        let mut initialized = false;
        for _attempt in 0..8 {
            for vi in &mut v {
                *vi = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
            }
            let v_norm = scaled_norm2(&v);
            let inv_v = 1.0 / v_norm;
            for vi in &mut v {
                *vi *= inv_v;
            }
            self.spmv(&v, &mut w)?;
            if scaled_norm2(&w) > 0.0 {
                initialized = true;
                break;
            }
        }
        if !initialized {
            return Ok(gersh);
        }

        for _iter in 0..num_iters {
            self.spmv(&v, &mut w)?;
            let w_norm = scaled_norm2(&w);
            if !w_norm.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max::spmv",
                ));
            }
            if w_norm == 0.0 {
                return Ok(gersh);
            }
            let inv_w = 1.0 / w_norm;
            for (vi, &wi) in v.iter_mut().zip(w.iter()) {
                *vi = wi * inv_w;
            }
        }

        self.spmv(&v, &mut w)?;
        let rq = dot_slice(&v, &w);
        if !rq.is_finite() {
            return Err(CoreError::non_finite("CsrMatrix::estimate_lambda_max::rq"));
        }
        Ok((safety_factor * rq.abs()).min(gersh))
    }

    /// Estimates $\lambda_{\max}(A)$ for a symmetric [`CsrMatrix`] via deterministic
    /// `num_steps`-step Lanczos tridiagonalization with two-pass MGS reorthogonalization
    /// and [`symmetric_eigen_jacobi`] on the $k \times k$ tridiagonal matrix, clamped by
    /// [`CsrMatrix::gershgorin_spectral_bound`].
    ///
    /// Avoids power iteration's hub-spectrum underestimation while remaining $O(k \cdot \mathrm{nnz})$.
    pub fn estimate_lambda_max_lanczos(
        &self,
        num_steps: usize,
        safety_factor: f64,
        seed: u64,
    ) -> CoreResult<f64> {
        if !safety_factor.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max_lanczos::safety_factor",
            ));
        }
        if safety_factor < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "safety_factor must be >= 1.0, got {safety_factor}"
            )));
        }
        let gersh = self.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max_lanczos::matrix",
            ));
        }
        let n = self.num_nodes;
        if n == 0 || gersh == 0.0 {
            return Ok(0.0);
        }
        if num_steps == 0 {
            return Ok(gersh);
        }
        if n == 1 {
            return Ok(self.diagonal[0].abs().min(gersh));
        }

        let k_max = num_steps.min(n);
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut q_vecs = vec![0.0_f64; k_max * n];
        for val in &mut q_vecs[..n] {
            *val = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
        }
        let q0_norm = scaled_norm2(&q_vecs[..n]);
        let inv_q0 = 1.0 / q0_norm;
        for val in &mut q_vecs[..n] {
            *val *= inv_q0;
        }

        let mut alphas = Vec::with_capacity(k_max);
        let mut betas = Vec::with_capacity(k_max.saturating_sub(1));
        let mut w = vec![0.0_f64; n];

        for j in 0..k_max {
            let q_j_start = j * n;
            self.spmv(&q_vecs[q_j_start..q_j_start + n], &mut w)?;

            if j > 0 {
                let beta_prev = betas[j - 1];
                if beta_prev != 0.0 {
                    let q_prev_start = (j - 1) * n;
                    for i in 0..n {
                        w[i] -= beta_prev * q_vecs[q_prev_start + i];
                    }
                }
            }

            let alpha_j = dot_slice(&q_vecs[q_j_start..q_j_start + n], &w);
            if !alpha_j.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max_lanczos::alpha",
                ));
            }
            alphas.push(alpha_j);

            if j + 1 == k_max {
                break;
            }

            // Two-pass Modified Gram-Schmidt reorthogonalization against all kept Lanczos vectors.
            for _pass in 0..2 {
                for prev in 0..=j {
                    let q_p = &q_vecs[prev * n..(prev + 1) * n];
                    let proj = dot_slice(q_p, &w);
                    for i in 0..n {
                        w[i] -= proj * q_p[i];
                    }
                }
            }

            let beta_j = scaled_norm2(&w);
            if !beta_j.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max_lanczos::beta",
                ));
            }

            if beta_j <= 1e-14 * gersh {
                // Invariant Krylov subspace reached; restart with an orthogonal random vector.
                let mut found = false;
                for _attempt in 0..8 {
                    let next_start = (j + 1) * n;
                    for val in &mut q_vecs[next_start..next_start + n] {
                        *val = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
                    }
                    for _pass in 0..2 {
                        for prev in 0..=j {
                            let (left, right) = q_vecs.split_at_mut(next_start);
                            let q_p = &left[prev * n..(prev + 1) * n];
                            let q_next = &mut right[..n];
                            let proj = dot_slice(q_p, q_next);
                            for i in 0..n {
                                q_next[i] -= proj * q_p[i];
                            }
                        }
                    }
                    let nrm = scaled_norm2(&q_vecs[next_start..next_start + n]);
                    if nrm > 1e-12 {
                        let inv = 1.0 / nrm;
                        for val in &mut q_vecs[next_start..next_start + n] {
                            *val *= inv;
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                betas.push(0.0);
            } else {
                betas.push(beta_j);
                let inv_beta = 1.0 / beta_j;
                let next_start = (j + 1) * n;
                for i in 0..n {
                    q_vecs[next_start + i] = w[i] * inv_beta;
                }
            }
        }

        let m = alphas.len();
        let mut t_mat = vec![0.0_f64; m * m];
        for i in 0..m {
            t_mat[i * m + i] = alphas[i];
            if i + 1 < m {
                let b = betas[i];
                t_mat[i * m + (i + 1)] = b;
                t_mat[(i + 1) * m + i] = b;
            }
        }
        let (evals, _) = symmetric_eigen_jacobi(&t_mat, m, 64, 1e-12)?;
        let max_eval = evals.iter().fold(0.0_f64, |acc, &e| acc.max(e.abs()));
        Ok((safety_factor * max_eval).min(gersh))
    }
}

#[inline]
pub(super) fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
pub(super) fn scaled_norm2(a: &[f64]) -> f64 {
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
