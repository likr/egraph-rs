use egraph_core::{CoreError, CoreResult, DeterministicRng};

use crate::eigen::symmetric_eigen_jacobi;
use crate::operator::LinearOperator;

use super::spectrum::{dot_slice, scaled_norm2};

/// Estimates $\lambda_{\max}(|A|)$ for a symmetric [`LinearOperator`] via deterministic
/// `num_steps`-step Lanczos tridiagonalization with two-pass MGS reorthogonalization
/// and [`symmetric_eigen_jacobi`] on the $k \times k$ tridiagonal matrix, scaled by
/// `safety_factor >= 1.0`.
///
/// Returns `0.0` when `op.dim() == 0` or the operator maps all probes to zero.
pub fn estimate_lambda_max_operator<O: LinearOperator<f64> + ?Sized>(
    op: &mut O,
    num_steps: usize,
    safety_factor: f64,
    seed: u64,
) -> CoreResult<f64> {
    if !safety_factor.is_finite() {
        return Err(CoreError::non_finite(
            "estimate_lambda_max_operator::safety_factor",
        ));
    }
    if safety_factor < 1.0 {
        return Err(CoreError::invalid_input(format!(
            "safety_factor must be >= 1.0, got {safety_factor}"
        )));
    }
    if num_steps == 0 {
        return Err(CoreError::invalid_input("num_steps must be >= 1"));
    }
    let n = op.dim();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        let mut y = [0.0_f64; 1];
        op.apply(&[1.0_f64], &mut y)?;
        if !y[0].is_finite() {
            return Err(CoreError::non_finite("estimate_lambda_max_operator::op"));
        }
        return Ok(safety_factor * y[0].abs());
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
    let mut scale_ref = 0.0_f64;

    for j in 0..k_max {
        let q_j_start = j * n;
        op.apply(&q_vecs[q_j_start..q_j_start + n], &mut w)?;
        let w_raw_norm = scaled_norm2(&w);
        if !w_raw_norm.is_finite() {
            return Err(CoreError::non_finite("estimate_lambda_max_operator::op"));
        }
        scale_ref = scale_ref.max(w_raw_norm);

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
            return Err(CoreError::non_finite("estimate_lambda_max_operator::alpha"));
        }
        alphas.push(alpha_j);

        if j + 1 == k_max {
            break;
        }

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
            return Err(CoreError::non_finite("estimate_lambda_max_operator::beta"));
        }

        if beta_j <= 1e-14 * scale_ref {
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
    Ok(safety_factor * max_eval)
}
