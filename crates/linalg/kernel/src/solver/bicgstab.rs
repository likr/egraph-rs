//! Batched BiCGSTAB solver implementation for linear systems $(I - \alpha P) Y = B$.

use crate::SparseSymmetricMatrix;

/// Workspace memory buffer reused across iterations during Batched BiCGSTAB solving.
///
/// Pre-allocating these vectors prevents repeated heap allocations during iterative solving steps.
#[derive(Debug, Clone)]
pub(crate) struct BicgstabSolverBuffers {
    pub r: Vec<f64>,
    pub r0_hat: Vec<f64>,
    pub p: Vec<f64>,
    pub v: Vec<f64>,
    pub s: Vec<f64>,
    pub t: Vec<f64>,
    pub temp: Vec<f64>,
}

impl BicgstabSolverBuffers {
    /// Creates pre-allocated zero buffers sized for n x m Row-Major arrays.
    pub fn new(n: usize, m: usize) -> Self {
        let size = n * m;
        Self {
            r: vec![0.0; size],
            r0_hat: vec![0.0; size],
            p: vec![0.0; size],
            v: vec![0.0; size],
            s: vec![0.0; size],
            t: vec![0.0; size],
            temp: vec![0.0; size],
        }
    }
}

/// Computes output matrix `y = P * v` where P = D^{-1} W is Markov transition matrix.
///
/// This application leverages degree normalization and sparse graph edge interactions.
#[allow(clippy::needless_range_loop)]
pub(crate) fn apply_p(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    v: &[f64],
    y: &mut [f64],
    m: usize,
) {
    let n = matrix.dim();
    y.fill(0.0);

    // Diagonal elements W_ii * V_{i,m}
    let diagonal = matrix.diagonal();
    for i in 0..n {
        let d_val = diagonal[i];
        if d_val != 0.0 {
            let v_row = &v[i * m..(i + 1) * m];
            let y_row = &mut y[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] += d_val * v_row[k];
            }
        }
    }

    // Off-diagonal edge interactions W_ij * V_{j,m} and W_ij * V_{i,m}
    for &(i, j, w) in matrix.edges() {
        let abs_w = w.abs();
        let v_i = &v[i * m..(i + 1) * m];
        let v_j = &v[j * m..(j + 1) * m];

        // Safely extract non-overlapping mutable slices for rows i and j
        if i < j {
            let (left, right) = y.split_at_mut(j * m);
            let y_i = &mut left[i * m..(i + 1) * m];
            let y_j = &mut right[0..m];
            for k in 0..m {
                y_i[k] += abs_w * v_j[k];
                y_j[k] += abs_w * v_i[k];
            }
        }
    }

    // Row degree normalization: Y = D^{-1} (W * V)
    for i in 0..n {
        let d = degrees[i];
        if d > 0.0 {
            let inv_d = 1.0 / d;
            let y_row = &mut y[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] *= inv_d;
            }
        }
    }
}

/// Computes `y = (I - alpha * P) * v` (SpMM for coefficient matrix A).
fn apply_a(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    alpha: f64,
    v: &[f64],
    y: &mut [f64],
    temp: &mut [f64],
    m: usize,
) {
    apply_p(matrix, degrees, v, temp, m);
    let size = matrix.dim() * m;
    for i in 0..size {
        y[i] = v[i] - alpha * temp[i];
    }
}

/// Solves (I - alpha * P) Y = B for M right-hand side columns using Batched BiCGSTAB.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn solve_batched_bicgstab(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    alpha: f64,
    b: &[f64],
    y: &mut [f64],
    m: usize,
    tol: f64,
    max_iter: usize,
    buffers: &mut BicgstabSolverBuffers,
) {
    let n = matrix.dim();
    let size = n * m;

    y.fill(0.0);
    buffers.r.copy_from_slice(b);
    buffers.r0_hat.copy_from_slice(b);
    buffers.p.fill(0.0);
    buffers.v.fill(0.0);

    let mut rho_prev: Vec<f64> = vec![1.0; m];
    let mut alpha_vec: Vec<f64> = vec![1.0; m];
    let mut omega_vec: Vec<f64> = vec![1.0; m];

    let b_norm: f64 = b.iter().map(|&val| val * val).sum::<f64>().sqrt();
    if b_norm == 0.0 {
        return;
    }

    for _iter in 0..max_iter {
        // Calculate per-column rho_k = dot(r0_hat_m, r_m)
        let mut rho_curr: Vec<f64> = vec![0.0; m];
        for i in 0..n {
            let r0_row = &buffers.r0_hat[i * m..(i + 1) * m];
            let r_row = &buffers.r[i * m..(i + 1) * m];
            for k in 0..m {
                rho_curr[k] += r0_row[k] * r_row[k];
            }
        }

        // Calculate beta_m = (rho_curr / rho_prev) * (alpha_prev / omega_prev)
        let mut beta: Vec<f64> = vec![0.0; m];
        for k in 0..m {
            if rho_prev[k].abs() > 1e-30 && omega_vec[k].abs() > 1e-30 {
                beta[k] = (rho_curr[k] / rho_prev[k]) * (alpha_vec[k] / omega_vec[k]);
            }
        }

        // Update P: p_m = r_m + beta_m * (p_m - omega_m * v_m)
        for i in 0..n {
            let r_row = &buffers.r[i * m..(i + 1) * m];
            let v_row = &buffers.v[i * m..(i + 1) * m];
            let p_row = &mut buffers.p[i * m..(i + 1) * m];
            for k in 0..m {
                p_row[k] = r_row[k] + beta[k] * (p_row[k] - omega_vec[k] * v_row[k]);
            }
        }

        // Compute V = A * P
        apply_a(
            matrix,
            degrees,
            alpha,
            &buffers.p,
            &mut buffers.v,
            &mut buffers.temp,
            m,
        );

        // Calculate alpha_m = rho_curr / dot(r0_hat_m, v_m)
        for k in 0..m {
            let mut v_dot = 0.0;
            for i in 0..n {
                v_dot += buffers.r0_hat[i * m + k] * buffers.v[i * m + k];
            }
            alpha_vec[k] = if v_dot.abs() > 1e-30 {
                rho_curr[k] / v_dot
            } else {
                0.0
            };
        }

        // Compute S = R - alpha * V
        for i in 0..n {
            let r_row = &buffers.r[i * m..(i + 1) * m];
            let v_row = &buffers.v[i * m..(i + 1) * m];
            let s_row = &mut buffers.s[i * m..(i + 1) * m];
            for k in 0..m {
                s_row[k] = r_row[k] - alpha_vec[k] * v_row[k];
            }
        }

        // Early check for residual convergence on S
        let s_norm: f64 = buffers.s.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if s_norm / b_norm <= tol {
            for i in 0..size {
                y[i] += alpha_vec[i % m] * buffers.p[i];
            }
            break;
        }

        // Compute T = A * S
        apply_a(
            matrix,
            degrees,
            alpha,
            &buffers.s,
            &mut buffers.t,
            &mut buffers.temp,
            m,
        );

        // Calculate omega_m = dot(t_m, s_m) / dot(t_m, t_m)
        for k in 0..m {
            let mut t_t = 0.0;
            let mut t_s = 0.0;
            for i in 0..n {
                let t_val = buffers.t[i * m + k];
                let s_val = buffers.s[i * m + k];
                t_t += t_val * t_val;
                t_s += t_val * s_val;
            }
            omega_vec[k] = if t_t > 1e-30 { t_s / t_t } else { 0.0 };
        }

        // Update solution Y += alpha * P + omega * S
        // and residual R = S - omega * T
        for i in 0..n {
            let p_row = &buffers.p[i * m..(i + 1) * m];
            let s_row = &buffers.s[i * m..(i + 1) * m];
            let t_row = &buffers.t[i * m..(i + 1) * m];
            let y_row = &mut y[i * m..(i + 1) * m];
            let r_row = &mut buffers.r[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] += alpha_vec[k] * p_row[k] + omega_vec[k] * s_row[k];
                r_row[k] = s_row[k] - omega_vec[k] * t_row[k];
            }
        }

        let r_norm: f64 = buffers.r.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if r_norm / b_norm <= tol {
            break;
        }

        rho_prev = rho_curr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bicgstab_identity_like() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);

        let mut degrees = matrix.diagonal().to_vec();
        for &(i, j, w) in matrix.edges() {
            degrees[i] += w;
            degrees[j] += w;
        }

        let alpha = 0.5;
        let m = 2;
        let n = 3;
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut y = vec![0.0; n * m];
        let mut buffers = BicgstabSolverBuffers::new(n, m);

        solve_batched_bicgstab(
            &matrix,
            &degrees,
            alpha,
            &b,
            &mut y,
            m,
            1e-7,
            100,
            &mut buffers,
        );

        // Verify residual ||b - (I - alpha P) y|| is small
        let mut ay = vec![0.0; n * m];
        apply_a(&matrix, &degrees, alpha, &y, &mut ay, &mut buffers.temp, m);
        for i in 0..(n * m) {
            assert!(
                (b[i] - ay[i]).abs() < 1e-5,
                "Residual at index {} too large: expected {}, got {}",
                i,
                b[i],
                ay[i]
            );
        }
    }
}
