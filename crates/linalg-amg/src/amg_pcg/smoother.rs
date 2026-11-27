use egraph_core::CoreResult;
use egraph_linalg::{CsrMatrix, symmetric_eigen_jacobi};

use super::config::AmgPcgConfig;
use super::{
    JACOBI_COARSEST_MAX_SWEEPS, JACOBI_COARSEST_TOL, SMOOTHER_DIAG_EPS, STALLED_COARSEST_SGS_SWEEPS,
};

#[derive(Debug, Clone)]
pub(super) enum CoarsestSolver {
    DensePinverse { n: usize, pinv: Vec<f64> },
    BoundedSgs { sweeps: usize, inv_diag: Vec<f64> },
}

#[inline]
pub(super) fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
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
pub(super) fn norm2(a: &[f64]) -> f64 {
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
pub(super) fn scaled_dot(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
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

pub(super) fn compute_safe_inv_diag(
    diag: &[f64],
    max_diag: f64,
    zero_mask: Option<&[bool]>,
) -> Vec<f64> {
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

pub(super) fn forward_gs_sweep(a: &CsrMatrix, inv_diag: &[f64], rhs: &[f64], x: &mut [f64]) {
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

pub(super) fn backward_gs_sweep(a: &CsrMatrix, inv_diag: &[f64], rhs: &[f64], x: &mut [f64]) {
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

pub(super) fn build_coarsest_solver(
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

pub(super) fn solve_coarsest(a: &CsrMatrix, solver: &CoarsestSolver, rhs: &[f64], sol: &mut [f64]) {
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
