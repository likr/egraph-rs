use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight};
use egraph_linalg::symmetric_eigen_jacobi;

use super::hde::HdeConfig;

pub(super) fn validate_positive_weights<W: EdgeWeight>(graph: &CsrGraph<W>) -> CoreResult<()> {
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.weight_to_f64();
            if !wf.is_finite() || wf <= 0.0 {
                return Err(CoreError::invalid_input(
                    "HDE edge weights must be finite and strictly positive",
                ));
            }
        }
    }
    Ok(())
}

/// Canonicalizes eigenvector orientation deterministically without depending on near-tie ordering.
fn canonicalize_eigenvector_sign(u: &mut [f64]) {
    let max_abs = u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0
        && let Some(&first_dom) = u.iter().find(|v| v.abs() > 0.5 * max_abs)
        && first_dom < 0.0
    {
        for v in u.iter_mut() {
            *v = -*v;
        }
    }
}

pub(super) trait PivotRowAccess {
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64;
}

impl PivotRowAccess for [u16] {
    #[inline]
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64 {
        let d = self[idx];
        let raw = if d == u16::MAX { fill } else { f64::from(d) };
        raw - mean
    }
}

impl PivotRowAccess for [u32] {
    #[inline]
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64 {
        let d = self[idx];
        let raw = if d == u32::MAX { fill } else { f64::from(d) };
        raw - mean
    }
}

impl PivotRowAccess for [f64] {
    #[inline]
    fn centered_val(&self, idx: usize, _fill: f64, _mean: f64) -> f64 {
        self[idx]
    }
}

pub(super) fn prepare_unweighted_row_stats<T: Copy + PartialEq>(
    row: &[T],
    sentinel: T,
    to_f64: fn(T) -> f64,
    n: usize,
) -> CoreResult<(f64, f64, f64)> {
    let mut max_finite = 0.0_f64;
    let mut has_finite = false;
    let mut has_nonfinite = false;
    for &d in row {
        if d != sentinel {
            has_finite = true;
            let df = to_f64(d);
            if df > max_finite {
                max_finite = df;
            }
        } else {
            has_nonfinite = true;
        }
    }
    let fill = if has_nonfinite {
        let f = if has_finite { max_finite + 1.0 } else { 0.0 };
        if !f.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "HDE finite distance fallback overflowed f64",
            ));
        }
        f
    } else {
        0.0
    };

    let inv_n = 1.0 / (n as f64);
    let mut sum = 0.0_f64;
    for &d in row {
        sum += if d == sentinel { fill } else { to_f64(d) };
    }
    let mut mean = sum * inv_n;
    if !mean.is_finite() {
        mean = row
            .iter()
            .map(|&d| (if d == sentinel { fill } else { to_f64(d) }) * inv_n)
            .sum::<f64>();
        if !mean.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "HDE column mean overflowed f64",
            ));
        }
    }

    let mut max_abs = 0.0_f64;
    for &d in row {
        let val = if d == sentinel { fill } else { to_f64(d) };
        let abs_v = (val - mean).abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }
    Ok((fill, mean, max_abs))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn hde_pca_project<T>(
    dists: &[T],
    n: usize,
    m_eff: usize,
    fills: &[f64],
    means: &[f64],
    max_abs_centered: f64,
    output_dims: usize,
    config: &HdeConfig,
    out_coords: &mut [f64],
) -> CoreResult<()>
where
    [T]: PivotRowAccess,
{
    if !max_abs_centered.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE centered distance overflowed f64",
        ));
    }
    if max_abs_centered == 0.0 {
        return Ok(());
    }

    // Unconditionally normalize centered entries into [-1, 1] via a two-step scale so
    // S = (1/n) X_hat X_hat^T and its Frobenius norm never overflow or underflow across all
    // positive finite f64 scales (including subnormals < 1e-300 and huge weights > 1e300).
    let (pre_scale, cov_scale) = if max_abs_centered < 1e-300 {
        let pre = 1e150_f64;
        (pre, 1.0 / (max_abs_centered * pre))
    } else if max_abs_centered > 1e300 {
        let pre = 1e-150_f64;
        (pre, 1.0 / (max_abs_centered * pre))
    } else {
        (1.0_f64, 1.0 / max_abs_centered)
    };
    if !cov_scale.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE covariance scale overflowed f64",
        ));
    }
    let scale_centered = |val: f64| -> f64 { (val * pre_scale) * cov_scale };

    let mut cov = vec![0.0_f64; m_eff * m_eff];
    let inv_n = 1.0 / (n as f64);
    for a in 0..m_eff {
        let row_a = &dists[a * n..(a + 1) * n];
        let fill_a = fills[a];
        let mean_a = means[a];
        for b in a..m_eff {
            let row_b = &dists[b * n..(b + 1) * n];
            let fill_b = fills[b];
            let mean_b = means[b];
            let mut dot = 0.0_f64;
            for v in 0..n {
                let va = scale_centered(row_a.centered_val(v, fill_a, mean_a));
                let vb = scale_centered(row_b.centered_val(v, fill_b, mean_b));
                dot += va * vb;
            }
            let s_ab = dot * inv_n;
            if !s_ab.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "HDE covariance entry overflowed f64",
                ));
            }
            cov[a * m_eff + b] = s_ab;
            cov[b * m_eff + a] = s_ab;
        }
    }

    let (evals, evecs) = symmetric_eigen_jacobi(&cov, m_eff, config.max_sweeps, config.tol)?;
    let used_dims = output_dims.min(m_eff);
    let lambda_0 = evals[m_eff - 1];
    if !lambda_0.is_finite() || lambda_0 <= 1e-300 {
        return Ok(());
    }

    let mut u_k = vec![0.0_f64; m_eff * used_dims];
    let mut mode_active = vec![false; used_dims];
    let mut col_buf = vec![0.0_f64; m_eff];
    for m in 0..used_dims {
        let eig_idx = m_eff - 1 - m;
        let lambda_m = evals[eig_idx];
        if m >= 1 && (!lambda_m.is_finite() || lambda_m <= 1e-12 * lambda_0) {
            continue;
        }
        for p in 0..m_eff {
            col_buf[p] = evecs[p * m_eff + eig_idx];
        }
        canonicalize_eigenvector_sign(&mut col_buf);
        for p in 0..m_eff {
            u_k[p * used_dims + m] = col_buf[p];
        }
        mode_active[m] = true;
    }

    // When config.normalize is enabled, project using the scaled O(1) matrix so downstream
    // isotropic normalization neither falls back to Gaussian noise on tiny weights (<= 1e-13)
    // nor overflows sum_sq on huge weights (>= 1e155).
    for p in 0..m_eff {
        let row_p = &dists[p * n..(p + 1) * n];
        let fill_p = fills[p];
        let mean_p = means[p];
        let u_row = &u_k[p * used_dims..(p + 1) * used_dims];
        for v in 0..n {
            let raw_x_pv = row_p.centered_val(v, fill_p, mean_p);
            let x_pv = if config.normalize {
                scale_centered(raw_x_pv)
            } else {
                raw_x_pv
            };
            let out_row = &mut out_coords[v * output_dims..v * output_dims + used_dims];
            for m in 0..used_dims {
                if mode_active[m] {
                    out_row[m] += x_pv * u_row[m];
                }
            }
        }
    }
    if !config.normalize && !out_coords.iter().all(|v| v.is_finite()) {
        return Err(CoreError::numerical_breakdown(
            "HDE unnormalized projection overflowed f64",
        ));
    }
    Ok(())
}
