//! Zero-allocation Chebyshev spectral filtering, Miller Bessel $e^{-t}I_k(t)$ evaluation,
//! DCT-II spectral square root, R1/R2 regularized DoG terms, and JL embeddings.

use std::f64::consts::PI;

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use crate::csr_matrix::{
    CsrMatrix, SpectralMomentStats, spectral_moment_stats, symmetric_normalized_laplacian,
};
use crate::shortest_path::select_maxmin_sp_pivots;

/// Computes the adaptive Chebyshev polynomial degree $M = \max(40, \lceil 3.5 t_{\max} + 30 \rceil)$.
#[inline]
pub fn adaptive_chebyshev_degree(max_t: f64) -> usize {
    if !max_t.is_finite() || max_t <= 0.0 {
        return 40;
    }
    let deg = (3.5 * max_t + 30.0).ceil();
    if !deg.is_finite() || deg >= (usize::MAX as f64) {
        return 40;
    }
    40usize.max(deg as usize)
}

/// Computes the adaptive Chebyshev polynomial degree scaled for spectrum $[0, \lambda_{\max}]$,
/// matching [`adaptive_chebyshev_degree`] bit-for-bit when `lambda_max == 2.0`.
#[inline]
pub fn adaptive_chebyshev_degree_scaled(max_t: f64, lambda_max: f64) -> usize {
    if !max_t.is_finite() || !lambda_max.is_finite() || max_t <= 0.0 || lambda_max <= 0.0 {
        return 40;
    }
    adaptive_chebyshev_degree(max_t * (0.5 * lambda_max))
}

/// Evaluates exponentially scaled modified Bessel functions $e^{-t} I_k(t)$ for $k = 0 \dots \mathrm{max\_k}$
/// in `f64` using Miller's backward recurrence with periodic rescaling above `1e100`.
///
/// For tiny $t \in (0, 10^{-8})$, evaluates the exact ascending power series to avoid
/// $2k/t$ overflow in the backward recurrence.
pub fn exp_scaled_bessel_i_all(max_k: usize, t: f64) -> CoreResult<Vec<f64>> {
    if !t.is_finite() {
        return Err(CoreError::non_finite("exp_scaled_bessel_i_all::t"));
    }
    if t < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "Bessel argument t must be >= 0, got {t}"
        )));
    }

    let mut out = vec![0.0_f64; max_k + 1];
    if t == 0.0 {
        out[0] = 1.0;
        return Ok(out);
    }

    if t < 1e-8 {
        let half_t = 0.5 * t;
        let x2 = half_t * half_t;
        let mut prefix = (-t).exp();
        for (k, val) in out.iter_mut().enumerate() {
            if k > 0 {
                prefix *= half_t / (k as f64);
                if prefix == 0.0 {
                    break;
                }
            }
            let kp1 = (k + 1) as f64;
            let kp2 = (k + 2) as f64;
            let series = 1.0 + x2 / kp1 + (x2 * x2) / (2.0 * kp1 * kp2);
            *val = prefix * series;
        }
        return Ok(out);
    }

    // Exact 2^-332 (~1.143e-100) avoids mantissa rounding during periodic rescaling.
    const RESCALE_THRESHOLD: f64 = 1e100;
    const RESCALE_FACTOR: f64 = f64::from_bits((1023_u64 - 332) << 52);

    let extra = (60.0 + 12.0 * t.sqrt() + 2.5 * (max_k as f64).sqrt()).ceil() as usize;
    let t_ceil = if t >= (usize::MAX as f64) {
        return Err(CoreError::invalid_input(
            "Bessel argument t exceeds maximum supported range",
        ));
    } else {
        t.ceil() as usize
    };
    let m_start = max_k.max(t_ceil) + extra;

    let mut i_next = 0.0_f64;
    let mut i_curr = 1.0_f64;
    let mut sum = 2.0_f64;

    for k in (1..=m_start).rev() {
        let i_prev = (2.0 * (k as f64) / t) * i_curr + i_next;
        i_next = i_curr;
        i_curr = i_prev;

        let idx = k - 1;
        if idx > 0 {
            sum += 2.0 * i_curr;
        } else {
            sum += i_curr;
        }
        if idx <= max_k {
            out[idx] = i_curr;
        }

        if i_curr > RESCALE_THRESHOLD {
            i_curr *= RESCALE_FACTOR;
            i_next *= RESCALE_FACTOR;
            sum *= RESCALE_FACTOR;
            if idx <= max_k {
                for v in &mut out[idx..=max_k] {
                    *v *= RESCALE_FACTOR;
                }
            }
        }
    }

    let inv_sum = 1.0 / sum;
    for v in &mut out {
        *v *= inv_sum;
    }
    Ok(out)
}

/// Evaluates a single exponentially scaled modified Bessel value $e^{-t} I_k(t)$.
pub fn exp_scaled_bessel_i(k: usize, t: f64) -> CoreResult<f64> {
    let all = exp_scaled_bessel_i_all(k, t)?;
    Ok(all[k])
}

fn validate_terms_and_lmax(terms: &[(f64, f64)], lambda_max: f64) -> CoreResult<()> {
    if !lambda_max.is_finite() {
        return Err(CoreError::non_finite("chebyshev::lambda_max"));
    }
    if lambda_max <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "lambda_max must be > 0, got {lambda_max}"
        )));
    }
    if terms.is_empty() {
        return Err(CoreError::invalid_input("filter terms must not be empty"));
    }
    for &(w, t) in terms {
        if !w.is_finite() || !t.is_finite() {
            return Err(CoreError::non_finite("chebyshev::terms"));
        }
        if t < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "diffusion time t must be >= 0, got {t}"
            )));
        }
    }
    Ok(())
}

/// Computes Chebyshev expansion coefficients for
/// $h(\lambda) = \sum_m w_m \exp(-t_m \lambda) \approx \sum_{k=0}^{\mathrm{degree}} c_k T_k(\frac{2}{\lambda_{\max}}\lambda - 1)$
/// over $\lambda \in [0, \lambda_{\max}]$, handling $t_m = 0.0$ exactly.
pub fn compute_chebyshev_bandpass_coefficients(
    terms: &[(f64, f64)],
    lambda_max: f64,
    degree: usize,
) -> CoreResult<Vec<f64>> {
    validate_terms_and_lmax(terms, lambda_max)?;

    let mut coeffs = vec![0.0_f64; degree + 1];
    let half_lmax = 0.5 * lambda_max;

    for &(w, t) in terms {
        if w == 0.0 {
            continue;
        }
        if t == 0.0 {
            coeffs[0] += w;
            continue;
        }
        let tau = t * half_lmax;
        let bessel = exp_scaled_bessel_i_all(degree, tau)?;
        coeffs[0] += w * bessel[0];
        for k in 1..=degree {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            coeffs[k] += 2.0 * w * sign * bessel[k];
        }
    }

    Ok(coeffs)
}

/// Computes Chebyshev expansion coefficients for $g(\lambda) = \sqrt{\max(h(\lambda), 0)}$
/// on $[0, \lambda_{\max}]$ via pure-Rust `f64` DCT-II at `n_quad = max(256, 4*(degree+1))`
/// Chebyshev-Gauss nodes, rejecting non-PSD filters with $\min h < -10^{-7}$.
pub fn compute_chebyshev_sqrt_filter_coefficients(
    terms: &[(f64, f64)],
    lambda_max: f64,
    degree: usize,
) -> CoreResult<Vec<f64>> {
    validate_terms_and_lmax(terms, lambda_max)?;

    let n_quad = 256usize.max(4 * (degree + 1));
    let n_quad_f = n_quad as f64;
    let half_lmax = 0.5 * lambda_max;

    let mut h_end_0 = 0.0_f64;
    let mut h_end_max = 0.0_f64;
    for &(w, t) in terms {
        if w == 0.0 {
            continue;
        }
        if t == 0.0 {
            h_end_0 += w;
            h_end_max += w;
        } else {
            h_end_0 += w;
            h_end_max += w * (-t * lambda_max).exp();
        }
    }
    let mut min_h = h_end_0.min(h_end_max);

    let mut coeffs = vec![0.0_f64; degree + 1];
    for j in 0..n_quad {
        let theta = PI * ((j as f64) + 0.5) / n_quad_f;
        let (s1, c1) = theta.sin_cos();
        let lambda = half_lmax * (c1 + 1.0);

        let mut h_val = 0.0_f64;
        for &(w, t) in terms {
            if w == 0.0 {
                continue;
            }
            if t == 0.0 {
                h_val += w;
            } else {
                h_val += w * (-t * lambda).exp();
            }
        }
        if h_val < min_h {
            min_h = h_val;
        }

        let g_val = h_val.max(0.0).sqrt();
        coeffs[0] += g_val;
        if degree >= 1 {
            // Unitary angle-addition rotation evaluates cos(k * theta) stably without repeated trig calls.
            let mut c = c1;
            let mut s = s1;
            coeffs[1] += g_val * c;
            for coeff in coeffs.iter_mut().take(degree + 1).skip(2) {
                let c_next = c * c1 - s * s1;
                let s_next = s * c1 + c * s1;
                c = c_next;
                s = s_next;
                *coeff += g_val * c;
            }
        }
    }

    if min_h < -1e-7 {
        return Err(CoreError::invalid_input(format!(
            "filter h(lambda) must be PSD on [0, {lambda_max}] for sqrt(h) expansion, got min(h)={min_h:.3e}"
        )));
    }

    coeffs[0] /= n_quad_f;
    let scale = 2.0 / n_quad_f;
    for c in &mut coeffs[1..] {
        *c *= scale;
    }
    Ok(coeffs)
}

/// Builds 2-term Difference-of-Gaussians (DoG) filter terms:
/// `[(1 - beta_floor, t1), (-alpha * (1 - beta_floor), t2)]`.
pub fn build_dog_terms(
    t1: f64,
    t2: f64,
    alpha: f64,
    beta_floor: f64,
) -> CoreResult<Vec<(f64, f64)>> {
    if !t1.is_finite() || !t2.is_finite() || !alpha.is_finite() || !beta_floor.is_finite() {
        return Err(CoreError::non_finite("build_dog_terms"));
    }
    if t1 <= 0.0 || t2 <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "DoG diffusion times t1 and t2 must be > 0, got t1={t1}, t2={t2}"
        )));
    }
    if alpha < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "DoG attenuation alpha must be >= 0, got alpha={alpha}"
        )));
    }
    if !(0.0..1.0).contains(&beta_floor) {
        return Err(CoreError::invalid_input(format!(
            "DoG beta_floor must satisfy 0.0 <= beta_floor < 1.0, got beta_floor={beta_floor}"
        )));
    }
    let scale = 1.0 - beta_floor;
    Ok(vec![(scale, t1), (-alpha * scale, t2)])
}

/// Builds parameter-free R1 regularized DoG terms from $\sigma_\lambda$:
/// $t_1 = \frac{1}{1 + \sigma_\lambda}$, $t_2 = \frac{4}{1 + \sigma_\lambda}$, $\alpha = \max(0, 1 - \sigma_\lambda)$.
pub fn build_regularized_dog_terms(
    sigma_lambda: f64,
    beta_floor: f64,
) -> CoreResult<Vec<(f64, f64)>> {
    if !sigma_lambda.is_finite() {
        return Err(CoreError::non_finite(
            "build_regularized_dog_terms::sigma_lambda",
        ));
    }
    if sigma_lambda < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "sigma_lambda must be >= 0, got {sigma_lambda}"
        )));
    }
    let t1 = 1.0 / (1.0 + sigma_lambda);
    let t2 = 4.0 / (1.0 + sigma_lambda);
    let alpha = (1.0 - sigma_lambda).max(0.0);
    build_dog_terms(t1, t2, alpha, beta_floor)
}

#[inline]
fn eval_chebyshev_poly_scalar(coeffs: &[f64], lambda: f64, lambda_max: f64) -> f64 {
    let x = (2.0 / lambda_max) * lambda - 1.0;
    let mut t_prev = 1.0;
    let mut acc = coeffs[0] * t_prev;
    if coeffs.len() > 1 {
        let mut t_curr = x;
        acc += coeffs[1] * t_curr;
        for &c in &coeffs[2..] {
            let t_next = 2.0 * x * t_curr - t_prev;
            acc += c * t_next;
            t_prev = t_curr;
            t_curr = t_next;
        }
    }
    acc
}

/// Projects a spectral filter $h(\lambda) = \sum_m w_m \exp(-t_m \lambda)$ onto the PSD cone
/// $h_+(\lambda) = (h(\lambda) + \delta_{\mathrm{psd}}) / (1 + \delta_{\mathrm{psd}}) \ge \epsilon_{\mathrm{psd}}$
/// across $\{0, \lambda_{\max}\}$ and `max(256, 4*(degree+1))` Chebyshev-Gauss quadrature nodes,
/// guarding both $h(\lambda)$ and its degree-`degree` Chebyshev truncation against undershoot.
///
/// Returns `(terms_psd, delta_psd, h_min)`.
#[allow(clippy::type_complexity)]
pub fn project_psd_spectral_filter(
    terms: &[(f64, f64)],
    lambda_max: f64,
    degree: usize,
    eps_psd: f64,
) -> CoreResult<(Vec<(f64, f64)>, f64, f64)> {
    validate_terms_and_lmax(terms, lambda_max)?;
    if !eps_psd.is_finite() {
        return Err(CoreError::non_finite(
            "project_psd_spectral_filter::eps_psd",
        ));
    }
    if eps_psd < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_psd must be >= 0, got {eps_psd}"
        )));
    }

    let coeffs_h = compute_chebyshev_bandpass_coefficients(terms, lambda_max, degree)?;
    let n_quad = 256usize.max(4 * (degree + 1));
    let n_quad_f = n_quad as f64;
    let half_lmax = 0.5 * lambda_max;

    let eval_h = |lam: f64| -> f64 {
        let mut s = 0.0_f64;
        for &(w, t) in terms {
            if w == 0.0 {
                continue;
            }
            if t == 0.0 {
                s += w;
            } else {
                s += w * (-t * lam).exp();
            }
        }
        s
    };

    let mut h_min = eval_h(0.0).min(eval_h(lambda_max));
    let mut p_min = eval_chebyshev_poly_scalar(&coeffs_h, 0.0, lambda_max).min(
        eval_chebyshev_poly_scalar(&coeffs_h, lambda_max, lambda_max),
    );

    for j in 0..n_quad {
        let theta = PI * ((j as f64) + 0.5) / n_quad_f;
        let lam = half_lmax * (theta.cos() + 1.0);
        let h_val = eval_h(lam);
        let p_val = eval_chebyshev_poly_scalar(&coeffs_h, lam, lambda_max);
        if h_val < h_min {
            h_min = h_val;
        }
        if p_val < p_min {
            p_min = p_val;
        }
    }

    let eff_min = h_min.min(p_min);
    if eff_min >= eps_psd {
        return Ok((terms.to_vec(), 0.0, h_min));
    }

    let delta_psd = eps_psd - eff_min;
    let inv_norm = 1.0 / (1.0 + delta_psd);
    let mut terms_psd = Vec::with_capacity(terms.len() + 1);
    terms_psd.push((delta_psd * inv_norm, 0.0));
    for &(w, t) in terms {
        if t == 0.0 {
            terms_psd[0].0 += w * inv_norm;
        } else {
            terms_psd.push((w * inv_norm, t));
        }
    }
    Ok((terms_psd, delta_psd, h_min))
}

const CHEBYSHEV_TILE_COLS: usize = 16;
const CHEBYSHEV_TAIL_L1_TOL: f64 = 1e-12;

/// Returns the shortest non-empty prefix length `len >= 1` of `coeffs` whose dropped tail
/// satisfies $\sum_{k=\mathrm{len}}^{d} |c_k| \le \mathrm{l1\_tail\_tol}$.
///
/// Since $|T_k(\tilde{\lambda})| \le 1$ on $[-1, 1]$, this bounds the dropped spectral
/// operator norm error by at most `l1_tail_tol`.
#[inline]
pub fn effective_chebyshev_len(coeffs: &[f64], l1_tail_tol: f64) -> usize {
    if coeffs.is_empty() {
        return 0;
    }
    if !l1_tail_tol.is_finite() || l1_tail_tol < 0.0 {
        return coeffs.len();
    }
    let mut tail_l1 = 0.0_f64;
    let mut keep = coeffs.len();
    while keep > 1 {
        let next_tail = tail_l1 + coeffs[keep - 1].abs();
        if next_tail <= l1_tail_tol {
            tail_l1 = next_tail;
            keep -= 1;
        } else {
            break;
        }
    }
    keep
}

fn validate_csr_indices_in_bounds(mat: &CsrMatrix) -> CoreResult<()> {
    let n = mat.num_nodes();
    for &col in mat.indices() {
        if (col as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                col as usize,
                n,
                "ChebyshevWorkspace::csr_indices",
            ));
        }
    }
    Ok(())
}

/// Zero-allocation-per-call tiled workspace (`t_prev`, `t_curr`, `t_next`, and L2-resident
/// 16-column cache-line tile accumulators) for fused Chebyshev SpMV / SpMM recurrences and JL embeddings.
#[derive(Debug, Clone, Default)]
pub struct ChebyshevWorkspace {
    t_prev: Vec<f64>,
    t_curr: Vec<f64>,
    t_next: Vec<f64>,
    out_a_tile: Vec<f64>,
    out_b_tile: Vec<f64>,
}

impl ChebyshevWorkspace {
    /// Creates an empty [`ChebyshevWorkspace`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`ChebyshevWorkspace`] pre-allocated for `capacity` scalar entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            t_prev: vec![0.0; capacity],
            t_curr: vec![0.0; capacity],
            t_next: vec![0.0; capacity],
            out_a_tile: Vec::new(),
            out_b_tile: Vec::new(),
        }
    }

    #[inline]
    fn ensure_len(&mut self, len: usize) {
        if self.t_prev.len() < len {
            self.t_prev.resize(len, 0.0);
        }
        if self.t_curr.len() < len {
            self.t_curr.resize(len, 0.0);
        }
        if self.t_next.len() < len {
            self.t_next.resize(len, 0.0);
        }
    }

    #[inline]
    fn ensure_tile_out(&mut self, len: usize, need_b: bool) {
        if self.out_a_tile.len() < len {
            self.out_a_tile.resize(len, 0.0);
        }
        if need_b && self.out_b_tile.len() < len {
            self.out_b_tile.resize(len, 0.0);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_step_const<const W: usize>(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        out_a: Option<&mut [f64]>,
        cb: f64,
        out_b: Option<&mut [f64]>,
        is_k1: bool,
    ) -> CoreResult<()> {
        let n = mat.num_nodes();
        let len = n
            .checked_mul(W)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace tile size overflow"))?;
        if curr.len() < len || next.len() < len || (!is_k1 && prev.len() < len) {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_const",
            ));
        }

        let (curr_rows, curr_rem) = curr[..len].as_chunks::<W>();
        let (next_rows, next_rem) = next[..len].as_chunks_mut::<W>();
        if curr_rows.len() != n
            || !curr_rem.is_empty()
            || next_rows.len() != n
            || !next_rem.is_empty()
        {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_const::chunks",
            ));
        }

        let prev_rows: &[[f64; W]] = if is_k1 {
            &[]
        } else {
            let (rows, rem) = prev[..len].as_chunks::<W>();
            if rows.len() != n || !rem.is_empty() {
                return Err(CoreError::dimension_mismatch(
                    len,
                    prev.len(),
                    "ChebyshevWorkspace::fused_step_const::prev",
                ));
            }
            rows
        };

        let mut oa_rows = match out_a {
            Some(oa) => {
                if oa.len() < len {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        oa.len(),
                        "ChebyshevWorkspace::fused_step_const::out_a",
                    ));
                }
                let (rows, rem) = oa[..len].as_chunks_mut::<W>();
                if rows.len() != n || !rem.is_empty() {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        oa.len(),
                        "ChebyshevWorkspace::fused_step_const::out_a",
                    ));
                }
                Some(rows)
            }
            None => None,
        };

        let mut ob_rows = match out_b {
            Some(ob) => {
                if ob.len() < len {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        ob.len(),
                        "ChebyshevWorkspace::fused_step_const::out_b",
                    ));
                }
                let (rows, rem) = ob[..len].as_chunks_mut::<W>();
                if rows.len() != n || !rem.is_empty() {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        ob.len(),
                        "ChebyshevWorkspace::fused_step_const::out_b",
                    ));
                }
                Some(rows)
            }
            None => None,
        };

        let indptr = mat.indptr();
        let indices = mat.indices();
        let values = mat.values();

        for (row, next_row) in next_rows.iter_mut().enumerate() {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut acc = [0.0_f64; W];
            for (&col_u32, &mval) in indices[start..end].iter().zip(&values[start..end]) {
                let x_row = &curr_rows[col_u32 as usize];
                for j in 0..W {
                    acc[j] += mval * x_row[j];
                }
            }

            let mut nxt = [0.0_f64; W];
            if let Some(alpha) = unshifted_scale {
                let curr_row = &curr_rows[row];
                if is_k1 {
                    for j in 0..W {
                        nxt[j] = alpha * acc[j] - curr_row[j];
                    }
                } else {
                    let prev_row = &prev_rows[row];
                    for j in 0..W {
                        nxt[j] = 2.0 * (alpha * acc[j] - curr_row[j]) - prev_row[j];
                    }
                }
            } else if is_k1 {
                nxt = acc;
            } else {
                let prev_row = &prev_rows[row];
                for j in 0..W {
                    nxt[j] = 2.0 * acc[j] - prev_row[j];
                }
            }

            *next_row = nxt;
            if let Some(oa) = oa_rows.as_deref_mut() {
                let row_oa = &mut oa[row];
                for j in 0..W {
                    row_oa[j] += ca * nxt[j];
                }
            }
            if let Some(ob) = ob_rows.as_deref_mut() {
                let row_ob = &mut ob[row];
                for j in 0..W {
                    row_ob[j] += cb * nxt[j];
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_step_tail(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        mut out_a: Option<&mut [f64]>,
        cb: f64,
        mut out_b: Option<&mut [f64]>,
        w: usize,
        is_k1: bool,
    ) -> CoreResult<()> {
        if w == 0 || w > CHEBYSHEV_TILE_COLS {
            return Err(CoreError::invalid_input(format!(
                "ChebyshevWorkspace tile width must be in 1..={CHEBYSHEV_TILE_COLS}, got {w}"
            )));
        }
        let n = mat.num_nodes();
        let len = n
            .checked_mul(w)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace tile size overflow"))?;
        if curr.len() < len || next.len() < len || (!is_k1 && prev.len() < len) {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_tail",
            ));
        }
        if let Some(oa) = out_a.as_deref()
            && oa.len() < len
        {
            return Err(CoreError::dimension_mismatch(
                len,
                oa.len(),
                "ChebyshevWorkspace::fused_step_tail::out_a",
            ));
        }
        if let Some(ob) = out_b.as_deref()
            && ob.len() < len
        {
            return Err(CoreError::dimension_mismatch(
                len,
                ob.len(),
                "ChebyshevWorkspace::fused_step_tail::out_b",
            ));
        }

        let indptr = mat.indptr();
        let indices = mat.indices();
        let values = mat.values();
        let curr = &curr[..len];
        let next = &mut next[..len];

        for row in 0..n {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut acc = [0.0_f64; CHEBYSHEV_TILE_COLS];
            for (&col_u32, &mval) in indices[start..end].iter().zip(&values[start..end]) {
                let x_off = (col_u32 as usize) * w;
                let x_row = &curr[x_off..x_off + w];
                for j in 0..w {
                    acc[j] += mval * x_row[j];
                }
            }

            let row_off = row * w;
            let mut nxt = [0.0_f64; CHEBYSHEV_TILE_COLS];
            if let Some(alpha) = unshifted_scale {
                let curr_row = &curr[row_off..row_off + w];
                if is_k1 {
                    for j in 0..w {
                        nxt[j] = alpha * acc[j] - curr_row[j];
                    }
                } else {
                    let prev_row = &prev[row_off..row_off + w];
                    for j in 0..w {
                        nxt[j] = 2.0 * (alpha * acc[j] - curr_row[j]) - prev_row[j];
                    }
                }
            } else if is_k1 {
                nxt[..w].copy_from_slice(&acc[..w]);
            } else {
                let prev_row = &prev[row_off..row_off + w];
                for j in 0..w {
                    nxt[j] = 2.0 * acc[j] - prev_row[j];
                }
            }

            next[row_off..row_off + w].copy_from_slice(&nxt[..w]);
            if let Some(oa) = out_a.as_deref_mut() {
                let row_oa = &mut oa[row_off..row_off + w];
                for j in 0..w {
                    row_oa[j] += ca * nxt[j];
                }
            }
            if let Some(ob) = out_b.as_deref_mut() {
                let row_ob = &mut ob[row_off..row_off + w];
                for j in 0..w {
                    row_ob[j] += cb * nxt[j];
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_fused_step(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        out_a: Option<&mut [f64]>,
        cb: f64,
        out_b: Option<&mut [f64]>,
        w: usize,
        is_k1: bool,
    ) -> CoreResult<()> {
        match w {
            16 => Self::fused_step_const::<16>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            8 => Self::fused_step_const::<8>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            4 => Self::fused_step_const::<4>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            1 => Self::fused_step_const::<1>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            _ => Self::fused_step_tail(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                w,
                is_k1,
            ),
        }
    }

    fn run_tile_degrees(
        &mut self,
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        coeffs_a: &[f64],
        coeffs_b: Option<&[f64]>,
        w: usize,
    ) -> CoreResult<()> {
        let n = mat.num_nodes();
        let tile_len = n * w;
        let max_len = coeffs_a.len().max(coeffs_b.map_or(0, |b| b.len()));
        if max_len <= 1 || n == 0 || w == 0 {
            return Ok(());
        }

        let ca1 = coeffs_a.get(1).copied().unwrap_or(0.0);
        let cb1 = coeffs_b.and_then(|cb| cb.get(1).copied()).unwrap_or(0.0);
        Self::dispatch_fused_step(
            mat,
            unshifted_scale,
            &[],
            &self.t_prev[..tile_len],
            &mut self.t_curr[..tile_len],
            ca1,
            if ca1 != 0.0 {
                Some(&mut self.out_a_tile[..tile_len])
            } else {
                None
            },
            cb1,
            if coeffs_b.is_some() && cb1 != 0.0 {
                Some(&mut self.out_b_tile[..tile_len])
            } else {
                None
            },
            w,
            true,
        )?;

        for k in 2..max_len {
            let ca = coeffs_a.get(k).copied().unwrap_or(0.0);
            let cb = coeffs_b.and_then(|b| b.get(k).copied()).unwrap_or(0.0);
            Self::dispatch_fused_step(
                mat,
                unshifted_scale,
                &self.t_prev[..tile_len],
                &self.t_curr[..tile_len],
                &mut self.t_next[..tile_len],
                ca,
                if ca != 0.0 {
                    Some(&mut self.out_a_tile[..tile_len])
                } else {
                    None
                },
                cb,
                if coeffs_b.is_some() && cb != 0.0 {
                    Some(&mut self.out_b_tile[..tile_len])
                } else {
                    None
                },
                w,
                false,
            )?;
            std::mem::swap(&mut self.t_prev, &mut self.t_curr);
            std::mem::swap(&mut self.t_curr, &mut self.t_next);
        }
        Ok(())
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\tilde{L}) v$ into `out` using a pre-shifted matrix
    /// $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$.
    pub fn apply_vec(
        &mut self,
        l_tilde: &CsrMatrix,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> CoreResult<()> {
        self.apply_batch(l_tilde, coeffs, v, out, 1)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\tilde{L}) V$ into `out` for row-major `(n, ncols)`
    /// matrices using a pre-shifted matrix $\tilde{L}$.
    pub fn apply_batch(
        &mut self,
        l_tilde: &CsrMatrix,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
    ) -> CoreResult<()> {
        self.apply_batch_dual_impl(l_tilde, None, coeffs, None, v, out, None, ncols)
    }

    /// Evaluates two Chebyshev polynomials sharing the same pre-shifted matrix $\tilde{L}$ and
    /// input matrix `v` in a single L2-cache-resident 8-column-tiled recurrence pass:
    /// `out_a = sum_k coeffs_a[k] T_k(L_tilde) V` and optionally `out_b = sum_k coeffs_b[k] T_k(L_tilde) V`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_batch_dual(
        &mut self,
        l_tilde: &CsrMatrix,
        coeffs_a: &[f64],
        coeffs_b: Option<&[f64]>,
        v: &[f64],
        out_a: &mut [f64],
        out_b: Option<&mut [f64]>,
        ncols: usize,
    ) -> CoreResult<()> {
        self.apply_batch_dual_impl(l_tilde, None, coeffs_a, coeffs_b, v, out_a, out_b, ncols)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\frac{2}{\lambda_{\max}} L - I) v$ directly from `L`
    /// without constructing an auxiliary shifted matrix.
    pub fn apply_vec_with_lambda_max(
        &mut self,
        l_sym: &CsrMatrix,
        lambda_max: f64,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> CoreResult<()> {
        self.apply_batch_with_lambda_max(l_sym, lambda_max, coeffs, v, out, 1)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\frac{2}{\lambda_{\max}} L - I) V$ directly from `L`
    /// for row-major `(n, ncols)` matrices without constructing an auxiliary shifted matrix.
    pub fn apply_batch_with_lambda_max(
        &mut self,
        l_sym: &CsrMatrix,
        lambda_max: f64,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
    ) -> CoreResult<()> {
        if !lambda_max.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::apply_batch_with_lambda_max::lambda_max",
            ));
        }
        if lambda_max <= 0.0 {
            return Err(CoreError::invalid_input(
                "lambda_max must be strictly positive",
            ));
        }
        self.apply_batch_dual_impl(
            l_sym,
            Some(2.0 / lambda_max),
            coeffs,
            None,
            v,
            out,
            None,
            ncols,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_batch_dual_impl(
        &mut self,
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        coeffs_a: &[f64],
        coeffs_b: Option<&[f64]>,
        v: &[f64],
        out_a: &mut [f64],
        mut out_b: Option<&mut [f64]>,
        ncols: usize,
    ) -> CoreResult<()> {
        if coeffs_a.is_empty() {
            return Err(CoreError::invalid_input(
                "Chebyshev coefficients must not be empty",
            ));
        }
        match (coeffs_b, out_b.as_deref()) {
            (Some(cb), Some(_)) => {
                if cb.is_empty() {
                    return Err(CoreError::invalid_input(
                        "Chebyshev secondary coefficients must not be empty",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(CoreError::invalid_input(
                    "coeffs_b and out_b must both be Some or both be None",
                ));
            }
        }

        let n = mat.num_nodes();
        let total = n
            .checked_mul(ncols)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace batch size overflow"))?;
        if v.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                v.len(),
                "ChebyshevWorkspace::v",
            ));
        }
        if out_a.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                out_a.len(),
                "ChebyshevWorkspace::out_a",
            ));
        }
        if let Some(ob) = out_b.as_deref()
            && ob.len() != total
        {
            return Err(CoreError::dimension_mismatch(
                total,
                ob.len(),
                "ChebyshevWorkspace::out_b",
            ));
        }
        if total == 0 {
            return Ok(());
        }
        validate_csr_indices_in_bounds(mat)?;

        let eff_a = effective_chebyshev_len(coeffs_a, CHEBYSHEV_TAIL_L1_TOL);
        let coeffs_a = &coeffs_a[..eff_a];
        let coeffs_b = coeffs_b.map(|cb| &cb[..effective_chebyshev_len(cb, CHEBYSHEV_TAIL_L1_TOL)]);

        let max_len = coeffs_a.len().max(coeffs_b.map_or(0, |b| b.len()));
        let ca0 = coeffs_a[0];
        let cb0_opt = coeffs_b.map(|cb| cb[0]);

        if ncols == 1 {
            self.ensure_len(n);
            self.t_prev[..n].copy_from_slice(v);
            for i in 0..n {
                out_a[i] = ca0 * v[i];
            }
            if let (Some(cb0), Some(ob)) = (cb0_opt, out_b.as_deref_mut()) {
                for i in 0..n {
                    ob[i] = cb0 * v[i];
                }
            }
            if max_len == 1 {
                return Ok(());
            }
            let ca1 = coeffs_a.get(1).copied().unwrap_or(0.0);
            let cb1 = coeffs_b.and_then(|cb| cb.get(1).copied()).unwrap_or(0.0);
            Self::fused_step_const::<1>(
                mat,
                unshifted_scale,
                &[],
                &self.t_prev[..n],
                &mut self.t_curr[..n],
                ca1,
                (ca1 != 0.0).then_some(&mut out_a[..n]),
                cb1,
                if cb1 != 0.0 {
                    out_b.as_deref_mut()
                } else {
                    None
                },
                true,
            )?;
            for k in 2..max_len {
                let ca = coeffs_a.get(k).copied().unwrap_or(0.0);
                let cb = coeffs_b.and_then(|b| b.get(k).copied()).unwrap_or(0.0);
                Self::fused_step_const::<1>(
                    mat,
                    unshifted_scale,
                    &self.t_prev[..n],
                    &self.t_curr[..n],
                    &mut self.t_next[..n],
                    ca,
                    (ca != 0.0).then_some(&mut out_a[..n]),
                    cb,
                    if cb != 0.0 {
                        out_b.as_deref_mut()
                    } else {
                        None
                    },
                    false,
                )?;
                std::mem::swap(&mut self.t_prev, &mut self.t_curr);
                std::mem::swap(&mut self.t_curr, &mut self.t_next);
            }
            return Ok(());
        }

        let max_tile_len = n * ncols.min(CHEBYSHEV_TILE_COLS);
        self.ensure_len(max_tile_len);
        self.ensure_tile_out(max_tile_len, coeffs_b.is_some());

        for col_start in (0..ncols).step_by(CHEBYSHEV_TILE_COLS) {
            let w = (ncols - col_start).min(CHEBYSHEV_TILE_COLS);
            let tile_len = n * w;

            if let Some(cb0) = cb0_opt {
                for row in 0..n {
                    let src = &v[row * ncols + col_start..row * ncols + col_start + w];
                    let prev_dst = &mut self.t_prev[row * w..row * w + w];
                    let oa_dst = &mut self.out_a_tile[row * w..row * w + w];
                    let ob_dst = &mut self.out_b_tile[row * w..row * w + w];
                    for j in 0..w {
                        let val = src[j];
                        prev_dst[j] = val;
                        oa_dst[j] = ca0 * val;
                        ob_dst[j] = cb0 * val;
                    }
                }
            } else {
                for row in 0..n {
                    let src = &v[row * ncols + col_start..row * ncols + col_start + w];
                    let prev_dst = &mut self.t_prev[row * w..row * w + w];
                    let oa_dst = &mut self.out_a_tile[row * w..row * w + w];
                    for j in 0..w {
                        let val = src[j];
                        prev_dst[j] = val;
                        oa_dst[j] = ca0 * val;
                    }
                }
            }

            self.run_tile_degrees(mat, unshifted_scale, coeffs_a, coeffs_b, w)?;

            for row in 0..n {
                let oa_src = &self.out_a_tile[row * w..row * w + w];
                out_a[row * ncols + col_start..row * ncols + col_start + w].copy_from_slice(oa_src);
            }
            if let Some(ob) = out_b.as_deref_mut() {
                for row in 0..n {
                    let ob_src = &self.out_b_tile[row * w..row * w + w];
                    ob[row * ncols + col_start..row * ncols + col_start + w]
                        .copy_from_slice(ob_src);
                }
            }
            let _ = tile_len;
        }
        Ok(())
    }

    /// Computes a Johnson-Lindenstrauss spectral embedding
    /// $Y = \sqrt{h}(L_{\mathrm{sym}}) Z$ (or $D^{-1/2} Y$ when `inv_sqrt_deg` is provided),
    /// where $Z \in \{\pm 1 / \sqrt{d}\}^{N \times d}$ is generated deterministically from `seed`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_kernel_jl_embedding(
        &mut self,
        l_sym: &CsrMatrix,
        terms: &[(f64, f64)],
        lambda_max: f64,
        degree: usize,
        jl_dim: usize,
        seed: u64,
        inv_sqrt_deg: Option<&[f64]>,
    ) -> CoreResult<Vec<f64>> {
        let n = l_sym.num_nodes();
        if jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        let total = n
            .checked_mul(jl_dim)
            .ok_or_else(|| CoreError::invalid_input("JL embedding size overflow"))?;
        let mut out = vec![0.0_f64; total];
        self.compute_kernel_jl_embedding_into(
            l_sym,
            terms,
            lambda_max,
            degree,
            jl_dim,
            seed,
            inv_sqrt_deg,
            &mut out,
        )?;
        Ok(out)
    }

    /// Computes a Johnson-Lindenstrauss spectral embedding using a pre-shifted matrix
    /// $\tilde{L} = \frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$ in L2-resident 8-column tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_kernel_jl_embedding_shifted(
        &mut self,
        l_tilde: &CsrMatrix,
        terms: &[(f64, f64)],
        lambda_max: f64,
        degree: usize,
        jl_dim: usize,
        seed: u64,
        inv_sqrt_deg: Option<&[f64]>,
    ) -> CoreResult<Vec<f64>> {
        let n = l_tilde.num_nodes();
        if jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        let total = n
            .checked_mul(jl_dim)
            .ok_or_else(|| CoreError::invalid_input("JL embedding size overflow"))?;
        let mut out = vec![0.0_f64; total];
        self.compute_kernel_jl_embedding_impl(
            l_tilde,
            None,
            terms,
            lambda_max,
            degree,
            jl_dim,
            seed,
            inv_sqrt_deg,
            &mut out,
        )?;
        Ok(out)
    }

    /// In-place variant of [`ChebyshevWorkspace::compute_kernel_jl_embedding`] writing into a
    /// pre-allocated row-major `(n, jl_dim)` buffer `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_kernel_jl_embedding_into(
        &mut self,
        l_sym: &CsrMatrix,
        terms: &[(f64, f64)],
        lambda_max: f64,
        degree: usize,
        jl_dim: usize,
        seed: u64,
        inv_sqrt_deg: Option<&[f64]>,
        out: &mut [f64],
    ) -> CoreResult<()> {
        if !lambda_max.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::compute_kernel_jl_embedding_into::lambda_max",
            ));
        }
        if lambda_max <= 0.0 {
            return Err(CoreError::invalid_input(
                "lambda_max must be strictly positive",
            ));
        }
        self.compute_kernel_jl_embedding_impl(
            l_sym,
            Some(2.0 / lambda_max),
            terms,
            lambda_max,
            degree,
            jl_dim,
            seed,
            inv_sqrt_deg,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_kernel_jl_embedding_impl(
        &mut self,
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        terms: &[(f64, f64)],
        lambda_max: f64,
        degree: usize,
        jl_dim: usize,
        seed: u64,
        inv_sqrt_deg: Option<&[f64]>,
        out: &mut [f64],
    ) -> CoreResult<()> {
        if jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        let n = mat.num_nodes();
        let total = n
            .checked_mul(jl_dim)
            .ok_or_else(|| CoreError::invalid_input("JL embedding size overflow"))?;
        if out.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                out.len(),
                "ChebyshevWorkspace::compute_kernel_jl_embedding_into::out",
            ));
        }
        if let Some(inv_d) = inv_sqrt_deg
            && inv_d.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                inv_d.len(),
                "ChebyshevWorkspace::compute_kernel_jl_embedding_into::inv_sqrt_deg",
            ));
        }

        let coeffs_full = compute_chebyshev_sqrt_filter_coefficients(terms, lambda_max, degree)?;
        if n == 0 {
            return Ok(());
        }
        validate_csr_indices_in_bounds(mat)?;
        let eff = effective_chebyshev_len(&coeffs_full, CHEBYSHEV_TAIL_L1_TOL);
        let coeffs = &coeffs_full[..eff];

        let mut rng = DeterministicRng::from_seed_u64(seed);
        let scale = 1.0 / (jl_dim as f64).sqrt();
        for dst in &mut out[..total] {
            *dst = if (rng.next_u64() & 1) == 0 {
                scale
            } else {
                -scale
            };
        }

        let max_tile_len = n * jl_dim.min(CHEBYSHEV_TILE_COLS);
        self.ensure_len(max_tile_len);
        self.ensure_tile_out(max_tile_len, false);

        let c0 = coeffs[0];
        for col_start in (0..jl_dim).step_by(CHEBYSHEV_TILE_COLS) {
            let w = (jl_dim - col_start).min(CHEBYSHEV_TILE_COLS);
            for row in 0..n {
                let z_src = &out[row * jl_dim + col_start..row * jl_dim + col_start + w];
                let prev_dst = &mut self.t_prev[row * w..row * w + w];
                let oa_dst = &mut self.out_a_tile[row * w..row * w + w];
                for j in 0..w {
                    let val = z_src[j];
                    prev_dst[j] = val;
                    oa_dst[j] = c0 * val;
                }
            }

            self.run_tile_degrees(mat, unshifted_scale, coeffs, None, w)?;

            if let Some(inv_d) = inv_sqrt_deg {
                for (row, &d_val) in inv_d.iter().enumerate() {
                    let oa_src = &self.out_a_tile[row * w..row * w + w];
                    let dst = &mut out[row * jl_dim + col_start..row * jl_dim + col_start + w];
                    for j in 0..w {
                        dst[j] = d_val * oa_src[j];
                    }
                }
            } else {
                for row in 0..n {
                    let oa_src = &self.out_a_tile[row * w..row * w + w];
                    out[row * jl_dim + col_start..row * jl_dim + col_start + w]
                        .copy_from_slice(oa_src);
                }
            }
        }
        Ok(())
    }

    /// Estimates row diagonal entries $\|Y_i\|_2^2$ for one or two spectral filters
    /// ($Y_a = \sqrt{h_a}(L_{\mathrm{sym}}) Z$, optionally $Y_b = \sqrt{h_b}(L_{\mathrm{sym}}) Z$)
    /// using a pre-shifted matrix $\tilde{L} = \frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$.
    ///
    /// Generates Rademacher $\pm 1/\sqrt{d}$ probes directly per 8-column tile via
    /// [`DeterministicRng::from_seed_u64`] without allocating any `(n, jl_dim)` matrices.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn compute_dual_jl_squared_norms_shifted(
        &mut self,
        l_tilde: &CsrMatrix,
        terms_a: &[(f64, f64)],
        degree_a: usize,
        terms_b: Option<(&[(f64, f64)], usize)>,
        lambda_max: f64,
        jl_dim: usize,
        seed: u64,
    ) -> CoreResult<(Vec<f64>, Option<Vec<f64>>)> {
        if jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        let coeffs_a_full =
            compute_chebyshev_sqrt_filter_coefficients(terms_a, lambda_max, degree_a)?;
        let coeffs_b_full = match terms_b {
            Some((tb, deg_b)) => Some(compute_chebyshev_sqrt_filter_coefficients(
                tb, lambda_max, deg_b,
            )?),
            None => None,
        };

        let n = l_tilde.num_nodes();
        if n == 0 {
            return Ok((Vec::new(), coeffs_b_full.as_ref().map(|_| Vec::new())));
        }
        validate_csr_indices_in_bounds(l_tilde)?;

        let eff_a = effective_chebyshev_len(&coeffs_a_full, CHEBYSHEV_TAIL_L1_TOL);
        let coeffs_a = &coeffs_a_full[..eff_a];
        let coeffs_b_trimmed = coeffs_b_full.as_deref().map(|cb| {
            let eff_b = effective_chebyshev_len(cb, CHEBYSHEV_TAIL_L1_TOL);
            &cb[..eff_b]
        });

        let mut norms_a = vec![0.0_f64; n];
        let mut norms_b = coeffs_b_trimmed.map(|_| vec![0.0_f64; n]);

        let max_tile_len = n
            .checked_mul(jl_dim.min(CHEBYSHEV_TILE_COLS))
            .ok_or_else(|| CoreError::invalid_input("JL tile size overflow"))?;
        self.ensure_len(max_tile_len);
        self.ensure_tile_out(max_tile_len, coeffs_b_trimmed.is_some());

        let mut rng = DeterministicRng::from_seed_u64(seed);
        let scale = 1.0 / (jl_dim as f64).sqrt();
        let ca0 = coeffs_a[0];
        let cb0_opt = coeffs_b_trimmed.map(|cb| cb[0]);

        for col_start in (0..jl_dim).step_by(CHEBYSHEV_TILE_COLS) {
            let w = (jl_dim - col_start).min(CHEBYSHEV_TILE_COLS);
            let tile_len = n * w;
            if let Some(cb0) = cb0_opt {
                for idx in 0..tile_len {
                    let val = if (rng.next_u64() & 1) == 0 {
                        scale
                    } else {
                        -scale
                    };
                    self.t_prev[idx] = val;
                    self.out_a_tile[idx] = ca0 * val;
                    self.out_b_tile[idx] = cb0 * val;
                }
            } else {
                for idx in 0..tile_len {
                    let val = if (rng.next_u64() & 1) == 0 {
                        scale
                    } else {
                        -scale
                    };
                    self.t_prev[idx] = val;
                    self.out_a_tile[idx] = ca0 * val;
                }
            }

            self.run_tile_degrees(l_tilde, None, coeffs_a, coeffs_b_trimmed, w)?;

            for (row, norm_a) in norms_a.iter_mut().enumerate() {
                let oa = &self.out_a_tile[row * w..row * w + w];
                let mut sq = 0.0_f64;
                for &val in oa {
                    sq += val * val;
                }
                *norm_a += sq;
            }
            if let Some(nb) = norms_b.as_deref_mut() {
                for (row, norm_b) in nb.iter_mut().enumerate() {
                    let ob = &self.out_b_tile[row * w..row * w + w];
                    let mut sq = 0.0_f64;
                    for &val in ob {
                        sq += val * val;
                    }
                    *norm_b += sq;
                }
            }
        }

        Ok((norms_a, norms_b))
    }

    /// Evaluates $\exp(-t A) V$ on an arbitrary symmetric [`CsrMatrix`] using dynamic
    /// spectrum scaling $[0, \hat{\lambda}_{\max}]$ (auto-estimated via Lanczos when
    /// `lambda_max` is `None`) and adaptive scaled Chebyshev degree (when `degree` is `None`).
    ///
    /// Handles empty (`num_nodes == 0`) and zero (`lambda_max == 0.0`) matrices analytically,
    /// returning the effective $\hat{\lambda}_{\max}$ used.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_heat_dynamic(
        &mut self,
        matrix: &CsrMatrix,
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
        t: f64,
        degree: Option<usize>,
        lambda_max: Option<f64>,
        lanczos_steps: usize,
        safety_factor: f64,
        seed: u64,
    ) -> CoreResult<f64> {
        if !t.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::apply_heat_dynamic::t",
            ));
        }
        if t < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "diffusion time t must be >= 0, got {t}"
            )));
        }
        let n = matrix.num_nodes();
        let total = n
            .checked_mul(ncols)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace batch size overflow"))?;
        if v.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                v.len(),
                "ChebyshevWorkspace::apply_heat_dynamic::v",
            ));
        }
        if out.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                out.len(),
                "ChebyshevWorkspace::apply_heat_dynamic::out",
            ));
        }

        let gersh = matrix.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::apply_heat_dynamic::matrix",
            ));
        }
        let lmax = match lambda_max {
            Some(lm) => {
                if !lm.is_finite() {
                    return Err(CoreError::non_finite(
                        "ChebyshevWorkspace::apply_heat_dynamic::lambda_max",
                    ));
                }
                if lm < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "lambda_max must be >= 0, got {lm}"
                    )));
                }
                if lm == 0.0 && gersh > 0.0 {
                    return Err(CoreError::invalid_input(
                        "lambda_max must be > 0 for non-zero matrices",
                    ));
                }
                lm
            }
            None => matrix.estimate_lambda_max_lanczos(lanczos_steps, safety_factor, seed)?,
        };

        if total == 0 {
            return Ok(lmax);
        }
        if lmax == 0.0 || t == 0.0 {
            out.copy_from_slice(v);
            return Ok(lmax);
        }

        let deg = degree.unwrap_or_else(|| adaptive_chebyshev_degree_scaled(t, lmax));
        let coeffs = compute_chebyshev_bandpass_coefficients(&[(1.0, t)], lmax, deg)?;
        self.apply_batch_with_lambda_max(matrix, lmax, &coeffs, v, out, ncols)?;
        Ok(lmax)
    }

    /// Computes a deterministic Johnson-Lindenstrauss spectral embedding
    /// $\sqrt{h_+}(A) Z$ (optionally left-multiplied by `scale_vec`) for an arbitrary symmetric
    /// [`CsrMatrix`] using dynamic spectrum scaling $[0, \hat{\lambda}_{\max}]$.
    ///
    /// Handles empty (`num_nodes == 0`) and zero (`lambda_max == 0.0`) matrices analytically,
    /// returning `(embedding, lambda_max_used)`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_dog_jl_dynamic(
        &mut self,
        matrix: &CsrMatrix,
        terms: &[(f64, f64)],
        lambda_max: Option<f64>,
        degree: Option<usize>,
        jl_dim: usize,
        eps_psd: f64,
        lanczos_steps: usize,
        safety_factor: f64,
        seed: u64,
        scale_vec: Option<&[f64]>,
    ) -> CoreResult<(Vec<f64>, f64)> {
        if jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        if !eps_psd.is_finite() || eps_psd < 0.0 {
            return Err(CoreError::invalid_input("eps_psd must be finite and >= 0"));
        }
        if terms.is_empty() {
            return Err(CoreError::invalid_input("filter terms must not be empty"));
        }
        if let Some(0) = degree {
            return Err(CoreError::invalid_input("degree must be >= 1"));
        }
        let mut max_t = 0.0_f64;
        for &(w, t) in terms {
            if !w.is_finite() || !t.is_finite() {
                return Err(CoreError::non_finite(
                    "ChebyshevWorkspace::compute_dog_jl_dynamic::terms",
                ));
            }
            if t < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "diffusion time t must be >= 0, got {t}"
                )));
            }
            if t > max_t {
                max_t = t;
            }
        }
        let n = matrix.num_nodes();
        if let Some(sv) = scale_vec {
            if sv.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    sv.len(),
                    "ChebyshevWorkspace::compute_dog_jl_dynamic::scale_vec",
                ));
            }
            if sv.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite(
                    "ChebyshevWorkspace::compute_dog_jl_dynamic::scale_vec",
                ));
            }
        }

        let gersh = matrix.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::compute_dog_jl_dynamic::matrix",
            ));
        }
        let lmax = match lambda_max {
            Some(lm) => {
                if !lm.is_finite() {
                    return Err(CoreError::non_finite(
                        "ChebyshevWorkspace::compute_dog_jl_dynamic::lambda_max",
                    ));
                }
                if lm < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "lambda_max must be >= 0, got {lm}"
                    )));
                }
                if lm == 0.0 && gersh > 0.0 {
                    return Err(CoreError::invalid_input(
                        "lambda_max must be > 0 for non-zero matrices",
                    ));
                }
                lm
            }
            None => matrix.estimate_lambda_max_lanczos(lanczos_steps, safety_factor, seed)?,
        };

        if n == 0 {
            return Ok((Vec::new(), lmax));
        }
        if lmax == 0.0 {
            let total = n
                .checked_mul(jl_dim)
                .ok_or_else(|| CoreError::invalid_input("JL embedding size overflow"))?;
            let h0: f64 = terms.iter().map(|&(w, _)| w).sum();
            let delta = (eps_psd - h0).max(0.0);
            let h_plus = ((h0 + delta) / (1.0 + delta)).max(0.0);
            let g0 = h_plus.sqrt();
            let mut rng = DeterministicRng::from_seed_u64(seed);
            let scale = g0 / (jl_dim as f64).sqrt();
            let mut out = vec![0.0_f64; total];
            for row in 0..n {
                let row_scale = scale_vec.map_or(1.0, |sv| sv[row]) * scale;
                for col in 0..jl_dim {
                    out[row * jl_dim + col] = if (rng.next_u64() & 1) == 0 {
                        row_scale
                    } else {
                        -row_scale
                    };
                }
            }
            return Ok((out, 0.0));
        }

        let deg = degree.unwrap_or_else(|| adaptive_chebyshev_degree_scaled(max_t, lmax));
        let (terms_psd, _, _) = project_psd_spectral_filter(terms, lmax, deg, eps_psd)?;
        let emb = self
            .compute_kernel_jl_embedding(matrix, &terms_psd, lmax, deg, jl_dim, seed, scale_vec)?;
        Ok((emb, lmax))
    }
}

/// Configuration for [`ChebyshevDogKernel`] implementing R1/R2 regularized Difference-of-Gaussians
/// with parameter-free `beta_auto` positivity floor and deterministic JL diagonal estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebyshevDogConfig {
    /// Optional short-range diffusion time $t_1$ (`None` uses R1 $1 / (1 + \sigma_\lambda)$).
    pub t1: Option<f64>,
    /// Optional long-range diffusion time $t_2$ (`None` uses R1 $4 / (1 + \sigma_\lambda)$).
    pub t2: Option<f64>,
    /// Optional DoG attenuation $\alpha$ (`None` uses R1 $\max(0, 1 - \sigma_\lambda)$).
    pub alpha: Option<f64>,
    /// Optional explicit uniform rank-1 floor $\beta \in [0, 1)$ (`None` computes `beta_auto`).
    pub beta_floor: Option<f64>,
    /// Optional explicit Chebyshev polynomial degree (`None` uses [`adaptive_chebyshev_degree`]).
    pub degree: Option<usize>,
    /// Number of Core-First MaxMin shortest-path landmark columns to evaluate.
    pub n_landmarks: usize,
    /// Minimum degree threshold for Core-First MaxMin landmark selection.
    pub min_degree: usize,
    /// Johnson-Lindenstrauss projection dimension $d$ for exact PSD diagonal estimation.
    pub jl_dim: usize,
    /// Minimum spectral floor $\epsilon_{\mathrm{psd}}$ enforced during R2 PSD projection.
    pub eps_psd: f64,
    /// Spectral upper bound $\lambda_{\max}$ for $L_{\mathrm{sym}}$ (default `2.0`).
    pub lambda_max: f64,
    /// Explicit deterministic seed controlling landmark selection and JL Rademacher probes.
    pub seed: u64,
}

impl Default for ChebyshevDogConfig {
    fn default() -> Self {
        Self {
            t1: None,
            t2: None,
            alpha: None,
            beta_floor: None,
            degree: None,
            n_landmarks: 64,
            min_degree: 2,
            jl_dim: 64,
            eps_psd: 1e-4,
            lambda_max: 2.0,
            seed: 42,
        }
    }
}

/// Pure $O(\mathrm{deg} \cdot (k_{\mathrm{lm}} + d_{\mathrm{JL}}) \cdot |E|)$ regularized DoG
/// diffusion kernel backed by [`ChebyshevWorkspace`] and deterministic JL embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevDogKernel {
    num_nodes: usize,
    t1: f64,
    t2: f64,
    alpha: f64,
    degree: usize,
    lambda_max: f64,
    delta_psd: f64,
    h_min: f64,
    beta_auto: f64,
    beta_eff: f64,
    moment_stats: SpectralMomentStats,
    terms_psd: Vec<(f64, f64)>,
    coeffs_h: Vec<f64>,
    landmarks: Vec<u32>,
    landmark_kernel: Vec<f64>,
    diag_estimate: Vec<f64>,
    jl_embedding: Vec<f64>,
    jl_dim: usize,
}

impl ChebyshevDogKernel {
    /// Builds a regularized DoG kernel from `graph` using Core-First MaxMin landmarks.
    pub fn build<W: Scalar>(graph: &CsrGraph<W>, config: &ChebyshevDogConfig) -> CoreResult<Self> {
        Self::build_internal(graph, config, None)
    }

    /// Builds a regularized DoG kernel using explicitly provided `landmarks`.
    pub fn build_with_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        config: &ChebyshevDogConfig,
        landmarks: &[u32],
    ) -> CoreResult<Self> {
        Self::build_internal(graph, config, Some(landmarks))
    }

    fn build_internal<W: Scalar>(
        graph: &CsrGraph<W>,
        config: &ChebyshevDogConfig,
        explicit_landmarks: Option<&[u32]>,
    ) -> CoreResult<Self> {
        if config.n_landmarks == 0 {
            return Err(CoreError::invalid_input("n_landmarks must be >= 1"));
        }
        if config.jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        if !config.eps_psd.is_finite() || config.eps_psd < 0.0 {
            return Err(CoreError::invalid_input("eps_psd must be finite and >= 0"));
        }
        if !config.lambda_max.is_finite() || config.lambda_max <= 0.0 {
            return Err(CoreError::invalid_input(
                "lambda_max must be finite and > 0",
            ));
        }
        if let Some(bf) = config.beta_floor
            && (!bf.is_finite() || !(0.0..1.0).contains(&bf))
        {
            return Err(CoreError::invalid_input(
                "beta_floor must be finite and in [0.0, 1.0)",
            ));
        }
        if let Some(deg) = config.degree
            && deg == 0
        {
            return Err(CoreError::invalid_input("degree must be >= 1"));
        }

        let moment_stats = spectral_moment_stats(graph);
        let s = moment_stats.sigma_lambda;
        let t1 = config.t1.unwrap_or(1.0 / (1.0 + s));
        let t2 = config.t2.unwrap_or(4.0 / (1.0 + s));
        let alpha = config.alpha.unwrap_or((1.0 - s).max(0.0));
        let base_terms = build_dog_terms(t1, t2, alpha, 0.0)?;

        let degree = config
            .degree
            .unwrap_or_else(|| adaptive_chebyshev_degree(t1.max(t2)));
        let (terms_psd, delta_psd, h_min) =
            project_psd_spectral_filter(&base_terms, config.lambda_max, degree, config.eps_psd)?;
        let coeffs_h =
            compute_chebyshev_bandpass_coefficients(&terms_psd, config.lambda_max, degree)?;

        let n = graph.num_nodes();
        if let Some(lms) = explicit_landmarks {
            if n >= 1 && lms.is_empty() {
                return Err(CoreError::invalid_input(
                    "explicit landmarks must not be empty",
                ));
            }
            for &p in lms {
                if (p as usize) >= n {
                    return Err(CoreError::index_out_of_bounds(
                        p as usize,
                        n,
                        "ChebyshevDogKernel::landmarks",
                    ));
                }
            }
        }

        if n == 0 {
            let beta_auto = 1e-4;
            let beta_eff = config.beta_floor.unwrap_or(beta_auto);
            return Ok(Self {
                num_nodes: 0,
                t1,
                t2,
                alpha,
                degree,
                lambda_max: config.lambda_max,
                delta_psd,
                h_min,
                beta_auto,
                beta_eff,
                moment_stats,
                terms_psd,
                coeffs_h,
                landmarks: Vec::new(),
                landmark_kernel: Vec::new(),
                diag_estimate: Vec::new(),
                jl_embedding: Vec::new(),
                jl_dim: config.jl_dim,
            });
        }

        if n == 1 {
            let beta_auto = 1e-4;
            let beta_eff = config.beta_floor.unwrap_or(beta_auto);
            let (l_sym, _) = symmetric_normalized_laplacian(graph);
            let mut ws = ChebyshevWorkspace::with_capacity(config.jl_dim);
            let jl_embedding = ws.compute_kernel_jl_embedding(
                &l_sym,
                &terms_psd,
                config.lambda_max,
                degree,
                config.jl_dim,
                config.seed,
                None,
            )?;
            let landmarks = match explicit_landmarks {
                Some(lms) => vec![lms[0]],
                None => vec![0],
            };
            return Ok(Self {
                num_nodes: 1,
                t1,
                t2,
                alpha,
                degree,
                lambda_max: config.lambda_max,
                delta_psd,
                h_min,
                beta_auto,
                beta_eff,
                moment_stats,
                terms_psd,
                coeffs_h,
                landmarks,
                landmark_kernel: vec![1.0],
                diag_estimate: vec![1.0],
                jl_embedding,
                jl_dim: config.jl_dim,
            });
        }

        let k_lm = config.n_landmarks.min(n);
        let landmarks: Vec<u32> = if let Some(lms) = explicit_landmarks {
            lms.iter().take(k_lm).copied().collect()
        } else {
            select_maxmin_sp_pivots(graph, k_lm, config.seed, config.min_degree)?
        };
        let k_lm_actual = landmarks.len();

        let (l_sym, _) = symmetric_normalized_laplacian(graph);
        let l_tilde = l_sym.scale_and_shift(2.0 / config.lambda_max, -1.0)?;

        let n_probes = config.jl_dim.max(k_lm_actual);
        let mut ws = ChebyshevWorkspace::with_capacity(n * n_probes.min(CHEBYSHEV_TILE_COLS));

        let mut e_lm = vec![0.0_f64; n * k_lm_actual];
        for (col, &lm) in landmarks.iter().enumerate() {
            e_lm[(lm as usize) * k_lm_actual + col] = 1.0;
        }
        let mut k_plus_lm = vec![0.0_f64; n * k_lm_actual];
        ws.apply_batch(&l_tilde, &coeffs_h, &e_lm, &mut k_plus_lm, k_lm_actual)?;

        let jl_embedding = ws.compute_kernel_jl_embedding_shifted(
            &l_tilde,
            &terms_psd,
            config.lambda_max,
            degree,
            n_probes,
            config.seed,
            None,
        )?;

        let mut k_u_plus = vec![0.0_f64; n];
        for (i, ku) in k_u_plus.iter_mut().enumerate() {
            let row = &jl_embedding[i * n_probes..(i + 1) * n_probes];
            *ku = row.iter().map(|&y| y * y).sum();
        }

        let mut k_min = f64::INFINITY;
        for &val in &k_plus_lm {
            if val < k_min {
                k_min = val;
            }
        }
        let mut k_p_plus = Vec::with_capacity(k_lm_actual);
        for (col, &lm) in landmarks.iter().enumerate() {
            k_p_plus.push(k_plus_lm[(lm as usize) * k_lm_actual + col]);
        }
        k_p_plus.sort_by(f64::total_cmp);
        let k_med = if k_p_plus.is_empty() {
            0.0
        } else if k_p_plus.len() % 2 == 1 {
            k_p_plus[k_p_plus.len() / 2]
        } else {
            let mid = k_p_plus.len() / 2;
            0.5 * (k_p_plus[mid - 1] + k_p_plus[mid])
        };

        let denom_beta = (1.0 - k_min).max(1e-12);
        let beta_auto = ((0.02 * k_med - k_min).max(0.0) / denom_beta).clamp(1e-4, 0.5);
        let beta_eff = config.beta_floor.unwrap_or(beta_auto);

        let one_minus_beta = 1.0 - beta_eff;
        for val in &mut k_plus_lm {
            *val = one_minus_beta * (*val) + beta_eff;
        }
        for val in &mut k_u_plus {
            *val = one_minus_beta * (*val) + beta_eff;
        }

        Ok(Self {
            num_nodes: n,
            t1,
            t2,
            alpha,
            degree,
            lambda_max: config.lambda_max,
            delta_psd,
            h_min,
            beta_auto,
            beta_eff,
            moment_stats,
            terms_psd,
            coeffs_h,
            landmarks,
            landmark_kernel: k_plus_lm,
            diag_estimate: k_u_plus,
            jl_embedding,
            jl_dim: n_probes,
        })
    }

    /// Returns the number of vertices $N$.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of evaluated landmark columns $k_{\mathrm{lm}}$.
    #[inline]
    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    /// Returns the JL embedding column dimension $d$.
    #[inline]
    pub fn jl_dim(&self) -> usize {
        self.jl_dim
    }

    /// Returns the selected landmark vertex IDs.
    #[inline]
    pub fn landmarks(&self) -> &[u32] {
        &self.landmarks
    }

    /// Returns the row-major `(num_nodes, num_landmarks)` effective kernel block $K_{\mathrm{eff}}[:, P]$.
    #[inline]
    pub fn landmark_kernel_matrix(&self) -> &[f64] {
        &self.landmark_kernel
    }

    /// Returns the positive JL-based diagonal estimate $\mathrm{diag}(K_{\mathrm{eff}})$ of length `num_nodes`.
    #[inline]
    pub fn diag_estimate(&self) -> &[f64] {
        &self.diag_estimate
    }

    /// Returns the row-major `(num_nodes, jl_dim)` spectral square-root embedding $\sqrt{h_+}(L_{\mathrm{sym}}) Z$.
    #[inline]
    pub fn jl_embedding(&self) -> &[f64] {
        &self.jl_embedding
    }

    /// Returns the exact $O(|E|)$ spectral moment statistics used to parameterize R1 DoG.
    #[inline]
    pub fn moment_stats(&self) -> &SpectralMomentStats {
        &self.moment_stats
    }

    /// Returns the PSD-projected spectral filter terms.
    #[inline]
    pub fn terms_psd(&self) -> &[(f64, f64)] {
        &self.terms_psd
    }

    /// Returns the effective short-range diffusion time $t_1$.
    #[inline]
    pub fn t1(&self) -> f64 {
        self.t1
    }

    /// Returns the effective long-range diffusion time $t_2$.
    #[inline]
    pub fn t2(&self) -> f64 {
        self.t2
    }

    /// Returns the effective DoG attenuation $\alpha$.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the Chebyshev polynomial degree used.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the PSD shift $\delta_{\mathrm{psd}}$ added during R2 projection.
    #[inline]
    pub fn delta_psd(&self) -> f64 {
        self.delta_psd
    }

    /// Returns the minimum sampled value $\min_\lambda h(\lambda)$ prior to PSD projection.
    #[inline]
    pub fn h_min(&self) -> f64 {
        self.h_min
    }

    /// Returns the parameter-free automatic positivity floor `beta_auto`.
    #[inline]
    pub fn beta_auto(&self) -> f64 {
        self.beta_auto
    }

    /// Returns the effective rank-1 positivity floor `beta_eff` actually applied.
    #[inline]
    pub fn beta_eff(&self) -> f64 {
        self.beta_eff
    }

    /// Applies the full regularized DoG kernel operator
    /// $(1 - \beta_{\mathrm{eff}}) h_+(L_{\mathrm{sym}}) V + \beta_{\mathrm{eff}} \mathbf{1}(\mathbf{1}^T V)$
    /// to row-major `(num_nodes, ncols)` matrix `v`.
    pub fn apply_batch(
        &self,
        l_sym: &CsrMatrix,
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
        ws: &mut ChebyshevWorkspace,
    ) -> CoreResult<()> {
        ws.apply_batch_with_lambda_max(l_sym, self.lambda_max, &self.coeffs_h, v, out, ncols)?;
        if self.num_nodes == 0 || ncols == 0 {
            return Ok(());
        }
        let one_minus_beta = 1.0 - self.beta_eff;
        for c in 0..ncols {
            let mut col_sum = 0.0_f64;
            for i in 0..self.num_nodes {
                col_sum += v[i * ncols + c];
            }
            let shift = self.beta_eff * col_sum;
            for i in 0..self.num_nodes {
                let idx = i * ncols + c;
                out[idx] = one_minus_beta * out[idx] + shift;
            }
        }
        Ok(())
    }

    /// Converts the landmark kernel block $K_{\mathrm{eff}}[:, P]$ into a row-major
    /// `(num_nodes, num_landmarks)` Normalized Varadhan Geodesic coordinate table
    /// $\left[-4 t_1 \ln \mathrm{clip}\!\left(\frac{K_{u p}}{\sqrt{K_{uu} K_{pp}}}, \epsilon_{\mathrm{sim}}, 1\right)\right]^p$.
    pub fn varadhan_landmark_table(
        &self,
        p_exp: f64,
        degree_normalize: bool,
        eps_sim: f64,
    ) -> CoreResult<Vec<f64>> {
        if !p_exp.is_finite() || p_exp <= 0.0 {
            return Err(CoreError::invalid_input("p_exp must be finite and > 0"));
        }
        if !eps_sim.is_finite() || eps_sim <= 0.0 || eps_sim > 1.0 {
            return Err(CoreError::invalid_input(
                "eps_sim must be finite and in (0, 1]",
            ));
        }
        let n = self.num_nodes;
        let k_lm = self.landmarks.len();
        if n == 0 || k_lm == 0 {
            return Ok(Vec::new());
        }

        let mut k_p_exact = Vec::with_capacity(k_lm);
        let mut max_kp = 1e-12_f64;
        for (col, &lm) in self.landmarks.iter().enumerate() {
            let val = self.landmark_kernel[(lm as usize) * k_lm + col].max(1e-12);
            if val > max_kp {
                max_kp = val;
            }
            k_p_exact.push(val);
        }
        let max_ku = self
            .diag_estimate
            .iter()
            .fold(1e-12_f64, |acc, &v| acc.max(v));
        let k_scale = max_ku.max(max_kp);

        let mut out = vec![0.0_f64; n * k_lm];
        for i in 0..n {
            let ku = self.diag_estimate[i].max(1e-12);
            let row_off = i * k_lm;
            for col in 0..k_lm {
                let denom = if degree_normalize {
                    (ku * k_p_exact[col]).sqrt().max(1e-12)
                } else {
                    k_scale
                };
                let sim = (self.landmark_kernel[row_off + col] / denom).clamp(eps_sim, 1.0);
                let log_val = (-4.0 * self.t1 * sim.ln()).max(0.0);
                out[row_off + col] = log_val.powf(p_exp);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_expm_neg(a: &[f64], n: usize, t: f64) -> Vec<f64> {
        let mut scaled = vec![0.0_f64; n * n];
        let mut max_row_sum = 0.0_f64;
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                let val = -t * a[i * n + j];
                scaled[i * n + j] = val;
                row_sum += val.abs();
            }
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }
        let s = if max_row_sum > 0.5 {
            (max_row_sum.log2().ceil().max(0.0) as u32) + 1
        } else {
            0
        };
        let pow_scale = 2.0_f64.powi(-(s as i32));
        for v in &mut scaled {
            *v *= pow_scale;
        }

        let mut res = vec![0.0_f64; n * n];
        let mut term = vec![0.0_f64; n * n];
        for i in 0..n {
            res[i * n + i] = 1.0;
            term[i * n + i] = 1.0;
        }
        let mut next = vec![0.0_f64; n * n];
        for k in 1..=35 {
            let inv_k = 1.0 / (k as f64);
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..n {
                        sum += term[i * n + p] * scaled[p * n + j];
                    }
                    next[i * n + j] = sum * inv_k;
                }
            }
            term.copy_from_slice(&next);
            for (dst, &val) in res.iter_mut().zip(term.iter()) {
                *dst += val;
            }
        }
        for _ in 0..s {
            next.fill(0.0);
            for i in 0..n {
                for p in 0..n {
                    let rip = res[i * n + p];
                    for j in 0..n {
                        next[i * n + j] += rip * res[p * n + j];
                    }
                }
            }
            res.copy_from_slice(&next);
        }
        res
    }

    #[test]
    fn test_bessel_miller_accuracy_to_200() {
        // Verify generating-function identities e^{t(cos(theta) - 1)} = e^{-t}I_0 + 2 sum e^{-t}I_k cos(k theta)
        // across t in [0, 200] including t=0, tiny t, and large t=200.
        for &t in &[0.0, 1e-12, 1e-5, 0.25, 1.0, 8.0, 32.0, 100.0, 200.0] {
            let max_k = adaptive_chebyshev_degree(t);
            let ive = exp_scaled_bessel_i_all(max_k, t).unwrap();
            for &theta in &[0.0, 0.7, PI * 0.5, 2.3, PI] {
                let exact = (t * (theta.cos() - 1.0)).exp();
                let mut approx = ive[0];
                for (k, &bk) in ive.iter().enumerate().skip(1) {
                    approx += 2.0 * bk * ((k as f64) * theta).cos();
                }
                let err = (approx - exact).abs();
                assert!(
                    err < 1e-14,
                    "Bessel generating identity error too large for t={t}, theta={theta}: {err:.3e}"
                );
            }
        }
        // Single-element helper matches full vector
        let single = exp_scaled_bessel_i(5, 12.5).unwrap();
        let all = exp_scaled_bessel_i_all(5, 12.5).unwrap();
        assert!((single - all[5]).abs() < 1e-15);
    }

    #[test]
    fn test_chebyshev_heat_and_bandpass_up_to_t32() {
        let g = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let (l_sym, _) = symmetric_normalized_laplacian(&g);
        let l_dense = l_sym.to_dense().unwrap();
        let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();
        let mut ws = ChebyshevWorkspace::with_capacity(6 * 6);

        let mut eye = vec![0.0_f64; 36];
        for i in 0..6 {
            eye[i * 6 + i] = 1.0;
        }
        let mut k_cheb = vec![0.0_f64; 36];

        for &t in &[0.0, 0.5, 4.0, 16.0, 32.0] {
            let terms = [(1.0, t)];
            let deg = adaptive_chebyshev_degree(t);
            let coeffs = compute_chebyshev_bandpass_coefficients(&terms, 2.0, deg).unwrap();
            ws.apply_batch(&l_tilde, &coeffs, &eye, &mut k_cheb, 6)
                .unwrap();

            let k_exact = dense_expm_neg(&l_dense, 6, t);
            for idx in 0..36 {
                let err = (k_cheb[idx] - k_exact[idx]).abs();
                assert!(
                    err < 1e-10,
                    "Chebyshev error at t={t}: got {} vs {} (err={err:.3e})",
                    k_cheb[idx],
                    k_exact[idx]
                );
            }
        }

        // Verify unshifted direct SpMM matches pre-shifted L_tilde SpMM on a mixed term
        let terms = [(1.2, 0.5), (-0.7, 3.2), (0.15, 0.0)];
        let coeffs = compute_chebyshev_bandpass_coefficients(&terms, 2.0, 60).unwrap();
        let v = vec![1.0, -0.5, 2.0, 0.3, -1.2, 0.8];
        let mut out_shifted = vec![0.0; 6];
        let mut out_direct = vec![0.0; 6];
        ws.apply_vec(&l_tilde, &coeffs, &v, &mut out_shifted)
            .unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs, &v, &mut out_direct)
            .unwrap();
        for i in 0..6 {
            assert!((out_shifted[i] - out_direct[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dct_sqrt_filter_and_psd_projection() {
        // Non-PSD DoG filter becomes strictly PSD after R2 projection
        let raw_dog = build_dog_terms(0.5, 2.0, 1.0, 0.0).unwrap();
        let (terms_psd, delta_psd, h_min) =
            project_psd_spectral_filter(&raw_dog, 2.0, 30, 1e-4).unwrap();
        assert!(h_min < 1e-4);
        assert!(delta_psd > 0.0);

        // Verify g(L)^2 == h_+(L) on a 6-node graph
        let g = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)],
            None,
            false,
            false,
        )
        .unwrap();
        let (l_sym, _) = symmetric_normalized_laplacian(&g);
        let mut ws = ChebyshevWorkspace::new();

        let deg = 140;
        let coeffs_h = compute_chebyshev_bandpass_coefficients(&terms_psd, 2.0, deg).unwrap();
        let coeffs_sqrt = compute_chebyshev_sqrt_filter_coefficients(&terms_psd, 2.0, deg).unwrap();

        let v = vec![1.0, -2.0, 0.5, 0.7, -0.3, 1.4];
        let mut hv = vec![0.0; 6];
        let mut sv = vec![0.0; 6];
        let mut ssv = vec![0.0; 6];
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_h, &v, &mut hv)
            .unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_sqrt, &v, &mut sv)
            .unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_sqrt, &sv, &mut ssv)
            .unwrap();

        for i in 0..6 {
            assert!(
                (ssv[i] - hv[i]).abs() < 2e-5,
                "sqrt(h)^2 mismatch at {i}: {} vs {}",
                ssv[i],
                hv[i]
            );
        }

        // Smooth strictly positive filter achieves machine-level accuracy even at degree 45
        let smooth_terms = [(0.6, 0.5), (0.4, 2.0)];
        let coeffs_hs = compute_chebyshev_bandpass_coefficients(&smooth_terms, 2.0, 45).unwrap();
        let coeffs_ss = compute_chebyshev_sqrt_filter_coefficients(&smooth_terms, 2.0, 45).unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_hs, &v, &mut hv)
            .unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_ss, &v, &mut sv)
            .unwrap();
        ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_ss, &sv, &mut ssv)
            .unwrap();
        for i in 0..6 {
            assert!(
                (ssv[i] - hv[i]).abs() < 1e-11,
                "smooth sqrt(h)^2 mismatch at {i}: {} vs {}",
                ssv[i],
                hv[i]
            );
        }

        // Strongly negative filter must be rejected by compute_chebyshev_sqrt_filter_coefficients
        let non_psd = [(-1.0, 0.5), (0.2, 2.0)];
        assert!(compute_chebyshev_sqrt_filter_coefficients(&non_psd, 2.0, 25).is_err());
    }

    #[test]
    fn test_chebyshev_dog_kernel_regularized_and_edge_cases() {
        let edges: Vec<(u32, u32)> = (0..11)
            .map(|i| (i, i + 1))
            .chain([(0, 11), (0, 6), (3, 9)])
            .collect();
        let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();

        let cfg = ChebyshevDogConfig::default();
        let k1 = ChebyshevDogKernel::build(&g, &cfg).unwrap();
        let k2 = ChebyshevDogKernel::build(&g, &cfg).unwrap();
        assert_eq!(k1, k2);
        assert!(k1.beta_eff() >= 1e-4 && k1.beta_eff() <= 0.5);
        assert!(k1.diag_estimate().iter().all(|&d| d.is_finite() && d > 0.0));
        assert!(
            k1.landmark_kernel_matrix()
                .iter()
                .all(|&v| v.is_finite() && v > 0.0)
        );
        let varadhan = k1.varadhan_landmark_table(0.5, true, 1e-12).unwrap();
        assert_eq!(varadhan.len(), 12 * k1.num_landmarks());
        assert!(varadhan.iter().all(|&v| v.is_finite() && v >= 0.0));

        // Verify kernel apply_batch matches landmark_kernel columns on unit basis vectors
        let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&g);
        let k_lm = k1.num_landmarks();
        let mut e_lm = vec![0.0_f64; 12 * k_lm];
        for (col, &lm) in k1.landmarks().iter().enumerate() {
            e_lm[(lm as usize) * k_lm + col] = 1.0;
        }
        let mut applied = vec![0.0_f64; 12 * k_lm];
        let mut ws = ChebyshevWorkspace::new();
        k1.apply_batch(&l_sym, &e_lm, &mut applied, k_lm, &mut ws)
            .unwrap();
        for (&a, &b) in applied.iter().zip(k1.landmark_kernel_matrix().iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Verify JL embedding with inv_sqrt_deg scaling
        let jl_plain = ws
            .compute_kernel_jl_embedding(&l_sym, k1.terms_psd(), 2.0, k1.degree(), 16, 77, None)
            .unwrap();
        let jl_scaled = ws
            .compute_kernel_jl_embedding(
                &l_sym,
                k1.terms_psd(),
                2.0,
                k1.degree(),
                16,
                77,
                Some(&inv_sqrt_deg),
            )
            .unwrap();
        for i in 0..12 {
            for c in 0..16 {
                assert!(
                    (jl_scaled[i * 16 + c] - inv_sqrt_deg[i] * jl_plain[i * 16 + c]).abs() < 1e-12
                );
            }
        }

        // Empty and singleton graphs
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let ke = ChebyshevDogKernel::build(&empty, &cfg).unwrap();
        assert_eq!(ke.num_nodes(), 0);
        assert!(ChebyshevDogKernel::build_with_landmarks(&empty, &cfg, &[0]).is_err());

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let ks = ChebyshevDogKernel::build(&single, &cfg).unwrap();
        assert_eq!(ks.num_nodes(), 1);
        assert_eq!(ks.diag_estimate(), &[1.0]);
        assert!(ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[]).is_err());
        assert!(ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[1]).is_err());
        assert_eq!(
            ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[0])
                .unwrap()
                .landmarks(),
            &[0]
        );
    }

    #[test]
    fn test_tiled_apply_batch_dual_and_shifted_jl_norms() {
        let edges: Vec<(u32, u32)> = (0..15)
            .map(|i| (i, i + 1))
            .chain([(0, 15), (0, 8), (4, 12), (2, 10)])
            .collect();
        let g = CsrGraph::<f64>::from_edges(16, &edges, None, false, false).unwrap();
        let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&g);
        let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();

        let coeffs_a = compute_chebyshev_bandpass_coefficients(&[(1.0, 0.8)], 2.0, 25).unwrap();
        let coeffs_b = compute_chebyshev_bandpass_coefficients(&[(1.0, 12.0)], 2.0, 75).unwrap();
        let mut ws = ChebyshevWorkspace::new();

        // Check ncols in {1, 8, 13} covering scalar, full tile, and partial trailing tile.
        for ncols in [1usize, 8, 13] {
            let mut v = vec![0.0_f64; 16 * ncols];
            for r in 0..16 {
                for c in 0..ncols {
                    v[r * ncols + c] = ((r * 7 + c * 3) as f64 * 0.37).sin();
                }
            }
            let mut out_a_dual = vec![0.0_f64; 16 * ncols];
            let mut out_b_dual = vec![0.0_f64; 16 * ncols];
            ws.apply_batch_dual(
                &l_tilde,
                &coeffs_a,
                Some(&coeffs_b),
                &v,
                &mut out_a_dual,
                Some(&mut out_b_dual),
                ncols,
            )
            .unwrap();

            let mut out_a_ref = vec![0.0_f64; 16 * ncols];
            let mut out_b_ref = vec![0.0_f64; 16 * ncols];
            ws.apply_batch_with_lambda_max(&l_sym, 2.0, &coeffs_a, &v, &mut out_a_ref, ncols)
                .unwrap();
            ws.apply_batch_with_lambda_max(&l_sym, 2.0, &coeffs_b, &v, &mut out_b_ref, ncols)
                .unwrap();

            for i in 0..(16 * ncols) {
                assert!((out_a_dual[i] - out_a_ref[i]).abs() < 1e-12);
                assert!((out_b_dual[i] - out_b_ref[i]).abs() < 1e-12);
            }
        }

        // Verify swapped polynomial lengths (coeffs_a longer than coeffs_b).
        let mut v13 = vec![0.5_f64; 16 * 13];
        v13[0] = 1.75;
        let mut out_long = vec![0.0_f64; 16 * 13];
        let mut out_short = vec![0.0_f64; 16 * 13];
        ws.apply_batch_dual(
            &l_tilde,
            &coeffs_b,
            Some(&coeffs_a),
            &v13,
            &mut out_long,
            Some(&mut out_short),
            13,
        )
        .unwrap();
        let mut ref_short = vec![0.0_f64; 16 * 13];
        ws.apply_batch(&l_tilde, &coeffs_a, &v13, &mut ref_short, 13)
            .unwrap();
        for (&got, &exp) in out_short.iter().zip(ref_short.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }

        // Verify shifted JL embedding matches unshifted JL embedding across tile widths.
        let terms_a = [(1.0, 0.8)];
        let terms_b = [(1.0, 12.0)];
        for jl_dim in [1usize, 5, 8, 13, 32] {
            let emb_unshifted = ws
                .compute_kernel_jl_embedding(
                    &l_sym,
                    &terms_a,
                    2.0,
                    25,
                    jl_dim,
                    99,
                    Some(&inv_sqrt_deg),
                )
                .unwrap();
            let emb_shifted = ws
                .compute_kernel_jl_embedding_shifted(
                    &l_tilde,
                    &terms_a,
                    2.0,
                    25,
                    jl_dim,
                    99,
                    Some(&inv_sqrt_deg),
                )
                .unwrap();
            for (&a, &b) in emb_unshifted.iter().zip(emb_shifted.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }

        // Verify compute_dual_jl_squared_norms_shifted accuracy and single-vs-dual consistency.
        let (norms_a_only, none_b) = ws
            .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_a, 25, None, 2.0, 13, 42)
            .unwrap();
        assert!(none_b.is_none());
        let (norms_a_dual, norms_b_dual) = ws
            .compute_dual_jl_squared_norms_shifted(
                &l_tilde,
                &terms_a,
                25,
                Some((&terms_b, 75)),
                2.0,
                13,
                42,
            )
            .unwrap();
        let norms_b_dual = norms_b_dual.unwrap();
        let (norms_b_only, _) = ws
            .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_b, 75, None, 2.0, 13, 42)
            .unwrap();
        for i in 0..16 {
            assert!((norms_a_only[i] - norms_a_dual[i]).abs() < 1e-14);
            assert!((norms_b_only[i] - norms_b_dual[i]).abs() < 1e-14);
            assert!(norms_a_dual[i] > 0.0 && norms_b_dual[i] > 0.0);
        }
        // For jl_dim <= 8, tile order equals row-major order so norms match exact JL row norms.
        let emb_7 = ws
            .compute_kernel_jl_embedding_shifted(&l_tilde, &terms_a, 2.0, 25, 7, 42, None)
            .unwrap();
        let (norms_7, _) = ws
            .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_a, 25, None, 2.0, 7, 42)
            .unwrap();
        for i in 0..16 {
            let row_sq: f64 = emb_7[i * 7..(i + 1) * 7].iter().map(|&x| x * x).sum();
            assert!((norms_7[i] - row_sq).abs() < 1e-12);
        }

        // Mismatched coeffs_b / out_b must return an error cleanly.
        assert!(
            ws.apply_batch_dual(
                &l_tilde,
                &coeffs_a,
                Some(&coeffs_b),
                &v13,
                &mut out_long,
                None,
                13
            )
            .is_err()
        );
    }

    #[test]
    fn test_effective_chebyshev_len_tail_truncation_and_16_col_tiling() {
        let coeffs_48 = compute_chebyshev_bandpass_coefficients(&[(1.0, 48.0)], 2.0, 75).unwrap();
        let eff_48 = effective_chebyshev_len(&coeffs_48, CHEBYSHEV_TAIL_L1_TOL);
        assert!(
            eff_48 <= 55,
            "expected effective_chebyshev_len(exp(-48 L)) <= 55, got {eff_48}"
        );

        let coeffs_sqrt_48 =
            compute_chebyshev_sqrt_filter_coefficients(&[(1.0, 48.0)], 2.0, 75).unwrap();
        let eff_sqrt_48 = effective_chebyshev_len(&coeffs_sqrt_48, CHEBYSHEV_TAIL_L1_TOL);
        assert!(
            eff_sqrt_48 <= 40,
            "expected effective_chebyshev_len(sqrt(exp(-48 L))) <= 40, got {eff_sqrt_48}"
        );

        // Verify truncated coefficients match untrimmed degree-75 Chebyshev evaluation < 1e-11
        let edges: Vec<(u32, u32)> = (0..19)
            .map(|i| (i, i + 1))
            .chain([(0, 19), (0, 10), (5, 15)])
            .collect();
        let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();
        let (l_sym, _) = symmetric_normalized_laplacian(&g);
        let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();

        let ncols = 16;
        let mut v = vec![0.0_f64; 20 * ncols];
        for r in 0..20 {
            for c in 0..ncols {
                v[r * ncols + c] = ((r * 5 + c * 11) as f64 * 0.23).cos();
            }
        }
        let mut ws = ChebyshevWorkspace::new();
        let mut out_band_trunc = vec![0.0_f64; 20 * ncols];
        let mut out_sqrt_trunc = vec![0.0_f64; 20 * ncols];
        ws.apply_batch_dual(
            &l_tilde,
            &coeffs_48,
            Some(&coeffs_sqrt_48),
            &v,
            &mut out_band_trunc,
            Some(&mut out_sqrt_trunc),
            ncols,
        )
        .unwrap();

        // Reference untrimmed degree-75 via scalar polynomial evaluation on dense eigendecomposition?
        // Even simpler: compute untrimmed degree-75 directly via CsrMatrix::spmm!
        let run_untrimmed = |coeffs: &[f64]| -> Vec<f64> {
            let mut t_prev = v.clone();
            let mut out = vec![0.0_f64; 20 * ncols];
            for (dst, &src) in out.iter_mut().zip(t_prev.iter()) {
                *dst = coeffs[0] * src;
            }
            let mut t_curr = vec![0.0_f64; 20 * ncols];
            l_tilde.spmm(&t_prev, &mut t_curr, ncols).unwrap();
            for (dst, &src) in out.iter_mut().zip(t_curr.iter()) {
                *dst += coeffs[1] * src;
            }
            let mut t_next = vec![0.0_f64; 20 * ncols];
            for &ck in &coeffs[2..] {
                l_tilde.spmm(&t_curr, &mut t_next, ncols).unwrap();
                for i in 0..(20 * ncols) {
                    let nxt = 2.0 * t_next[i] - t_prev[i];
                    t_next[i] = nxt;
                    out[i] += ck * nxt;
                }
                std::mem::swap(&mut t_prev, &mut t_curr);
                std::mem::swap(&mut t_curr, &mut t_next);
            }
            out
        };

        let ref_band = run_untrimmed(&coeffs_48);
        let ref_sqrt = run_untrimmed(&coeffs_sqrt_48);
        for (&got, &exp) in out_band_trunc.iter().zip(ref_band.iter()) {
            assert!((got - exp).abs() < 1e-11);
        }
        for (&got, &exp) in out_sqrt_trunc.iter().zip(ref_sqrt.iter()) {
            assert!((got - exp).abs() < 1e-11);
        }

        // Check edge cases of effective_chebyshev_len
        assert_eq!(effective_chebyshev_len(&[], 1e-12), 0);
        assert_eq!(effective_chebyshev_len(&[1.0], 1e-12), 1);
        assert_eq!(effective_chebyshev_len(&[1.0, 0.5, 1e-13], 1e-12), 2);
    }

    #[test]
    fn test_adaptive_degree_scaled_and_dynamic_spectrum_star_graph() {
        // Bit-exact match at lambda_max == 2.0
        for &t in &[0.0, 0.5, 1.0, 4.5, 16.0, 48.0] {
            assert_eq!(
                adaptive_chebyshev_degree_scaled(t, 2.0),
                adaptive_chebyshev_degree(t)
            );
        }
        assert!(adaptive_chebyshev_degree_scaled(2.0, 100.0) > adaptive_chebyshev_degree(2.0));

        // Zero-matrix fast path (edgeless graph under combinatorial Laplacian)
        let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let l_zero = crate::csr_matrix::combinatorial_laplacian(&edgeless);
        let mut ws = ChebyshevWorkspace::new();
        let v_zero = vec![1.0, -2.0, 3.0, -4.0];
        let mut out_zero = vec![0.0; 4];
        let lmax_zero = ws
            .apply_heat_dynamic(
                &l_zero,
                &v_zero,
                &mut out_zero,
                1,
                2.5,
                None,
                None,
                16,
                1.05,
                42,
            )
            .unwrap();
        assert_eq!(lmax_zero, 0.0);
        assert_eq!(out_zero, v_zero);

        let terms_zero = build_dog_terms(1.0, 4.0, 0.5, 0.0).unwrap();
        let (emb_zero, lmax_emb_zero) = ws
            .compute_dog_jl_dynamic(
                &l_zero,
                &terms_zero,
                None,
                None,
                8,
                1e-4,
                16,
                1.05,
                42,
                None,
            )
            .unwrap();
        assert_eq!(lmax_emb_zero, 0.0);
        assert_eq!(emb_zero.len(), 32);
        assert!(emb_zero.iter().all(|v| v.is_finite() && v.abs() > 0.0));

        // Star graph K_{1, 50}: L_comb has lambda_max = 51.0 >> 2.0
        let n_star = 51usize;
        let edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0, i as u32)).collect();
        let star = CsrGraph::<f64>::from_edges(n_star, &edges, None, false, false).unwrap();
        let l_star = crate::csr_matrix::combinatorial_laplacian(&star);
        let l_dense = l_star.to_dense().unwrap();
        let (evals, evecs) =
            crate::eigen::symmetric_eigen_jacobi(&l_dense, n_star, 64, 1e-12).unwrap();

        let t = 0.25;
        let v: Vec<f64> = (0..n_star).map(|i| ((i as f64) * 0.31).sin()).collect();
        let mut out = vec![0.0; n_star];
        let lmax_used = ws
            .apply_heat_dynamic(&l_star, &v, &mut out, 1, t, None, None, 16, 1.05, 42)
            .unwrap();
        assert!(lmax_used >= 51.0);

        // Exact exp(-t * L) v = V diag(exp(-t * lambda)) V^T v
        let mut proj = vec![0.0; n_star];
        for m in 0..n_star {
            let mut s = 0.0;
            for i in 0..n_star {
                s += evecs[i * n_star + m] * v[i];
            }
            proj[m] = s * (-t * evals[m]).exp();
        }
        let mut exact = vec![0.0; n_star];
        for i in 0..n_star {
            let mut s = 0.0;
            for m in 0..n_star {
                s += evecs[i * n_star + m] * proj[m];
            }
            exact[i] = s;
        }
        let max_err = out
            .iter()
            .zip(exact.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-8, "star heat dynamic max_err={max_err}");

        // Dynamic DoG JL embedding on star graph (lambda_max >> 2)
        let (dog_emb, lmax_dog) = ws
            .compute_dog_jl_dynamic(
                &l_star,
                &terms_zero,
                None,
                None,
                8,
                1e-4,
                16,
                1.05,
                42,
                None,
            )
            .unwrap();
        assert!(lmax_dog >= 51.0);
        assert_eq!(dog_emb.len(), n_star * 8);
        assert!(dog_emb.iter().all(|v| v.is_finite()));
    }
}
