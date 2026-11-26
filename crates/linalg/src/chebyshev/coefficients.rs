use std::f64::consts::PI;

use egraph_core::{CoreError, CoreResult};

pub(super) const CHEBYSHEV_TAIL_L1_TOL: f64 = 1e-12;

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
