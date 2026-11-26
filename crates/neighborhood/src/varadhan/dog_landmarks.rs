//! Regularized Chebyshev DoG Varadhan landmark computation and optional coarse diffusive
//! scaffold evaluation.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::{
    ChebyshevWorkspace, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
    DEFAULT_SPECTRUM_LANCZOS_STEPS, SpectralMomentStats, adaptive_chebyshev_degree_scaled,
    build_dog_terms, compute_chebyshev_bandpass_coefficients, project_psd_spectral_filter,
    select_maxmin_sp_pivots, spectral_moment_stats, symmetric_normalized_laplacian,
};

use super::DogVaradhanLandmarks;
use super::config::{DogVaradhanConfig, DogVaradhanStats, validate_dog_varadhan_config};

fn build_varadhan_table(
    k_lm_mat: &[f64],
    k_u: &[f64],
    k_p: &[f64],
    t_eff: f64,
    config: &DogVaradhanConfig,
) -> Vec<f64> {
    let n = k_u.len();
    let k_lm = k_p.len();
    if n == 0 || k_lm == 0 {
        return Vec::new();
    }
    let max_ku = k_u.iter().fold(1e-12_f64, |acc, &v| acc.max(v));
    let max_kp = k_p.iter().fold(1e-12_f64, |acc, &v| acc.max(v));
    let k_scale = max_ku.max(max_kp);

    let mut out = vec![0.0_f64; n * k_lm];
    for (i, &ku_raw) in k_u.iter().enumerate() {
        let ku = ku_raw.max(1e-12);
        let row_off = i * k_lm;
        for (col, &kp_raw) in k_p.iter().enumerate() {
            let denom = if config.degree_normalize {
                (ku * kp_raw.max(1e-12)).sqrt().max(1e-12)
            } else {
                k_scale
            };
            let sim = (k_lm_mat[row_off + col] / denom).clamp(config.eps_sim, 1.0);
            let log_val = (-4.0 * t_eff * sim.ln()).max(0.0);
            out[row_off + col] = log_val.powf(config.p_exp);
        }
    }
    out
}

/// Computes pure $O(|E| + \mathrm{deg}\cdot k \cdot |E|)$ regularized DoG Varadhan landmark
/// tables combining `egraph-linalg`'s R1/R2 PSD-projected DoG filter, single-pass dual tiled
/// Chebyshev SpMM + JL diagonal estimation, `beta_auto` positivity shift, and an optional
/// coarse heat scaffold at `config.t_coarse` (`48.0`).
pub fn compute_regularized_dog_varadhan_landmarks<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &DogVaradhanConfig,
) -> CoreResult<(DogVaradhanLandmarks, DogVaradhanStats)> {
    validate_dog_varadhan_config(config)?;
    if !config.lambda_max.is_finite() || config.lambda_max <= 0.0 {
        return Err(CoreError::invalid_input(
            "lambda_max must be finite and > 0",
        ));
    }

    let moment_stats = spectral_moment_stats(graph);
    let (l_sym, _) = symmetric_normalized_laplacian(graph);
    compute_dog_varadhan_landmarks_impl(
        graph,
        &l_sym,
        config.lambda_max,
        moment_stats.sigma_lambda,
        moment_stats,
        config,
    )
}

/// Computes regularized DoG Varadhan landmark tables from a custom symmetric PSD [`CsrMatrix`]
/// (such as [`egraph_linalg::anisotropic_laplacian`]) with dynamic $[0, \hat{\lambda}_{\max}]$
/// spectrum scaling.
///
/// When `lambda_max` is `None`, estimates $\hat{\lambda}_{\max}$ via 16-step deterministic
/// Lanczos (`1.05` safety margin clamped by Gershgorin) and scales adaptive Chebyshev degrees
/// to $[0, \hat{\lambda}_{\max}]$. Set `config.t_coarse = 0.0` to evaluate pure $L_\alpha$ DoG
/// without a coarse scaffold.
pub fn compute_regularized_dog_varadhan_landmarks_from_matrix<W: Scalar>(
    graph: &CsrGraph<W>,
    matrix: &CsrMatrix,
    lambda_max: Option<f64>,
    config: &DogVaradhanConfig,
) -> CoreResult<(DogVaradhanLandmarks, DogVaradhanStats)> {
    validate_dog_varadhan_config(config)?;
    let n = graph.num_nodes();
    if matrix.num_nodes() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            matrix.num_nodes(),
            "compute_regularized_dog_varadhan_landmarks_from_matrix::matrix",
        ));
    }
    if !matrix.is_symmetric(1e-10) {
        return Err(CoreError::invalid_input(
            "matrix must be symmetric for DoG Varadhan landmarks",
        ));
    }

    let mut moment_stats = spectral_moment_stats(graph);
    if n > 0 {
        let n_f = n as f64;
        let mu1 = matrix.trace() / n_f;
        let frob_sq: f64 = matrix.values().iter().map(|&v| v * v).sum();
        let sigma_sq = (frob_sq / n_f - mu1 * mu1).max(0.0);
        moment_stats.sigma_lambda_sq = sigma_sq;
        moment_stats.sigma_lambda = sigma_sq.max(1e-6).sqrt();
    }

    let gersh = matrix.gershgorin_spectral_bound();
    if !gersh.is_finite() {
        return Err(CoreError::non_finite(
            "compute_regularized_dog_varadhan_landmarks_from_matrix::matrix",
        ));
    }
    let lmax = match lambda_max {
        Some(lm) => {
            if !lm.is_finite() {
                return Err(CoreError::non_finite(
                    "compute_regularized_dog_varadhan_landmarks_from_matrix::lambda_max",
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
        None => matrix.estimate_lambda_max_lanczos(
            DEFAULT_SPECTRUM_LANCZOS_STEPS,
            DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
            config.seed,
        )?,
    };

    compute_dog_varadhan_landmarks_impl(
        graph,
        matrix,
        lmax,
        moment_stats.sigma_lambda,
        moment_stats,
        config,
    )
}

fn compute_dog_varadhan_landmarks_impl<W: Scalar>(
    graph: &CsrGraph<W>,
    matrix: &CsrMatrix,
    lambda_max: f64,
    s: f64,
    moment_stats: SpectralMomentStats,
    config: &DogVaradhanConfig,
) -> CoreResult<(DogVaradhanLandmarks, DogVaradhanStats)> {
    let t1 = config.t1.unwrap_or(1.0 / (1.0 + s));
    let t2 = config.t2.unwrap_or(4.0 / (1.0 + s));
    let alpha = config.alpha.unwrap_or((1.0 - s).max(0.0));
    let base_terms = build_dog_terms(t1, t2, alpha, 0.0)?;

    let deg_fine = config.degree.unwrap_or_else(|| {
        if lambda_max > 0.0 {
            adaptive_chebyshev_degree_scaled(t1.max(t2), lambda_max)
        } else {
            40
        }
    });

    let n = graph.num_nodes();
    if n == 0 {
        let (delta_psd, h_min) = if lambda_max > 0.0 {
            let (_, d, hm) =
                project_psd_spectral_filter(&base_terms, lambda_max, deg_fine, config.eps_psd)?;
            (d, hm)
        } else {
            let h0: f64 = base_terms.iter().map(|&(w, _)| w).sum();
            ((config.eps_psd - h0).max(0.0), h0)
        };
        let beta_auto = 1e-4;
        let beta_eff = config.beta_floor.unwrap_or(beta_auto);
        let stats = DogVaradhanStats {
            moment_stats,
            t1,
            t2,
            alpha,
            degree: deg_fine,
            delta_psd,
            h_min,
            k_min: 0.0,
            k_med: 0.0,
            beta_auto,
            beta_eff,
            n_landmarks: 0,
            t_coarse: config.t_coarse,
            used_coarse_scaffold: false,
        };
        return Ok((
            DogVaradhanLandmarks {
                num_nodes: 0,
                landmarks: Vec::new(),
                d_dog: Vec::new(),
                d_coarse: None,
            },
            stats,
        ));
    }

    if n == 1 {
        let (delta_psd, h_min) = if lambda_max > 0.0 {
            let (_, d, hm) =
                project_psd_spectral_filter(&base_terms, lambda_max, deg_fine, config.eps_psd)?;
            (d, hm)
        } else {
            let h0: f64 = base_terms.iter().map(|&(w, _)| w).sum();
            ((config.eps_psd - h0).max(0.0), h0)
        };
        let beta_auto = 1e-4;
        let beta_eff = config.beta_floor.unwrap_or(beta_auto);
        let stats = DogVaradhanStats {
            moment_stats,
            t1,
            t2,
            alpha,
            degree: deg_fine,
            delta_psd,
            h_min,
            k_min: 1.0,
            k_med: 1.0,
            beta_auto,
            beta_eff,
            n_landmarks: 1,
            t_coarse: config.t_coarse,
            used_coarse_scaffold: false,
        };
        return Ok((
            DogVaradhanLandmarks {
                num_nodes: 1,
                landmarks: vec![0],
                d_dog: vec![0.0],
                d_coarse: None,
            },
            stats,
        ));
    }

    let k_lm = config.n_landmarks.min(n);
    let landmarks = select_maxmin_sp_pivots(graph, k_lm, config.seed, config.min_degree)?;
    let k_lm_actual = landmarks.len();
    let use_coarse = t1.max(t2) < 32.0 && config.t_coarse > 0.0;

    let (delta_psd, h_min, mut k_plus_lm, mut k_u_plus, k_c_opt, k_u_c_opt) = if lambda_max == 0.0 {
        let h0: f64 = base_terms.iter().map(|&(w, _)| w).sum();
        let d_psd = (config.eps_psd - h0).max(0.0);
        let h_plus_0 = ((h0 + d_psd) / (1.0 + d_psd)).max(0.0);
        let mut kp = vec![0.0_f64; n * k_lm_actual];
        for (col, &lm) in landmarks.iter().enumerate() {
            kp[(lm as usize) * k_lm_actual + col] = h_plus_0;
        }
        let ku = vec![h_plus_0; n];
        let (kc, kuc) = if use_coarse {
            let mut kc_tbl = vec![0.0_f64; n * k_lm_actual];
            for (col, &lm) in landmarks.iter().enumerate() {
                kc_tbl[(lm as usize) * k_lm_actual + col] = 1.0;
            }
            (Some(kc_tbl), Some(vec![1.0_f64; n]))
        } else {
            (None, None)
        };
        (d_psd, h0, kp, ku, kc, kuc)
    } else {
        let (terms_psd, delta_psd, h_min) =
            project_psd_spectral_filter(&base_terms, lambda_max, deg_fine, config.eps_psd)?;
        let coeffs_fine =
            compute_chebyshev_bandpass_coefficients(&terms_psd, lambda_max, deg_fine)?;
        let l_tilde = matrix.scale_and_shift(2.0 / lambda_max, -1.0)?;

        let deg_c = match config.degree {
            Some(d) => d.max(75),
            None => deg_fine.max(adaptive_chebyshev_degree_scaled(
                config.t_coarse,
                lambda_max,
            )),
        };
        let coarse_terms = [(1.0, config.t_coarse)];
        let coeffs_c_opt = if use_coarse {
            Some(compute_chebyshev_bandpass_coefficients(
                &coarse_terms,
                lambda_max,
                deg_c,
            )?)
        } else {
            None
        };

        let mut e_lm = vec![0.0_f64; n * k_lm_actual];
        for (col, &lm) in landmarks.iter().enumerate() {
            e_lm[(lm as usize) * k_lm_actual + col] = 1.0;
        }

        let mut ws = ChebyshevWorkspace::with_capacity(n * k_lm_actual.min(8));
        let mut k_plus_lm = vec![0.0_f64; n * k_lm_actual];
        let mut k_c_opt = if use_coarse {
            Some(vec![0.0_f64; n * k_lm_actual])
        } else {
            None
        };
        ws.apply_batch_dual(
            &l_tilde,
            &coeffs_fine,
            coeffs_c_opt.as_deref(),
            &e_lm,
            &mut k_plus_lm,
            k_c_opt.as_deref_mut(),
            k_lm_actual,
        )?;

        let n_probes = config.jl_dim.max(k_lm_actual);
        let (k_u_plus, k_u_c_opt) = ws.compute_dual_jl_squared_norms_shifted(
            &l_tilde,
            &terms_psd,
            deg_fine,
            use_coarse.then_some((&coarse_terms[..], deg_c)),
            lambda_max,
            n_probes,
            config.seed,
        )?;
        (delta_psd, h_min, k_plus_lm, k_u_plus, k_c_opt, k_u_c_opt)
    };

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
    let mut k_p_eff = Vec::with_capacity(k_lm_actual);
    for (col, &lm) in landmarks.iter().enumerate() {
        k_p_eff.push(k_plus_lm[(lm as usize) * k_lm_actual + col]);
    }
    let d_dog = build_varadhan_table(&k_plus_lm, &k_u_plus, &k_p_eff, t1, config);

    let d_coarse = match (k_c_opt, k_u_c_opt) {
        (Some(k_c), Some(k_u_c)) => {
            let mut k_p_c = Vec::with_capacity(k_lm_actual);
            for (col, &lm) in landmarks.iter().enumerate() {
                k_p_c.push(k_c[(lm as usize) * k_lm_actual + col]);
            }
            Some(build_varadhan_table(
                &k_c,
                &k_u_c,
                &k_p_c,
                config.t_coarse,
                config,
            ))
        }
        _ => None,
    };

    let stats = DogVaradhanStats {
        moment_stats,
        t1,
        t2,
        alpha,
        degree: deg_fine,
        delta_psd,
        h_min,
        k_min,
        k_med,
        beta_auto,
        beta_eff,
        n_landmarks: k_lm_actual,
        t_coarse: config.t_coarse,
        used_coarse_scaffold: d_coarse.is_some(),
    };

    Ok((
        DogVaradhanLandmarks {
            num_nodes: n,
            landmarks,
            d_dog,
            d_coarse,
        },
        stats,
    ))
}
