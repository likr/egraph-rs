use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use crate::csr_matrix::{spectral_moment_stats, symmetric_normalized_laplacian};
use crate::shortest_path::select_maxmin_sp_pivots;

use super::coefficients::{
    adaptive_chebyshev_degree, build_dog_terms, compute_chebyshev_bandpass_coefficients,
    project_psd_spectral_filter,
};
use super::dog_kernel::{ChebyshevDogConfig, ChebyshevDogKernel};
use super::workspace::ChebyshevWorkspace;
use super::workspace_kernels::CHEBYSHEV_TILE_COLS;

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
}
