use egraph_core::{CoreError, CoreResult, DeterministicRng};

use crate::csr_matrix::CsrMatrix;

use super::coefficients::{
    adaptive_chebyshev_degree_scaled, compute_chebyshev_bandpass_coefficients,
    project_psd_spectral_filter,
};
use super::workspace::ChebyshevWorkspace;

impl ChebyshevWorkspace {
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
