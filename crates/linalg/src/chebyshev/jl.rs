use egraph_core::{CoreError, CoreResult, DeterministicRng};

use crate::csr_matrix::CsrMatrix;

use super::coefficients::{
    CHEBYSHEV_TAIL_L1_TOL, compute_chebyshev_sqrt_filter_coefficients, effective_chebyshev_len,
};
use super::workspace::ChebyshevWorkspace;
use super::workspace_kernels::{CHEBYSHEV_TILE_COLS, validate_csr_indices_in_bounds};

impl ChebyshevWorkspace {
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
}
