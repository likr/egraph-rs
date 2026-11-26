use egraph_core::{CoreError, CoreResult};

use crate::csr_matrix::CsrMatrix;

use super::coefficients::{CHEBYSHEV_TAIL_L1_TOL, effective_chebyshev_len};
use super::workspace_kernels::{CHEBYSHEV_TILE_COLS, validate_csr_indices_in_bounds};

/// Zero-allocation-per-call tiled workspace (`t_prev`, `t_curr`, `t_next`, and L2-resident
/// 16-column cache-line tile accumulators) for fused Chebyshev SpMV / SpMM recurrences and JL embeddings.
#[derive(Debug, Clone, Default)]
pub struct ChebyshevWorkspace {
    pub(super) t_prev: Vec<f64>,
    pub(super) t_curr: Vec<f64>,
    pub(super) t_next: Vec<f64>,
    pub(super) out_a_tile: Vec<f64>,
    pub(super) out_b_tile: Vec<f64>,
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
    pub(super) fn ensure_len(&mut self, len: usize) {
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
    pub(super) fn ensure_tile_out(&mut self, len: usize, need_b: bool) {
        if self.out_a_tile.len() < len {
            self.out_a_tile.resize(len, 0.0);
        }
        if need_b && self.out_b_tile.len() < len {
            self.out_b_tile.resize(len, 0.0);
        }
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
}
