use egraph_core::{CoreError, CoreResult};

use crate::operator::LinearOperator;

use super::coefficients::{CHEBYSHEV_TAIL_L1_TOL, effective_chebyshev_len};
use super::workspace::ChebyshevWorkspace;

impl ChebyshevWorkspace {
    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\tilde{L}) v$ into `out` using a pre-shifted
    /// [`LinearOperator<f64>`] $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$.
    pub fn apply_vec_operator<O: LinearOperator<f64> + ?Sized>(
        &mut self,
        l_tilde: &mut O,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> CoreResult<()> {
        self.apply_batch_operator(l_tilde, coeffs, v, out, 1)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\tilde{L}) V$ into `out` for row-major `(n, ncols)`
    /// matrices using a pre-shifted [`LinearOperator<f64>`] $\tilde{L}$.
    pub fn apply_batch_operator<O: LinearOperator<f64> + ?Sized>(
        &mut self,
        l_tilde: &mut O,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
    ) -> CoreResult<()> {
        self.apply_batch_operator_impl(l_tilde, None, coeffs, v, out, ncols)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\frac{2}{\lambda_{\max}} L - I) v$ using an
    /// unshifted [`LinearOperator<f64>`] representing $L$.
    pub fn apply_vec_operator_with_lambda_max<O: LinearOperator<f64> + ?Sized>(
        &mut self,
        op: &mut O,
        lambda_max: f64,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> CoreResult<()> {
        self.apply_batch_operator_with_lambda_max(op, lambda_max, coeffs, v, out, 1)
    }

    /// Evaluates $\sum_{k=0}^{d} c_k T_k(\frac{2}{\lambda_{\max}} L - I) V$ for row-major
    /// `(n, ncols)` matrices using an unshifted [`LinearOperator<f64>`] representing $L$.
    pub fn apply_batch_operator_with_lambda_max<O: LinearOperator<f64> + ?Sized>(
        &mut self,
        op: &mut O,
        lambda_max: f64,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
    ) -> CoreResult<()> {
        if !lambda_max.is_finite() {
            return Err(CoreError::non_finite(
                "ChebyshevWorkspace::apply_batch_operator_with_lambda_max::lambda_max",
            ));
        }
        if lambda_max <= 0.0 {
            return Err(CoreError::invalid_input(
                "lambda_max must be strictly positive",
            ));
        }
        self.apply_batch_operator_impl(op, Some(2.0 / lambda_max), coeffs, v, out, ncols)
    }

    fn apply_batch_operator_impl<O: LinearOperator<f64> + ?Sized>(
        &mut self,
        op: &mut O,
        unshifted_scale: Option<f64>,
        coeffs: &[f64],
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
    ) -> CoreResult<()> {
        if coeffs.is_empty() {
            return Err(CoreError::invalid_input(
                "Chebyshev coefficients must not be empty",
            ));
        }
        let n = op.dim();
        let total = n
            .checked_mul(ncols)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace batch size overflow"))?;
        if v.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                v.len(),
                "ChebyshevWorkspace::apply_batch_operator::v",
            ));
        }
        if out.len() != total {
            return Err(CoreError::dimension_mismatch(
                total,
                out.len(),
                "ChebyshevWorkspace::apply_batch_operator::out",
            ));
        }
        if total == 0 {
            return Ok(());
        }

        let eff = effective_chebyshev_len(coeffs, CHEBYSHEV_TAIL_L1_TOL);
        let coeffs = &coeffs[..eff];
        let c0 = coeffs[0];

        self.ensure_len(n);
        self.ensure_tile_out(n, false);

        for c in 0..ncols {
            if ncols == 1 {
                self.t_prev[..n].copy_from_slice(v);
            } else {
                for row in 0..n {
                    self.t_prev[row] = v[row * ncols + c];
                }
            }
            for row in 0..n {
                self.out_a_tile[row] = c0 * self.t_prev[row];
            }

            if coeffs.len() > 1 {
                op.apply(&self.t_prev[..n], &mut self.t_curr[..n])?;
                if let Some(alpha) = unshifted_scale {
                    for i in 0..n {
                        self.t_curr[i] = alpha * self.t_curr[i] - self.t_prev[i];
                    }
                }
                let c1 = coeffs[1];
                if c1 != 0.0 {
                    for i in 0..n {
                        self.out_a_tile[i] += c1 * self.t_curr[i];
                    }
                }

                for &ck in &coeffs[2..] {
                    op.apply(&self.t_curr[..n], &mut self.t_next[..n])?;
                    if let Some(alpha) = unshifted_scale {
                        for i in 0..n {
                            self.t_next[i] =
                                2.0 * (alpha * self.t_next[i] - self.t_curr[i]) - self.t_prev[i];
                        }
                    } else {
                        for i in 0..n {
                            self.t_next[i] = 2.0 * self.t_next[i] - self.t_prev[i];
                        }
                    }
                    if ck != 0.0 {
                        for i in 0..n {
                            self.out_a_tile[i] += ck * self.t_next[i];
                        }
                    }
                    std::mem::swap(&mut self.t_prev, &mut self.t_curr);
                    std::mem::swap(&mut self.t_curr, &mut self.t_next);
                }
            }

            if ncols == 1 {
                out.copy_from_slice(&self.out_a_tile[..n]);
            } else {
                for row in 0..n {
                    out[row * ncols + c] = self.out_a_tile[row];
                }
            }
        }
        Ok(())
    }
}
