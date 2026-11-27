use std::fmt;

use egraph_core::{CoreError, CoreResult, Scalar};

use crate::csr_matrix::CsrMatrix;

/// Matrix-free square linear operator $A : \mathbb{R}^n \to \mathbb{R}^n$.
///
/// Implemented by [`CsrMatrix`], closure adapters ([`FnOperator`]), and custom
/// matrix-free operators (such as surface Laplacians, shifted operators, or
/// Krylov preconditioners).
pub trait LinearOperator<S: Scalar = f64> {
    /// Returns the square dimension $n$ of the operator domain and codomain.
    fn dim(&self) -> usize;

    /// Computes $y = A x$, validating that `x.len() == self.dim()` and `y.len() == self.dim()`.
    fn apply(&mut self, x: &[S], y: &mut [S]) -> CoreResult<()>;
}

impl<S: Scalar, T: LinearOperator<S> + ?Sized> LinearOperator<S> for &mut T {
    #[inline]
    fn dim(&self) -> usize {
        (**self).dim()
    }

    #[inline]
    fn apply(&mut self, x: &[S], y: &mut [S]) -> CoreResult<()> {
        (**self).apply(x, y)
    }
}

impl<S: Scalar, T: LinearOperator<S> + ?Sized> LinearOperator<S> for Box<T> {
    #[inline]
    fn dim(&self) -> usize {
        (**self).dim()
    }

    #[inline]
    fn apply(&mut self, x: &[S], y: &mut [S]) -> CoreResult<()> {
        (**self).apply(x, y)
    }
}

impl LinearOperator<f64> for CsrMatrix {
    #[inline]
    fn dim(&self) -> usize {
        self.num_nodes()
    }

    #[inline]
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> CoreResult<()> {
        self.spmv(x, y)
    }
}

impl LinearOperator<f64> for &CsrMatrix {
    #[inline]
    fn dim(&self) -> usize {
        self.num_nodes()
    }

    #[inline]
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> CoreResult<()> {
        self.spmv(x, y)
    }
}

/// Adapts a closure `FnMut(&[S], &mut [S]) -> CoreResult<()>` with fixed dimension
/// into a [`LinearOperator`].
#[derive(Clone)]
pub struct FnOperator<F> {
    dim: usize,
    apply_fn: F,
}

impl<F> FnOperator<F> {
    /// Creates a new closure-backed linear operator of dimension `dim`.
    pub fn new(dim: usize, apply_fn: F) -> Self {
        Self { dim, apply_fn }
    }

    /// Returns the square operator dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<F> fmt::Debug for FnOperator<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnOperator")
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl<S: Scalar, F> LinearOperator<S> for FnOperator<F>
where
    F: FnMut(&[S], &mut [S]) -> CoreResult<()>,
{
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn apply(&mut self, x: &[S], y: &mut [S]) -> CoreResult<()> {
        if x.len() != self.dim {
            return Err(CoreError::dimension_mismatch(
                self.dim,
                x.len(),
                "FnOperator::apply::x",
            ));
        }
        if y.len() != self.dim {
            return Err(CoreError::dimension_mismatch(
                self.dim,
                y.len(),
                "FnOperator::apply::y",
            ));
        }
        (self.apply_fn)(x, y)
    }
}

/// Internal shifted-negated operator $M x = \sigma x - A x$ used to map smallest
/// eigenvalues of $A$ to largest eigenvalues of $M$.
pub(crate) struct ShiftedOperator<'a, O: LinearOperator<f64> + ?Sized> {
    op: &'a mut O,
    shift: f64,
    tmp: Vec<f64>,
}

impl<'a, O: LinearOperator<f64> + ?Sized> ShiftedOperator<'a, O> {
    pub(crate) fn new(op: &'a mut O, shift: f64) -> Self {
        let n = op.dim();
        Self {
            op,
            shift,
            tmp: vec![0.0_f64; n],
        }
    }
}

impl<O: LinearOperator<f64> + ?Sized> LinearOperator<f64> for ShiftedOperator<'_, O> {
    #[inline]
    fn dim(&self) -> usize {
        self.op.dim()
    }

    #[inline]
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> CoreResult<()> {
        let n = self.op.dim();
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "ShiftedOperator::apply::x",
            ));
        }
        if y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                y.len(),
                "ShiftedOperator::apply::y",
            ));
        }
        self.op.apply(x, &mut self.tmp)?;
        for i in 0..n {
            y[i] = self.shift * x[i] - self.tmp[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_operator_csr_fn_and_blankets() {
        let mut mat = CsrMatrix::from_csr(
            3,
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, 3.0],
        )
        .unwrap();
        let x = [1.0, 2.0, -1.0];
        let mut y_csr = [0.0; 3];
        mat.apply(&x, &mut y_csr).unwrap();
        assert_eq!(y_csr, [0.0, 3.0, -3.0]);

        let mut dyn_ref: &mut dyn LinearOperator<f64> = &mut mat;
        let mut y_ref = [0.0; 3];
        (&mut dyn_ref).apply(&x, &mut y_ref).unwrap();
        assert_eq!(y_ref, y_csr);

        let mut boxed: Box<dyn LinearOperator<f64>> =
            Box::new(FnOperator::new(3, |vx: &[f64], vy: &mut [f64]| {
                vy[0] = 2.0 * vx[0] - vx[1];
                vy[1] = -vx[0] + 2.0 * vx[1];
                vy[2] = 3.0 * vx[2];
                Ok(())
            }));
        let mut y_box = [0.0; 3];
        boxed.apply(&x, &mut y_box).unwrap();
        assert_eq!(y_box, y_csr);
        assert_eq!(boxed.dim(), 3);

        // Dimension mismatch validation on FnOperator
        let mut bad_out = [0.0; 2];
        assert!(boxed.apply(&x, &mut bad_out).is_err());
        assert!(boxed.apply(&[1.0, 2.0], &mut y_box).is_err());

        // Immutable &CsrMatrix adapter
        let mut mat_ref: &CsrMatrix = &mat;
        let mut y_imm = [0.0; 3];
        mat_ref.apply(&x, &mut y_imm).unwrap();
        assert_eq!(y_imm, y_csr);
    }
}
