use super::matrix::CsrMatrix;
use ndarray::Array1;

/// Weighted Jacobi smoother
/// Performs `iterations` sweeps of weighted Jacobi on `A x = b`
pub fn jacobi_smooth<S>(
    a: &CsrMatrix<S>,
    b: &Array1<S>,
    x: &mut Array1<S>,
    omega: S,
    iterations: usize,
) where
    S: Copy
        + num_traits::Float
        + num_traits::Zero
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Default,
{
    let n = a.rows;
    let diag = a.diagonal();
    let mut x_new = Array1::zeros(n);
    let mut residual = Array1::zeros(n);

    for _ in 0..iterations {
        // compute residual = b - A*x
        a.multiply_into(x, &mut residual);
        for i in 0..n {
            residual[i] = b[i] - residual[i];
        }

        // x_new = x + omega * D^{-1} * residual
        for i in 0..n {
            let d = diag[i];
            let inv_d = if d != S::zero() {
                S::one() / d
            } else {
                S::zero()
            };
            x_new[i] = x[i] + omega * inv_d * residual[i];
        }

        // update x
        x.assign(&x_new);
    }
}
