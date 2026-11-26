use ndarray::prelude::*;
use petgraph_drawing::DrawingValue;

/// Computes the optimal step length (alpha) in the conjugate gradient method.
///
/// The line search finds the value of alpha that minimizes the function value
/// when moving in the direction d.
///
/// # Arguments
///
/// * `a` - The coefficient matrix
/// * `dx` - The gradient vector
/// * `d` - The search direction
///
/// # Returns
///
/// The optimal step length alpha
fn line_search<S: DrawingValue>(a: &Array2<S>, dx: &Array1<S>, d: &Array1<S>) -> S {
    let n = dx.len();
    let mut alpha = -d.dot(dx);
    let mut s = S::zero();
    for i in 0..n {
        for j in 0..n {
            s += d[i] * d[j] * a[[i, j]];
        }
    }
    alpha /= s;
    alpha
}

/// Computes the gradient (delta_f) of the quadratic function f(x) = (1/2)x^T A x - b^T x.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The vector b
/// * `x` - The current position x
/// * `dx` - Output parameter where the gradient will be stored
fn delta_f<S: DrawingValue>(a: &Array2<S>, b: &Array1<S>, x: &Array1<S>, dx: &mut Array1<S>) {
    let n = b.len();
    for i in 0..n {
        dx[i] = S::zero();
        for j in 0..n {
            dx[i] += a[[i, j]] * x[j];
        }
        dx[i] -= b[i];
    }
}

/// Solves a system of linear equations Ax = b using the conjugate gradient method.
///
/// The conjugate gradient method is an iterative algorithm for solving systems
/// of the form Ax = b where A is a symmetric positive-definite matrix.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
/// * `x` - The initial guess for x, which will be updated with the solution
/// * `epsilon` - The convergence threshold (algorithm stops when the residual norm is less than this value)
pub fn conjugate_gradient<S: DrawingValue>(
    a: &Array2<S>,
    b: &Array1<S>,
    x: &mut Array1<S>,
    epsilon: S,
) {
    let n = b.len();
    if n == 0 {
        return;
    }
    let mut dx = Array1::zeros(n);
    let mut d = Array1::zeros(n);
    delta_f(a, b, x, &mut dx);
    for i in 0..n {
        d[i] = -dx[i];
    }
    let mut dx_norm0 = dx.dot(&dx);
    // Exit early before line_search if the residual is already zero/converged (avoids 0/0 division on singular/exact inputs).
    if !dx_norm0.is_finite() || dx_norm0 < epsilon || dx_norm0 == S::zero() {
        return;
    }
    for _ in 0..n {
        let alpha = line_search(a, &dx, &d);
        if !alpha.is_finite() || (0..n).any(|i| !(x[i] + alpha * d[i]).is_finite()) {
            break;
        }
        for i in 0..n {
            x[i] += alpha * d[i];
        }
        delta_f(a, b, x, &mut dx);
        let dx_norm = dx.dot(&dx);
        if !dx_norm.is_finite() || dx_norm < epsilon || dx_norm0 == S::zero() {
            break;
        }
        let beta = dx_norm / dx_norm0;
        if !beta.is_finite() {
            break;
        }
        dx_norm0 = dx_norm;
        for i in 0..n {
            d[i] = beta * d[i] - dx[i];
        }
    }
}
