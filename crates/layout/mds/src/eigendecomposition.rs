use ndarray::prelude::*;

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity measures the cosine of the angle between two vectors,
/// providing a value between -1 and 1 that indicates how similar their directions are.
/// A value of 1 means the vectors are identical in direction, 0 means they are orthogonal,
/// and -1 means they are in opposite directions.
///
/// # Parameters
///
/// * `a`: First vector
/// * `b`: Second vector
///
/// # Returns
///
/// The cosine similarity value between the two vectors
fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let ab = a.dot(b);
    let aa = a.dot(a);
    let bb = b.dot(b);
    ab / (aa * bb).sqrt()
}

/// Performs power iteration to find the dominant eigenvector and eigenvalue of a matrix.
///
/// Power iteration is an algorithm for computing the dominant eigenvalue and corresponding
/// eigenvector of a matrix. It works by repeatedly multiplying an initial vector by the
/// matrix and normalizing the result, which converges to the eigenvector corresponding
/// to the largest eigenvalue (in absolute value).
///
/// # Parameters
///
/// * `a`: The input matrix
/// * `eps`: Convergence threshold - stops iterating when the cosine similarity between
///          consecutive iterations is close enough to 1
///
/// # Returns
///
/// A tuple containing:
/// - The dominant eigenvalue
/// - The corresponding normalized eigenvector
fn power_iteration(a: &Array2<f32>, eps: f32) -> (f32, Array1<f32>) {
    let n = a.shape()[0];
    let mut x = Array1::from_elem(n, 1. / n as f32);
    let mut x_next = a.dot(&x);
    for _ in 0..10 {
        if 1. - cos(&x_next, &x) < eps {
            break;
        }
        x_next /= x_next.dot(&x_next).sqrt();
        x = x_next;
        x_next = a.dot(&x);
    }
    let e = x_next.dot(&x_next) / x_next.dot(&x);
    x_next /= x_next.dot(&x_next).sqrt();
    (e, x_next)
}

/// Performs eigendecomposition of a matrix to find the top k eigenvalues and eigenvectors.
///
/// This implementation uses a deflation method based on power iteration:
/// 1. Find the dominant eigenvector/eigenvalue using power iteration
/// 2. Deflate the matrix by removing the component in the direction of the found eigenvector
/// 3. Repeat steps 1-2 to find the next eigenvector/eigenvalue
///
/// This approach is suitable for MDS where we only need the top k eigenvalues and
/// eigenvectors, rather than the full decomposition.
///
/// # Parameters
///
/// * `a`: The input matrix to decompose
/// * `k`: The number of eigenvalues/eigenvectors to compute
/// * `eps`: Convergence threshold for the power iteration
///
/// # Returns
///
/// A tuple containing:
/// - An array of the k largest eigenvalues
/// - A matrix where each column is the corresponding eigenvector
pub fn eigendecomposition(a: &Array2<f32>, k: usize, eps: f32) -> (Array1<f32>, Array2<f32>) {
    let n = a.shape()[0];
    let mut b = a.clone();
    let mut e = Array1::zeros(k);
    let mut v = Array2::zeros((n, k));
    let (ei, vi) = power_iteration(&b, eps);
    e[0] = ei;
    v.slice_mut(s![.., 0]).assign(&vi);
    for i in 1..k {
        for r in 0..n {
            for c in 0..n {
                b[[r, c]] -= e[i - 1] * v[[r, i - 1]] * v[[c, i - 1]];
            }
        }
        let (ei, vi) = power_iteration(&b, eps);
        e[i] = ei;
        v.slice_mut(s![.., i]).assign(&vi);
    }
    (e, v)
}
