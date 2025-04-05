use ndarray::prelude::*;

/// Applies double centering transformation to a squared distance matrix.
///
/// Double centering is a mathematical operation that transforms a squared distance matrix
/// into a matrix of dot products (also known as a Gram matrix or a kernel matrix).
/// This is a key step in the MDS algorithm, as it allows working with inner products
/// rather than distances, which enables the use of eigendecomposition to find the
/// optimal configuration of points.
///
/// The double centering operation is defined as:
/// B = -0.5 * (I - 1/n * 11ᵀ) * D * (I - 1/n * 11ᵀ)
///
/// Where:
/// - D is the squared distance matrix
/// - I is the identity matrix
/// - 1 is a vector of ones
/// - n is the number of points
///
/// # Parameters
///
/// * `delta`: The squared distance matrix to center
///
/// # Returns
///
/// The double-centered matrix (kernel matrix)
pub fn double_centering(delta: &Array2<f32>) -> Array2<f32> {
    let n = delta.shape()[0];
    let k = delta.shape()[1];
    let sum_col = delta.mean_axis(Axis(1)).unwrap();
    let sum_row = delta.mean_axis(Axis(0)).unwrap();
    let sum_all = sum_col.mean().unwrap();
    let mut c = Array::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            c[[i, j]] = (sum_col[i] + sum_row[j] - delta[[i, j]] - sum_all) / 2.;
        }
    }
    c
}
