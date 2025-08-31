use ndarray::prelude::*;
use petgraph_drawing::DrawingValue;

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
fn cos<S: DrawingValue>(a: &Array1<S>, b: &Array1<S>) -> S {
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
fn power_iteration<S: DrawingValue>(a: &Array2<S>, eps: S) -> (S, Array1<S>) {
    let n = a.shape()[0];
    let mut x = Array1::from_elem(n, (S::one() / S::from_usize(n).unwrap()).sqrt());
    let mut x_next = a.dot(&x);

    // Check if the matrix is all zeros or very close to zero
    let norm_x_next = x_next.dot(&x_next).sqrt();
    if norm_x_next < (1e-10).into() {
        // For zero or near-zero matrices, return zero eigenvalue and a normalized vector
        return (S::zero(), x);
    }

    // Normalize x_next
    x_next /= norm_x_next;

    for _ in 0..10 {
        if S::one() - cos(&x_next, &x) < eps {
            break;
        }
        x = x_next.clone();
        x_next = a.dot(&x);

        // Check if x_next is close to zero
        let norm_x_next = x_next.dot(&x_next).sqrt();
        if norm_x_next < (1e-10).into() {
            // Ensure x is normalized before returning
            let norm_x = x.dot(&x).sqrt();
            if norm_x > S::zero() {
                x /= norm_x;
            }
            return (S::zero(), x);
        }

        // Normalize x_next
        x_next /= norm_x_next;
    }

    // Calculate eigenvalue using the Rayleigh quotient
    let x_dot_ax = x.dot(&a.dot(&x));
    let x_dot_x = x.dot(&x);

    // Avoid division by very small numbers
    let e = if x_dot_x < (1e-10).into() {
        S::zero()
    } else {
        x_dot_ax / x_dot_x
    };

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
pub fn eigendecomposition<S: DrawingValue>(
    a: &Array2<S>,
    k: usize,
    eps: S,
) -> (Array1<S>, Array2<S>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Test power_iteration with a simple identity matrix
    #[test]
    fn test_power_iteration_identity() {
        // Identity matrix should have eigenvalue 1 and any vector as eigenvector
        let a = Array2::<f64>::eye(3);
        let (eigenvalue, eigenvector) = power_iteration(&a, 1e-6);

        assert!(eigenvalue.is_finite(), "Eigenvalue should be finite");
        assert!(
            (eigenvalue - 1.0).abs() < 1e-3,
            "Eigenvalue should be close to 1.0"
        );

        // Check that eigenvector is normalized
        let norm = eigenvector.dot(&eigenvector);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "Eigenvector should be normalized"
        );

        // Check that Av = λv
        let av = a.dot(&eigenvector);
        for i in 0..eigenvector.len() {
            assert!(
                (av[i] - eigenvalue * eigenvector[i]).abs() < 1e-6,
                "Av should equal λv"
            );
        }
    }

    /// Test power_iteration with a matrix that has a clear dominant eigenvalue
    #[test]
    fn test_power_iteration_diagonal() {
        // Diagonal matrix with dominant eigenvalue 5.0
        let mut a = Array2::<f64>::zeros((3, 3));
        a[[0, 0]] = 5.0;
        a[[1, 1]] = 2.0;
        a[[2, 2]] = 1.0;

        let (eigenvalue, eigenvector) = power_iteration(&a, 1e-6);

        assert!(eigenvalue.is_finite(), "Eigenvalue should be finite");
        assert!(
            (eigenvalue - 5.0).abs() < 1e-3,
            "Eigenvalue should be close to 5.0"
        );

        // Check that eigenvector is normalized
        let norm = eigenvector.dot(&eigenvector);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "Eigenvector should be normalized"
        );

        // The eigenvector should be close to [1, 0, 0]
        assert!(
            eigenvector[0].abs() > 0.99,
            "First component should be dominant"
        );
        assert!(
            eigenvector[1].abs() < 0.01,
            "Second component should be close to 0"
        );
        assert!(
            eigenvector[2].abs() < 0.01,
            "Third component should be close to 0"
        );
    }

    /// Test power_iteration with a zero matrix
    #[test]
    fn test_power_iteration_zero_matrix() {
        // Zero matrix should have eigenvalue 0
        let a = Array2::<f64>::zeros((3, 3));
        let (eigenvalue, eigenvector) = power_iteration(&a, 1e-6);

        println!("Zero matrix eigenvalue: {}", eigenvalue);
        println!("Zero matrix eigenvector: {:?}", eigenvector);

        assert!(eigenvalue.is_finite(), "Eigenvalue should be finite");
        assert!(eigenvalue.abs() < 1e-6, "Eigenvalue should be close to 0.0");

        // Check that eigenvector is normalized
        let norm = eigenvector.dot(&eigenvector);
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Eigenvector should be normalized"
        );
    }

    /// Test power_iteration with a matrix that has all NaN values
    #[test]
    fn test_power_iteration_nan_matrix() {
        // Matrix with all NaN values
        let mut a = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                a[[i, j]] = f32::NAN;
            }
        }

        let (eigenvalue, eigenvector) = power_iteration(&a, 1e-6);

        println!("NaN matrix eigenvalue: {}", eigenvalue);
        println!("NaN matrix eigenvector: {:?}", eigenvector);

        // With the current implementation, we expect NaN results
        // This test documents the current behavior
        assert!(
            eigenvalue.is_nan(),
            "Eigenvalue is expected to be NaN with current implementation"
        );
    }

    /// Test power_iteration with a matrix similar to what's used in PivotMDS test case
    #[test]
    fn test_power_iteration_pivot_mds_like() {
        // Create a matrix similar to what might be generated in the PivotMDS test case
        // This is a small matrix with a specific structure that might cause issues
        let mut a = Array2::<f64>::zeros((2, 2));
        a[[0, 0]] = 0.0;
        a[[0, 1]] = 0.0;
        a[[1, 0]] = 0.0;
        a[[1, 1]] = 0.0;

        let (eigenvalue, eigenvector) = power_iteration(&a, 1e-6);

        println!("PivotMDS-like matrix: {:?}", a);
        println!("PivotMDS-like eigenvalue: {}", eigenvalue);
        println!("PivotMDS-like eigenvector: {:?}", eigenvector);

        // With the current implementation, we expect a zero eigenvalue
        assert!(eigenvalue.is_finite(), "Eigenvalue should be finite");
    }
}
