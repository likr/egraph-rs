use super::matrix::CsrMatrix;

/// Computes the smoothed prolongator $P = (I - \omega D^{-1} A) P_0$
pub fn smoothed_prolongator<S>(a: &CsrMatrix<S>, p0: &CsrMatrix<S>, omega: S) -> CsrMatrix<S>
where
    S: Copy
        + num_traits::Float
        + num_traits::Zero
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Default
        + std::cmp::PartialOrd,
{
    // Compute D^{-1} A
    let mut d_inv_a = CsrMatrix::new(a.rows, a.cols);
    d_inv_a.indptr = a.indptr.clone();
    d_inv_a.indices = a.indices.clone();
    d_inv_a.data = vec![S::zero(); a.data.len()];

    let diag = a.diagonal();

    for i in 0..a.rows {
        let d_inv = if diag[i] != S::zero() {
            S::one() / diag[i]
        } else {
            S::zero()
        };

        for j_ptr in a.indptr[i]..a.indptr[i + 1] {
            d_inv_a.data[j_ptr] = a.data[j_ptr] * d_inv;
        }
    }

    // Compute I - \omega D^{-1} A
    // Since we only need to multiply it by P0, we can do it directly:
    // P = P0 - \omega (D^{-1} A) P0
    // So P = P0 - ( \omega D^{-1} A ) * P0

    // Scale D^{-1} A by \omega
    for d in &mut d_inv_a.data {
        *d = *d * omega;
    }

    let scaled_d_inv_a_p0 = d_inv_a.mul_csr(p0);

    // Now P = P0 - scaled_d_inv_a_p0
    // Both P0 and scaled_d_inv_a_p0 have the same dimensions (rows = a.rows, cols = p0.cols)
    // We can add them up.

    let mut indptr = vec![0; p0.rows + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    let mut row_marker = vec![-1_isize; p0.cols];
    let mut row_values = vec![S::zero(); p0.cols];

    for i in 0..p0.rows {
        let mut row_indices = Vec::new();

        // Add P0
        for j_ptr in p0.indptr[i]..p0.indptr[i + 1] {
            let j = p0.indices[j_ptr];
            if row_marker[j] != i as isize {
                row_marker[j] = i as isize;
                row_values[j] = p0.data[j_ptr];
                row_indices.push(j);
            } else {
                row_values[j] += p0.data[j_ptr];
            }
        }

        // Subtract scaled_d_inv_a_p0
        for j_ptr in scaled_d_inv_a_p0.indptr[i]..scaled_d_inv_a_p0.indptr[i + 1] {
            let j = scaled_d_inv_a_p0.indices[j_ptr];
            if row_marker[j] != i as isize {
                row_marker[j] = i as isize;
                row_values[j] = -scaled_d_inv_a_p0.data[j_ptr];
                row_indices.push(j);
            } else {
                row_values[j] -= scaled_d_inv_a_p0.data[j_ptr];
            }
        }

        row_indices.sort_unstable();
        for &j in &row_indices {
            let val = row_values[j];
            if val.abs() > S::from(1e-12).unwrap() {
                // filter tiny values
                indices.push(j);
                data.push(val);
            }
        }
        indptr[i + 1] = indices.len();
    }

    CsrMatrix {
        rows: p0.rows,
        cols: p0.cols,
        indptr,
        indices,
        data,
    }
}

/// Estimates the spectral radius of $D^{-1} A$ using Gershgorin circle theorem.
pub fn estimate_spectral_radius<S>(a: &CsrMatrix<S>) -> S
where
    S: Copy
        + num_traits::Float
        + num_traits::Zero
        + std::ops::AddAssign
        + Default
        + std::cmp::PartialOrd,
{
    let diag = a.diagonal();
    let mut max_radius = S::zero();

    for i in 0..a.rows {
        let d = diag[i];
        if d == S::zero() {
            continue;
        }
        let d_inv = S::one() / d;
        let mut row_sum = S::zero();
        for j_ptr in a.indptr[i]..a.indptr[i + 1] {
            row_sum += (a.data[j_ptr] * d_inv).abs();
        }
        if row_sum > max_radius {
            max_radius = row_sum;
        }
    }

    // In many cases for Laplacians, max_radius is bounded by 2.0.
    // Gershgorin circle theorem says eigenvalues are in sum of absolute values of row entries.
    max_radius
}
