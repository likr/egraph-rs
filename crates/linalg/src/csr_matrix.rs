//! Compressed Sparse Row (`CsrMatrix`) representation, graph Laplacians, and $O(|E|)$ spectral moments.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use crate::eigen::symmetric_eigen_jacobi;

/// Default number of deterministic power-iteration steps in [`CsrMatrix::estimate_lambda_max`].
pub const DEFAULT_SPECTRUM_POWER_ITERS: usize = 10;
/// Default multiplicative safety margin in [`CsrMatrix::estimate_lambda_max`].
pub const DEFAULT_SPECTRUM_SAFETY_FACTOR: f64 = 1.02;
/// Default number of Lanczos steps in [`CsrMatrix::estimate_lambda_max_lanczos`].
pub const DEFAULT_SPECTRUM_LANCZOS_STEPS: usize = 16;
/// Default multiplicative safety margin in [`CsrMatrix::estimate_lambda_max_lanczos`].
pub const DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR: f64 = 1.05;

/// Supported graph Laplacian variants for sparse CSR construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaplacianKind {
    /// Combinatorial (unnormalized) Laplacian $L = D - A$.
    Combinatorial,
    /// Symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$.
    SymmetricNormalized,
    /// Random-walk normalized Laplacian $L_{\mathrm{rw}} = D^{-1} (D - A)$.
    RandomWalk,
}

/// Exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMomentStats {
    /// Exact spectral variance $\sigma_\lambda^2 = \frac{1}{N}\|L_{\mathrm{sym}}\|_F^2 - 1 = \frac{2}{N}\sum_{(u,v)\in E}\frac{1}{d_u d_v}$.
    pub sigma_lambda_sq: f64,
    /// Clamped spectral standard deviation $\sigma_\lambda = \sqrt{\max(\sigma_\lambda^2, 10^{-6})}$.
    pub sigma_lambda: f64,
    /// Coefficient of variation of node degrees ($\mathrm{std}(d) / \mathrm{mean}(d)$).
    pub deg_cv: f64,
    /// Mean unweighted simple degree across all vertices.
    pub mean_degree: f64,
    /// Fraction of vertices with degree equal to 1.
    pub leaf_fraction: f64,
}

/// Double-precision square Compressed Sparse Row matrix with sorted column indices per row.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    num_nodes: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f64>,
    diagonal: Vec<f64>,
}

impl CsrMatrix {
    /// Constructs a [`CsrMatrix`] from raw CSR components, validating structure, finiteness,
    /// and strictly ascending column indices per row.
    pub fn from_csr(
        num_nodes: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        values: Vec<f64>,
    ) -> CoreResult<Self> {
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        if indptr.len() != num_nodes + 1 {
            return Err(CoreError::dimension_mismatch(
                num_nodes + 1,
                indptr.len(),
                "CsrMatrix::from_csr::indptr",
            ));
        }
        if indices.len() != values.len() {
            return Err(CoreError::dimension_mismatch(
                indices.len(),
                values.len(),
                "CsrMatrix::from_csr::values",
            ));
        }
        if indptr[0] != 0 {
            return Err(CoreError::invalid_csr(format!(
                "indptr[0] must be 0, got {}",
                indptr[0]
            )));
        }
        for (row, win) in indptr.windows(2).enumerate() {
            if win[0] > win[1] {
                return Err(CoreError::invalid_csr(format!(
                    "indptr is not monotonic at row {row}: {} > {}",
                    win[0], win[1]
                )));
            }
        }
        let nnz = indptr[num_nodes] as usize;
        if nnz != indices.len() {
            return Err(CoreError::invalid_csr(format!(
                "indptr[num_nodes] ({nnz}) != indices.len() ({})",
                indices.len()
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("CsrMatrix::from_csr::values"));
        }

        let mut diagonal = vec![0.0_f64; num_nodes];
        for row in 0..num_nodes {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut prev_col: Option<u32> = None;
            for pos in start..end {
                let col = indices[pos];
                if (col as usize) >= num_nodes {
                    return Err(CoreError::index_out_of_bounds(
                        col as usize,
                        num_nodes,
                        "CsrMatrix::from_csr::indices",
                    ));
                }
                if let Some(prev) = prev_col
                    && col <= prev
                {
                    return Err(CoreError::invalid_csr(format!(
                        "CSR row {row} column indices must be strictly ascending: {prev} >= {col}"
                    )));
                }
                prev_col = Some(col);
                if col as usize == row {
                    diagonal[row] = values[pos];
                }
            }
        }

        Ok(Self {
            num_nodes,
            indptr,
            indices,
            values,
            diagonal,
        })
    }

    /// Constructs the $n \times n$ identity matrix $I_n$.
    pub fn identity(num_nodes: usize) -> CoreResult<Self> {
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        let mut indptr = Vec::with_capacity(num_nodes + 1);
        let mut indices = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            indptr.push(i as u32);
            indices.push(i as u32);
        }
        indptr.push(num_nodes as u32);
        Ok(Self {
            num_nodes,
            indptr,
            indices,
            values: vec![1.0; num_nodes],
            diagonal: vec![1.0; num_nodes],
        })
    }

    /// Returns the number of rows/columns $n$.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns the CSR row pointer slice of length `num_nodes + 1`.
    #[inline]
    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    /// Returns the sorted column index slice of length `nnz`.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Returns the non-zero value slice aligned with `indices()`.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the cached main diagonal entries of length `num_nodes`.
    #[inline]
    pub fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    /// Computes sparse matrix-vector multiplication $y = A x$.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) -> CoreResult<()> {
        if x.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                x.len(),
                "CsrMatrix::spmv::x",
            ));
        }
        if y.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                y.len(),
                "CsrMatrix::spmv::y",
            ));
        }
        if self.num_nodes == 0 {
            return Ok(());
        }

        for (row, y_out) in y.iter_mut().enumerate() {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let mut sum = 0.0_f64;
            for pos in start..end {
                let col = self.indices[pos] as usize;
                sum += self.values[pos] * x[col];
            }
            *y_out = sum;
        }
        Ok(())
    }

    /// Computes sparse matrix-matrix multiplication $Y = A X$ for row-major `(num_nodes, ncols)` matrices.
    pub fn spmm(&self, x: &[f64], y: &mut [f64], ncols: usize) -> CoreResult<()> {
        let expected = self
            .num_nodes
            .checked_mul(ncols)
            .ok_or_else(|| CoreError::invalid_input("CsrMatrix::spmm size overflow"))?;
        if x.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                x.len(),
                "CsrMatrix::spmm::x",
            ));
        }
        if y.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                y.len(),
                "CsrMatrix::spmm::y",
            ));
        }
        if self.num_nodes == 0 || ncols == 0 {
            return Ok(());
        }
        if ncols == 1 {
            return self.spmv(x, y);
        }

        for row in 0..self.num_nodes {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let y_row = &mut y[row * ncols..(row + 1) * ncols];
            y_row.fill(0.0);
            for pos in start..end {
                let col = self.indices[pos] as usize;
                let val = self.values[pos];
                let x_row = &x[col * ncols..(col + 1) * ncols];
                for c in 0..ncols {
                    y_row[c] += val * x_row[c];
                }
            }
        }
        Ok(())
    }

    /// Converts the sparse matrix to a flat row-major `num_nodes * num_nodes` dense buffer.
    pub fn to_dense(&self) -> CoreResult<Vec<f64>> {
        let total = self
            .num_nodes
            .checked_mul(self.num_nodes)
            .ok_or_else(|| CoreError::invalid_input("CsrMatrix::to_dense size overflow"))?;
        let mut dense = vec![0.0_f64; total];
        for row in 0..self.num_nodes {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_off = row * self.num_nodes;
            for pos in start..end {
                let col = self.indices[pos] as usize;
                dense[row_off + col] = self.values[pos];
            }
        }
        Ok(dense)
    }

    /// Constructs the shifted and scaled sparse matrix `scale * A + shift * I`.
    ///
    /// Exact zero diagonal entries are omitted from the CSR storage while preserving
    /// accurate entries in [`CsrMatrix::diagonal`].
    pub fn scale_and_shift(&self, scale: f64, shift: f64) -> CoreResult<Self> {
        if !scale.is_finite() {
            return Err(CoreError::non_finite("CsrMatrix::scale_and_shift::scale"));
        }
        if !shift.is_finite() {
            return Err(CoreError::non_finite("CsrMatrix::scale_and_shift::shift"));
        }
        let n = self.num_nodes;
        let mut out_indptr = Vec::with_capacity(n + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::with_capacity(self.indices.len() + n);
        let mut out_values = Vec::with_capacity(self.values.len() + n);
        let mut out_diag = Vec::with_capacity(n);

        for row in 0..n {
            let row_u32 = row as u32;
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let mut diag_handled = false;
            let mut diag_val = shift;

            for pos in start..end {
                let col = self.indices[pos];
                if !diag_handled && col > row_u32 {
                    diag_handled = true;
                    if diag_val != 0.0 {
                        out_indices.push(row_u32);
                        out_values.push(diag_val);
                    }
                }
                if col == row_u32 {
                    diag_handled = true;
                    diag_val = scale * self.values[pos] + shift;
                    if diag_val != 0.0 {
                        out_indices.push(row_u32);
                        out_values.push(diag_val);
                    }
                } else {
                    let v = scale * self.values[pos];
                    if v != 0.0 {
                        out_indices.push(col);
                        out_values.push(v);
                    }
                }
            }

            if !diag_handled && diag_val != 0.0 {
                out_indices.push(row_u32);
                out_values.push(diag_val);
            }

            out_diag.push(diag_val);
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("scaled matrix nnz exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(Self {
            num_nodes: n,
            indptr: out_indptr,
            indices: out_indices,
            values: out_values,
            diagonal: out_diag,
        })
    }

    /// Returns `true` if $|A_{ij} - A_{ji}| \le \mathrm{tol}$ for all entries `(i, j)`.
    ///
    /// Returns `false` if `tol` is non-finite or negative.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !tol.is_finite() || tol < 0.0 {
            return false;
        }
        let n = self.num_nodes;
        for row in 0..n {
            let row_u32 = row as u32;
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for pos in start..end {
                let col = self.indices[pos] as usize;
                if col == row {
                    continue;
                }
                let val = self.values[pos];
                let t_start = self.indptr[col] as usize;
                let t_end = self.indptr[col + 1] as usize;
                let t_cols = &self.indices[t_start..t_end];
                let rev = match t_cols.binary_search(&row_u32) {
                    Ok(idx) => self.values[t_start + idx],
                    Err(_) => 0.0,
                };
                if (val - rev).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the explicit symmetric part $\frac{1}{2}(A + A^T)$ in canonical sorted CSR form,
    /// pruning exact zero entries from CSR storage while keeping [`CsrMatrix::diagonal`] accurate.
    pub fn symmetrize(&self) -> CoreResult<Self> {
        let n = self.num_nodes;
        if n == 0 {
            return Ok(Self {
                num_nodes: 0,
                indptr: vec![0],
                indices: Vec::new(),
                values: Vec::new(),
                diagonal: Vec::new(),
            });
        }

        // Compute A^T in O(n + nnz) without HashMap; scanning rows 0..n in order yields
        // strictly ascending column indices in every row of A^T.
        let nnz = self.indices.len();
        let mut col_counts = vec![0u32; n];
        for &col in &self.indices {
            col_counts[col as usize] += 1;
        }
        let mut t_indptr = Vec::with_capacity(n + 1);
        t_indptr.push(0u32);
        let mut running = 0u32;
        for &cnt in &col_counts {
            running = running
                .checked_add(cnt)
                .ok_or_else(|| CoreError::invalid_csr("transpose nnz exceeds u32::MAX"))?;
            t_indptr.push(running);
        }
        let mut t_cursor = t_indptr[..n].to_vec();
        let mut t_indices = vec![0u32; nnz];
        let mut t_values = vec![0.0_f64; nnz];
        for row in 0..n {
            let row_u32 = row as u32;
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for pos in start..end {
                let col = self.indices[pos] as usize;
                let dst = t_cursor[col] as usize;
                t_indices[dst] = row_u32;
                t_values[dst] = self.values[pos];
                t_cursor[col] += 1;
            }
        }

        // Two-pointer merge of sorted row u of A and sorted row u of A^T.
        let mut out_indptr = Vec::with_capacity(n + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::with_capacity(nnz);
        let mut out_values = Vec::with_capacity(nnz);
        let mut out_diag = vec![0.0_f64; n];

        for row in 0..n {
            let row_u32 = row as u32;
            let mut p1 = self.indptr[row] as usize;
            let e1 = self.indptr[row + 1] as usize;
            let mut p2 = t_indptr[row] as usize;
            let e2 = t_indptr[row + 1] as usize;

            while p1 < e1 || p2 < e2 {
                let (col, avg) = if p2 >= e2 || (p1 < e1 && self.indices[p1] < t_indices[p2]) {
                    let c = self.indices[p1];
                    let v = 0.5 * self.values[p1];
                    p1 += 1;
                    (c, v)
                } else if p1 >= e1 || t_indices[p2] < self.indices[p1] {
                    let c = t_indices[p2];
                    let v = 0.5 * t_values[p2];
                    p2 += 1;
                    (c, v)
                } else {
                    let c = self.indices[p1];
                    let v = 0.5 * (self.values[p1] + t_values[p2]);
                    p1 += 1;
                    p2 += 1;
                    (c, v)
                };

                if col == row_u32 {
                    out_diag[row] = avg;
                }
                if avg != 0.0 {
                    out_indices.push(col);
                    out_values.push(avg);
                }
            }

            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("symmetrized matrix nnz exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(Self {
            num_nodes: n,
            indptr: out_indptr,
            indices: out_indices,
            values: out_values,
            diagonal: out_diag,
        })
    }

    /// Returns the matrix trace $\mathrm{tr}(A) = \sum_i A_{ii}$.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.diagonal.iter().copied().sum()
    }

    /// Returns the symmetric Gershgorin spectral upper bound
    /// $\max_i \left(|A_{ii}| + \sum_{j \neq i} |A_{ij}|\right)$.
    ///
    /// Returns `0.0` for empty (`num_nodes == 0`) or zero matrices.
    pub fn gershgorin_spectral_bound(&self) -> f64 {
        let n = self.num_nodes;
        if n == 0 {
            return 0.0;
        }
        let mut max_bound = 0.0_f64;
        for row in 0..n {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_u32 = row as u32;
            let mut row_abs_sum = 0.0_f64;
            let mut saw_diag = false;
            for pos in start..end {
                let abs_val = self.values[pos].abs();
                if !abs_val.is_finite() {
                    return f64::NAN;
                }
                if self.indices[pos] == row_u32 {
                    saw_diag = true;
                }
                row_abs_sum += abs_val;
            }
            if !saw_diag {
                let d_abs = self.diagonal[row].abs();
                if !d_abs.is_finite() {
                    return f64::NAN;
                }
                row_abs_sum += d_abs;
            }
            if row_abs_sum > max_bound {
                max_bound = row_abs_sum;
            }
        }
        max_bound
    }

    /// Estimates $\lambda_{\max}(A)$ for a symmetric [`CsrMatrix`] using deterministic
    /// Rademacher power iteration clamped by [`CsrMatrix::gershgorin_spectral_bound`],
    /// scaled by `safety_factor >= 1.0`.
    ///
    /// Returns `0.0` for empty or zero matrices, and `gershgorin_spectral_bound()` when `num_iters == 0`.
    pub fn estimate_lambda_max(
        &self,
        num_iters: usize,
        safety_factor: f64,
        seed: u64,
    ) -> CoreResult<f64> {
        if !safety_factor.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max::safety_factor",
            ));
        }
        if safety_factor < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "safety_factor must be >= 1.0, got {safety_factor}"
            )));
        }
        let gersh = self.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max::matrix",
            ));
        }
        let n = self.num_nodes;
        if n == 0 || gersh == 0.0 {
            return Ok(0.0);
        }
        if num_iters == 0 {
            return Ok(gersh);
        }
        if n == 1 {
            return Ok(self.diagonal[0].abs().min(gersh));
        }

        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut v = vec![0.0_f64; n];
        let mut w = vec![0.0_f64; n];

        let mut initialized = false;
        for _attempt in 0..8 {
            for vi in &mut v {
                *vi = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
            }
            let v_norm = scaled_norm2(&v);
            let inv_v = 1.0 / v_norm;
            for vi in &mut v {
                *vi *= inv_v;
            }
            self.spmv(&v, &mut w)?;
            if scaled_norm2(&w) > 0.0 {
                initialized = true;
                break;
            }
        }
        if !initialized {
            return Ok(gersh);
        }

        for _iter in 0..num_iters {
            self.spmv(&v, &mut w)?;
            let w_norm = scaled_norm2(&w);
            if !w_norm.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max::spmv",
                ));
            }
            if w_norm == 0.0 {
                return Ok(gersh);
            }
            let inv_w = 1.0 / w_norm;
            for (vi, &wi) in v.iter_mut().zip(w.iter()) {
                *vi = wi * inv_w;
            }
        }

        self.spmv(&v, &mut w)?;
        let rq = dot_slice(&v, &w);
        if !rq.is_finite() {
            return Err(CoreError::non_finite("CsrMatrix::estimate_lambda_max::rq"));
        }
        Ok((safety_factor * rq.abs()).min(gersh))
    }

    /// Estimates $\lambda_{\max}(A)$ for a symmetric [`CsrMatrix`] via deterministic
    /// `num_steps`-step Lanczos tridiagonalization with two-pass MGS reorthogonalization
    /// and [`symmetric_eigen_jacobi`] on the $k \times k$ tridiagonal matrix, clamped by
    /// [`CsrMatrix::gershgorin_spectral_bound`].
    ///
    /// Avoids power iteration's hub-spectrum underestimation while remaining $O(k \cdot \mathrm{nnz})$.
    pub fn estimate_lambda_max_lanczos(
        &self,
        num_steps: usize,
        safety_factor: f64,
        seed: u64,
    ) -> CoreResult<f64> {
        if !safety_factor.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max_lanczos::safety_factor",
            ));
        }
        if safety_factor < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "safety_factor must be >= 1.0, got {safety_factor}"
            )));
        }
        let gersh = self.gershgorin_spectral_bound();
        if !gersh.is_finite() {
            return Err(CoreError::non_finite(
                "CsrMatrix::estimate_lambda_max_lanczos::matrix",
            ));
        }
        let n = self.num_nodes;
        if n == 0 || gersh == 0.0 {
            return Ok(0.0);
        }
        if num_steps == 0 {
            return Ok(gersh);
        }
        if n == 1 {
            return Ok(self.diagonal[0].abs().min(gersh));
        }

        let k_max = num_steps.min(n);
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut q_vecs = vec![0.0_f64; k_max * n];
        for val in &mut q_vecs[..n] {
            *val = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
        }
        let q0_norm = scaled_norm2(&q_vecs[..n]);
        let inv_q0 = 1.0 / q0_norm;
        for val in &mut q_vecs[..n] {
            *val *= inv_q0;
        }

        let mut alphas = Vec::with_capacity(k_max);
        let mut betas = Vec::with_capacity(k_max.saturating_sub(1));
        let mut w = vec![0.0_f64; n];

        for j in 0..k_max {
            let q_j_start = j * n;
            self.spmv(&q_vecs[q_j_start..q_j_start + n], &mut w)?;

            if j > 0 {
                let beta_prev = betas[j - 1];
                if beta_prev != 0.0 {
                    let q_prev_start = (j - 1) * n;
                    for i in 0..n {
                        w[i] -= beta_prev * q_vecs[q_prev_start + i];
                    }
                }
            }

            let alpha_j = dot_slice(&q_vecs[q_j_start..q_j_start + n], &w);
            if !alpha_j.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max_lanczos::alpha",
                ));
            }
            alphas.push(alpha_j);

            if j + 1 == k_max {
                break;
            }

            // Two-pass Modified Gram-Schmidt reorthogonalization against all kept Lanczos vectors.
            for _pass in 0..2 {
                for prev in 0..=j {
                    let q_p = &q_vecs[prev * n..(prev + 1) * n];
                    let proj = dot_slice(q_p, &w);
                    for i in 0..n {
                        w[i] -= proj * q_p[i];
                    }
                }
            }

            let beta_j = scaled_norm2(&w);
            if !beta_j.is_finite() {
                return Err(CoreError::non_finite(
                    "CsrMatrix::estimate_lambda_max_lanczos::beta",
                ));
            }

            if beta_j <= 1e-14 * gersh {
                // Invariant Krylov subspace reached; restart with an orthogonal random vector.
                let mut found = false;
                for _attempt in 0..8 {
                    let next_start = (j + 1) * n;
                    for val in &mut q_vecs[next_start..next_start + n] {
                        *val = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
                    }
                    for _pass in 0..2 {
                        for prev in 0..=j {
                            let (left, right) = q_vecs.split_at_mut(next_start);
                            let q_p = &left[prev * n..(prev + 1) * n];
                            let q_next = &mut right[..n];
                            let proj = dot_slice(q_p, q_next);
                            for i in 0..n {
                                q_next[i] -= proj * q_p[i];
                            }
                        }
                    }
                    let nrm = scaled_norm2(&q_vecs[next_start..next_start + n]);
                    if nrm > 1e-12 {
                        let inv = 1.0 / nrm;
                        for val in &mut q_vecs[next_start..next_start + n] {
                            *val *= inv;
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                betas.push(0.0);
            } else {
                betas.push(beta_j);
                let inv_beta = 1.0 / beta_j;
                let next_start = (j + 1) * n;
                for i in 0..n {
                    q_vecs[next_start + i] = w[i] * inv_beta;
                }
            }
        }

        let m = alphas.len();
        let mut t_mat = vec![0.0_f64; m * m];
        for i in 0..m {
            t_mat[i * m + i] = alphas[i];
            if i + 1 < m {
                let b = betas[i];
                t_mat[i * m + (i + 1)] = b;
                t_mat[(i + 1) * m + i] = b;
            }
        }
        let (evals, _) = symmetric_eigen_jacobi(&t_mat, m, 64, 1e-12)?;
        let max_eval = evals.iter().fold(0.0_f64, |acc, &e| acc.max(e.abs()));
        Ok((safety_factor * max_eval).min(gersh))
    }
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn scaled_norm2(a: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for &x in a {
        let ax = x.abs();
        if !ax.is_finite() {
            return f64::NAN;
        }
        if ax > max_abs {
            max_abs = ax;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    if (1e-100..=1e100).contains(&max_abs) {
        return dot_slice(a, a).sqrt();
    }
    let sum_sq: f64 = a
        .iter()
        .map(|&x| {
            let s = x / max_abs;
            s * s
        })
        .sum();
    max_abs * sum_sq.sqrt()
}

/// Extracts canonical undirected simple adjacency rows `(col, weight)` and vertex degrees,
/// filtering self-loops and non-positive or non-finite edge weights.
fn canonical_undirected_adjacency<W: Scalar>(
    graph: &CsrGraph<W>,
    binarize: bool,
) -> (Vec<Vec<(u32, f64)>>, Vec<f64>) {
    let n = graph.num_nodes();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut rows = vec![Vec::new(); n];
    let weights_opt = graph.weights();
    let directed = graph.is_directed();

    for u in 0..n {
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos] as usize;
            if v == u || v >= n {
                continue;
            }
            let w = match weights_opt {
                Some(ws) => {
                    let wf = ws[pos].to_f64();
                    if !wf.is_finite() || wf <= 0.0 {
                        continue;
                    }
                    if binarize { 1.0 } else { wf }
                }
                None => 1.0,
            };
            rows[u].push((v as u32, w));
            if directed {
                rows[v].push((u as u32, w));
            }
        }
    }

    let mut degrees = vec![0.0_f64; n];
    for (u, row) in rows.iter_mut().enumerate() {
        row.sort_unstable_by_key(|&(v, _)| v);
        let mut write = 0usize;
        for read in 0..row.len() {
            let (v, w) = row[read];
            if write > 0 && row[write - 1].0 == v {
                row[write - 1].1 = row[write - 1].1.max(w);
            } else {
                row[write] = (v, w);
                write += 1;
            }
        }
        row.truncate(write);
        degrees[u] = row.iter().map(|&(_, w)| w).sum();
    }

    (rows, degrees)
}

/// Checks whether an undirected `CsrGraph` can be streamed in a single zero-bucket pass
/// (already sorted rows and strictly positive finite weights).
fn is_fast_undirected_clean<W: Scalar>(graph: &CsrGraph<W>) -> bool {
    if graph.is_directed() {
        return false;
    }
    if let Some(ws) = graph.weights()
        && ws.iter().any(|w| {
            let wf = w.to_f64();
            !wf.is_finite() || wf <= 0.0
        })
    {
        return false;
    }
    for u in 0..graph.num_nodes() {
        let row = graph.neighbors(u);
        if !row.windows(2).all(|win| win[0] < win[1]) {
            return false;
        }
    }
    true
}

/// Constructs the symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$
/// and inverse square-root degree vector `inv_sqrt_deg`.
///
/// Edge weights are filtered to strictly positive finite values (`> 0.0`); unweighted graphs use
/// unit strength `1.0`. Isolated vertices receive `inv_sqrt_deg[i] = 0.0` and `L_sym[i, i] = 1.0`.
pub fn symmetric_normalized_laplacian<W: Scalar>(graph: &CsrGraph<W>) -> (CsrMatrix, Vec<f64>) {
    let n = graph.num_nodes();
    if n == 0 {
        return (
            CsrMatrix {
                num_nodes: 0,
                indptr: vec![0],
                indices: Vec::new(),
                values: Vec::new(),
                diagonal: Vec::new(),
            },
            Vec::new(),
        );
    }

    if is_fast_undirected_clean(graph) {
        let mut degrees = vec![0.0_f64; n];
        let weights_opt = graph.weights();
        for (u, deg_u) in degrees.iter_mut().enumerate() {
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            *deg_u = match weights_opt {
                Some(ws) => ws[start..end].iter().map(|w| w.to_f64()).sum(),
                None => (end - start) as f64,
            };
        }
        let inv_sqrt_deg: Vec<f64> = degrees
            .iter()
            .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 0.0 })
            .collect();

        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u32);
        let mut indices = Vec::with_capacity(graph.nnz() + n);
        let mut values = Vec::with_capacity(graph.nnz() + n);

        for (u, &inv_u) in inv_sqrt_deg.iter().enumerate() {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut diag_inserted = false;

            for pos in start..end {
                let v = graph.indices()[pos];
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(1.0);
                    diag_inserted = true;
                }
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                let entry = -w * inv_u * inv_sqrt_deg[v as usize];
                indices.push(v);
                values.push(entry);
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(1.0);
            }
            indptr.push(indices.len() as u32);
        }

        let matrix = CsrMatrix {
            num_nodes: n,
            indptr,
            indices,
            values,
            diagonal: vec![1.0; n],
        };
        return (matrix, inv_sqrt_deg);
    }

    let (rows, degrees) = canonical_undirected_adjacency(graph, false);
    let inv_sqrt_deg: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 0.0 })
        .collect();

    let total_edges: usize = rows.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(total_edges + n);
    let mut values = Vec::with_capacity(total_edges + n);

    for (u, row) in rows.iter().enumerate() {
        let u_u32 = u as u32;
        let inv_u = inv_sqrt_deg[u];
        let mut diag_inserted = false;

        for &(v, w) in row {
            if !diag_inserted && v > u_u32 {
                indices.push(u_u32);
                values.push(1.0);
                diag_inserted = true;
            }
            indices.push(v);
            values.push(-w * inv_u * inv_sqrt_deg[v as usize]);
        }
        if !diag_inserted {
            indices.push(u_u32);
            values.push(1.0);
        }
        indptr.push(indices.len() as u32);
    }

    let matrix = CsrMatrix {
        num_nodes: n,
        indptr,
        indices,
        values,
        diagonal: vec![1.0; n],
    };
    (matrix, inv_sqrt_deg)
}

/// Constructs the combinatorial (unnormalized) graph Laplacian $L = D - A$.
pub fn combinatorial_laplacian<W: Scalar>(graph: &CsrGraph<W>) -> CsrMatrix {
    let n = graph.num_nodes();
    if n == 0 {
        return CsrMatrix {
            num_nodes: 0,
            indptr: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
            diagonal: Vec::new(),
        };
    }

    if is_fast_undirected_clean(graph) {
        let weights_opt = graph.weights();
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u32);
        let mut indices = Vec::with_capacity(graph.nnz() + n);
        let mut values = Vec::with_capacity(graph.nnz() + n);
        let mut diagonal = Vec::with_capacity(n);

        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let deg_u: f64 = match weights_opt {
                Some(ws) => ws[start..end].iter().map(|w| w.to_f64()).sum(),
                None => (end - start) as f64,
            };
            diagonal.push(deg_u);

            let mut diag_inserted = deg_u == 0.0;
            for pos in start..end {
                let v = graph.indices()[pos];
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(deg_u);
                    diag_inserted = true;
                }
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                indices.push(v);
                values.push(-w);
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(deg_u);
            }
            indptr.push(indices.len() as u32);
        }

        return CsrMatrix {
            num_nodes: n,
            indptr,
            indices,
            values,
            diagonal,
        };
    }

    let (rows, degrees) = canonical_undirected_adjacency(graph, false);
    let total_edges: usize = rows.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(total_edges + n);
    let mut values = Vec::with_capacity(total_edges + n);

    for (u, row) in rows.iter().enumerate() {
        let u_u32 = u as u32;
        let deg_u = degrees[u];
        let mut diag_inserted = deg_u == 0.0;

        for &(v, w) in row {
            if !diag_inserted && v > u_u32 {
                indices.push(u_u32);
                values.push(deg_u);
                diag_inserted = true;
            }
            indices.push(v);
            values.push(-w);
        }
        if !diag_inserted {
            indices.push(u_u32);
            values.push(deg_u);
        }
        indptr.push(indices.len() as u32);
    }

    CsrMatrix {
        num_nodes: n,
        indptr,
        indices,
        values,
        diagonal: degrees,
    }
}

/// Constructs the random-walk normalized graph Laplacian $L_{\mathrm{rw}} = D^{-1}(D - A)$.
///
/// Isolated nodes (`deg == 0`) have zero rows and zero diagonal entries.
pub fn random_walk_laplacian<W: Scalar>(graph: &CsrGraph<W>) -> CsrMatrix {
    let n = graph.num_nodes();
    if n == 0 {
        return CsrMatrix {
            num_nodes: 0,
            indptr: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
            diagonal: Vec::new(),
        };
    }

    let (rows, degrees) = canonical_undirected_adjacency(graph, false);
    let total_edges: usize = rows.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(total_edges + n);
    let mut values = Vec::with_capacity(total_edges + n);
    let mut diagonal = vec![0.0_f64; n];

    for (u, row) in rows.iter().enumerate() {
        let deg_u = degrees[u];
        if deg_u > 0.0 {
            let u_u32 = u as u32;
            let inv_d = 1.0 / deg_u;
            diagonal[u] = 1.0;
            let mut diag_inserted = false;
            for &(v, w) in row {
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(1.0);
                    diag_inserted = true;
                }
                indices.push(v);
                values.push(-w * inv_d);
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(1.0);
            }
        }
        indptr.push(indices.len() as u32);
    }

    CsrMatrix {
        num_nodes: n,
        indptr,
        indices,
        values,
        diagonal,
    }
}

/// Configuration for Coifman-Lafon anisotropic graph Laplacian construction ($L_\alpha$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnisotropicLaplacianConfig {
    /// Density reweighting exponent `alpha` in $A^{(\alpha)}_{ij} = \tilde{d}_i^{-\alpha} A_{ij} \tilde{d}_j^{-\alpha}$.
    /// `0.0` recovers standard adjacency; `0.5` gives Fokker-Planck; `1.0` gives Laplace-Beltrami.
    pub alpha: f64,
    /// Outer symmetric degree normalization exponent `beta` in $S (D^{(\alpha)} - A^{(\alpha)}) S$.
    /// `0.0` gives unnormalized $D^{(\alpha)} - A^{(\alpha)}$; `-0.5` gives symmetric normalized form.
    pub beta: f64,
    /// Additive degree regularization $c_{\mathrm{deg}} \ge 0.0$ in $\tilde{d}_i = \min(d_i, d_{\mathrm{cap}}) + c_{\mathrm{deg}}$.
    pub degree_reg: f64,
    /// Optional degree saturation cap $d_{\mathrm{cap}} > 0.0$ applied before `degree_reg`.
    pub degree_cap: Option<f64>,
    /// When `true` (default), outer scaling $S$ uses reweighted degrees $(d^{(\alpha)}_i)^\beta$,
    /// guaranteeing spectrum $\subseteq [0, 2]$ whenever `beta == -0.5`.
    /// When `false`, outer scaling $S$ uses saturated raw degrees $\tilde{d}_i^\beta$.
    pub reweighted_degree_norm: bool,
}

impl Default for AnisotropicLaplacianConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.0,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: true,
        }
    }
}

/// Output of [`anisotropic_laplacian`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicLaplacianResult {
    /// Symmetric PSD sparse matrix $S (D^{(\alpha)} - A^{(\alpha)}) S$.
    pub matrix: CsrMatrix,
    /// Near-nullspace vector $u$ satisfying `matrix * u == 0` on every non-isolated component
    /// (`1.0 / s_i` when `s_i > 0`, `0.0` for isolated unit-diagonal rows, `1.0` for isolated zero rows).
    pub nullspace: Vec<f64>,
    /// Outer symmetric scaling factors $s_i$ (`1.0` when `beta == 0.0`).
    pub outer_scale: Vec<f64>,
}

fn is_fast_undirected_clean_loopless<W: Scalar>(graph: &CsrGraph<W>) -> bool {
    if !is_fast_undirected_clean(graph) {
        return false;
    }
    for u in 0..graph.num_nodes() {
        if graph.neighbors(u).binary_search(&(u as u32)).is_ok() {
            return false;
        }
    }
    true
}

/// Constructs a symmetric positive semidefinite Coifman-Lafon anisotropic graph Laplacian
/// $L_{\mathrm{aniso}} = S (D^{(\alpha)} - A^{(\alpha)}) S$ from `graph`.
///
/// Special cases:
/// - `(alpha=0.0, beta=0.0, degree_reg=0.0, degree_cap=None)`: exact combinatorial Laplacian $L = D - A$.
/// - `(alpha=0.0, beta=-0.5, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=true)`:
///   exact symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$.
/// - `(alpha=0.0, beta=-p, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=false)`:
///   degree-power interpolation $L_{\mathrm{pow}}(p) = D^{-p}(D - A)D^{-p}$.
/// - `(alpha, beta=-0.5, reweighted_degree_norm=true)`: Coifman-Lafon symmetric normalized
///   operator with spectrum strictly bounded in $[0, 2]$.
pub fn anisotropic_laplacian<W: Scalar>(
    graph: &CsrGraph<W>,
    config: AnisotropicLaplacianConfig,
) -> CoreResult<AnisotropicLaplacianResult> {
    if !config.alpha.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::alpha"));
    }
    if !config.beta.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::beta"));
    }
    if !config.degree_reg.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::degree_reg"));
    }
    if config.degree_reg < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "degree_reg must be >= 0.0, got {}",
            config.degree_reg
        )));
    }
    if let Some(cap) = config.degree_cap {
        if !cap.is_finite() {
            return Err(CoreError::non_finite("anisotropic_laplacian::degree_cap"));
        }
        if cap <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "degree_cap must be > 0.0, got {cap}"
            )));
        }
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(AnisotropicLaplacianResult {
            matrix: CsrMatrix {
                num_nodes: 0,
                indptr: vec![0],
                indices: Vec::new(),
                values: Vec::new(),
                diagonal: Vec::new(),
            },
            nullspace: Vec::new(),
            outer_scale: Vec::new(),
        });
    }

    let is_fast = is_fast_undirected_clean_loopless(graph);
    let (rows_opt, raw_degrees) = if is_fast {
        let mut deg = vec![0.0_f64; n];
        let weights_opt = graph.weights();
        for (u, d_u) in deg.iter_mut().enumerate() {
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            *d_u = match weights_opt {
                Some(ws) => ws[start..end].iter().map(|w| w.to_f64()).sum(),
                None => (end - start) as f64,
            };
        }
        (None, deg)
    } else {
        let (rows, deg) = canonical_undirected_adjacency(graph, false);
        (Some(rows), deg)
    };

    let mut d_eff = vec![0.0_f64; n];
    let mut d_neg_alpha = vec![0.0_f64; n];
    for i in 0..n {
        let d_sat = match config.degree_cap {
            Some(cap) => raw_degrees[i].min(cap),
            None => raw_degrees[i],
        };
        let eff = d_sat + config.degree_reg;
        d_eff[i] = eff;
        d_neg_alpha[i] = if config.alpha == 0.0 {
            1.0
        } else if eff > 0.0 {
            eff.powf(-config.alpha)
        } else {
            0.0
        };
    }

    let mut d_alpha = vec![0.0_f64; n];
    if config.alpha == 0.0 {
        d_alpha.copy_from_slice(&raw_degrees);
    } else if let Some(ref rows) = rows_opt {
        for (u, row) in rows.iter().enumerate() {
            let du = d_neg_alpha[u];
            if du == 0.0 {
                continue;
            }
            let mut sum = 0.0_f64;
            for &(v, w) in row {
                sum += du * w * d_neg_alpha[v as usize];
            }
            d_alpha[u] = sum;
        }
    } else {
        let weights_opt = graph.weights();
        for (u, d_a_u) in d_alpha.iter_mut().enumerate() {
            let du = d_neg_alpha[u];
            if du == 0.0 {
                continue;
            }
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut sum = 0.0_f64;
            for pos in start..end {
                let v = graph.indices()[pos] as usize;
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                sum += du * w * d_neg_alpha[v];
            }
            *d_a_u = sum;
        }
    }

    let is_sym_norm_half = (config.beta + 0.5).abs() <= 1e-12;
    let use_unit_diag_isolated = config.reweighted_degree_norm && is_sym_norm_half;

    let mut outer_scale = vec![0.0_f64; n];
    let mut nullspace = vec![0.0_f64; n];
    for i in 0..n {
        let base_deg = if config.reweighted_degree_norm {
            d_alpha[i]
        } else {
            d_eff[i]
        };
        let s_i = if config.beta == 0.0 {
            1.0
        } else if base_deg > 0.0 {
            if is_sym_norm_half {
                base_deg.sqrt().recip()
            } else {
                base_deg.powf(config.beta)
            }
        } else {
            0.0
        };
        outer_scale[i] = s_i;
        nullspace[i] = if s_i > 0.0 {
            if config.beta == 0.0 {
                1.0
            } else if is_sym_norm_half {
                base_deg.sqrt()
            } else {
                1.0 / s_i
            }
        } else if use_unit_diag_isolated {
            0.0
        } else {
            1.0
        };
    }

    let est_edges = match &rows_opt {
        Some(rows) => rows.iter().map(Vec::len).sum(),
        None => graph.nnz(),
    };
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(est_edges + n);
    let mut values = Vec::with_capacity(est_edges + n);
    let mut diagonal = vec![0.0_f64; n];

    if let Some(ref rows) = rows_opt {
        for (u, row) in rows.iter().enumerate() {
            let u_u32 = u as u32;
            let scale_u = outer_scale[u] * d_neg_alpha[u];
            let diag_u = if use_unit_diag_isolated {
                1.0
            } else {
                outer_scale[u] * d_alpha[u] * outer_scale[u]
            };
            diagonal[u] = diag_u;
            let mut diag_inserted = !use_unit_diag_isolated && diag_u == 0.0;

            for &(v, w) in row {
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(diag_u);
                    diag_inserted = true;
                }
                let entry = -(scale_u * w * d_neg_alpha[v as usize] * outer_scale[v as usize]);
                indices.push(v);
                values.push(if entry == 0.0 { 0.0 } else { entry });
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(diag_u);
            }
            let ptr = u32::try_from(indices.len()).map_err(|_| {
                CoreError::invalid_csr("anisotropic_laplacian nnz exceeds u32::MAX")
            })?;
            indptr.push(ptr);
        }
    } else {
        let weights_opt = graph.weights();
        for u in 0..n {
            let u_u32 = u as u32;
            let scale_u = outer_scale[u] * d_neg_alpha[u];
            let diag_u = if use_unit_diag_isolated {
                1.0
            } else {
                outer_scale[u] * d_alpha[u] * outer_scale[u]
            };
            diagonal[u] = diag_u;
            let mut diag_inserted = !use_unit_diag_isolated && diag_u == 0.0;

            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos];
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(diag_u);
                    diag_inserted = true;
                }
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                let entry = -(scale_u * w * d_neg_alpha[v as usize] * outer_scale[v as usize]);
                indices.push(v);
                values.push(if entry == 0.0 { 0.0 } else { entry });
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(diag_u);
            }
            let ptr = u32::try_from(indices.len()).map_err(|_| {
                CoreError::invalid_csr("anisotropic_laplacian nnz exceeds u32::MAX")
            })?;
            indptr.push(ptr);
        }
    }

    Ok(AnisotropicLaplacianResult {
        matrix: CsrMatrix {
            num_nodes: n,
            indptr,
            indices,
            values,
            diagonal,
        },
        nullspace,
        outer_scale,
    })
}

/// Constructs a CSR graph Laplacian of the requested [`LaplacianKind`].
pub fn laplacian<W: Scalar>(graph: &CsrGraph<W>, kind: LaplacianKind) -> CsrMatrix {
    match kind {
        LaplacianKind::Combinatorial => combinatorial_laplacian(graph),
        LaplacianKind::SymmetricNormalized => symmetric_normalized_laplacian(graph).0,
        LaplacianKind::RandomWalk => random_walk_laplacian(graph),
    }
}

/// Constructs the Chebyshev-scaled shifted Laplacian
/// $\tilde{L} = \frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$ mapping eigenvalues in
/// $[0, \lambda_{\max}]$ into $[-1, 1]$.
pub fn shifted_normalized_laplacian(l_sym: &CsrMatrix, lambda_max: f64) -> CoreResult<CsrMatrix> {
    if !lambda_max.is_finite() {
        return Err(CoreError::non_finite(
            "shifted_normalized_laplacian::lambda_max",
        ));
    }
    if lambda_max <= 0.0 {
        return Err(CoreError::invalid_input(
            "lambda_max must be strictly positive",
        ));
    }
    l_sym.scale_and_shift(2.0 / lambda_max, -1.0)
}

/// Computes exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$ without eigensolvers.
pub fn spectral_moment_stats<W: Scalar>(graph: &CsrGraph<W>) -> SpectralMomentStats {
    let n = graph.num_nodes();
    if n == 0 {
        return SpectralMomentStats {
            sigma_lambda_sq: 0.0,
            sigma_lambda: 1e-6_f64.sqrt(),
            deg_cv: 0.0,
            mean_degree: 0.0,
            leaf_fraction: 0.0,
        };
    }

    if is_fast_undirected_clean(graph) {
        let mut deg = vec![0usize; n];
        let mut deg_sum = 0usize;
        let mut leaf_count = 0usize;
        for (u, d_u) in deg.iter_mut().enumerate() {
            let d = graph.degree(u);
            *d_u = d;
            deg_sum += d;
            if d == 1 {
                leaf_count += 1;
            }
        }
        let n_f64 = n as f64;
        let mean_degree = (deg_sum as f64) / n_f64;
        let deg_cv = if mean_degree > 0.0 {
            let var_deg = deg
                .iter()
                .map(|&d| {
                    let diff = (d as f64) - mean_degree;
                    diff * diff
                })
                .sum::<f64>()
                / n_f64;
            var_deg.sqrt() / mean_degree
        } else {
            0.0
        };
        let leaf_fraction = (leaf_count as f64) / n_f64;

        let sigma_lambda_sq = if n <= 1 || graph.nnz() == 0 {
            0.0
        } else {
            let inv_deg: Vec<f64> = deg
                .iter()
                .map(|&d| if d > 0 { 1.0 / (d as f64) } else { 0.0 })
                .collect();
            let mut sum_inv = 0.0_f64;
            for (u, &inv_u) in inv_deg.iter().enumerate() {
                if inv_u == 0.0 {
                    continue;
                }
                for &v in graph.neighbors(u) {
                    sum_inv += inv_u * inv_deg[v as usize];
                }
            }
            sum_inv / n_f64
        };

        return SpectralMomentStats {
            sigma_lambda_sq,
            sigma_lambda: sigma_lambda_sq.max(1e-6).sqrt(),
            deg_cv,
            mean_degree,
            leaf_fraction,
        };
    }

    let (rows, deg_f64) = canonical_undirected_adjacency(graph, true);
    let n_f64 = n as f64;
    let deg_sum: f64 = deg_f64.iter().sum();
    let mean_degree = deg_sum / n_f64;
    let deg_cv = if mean_degree > 0.0 {
        let var_deg = deg_f64
            .iter()
            .map(|&d| {
                let diff = d - mean_degree;
                diff * diff
            })
            .sum::<f64>()
            / n_f64;
        var_deg.sqrt() / mean_degree
    } else {
        0.0
    };
    let leaf_count = deg_f64.iter().filter(|&&d| (d - 1.0).abs() < 1e-12).count();
    let leaf_fraction = (leaf_count as f64) / n_f64;

    let nnz: usize = rows.iter().map(Vec::len).sum();
    let sigma_lambda_sq = if n <= 1 || nnz == 0 {
        0.0
    } else {
        let inv_deg: Vec<f64> = deg_f64
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();
        let mut sum_inv = 0.0_f64;
        for (u, row) in rows.iter().enumerate() {
            let inv_u = inv_deg[u];
            if inv_u == 0.0 {
                continue;
            }
            for &(v, _) in row {
                sum_inv += inv_u * inv_deg[v as usize];
            }
        }
        sum_inv / n_f64
    };

    SpectralMomentStats {
        sigma_lambda_sq,
        sigma_lambda: sigma_lambda_sq.max(1e-6).sqrt(),
        deg_cv,
        mean_degree,
        leaf_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> CsrGraph<f64> {
        let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1))
            .map(|i| (i as u32, (i + 1) as u32))
            .collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_laplacians_and_spmv_spmm() {
        // 4-node graph with isolated node 3: triangle 0-1-2 + isolated 3
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let (l_sym, inv_sqrt) = symmetric_normalized_laplacian(&g);
        assert_eq!(l_sym.num_nodes(), 4);
        assert_eq!(l_sym.diagonal(), &[1.0, 1.0, 1.0, 1.0]);
        assert!((inv_sqrt[0] - (0.5_f64).sqrt()).abs() < 1e-12);
        assert_eq!(inv_sqrt[3], 0.0);

        // Stationary vector [sqrt(2), sqrt(2), sqrt(2), 0] is in nullspace of L_sym
        let x = vec![2.0_f64.sqrt(), 2.0_f64.sqrt(), 2.0_f64.sqrt(), 0.0];
        let mut y = vec![0.0; 4];
        l_sym.spmv(&x, &mut y).unwrap();
        for &val in &y {
            assert!(val.abs() < 1e-12);
        }

        // Isolated node unit vector maps to 1.0 under L_sym
        let e3 = vec![0.0, 0.0, 0.0, 1.0];
        l_sym.spmv(&e3, &mut y).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0, 1.0]);

        // Test SpMM with 2 columns
        let x_batch = vec![
            x[0], e3[0], // row 0
            x[1], e3[1], // row 1
            x[2], e3[2], // row 2
            x[3], e3[3], // row 3
        ];
        let mut y_batch = vec![0.0; 8];
        l_sym.spmm(&x_batch, &mut y_batch, 2).unwrap();
        for i in 0..4 {
            assert!(y_batch[2 * i].abs() < 1e-12);
            assert!((y_batch[2 * i + 1] - e3[i]).abs() < 1e-12);
        }

        // Shifted Laplacian L_tilde = L_sym - I has 0 diagonal
        let l_tilde = shifted_normalized_laplacian(&l_sym, 2.0).unwrap();
        assert_eq!(l_tilde.diagonal(), &[0.0, 0.0, 0.0, 0.0]);
        let mut y_tilde = vec![0.0; 4];
        l_tilde.spmv(&e3, &mut y_tilde).unwrap();
        assert_eq!(y_tilde, vec![0.0, 0.0, 0.0, 0.0]);

        // Combinatorial and random-walk Laplacians
        let l_comb = laplacian(&g, LaplacianKind::Combinatorial);
        assert_eq!(l_comb.diagonal(), &[2.0, 2.0, 2.0, 0.0]);
        l_comb.spmv(&[1.0, 1.0, 1.0, 5.0], &mut y).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0, 0.0]);

        let l_rw = laplacian(&g, LaplacianKind::RandomWalk);
        assert_eq!(l_rw.diagonal(), &[1.0, 1.0, 1.0, 0.0]);
        l_rw.spmv(&[1.0, 1.0, 1.0, 5.0], &mut y).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_weighted_and_filtered_laplacian() {
        // Negative / zero / non-positive weights are ignored cleanly
        let g = CsrGraph::<f64>::from_edges(
            3,
            &[(0, 1), (1, 2), (0, 2)],
            Some(&[4.0, 0.0, -2.0]),
            false,
            false,
        )
        .unwrap();
        let (l_sym, inv_sqrt) = symmetric_normalized_laplacian(&g);
        assert!((inv_sqrt[0] - 0.5).abs() < 1e-12);
        assert!((inv_sqrt[1] - 0.5).abs() < 1e-12);
        assert_eq!(inv_sqrt[2], 0.0);
        let dense = l_sym.to_dense().unwrap();
        assert!((dense[1] - (-1.0)).abs() < 1e-12);
        assert!((dense[3] - (-1.0)).abs() < 1e-12);
        assert_eq!(dense[8], 1.0);
    }

    #[test]
    fn test_spectral_moment_stats_exact_frobenius_identity() {
        let g = path_graph(6);
        let stats = spectral_moment_stats(&g);
        let (l_sym, _) = symmetric_normalized_laplacian(&g);
        let dense = l_sym.to_dense().unwrap();
        let n = g.num_nodes() as f64;
        let frob_sq: f64 = dense.iter().map(|&v| v * v).sum();
        let trace: f64 = (0..g.num_nodes()).map(|i| dense[i * 6 + i]).sum();
        let mu1 = trace / n;
        let expected_var = frob_sq / n - mu1 * mu1;
        assert!(
            (stats.sigma_lambda_sq - expected_var).abs() < 1e-12,
            "got {} vs expected {expected_var}",
            stats.sigma_lambda_sq
        );
        assert!((stats.sigma_lambda - expected_var.sqrt()).abs() < 1e-12);
        assert!((stats.mean_degree - (10.0 / 6.0)).abs() < 1e-12);
        assert!((stats.leaf_fraction - (2.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_empty_and_singleton_csr_matrix() {
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let (l_empty, inv_empty) = symmetric_normalized_laplacian(&empty);
        assert_eq!(l_empty.num_nodes(), 0);
        assert_eq!(l_empty.nnz(), 0);
        assert!(inv_empty.is_empty());
        let empty_stats = spectral_moment_stats(&empty);
        assert_eq!(empty_stats.sigma_lambda_sq, 0.0);
        assert!((empty_stats.sigma_lambda - 1e-3).abs() < 1e-12);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let (l_single, inv_single) = symmetric_normalized_laplacian(&single);
        assert_eq!(l_single.num_nodes(), 1);
        assert_eq!(l_single.diagonal(), &[1.0]);
        assert_eq!(inv_single, vec![0.0]);
        let single_stats = spectral_moment_stats(&single);
        assert_eq!(single_stats.sigma_lambda_sq, 0.0);
        assert!((single_stats.sigma_lambda - 1e-3).abs() < 1e-12);

        assert!(shifted_normalized_laplacian(&l_single, 0.0).is_err());
        assert!(shifted_normalized_laplacian(&l_single, f64::NAN).is_err());
    }

    #[test]
    fn test_is_symmetric_and_symmetrize() {
        // Asymmetric 3x3 matrix:
        // [ 2.0, -1.0,  0.0]
        // [-3.0,  4.0,  2.0]
        // [ 0.0, -2.0,  1.0]
        let asym = CsrMatrix::from_csr(
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -3.0, 4.0, 2.0, -2.0, 1.0],
        )
        .unwrap();
        assert!(!asym.is_symmetric(1e-12));
        assert!(!asym.is_symmetric(1.5));
        assert!(asym.is_symmetric(4.0));
        assert!(!asym.is_symmetric(-1.0));
        assert!(!asym.is_symmetric(f64::NAN));

        let sym = asym.symmetrize().unwrap();
        assert!(sym.is_symmetric(0.0));
        assert_eq!(sym.diagonal(), &[2.0, 4.0, 1.0]);
        // (1, 2) and (2, 1) averaged to (2 + (-2)) * 0.5 = 0.0 and should be pruned!
        let dense = sym.to_dense().unwrap();
        assert_eq!(dense, vec![2.0, -2.0, 0.0, -2.0, 4.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(sym.nnz(), 5);

        let empty = CsrMatrix::from_csr(0, vec![0], vec![], vec![]).unwrap();
        assert!(empty.is_symmetric(0.0));
        assert_eq!(empty.symmetrize().unwrap().num_nodes(), 0);
    }

    #[test]
    fn test_anisotropic_laplacian_special_cases_and_psd() {
        // Weighted graph with hub and isolated vertex 4
        let g = CsrGraph::<f64>::from_edges(
            5,
            &[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
            Some(&[2.0, 1.0, 3.0, 1.5, 0.5]),
            false,
            false,
        )
        .unwrap();

        // 1. (0, 0) recovers combinatorial_laplacian bit-for-bit
        let l_comb = combinatorial_laplacian(&g);
        let res_comb = anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                alpha: 0.0,
                beta: 0.0,
                degree_reg: 0.0,
                degree_cap: None,
                reweighted_degree_norm: true,
            },
        )
        .unwrap();
        assert_eq!(res_comb.matrix, l_comb);
        assert_eq!(res_comb.nullspace, vec![1.0; 5]);

        // 2. (0, -0.5, reweighted=true) recovers symmetric_normalized_laplacian
        let (l_sym, _) = symmetric_normalized_laplacian(&g);
        let res_sym = anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                alpha: 0.0,
                beta: -0.5,
                degree_reg: 0.0,
                degree_cap: None,
                reweighted_degree_norm: true,
            },
        )
        .unwrap();
        assert_eq!(res_sym.matrix, l_sym);
        let mut y_ns = vec![0.0; 5];
        res_sym.matrix.spmv(&res_sym.nullspace, &mut y_ns).unwrap();
        for &v in &y_ns {
            assert!(v.abs() < 1e-13);
        }

        // 3. Connected graph: (0, -0.5, reweighted=false) also matches L_sym
        let g_conn = path_graph(6);
        let (l_sym_conn, _) = symmetric_normalized_laplacian(&g_conn);
        let res_sym_unrew = anisotropic_laplacian(
            &g_conn,
            AnisotropicLaplacianConfig {
                alpha: 0.0,
                beta: -0.5,
                degree_reg: 0.0,
                degree_cap: None,
                reweighted_degree_norm: false,
            },
        )
        .unwrap();
        let diff_sym: f64 = l_sym_conn
            .to_dense()
            .unwrap()
            .iter()
            .zip(res_sym_unrew.matrix.to_dense().unwrap().iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(diff_sym < 1e-14);

        // 4. L_pow(p) = D^{-p}(D - A)D^{-p} at (0.0, -p, reweighted=false)
        let p = 0.6;
        let res_pow = anisotropic_laplacian(
            &g_conn,
            AnisotropicLaplacianConfig {
                alpha: 0.0,
                beta: -p,
                degree_reg: 0.0,
                degree_cap: None,
                reweighted_degree_norm: false,
            },
        )
        .unwrap();
        let l_comb_conn = combinatorial_laplacian(&g_conn).to_dense().unwrap();
        let d_pow: Vec<f64> = (0..6).map(|i| (g_conn.degree(i) as f64).powf(-p)).collect();
        let dense_pow = res_pow.matrix.to_dense().unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let expected = d_pow[i] * l_comb_conn[i * 6 + j] * d_pow[j];
                assert!((dense_pow[i * 6 + j] - expected).abs() < 1e-13);
            }
        }

        // 5. CL_sym at (alpha, -0.5, reweighted=true) is symmetric, PSD, and spectrum in [0, 2]
        for &alpha in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let cl_sym = anisotropic_laplacian(
                &g,
                AnisotropicLaplacianConfig {
                    alpha,
                    beta: -0.5,
                    degree_reg: 0.0,
                    degree_cap: None,
                    reweighted_degree_norm: true,
                },
            )
            .unwrap();
            assert!(cl_sym.matrix.is_symmetric(1e-13));
            let dense = cl_sym.matrix.to_dense().unwrap();
            let (evals, _) = symmetric_eigen_jacobi(&dense, 5, 64, 1e-12).unwrap();
            assert!(evals.iter().all(|&ev| (-1e-12..=2.0 + 1e-12).contains(&ev)));
            cl_sym.matrix.spmv(&cl_sym.nullspace, &mut y_ns).unwrap();
            assert!(y_ns.iter().all(|&v| v.abs() < 1e-12));

            // Also verify unnormalized (beta = 0.0) with degree_reg & degree_cap is symmetric PSD
            let cl_reg = anisotropic_laplacian(
                &g,
                AnisotropicLaplacianConfig {
                    alpha,
                    beta: 0.0,
                    degree_reg: 0.5,
                    degree_cap: Some(3.0),
                    reweighted_degree_norm: true,
                },
            )
            .unwrap();
            assert!(cl_reg.matrix.is_symmetric(1e-13));
            let dense_reg = cl_reg.matrix.to_dense().unwrap();
            let (evals_reg, _) = symmetric_eigen_jacobi(&dense_reg, 5, 64, 1e-12).unwrap();
            assert!(evals_reg.iter().all(|&ev| ev >= -1e-12));
        }

        // 6. Validation errors
        assert!(
            anisotropic_laplacian(
                &g,
                AnisotropicLaplacianConfig {
                    alpha: f64::NAN,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            anisotropic_laplacian(
                &g,
                AnisotropicLaplacianConfig {
                    degree_reg: -0.1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            anisotropic_laplacian(
                &g,
                AnisotropicLaplacianConfig {
                    degree_cap: Some(0.0),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn test_spectrum_estimation_gershgorin_power_lanczos_and_trace() {
        // Empty and zero matrices
        let empty = CsrMatrix::from_csr(0, vec![0], vec![], vec![]).unwrap();
        assert_eq!(empty.trace(), 0.0);
        assert_eq!(empty.gershgorin_spectral_bound(), 0.0);
        assert_eq!(empty.estimate_lambda_max(10, 1.02, 42).unwrap(), 0.0);
        assert_eq!(
            empty.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap(),
            0.0
        );

        let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let zero_lap = combinatorial_laplacian(&edgeless);
        assert_eq!(zero_lap.trace(), 0.0);
        assert_eq!(zero_lap.gershgorin_spectral_bound(), 0.0);
        assert_eq!(zero_lap.estimate_lambda_max(10, 1.02, 42).unwrap(), 0.0);
        assert_eq!(
            zero_lap.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap(),
            0.0
        );

        // Star graph K_{1, 50}: L_comb has exact lambda_max = 51.0, trace = 100.0
        let n_star = 51usize;
        let star_edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0, i as u32)).collect();
        let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
        let l_star = combinatorial_laplacian(&star);
        assert!((l_star.trace() - 100.0).abs() < 1e-12);
        let gersh = l_star.gershgorin_spectral_bound();
        assert!((gersh - 100.0).abs() < 1e-12);
        let pi = l_star.estimate_lambda_max(10, 1.02, 42).unwrap();
        let lanc = l_star.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap();
        assert!(pi >= 50.0 && pi <= gersh);
        assert!(lanc >= 51.0 && lanc <= (1.05 * 51.0 + 1e-9).min(gersh));

        // 2-node graph (tests constant-vector Rademacher edge case!)
        let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let l2 = combinatorial_laplacian(&g2);
        assert!((l2.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap() - 2.0).abs() < 1e-12);
        assert!(l2.estimate_lambda_max(10, 1.02, 42).unwrap() >= 2.0 - 1e-12);

        // Validation
        assert!(l_star.estimate_lambda_max(10, 0.99, 42).is_err());
        assert!(l_star.estimate_lambda_max_lanczos(16, 0.99, 42).is_err());
        assert!((l_star.estimate_lambda_max(0, 1.0, 42).unwrap() - gersh).abs() < 1e-12);
        assert!((l_star.estimate_lambda_max_lanczos(0, 1.0, 42).unwrap() - gersh).abs() < 1e-12);
    }
}
