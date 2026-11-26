use egraph_core::{CoreError, CoreResult};

/// Double-precision square Compressed Sparse Row matrix with sorted column indices per row.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) values: Vec<f64>,
    pub(super) diagonal: Vec<f64>,
}

impl CsrMatrix {
    /// Internal constructor for pre-validated CSR buffers produced within `csr_matrix`.
    #[inline]
    pub(super) fn from_validated_parts(
        num_nodes: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        values: Vec<f64>,
        diagonal: Vec<f64>,
    ) -> Self {
        Self {
            num_nodes,
            indptr,
            indices,
            values,
            diagonal,
        }
    }

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
}
