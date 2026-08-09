use ndarray::Array1;
use petgraph_distance::SparseSymmetricMatrix;

/// A general Compressed Sparse Row (CSR) matrix representation.
#[derive(Debug, Clone)]
pub struct CsrMatrix<S> {
    pub rows: usize,
    pub cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<S>,
}

impl<S> CsrMatrix<S>
where
    S: Copy + num_traits::Float + num_traits::Zero + std::ops::AddAssign + Default,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Convert from petgraph_distance::SparseSymmetricMatrix
    pub fn from_symmetric(matrix: &SparseSymmetricMatrix<S>) -> Self {
        let n = matrix.dim();
        let mut row_counts = vec![0; n];

        // Diagonal
        for i in 0..n {
            row_counts[i] += 1;
        }

        // Off-diagonal (symmetric)
        for &(i, j, _val) in matrix.edges() {
            row_counts[i] += 1;
            row_counts[j] += 1;
        }

        let mut indptr = vec![0; n + 1];
        for i in 0..n {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        let nnz = indptr[n];
        let mut indices = vec![0; nnz];
        let mut data = vec![S::zero(); nnz];

        let mut current_row_ptr = indptr[..n].to_vec();

        // Add diagonals
        for i in 0..n {
            let ptr = current_row_ptr[i];
            indices[ptr] = i;
            data[ptr] = matrix.diagonal()[i];
            current_row_ptr[i] += 1;
        }

        // Add edges
        for &(i, j, val) in matrix.edges() {
            let ptr_i = current_row_ptr[i];
            indices[ptr_i] = j;
            data[ptr_i] = val;
            current_row_ptr[i] += 1;

            let ptr_j = current_row_ptr[j];
            indices[ptr_j] = i;
            data[ptr_j] = val;
            current_row_ptr[j] += 1;
        }

        // Sort indices in each row for better access patterns and SpGEMM
        for r in 0..n {
            let start = indptr[r];
            let end = indptr[r + 1];
            let mut row_entries: Vec<_> = indices[start..end]
                .iter()
                .zip(data[start..end].iter())
                .map(|(&idx, &val)| (idx, val))
                .collect();
            row_entries.sort_by_key(|&(idx, _)| idx);

            for (k, (idx, val)) in row_entries.into_iter().enumerate() {
                indices[start + k] = idx;
                data[start + k] = val;
            }
        }

        Self {
            rows: n,
            cols: n,
            indptr,
            indices,
            data,
        }
    }

    /// Matrix-vector multiplication
    pub fn multiply(&self, x: &Array1<S>) -> Array1<S> {
        let mut y = Array1::zeros(self.rows);
        self.multiply_into(x, &mut y);
        y
    }

    /// Matrix-vector multiplication in place
    pub fn multiply_into(&self, x: &Array1<S>, y: &mut Array1<S>) {
        for i in 0..self.rows {
            let mut sum = S::zero();
            for j in self.indptr[i]..self.indptr[i + 1] {
                sum += self.data[j] * x[self.indices[j]];
            }
            y[i] = sum;
        }
    }

    /// Extract diagonal elements
    pub fn diagonal(&self) -> Array1<S> {
        let n = self.rows.min(self.cols);
        let mut diag = Array1::zeros(n);
        for i in 0..n {
            for j in self.indptr[i]..self.indptr[i + 1] {
                if self.indices[j] == i {
                    diag[i] = self.data[j];
                    break;
                }
            }
        }
        diag
    }

    /// Matrix transpose
    pub fn transpose(&self) -> Self {
        let mut col_counts = vec![0; self.cols];
        for &col in &self.indices {
            col_counts[col] += 1;
        }

        let mut indptr = vec![0; self.cols + 1];
        for i in 0..self.cols {
            indptr[i + 1] = indptr[i] + col_counts[i];
        }

        let mut indices = vec![0; self.data.len()];
        let mut data = vec![S::zero(); self.data.len()];

        let mut current_col_ptr = indptr[..self.cols].to_vec();

        for r in 0..self.rows {
            for j in self.indptr[r]..self.indptr[r + 1] {
                let c = self.indices[j];
                let ptr = current_col_ptr[c];
                indices[ptr] = r;
                data[ptr] = self.data[j];
                current_col_ptr[c] += 1;
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            indptr,
            indices,
            data,
        }
    }

    /// Matrix-matrix multiplication (SpGEMM). `self * other`.
    pub fn mul_csr(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut indptr = vec![0; self.rows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        let mut row_marker = vec![-1_isize; other.cols];
        let mut row_values = vec![S::zero(); other.cols];

        for i in 0..self.rows {
            let mut row_nnz = 0;
            let mut row_indices = Vec::new();

            for j_ptr in self.indptr[i]..self.indptr[i + 1] {
                let k = self.indices[j_ptr];
                let val_ik = self.data[j_ptr];

                for k_ptr in other.indptr[k]..other.indptr[k + 1] {
                    let j = other.indices[k_ptr];
                    let val_kj = other.data[k_ptr];

                    if row_marker[j] != i as isize {
                        row_marker[j] = i as isize;
                        row_values[j] = val_ik * val_kj;
                        row_indices.push(j);
                        row_nnz += 1;
                    } else {
                        row_values[j] += val_ik * val_kj;
                    }
                }
            }

            row_indices.sort_unstable();
            for &j in &row_indices {
                // Filter near-zero values to maintain sparsity?
                // Let's just keep everything for now.
                indices.push(j);
                data.push(row_values[j]);
            }
            indptr[i + 1] = indptr[i] + row_nnz;
        }

        Self {
            rows: self.rows,
            cols: other.cols,
            indptr,
            indices,
            data,
        }
    }
}
