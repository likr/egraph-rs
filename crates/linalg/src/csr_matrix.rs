//! Compressed Sparse Row (`CsrMatrix`) representation, graph Laplacians, and $O(|E|)$ spectral moments.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

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
}
