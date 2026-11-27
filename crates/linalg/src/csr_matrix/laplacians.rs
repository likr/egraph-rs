use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use super::matrix::CsrMatrix;

/// Supported graph Laplacian variants for sparse CSR construction.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaplacianKind {
    /// Combinatorial (unnormalized) Laplacian $L = D - A$.
    Combinatorial,
    /// Symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$.
    SymmetricNormalized,
    /// Random-walk normalized Laplacian $L_{\mathrm{rw}} = D^{-1} (D - A)$.
    RandomWalk,
}

/// Extracts canonical undirected simple adjacency rows `(col, weight)` and vertex degrees,
/// filtering self-loops and non-positive or non-finite edge weights.
pub(super) fn canonical_undirected_adjacency<W: Scalar>(
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
pub(super) fn is_fast_undirected_clean<W: Scalar>(graph: &CsrGraph<W>) -> bool {
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
            CsrMatrix::from_validated_parts(0, vec![0], Vec::new(), Vec::new(), Vec::new()),
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

        let matrix = CsrMatrix::from_validated_parts(n, indptr, indices, values, vec![1.0; n]);
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

    let matrix = CsrMatrix::from_validated_parts(n, indptr, indices, values, vec![1.0; n]);
    (matrix, inv_sqrt_deg)
}

/// Constructs the combinatorial (unnormalized) graph Laplacian $L = D - A$.
pub fn combinatorial_laplacian<W: Scalar>(graph: &CsrGraph<W>) -> CsrMatrix {
    let n = graph.num_nodes();
    if n == 0 {
        return CsrMatrix::from_validated_parts(0, vec![0], Vec::new(), Vec::new(), Vec::new());
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

        return CsrMatrix::from_validated_parts(n, indptr, indices, values, diagonal);
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

    CsrMatrix::from_validated_parts(n, indptr, indices, values, degrees)
}

/// Constructs the random-walk normalized graph Laplacian $L_{\mathrm{rw}} = D^{-1}(D - A)$.
///
/// Isolated nodes (`deg == 0`) have zero rows and zero diagonal entries.
pub fn random_walk_laplacian<W: Scalar>(graph: &CsrGraph<W>) -> CsrMatrix {
    let n = graph.num_nodes();
    if n == 0 {
        return CsrMatrix::from_validated_parts(0, vec![0], Vec::new(), Vec::new(), Vec::new());
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

    CsrMatrix::from_validated_parts(n, indptr, indices, values, diagonal)
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
