//! Symmetric pair scatter and row-normalized CSR probability symmetrization.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};

use super::config::SparseProbabilities;

/// Symmetrizes undirected pairs `(u, v, p_val)` with `u < v` into canonical sorted symmetric CSR
/// [`SparseProbabilities`] without per-row `HashMap`s.
pub(super) fn scatter_symmetric_pairs(
    n_nodes: usize,
    mut pairs: Vec<(u32, u32, f64)>,
    scale_by_two_n: bool,
) -> CoreResult<SparseProbabilities> {
    if n_nodes < 2 || pairs.is_empty() {
        return Ok(SparseProbabilities::empty(n_nodes));
    }

    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let inv_two_n = if scale_by_two_n {
        1.0 / (2.0 * (n_nodes as f64))
    } else {
        1.0
    };

    let mut merged: Vec<(u32, u32, f64)> = Vec::with_capacity(pairs.len());
    let mut deg = vec![0usize; n_nodes];

    let mut i = 0usize;
    while i < pairs.len() {
        let (u, v, mut acc) = pairs[i];
        i += 1;
        while i < pairs.len() && pairs[i].0 == u && pairs[i].1 == v {
            acc += pairs[i].2;
            i += 1;
        }
        let p_sym = acc * inv_two_n;
        if p_sym > 0.0 {
            merged.push((u, v, p_sym));
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
    }

    if merged.is_empty() {
        return Ok(SparseProbabilities::empty(n_nodes));
    }

    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0u32);
    let mut total = 0usize;
    for d in &deg {
        total += *d;
        let ptr = u32::try_from(total)
            .map_err(|_| CoreError::invalid_csr("SparseProbabilities nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }
    let mut cursor: Vec<usize> = indptr[..n_nodes].iter().map(|&p| p as usize).collect();
    let mut indices = vec![0u32; total];
    let mut values = vec![0.0_f64; total];

    // Iterating pairs sorted by (u, v) with u < v inserts neighbors < x before neighbors > x
    // in strictly ascending column order without per-row sorting.
    for (u, v, p_sym) in merged {
        let pu = cursor[u as usize];
        indices[pu] = v;
        values[pu] = p_sym;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize];
        indices[pv] = u;
        values[pv] = p_sym;
        cursor[v as usize] += 1;
    }

    Ok(SparseProbabilities::from_validated_parts(
        n_nodes, indptr, indices, values,
    ))
}

/// Computes row-normalized symmetrized joint probabilities
/// $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ from arbitrary CSR
/// directed affinities `(indptr, indices, weights)`.
///
/// Self-loops and non-positive weights are dropped before row normalization; duplicate entries in
/// the same row are summed via a reusable `VisitStamp` without per-row `HashMap`s.
pub fn symmetrize_row_normalized(
    n_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f64],
) -> CoreResult<SparseProbabilities> {
    if u32::try_from(n_nodes).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if indptr.len() != n_nodes + 1 {
        return Err(CoreError::dimension_mismatch(
            n_nodes + 1,
            indptr.len(),
            "symmetrize_row_normalized::indptr",
        ));
    }
    if indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    for row in 0..n_nodes {
        if indptr[row + 1] < indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "indptr must be non-decreasing at row {row}: {} > {}",
                indptr[row],
                indptr[row + 1]
            )));
        }
    }
    let nnz = indptr[n_nodes] as usize;
    if indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            indices.len(),
            "symmetrize_row_normalized::indices",
        ));
    }
    if weights.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            weights.len(),
            "symmetrize_row_normalized::weights",
        ));
    }
    for (&idx, &w) in indices.iter().zip(weights.iter()) {
        if (idx as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n_nodes,
                "symmetrize_row_normalized::indices",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite("symmetrize_row_normalized::weights"));
        }
        if w < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "weights must be non-negative, got {w}"
            )));
        }
    }

    if n_nodes < 2 {
        return Ok(SparseProbabilities::empty(n_nodes));
    }

    let mut seen = VisitStamp::new(n_nodes);
    let mut seen_pos = vec![0usize; n_nodes];
    let mut row_entries: Vec<(u32, f64)> = Vec::new();
    let mut pairs: Vec<(u32, u32, f64)> = Vec::with_capacity(nnz);

    for i in 0..n_nodes {
        seen.clear();
        row_entries.clear();
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        for pos in start..end {
            let j = indices[pos];
            let j_usize = j as usize;
            let w = weights[pos];
            if j_usize == i || w <= 0.0 {
                continue;
            }
            if seen.visit(j_usize) {
                seen_pos[j_usize] = row_entries.len();
                row_entries.push((j, w));
            } else {
                row_entries[seen_pos[j_usize]].1 += w;
            }
        }
        let row_sum: f64 = row_entries.iter().map(|&(_, w)| w).sum();
        if row_sum > 0.0 {
            let inv = 1.0 / row_sum;
            let i_u32 = i as u32;
            for &(j, w) in &row_entries {
                let a_tilde = w * inv;
                let (u, v) = if i_u32 < j { (i_u32, j) } else { (j, i_u32) };
                pairs.push((u, v, a_tilde));
            }
        }
    }

    scatter_symmetric_pairs(n_nodes, pairs, true)
}

/// Computes row-normalized symmetrized graph probabilities
/// $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ directly from a
/// [`CsrGraph<W>`] (Böhm et al. 2025 `graph t-SNE` normalization).
pub fn graph_row_normalized_probabilities<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<SparseProbabilities> {
    let n = graph.num_nodes();
    if n < 2 {
        return Ok(SparseProbabilities::empty(n));
    }
    let weights: Vec<f64> = match graph.weights() {
        Some(ws) => ws.iter().map(|w| w.to_f64()).collect(),
        None => vec![1.0; graph.nnz()],
    };
    symmetrize_row_normalized(n, graph.indptr(), graph.indices(), &weights)
}
