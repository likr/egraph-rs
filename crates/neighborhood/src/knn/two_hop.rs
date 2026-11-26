//! Canonical unweighted 1-hop adjacency, single-pass `VisitStamp` 1+2-hop CSR extraction,
//! and uncapped pure 1+2-hop [`KnnGraph`] construction.

use egraph_core::{CoreError, CoreResult, CsrGraph, VisitStamp};

use super::graph::{KnnGraph, KnnStats};

/// Canonicalizes any [`CsrGraph<W>`] into a self-loop-free, sorted, simple symmetric unweighted
/// [`CsrGraph<f64>`] (`weights: None`).
pub(crate) fn canonical_unweighted_1hop<W>(graph: &CsrGraph<W>) -> CoreResult<CsrGraph<f64>> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }

    // Fast path when the input is already undirected with strictly ascending rows (no self-loops
    // or duplicates are permitted in clean undirected CsrGraph instances).
    if !graph.is_directed() {
        let mut all_sorted_simple = true;
        for u in 0..n {
            let u32_u = u as u32;
            let row = graph.neighbors(u);
            if !row.windows(2).all(|w| w[0] < w[1]) || row.contains(&u32_u) {
                all_sorted_simple = false;
                break;
            }
        }
        if all_sorted_simple {
            return CsrGraph::from_csr(graph.indptr(), graph.indices(), None, false, false);
        }
    }

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (u, bucket) in buckets.iter_mut().enumerate() {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u && (v as usize) < n {
                bucket.push(v);
            }
        }
    }
    if graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v in graph.neighbors(u) {
                if v != u32_u && (v as usize) < n {
                    buckets[v as usize].push(u32_u);
                }
            }
        }
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    for row in &mut buckets {
        row.sort_unstable();
        row.dedup();
        indices.extend_from_slice(row);
        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("canonical 1-hop nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    CsrGraph::from_csr(&indptr, &indices, None, false, false)
}

/// Extracts disjoint, self-loop-free, sorted simple symmetric unweighted 1-hop and pure 2-hop
/// `CsrGraph<f64>` graphs (`N_2(u) \setminus (\{u\} \cup N_1(u))`) using a single reusable
/// `VisitStamp` without dense matrices or per-row `HashMap`s.
pub fn extract_1hop_2hop_csr<W>(graph: &CsrGraph<W>) -> CoreResult<(CsrGraph<f64>, CsrGraph<f64>)> {
    let adj1 = canonical_unweighted_1hop(graph)?;
    let n = adj1.num_nodes();
    if n == 0 || adj1.nnz() == 0 {
        let indptr2 = vec![0u32; n + 1];
        let adj2 = CsrGraph::from_csr(&indptr2, &[], None, false, false)?;
        return Ok((adj1, adj2));
    }

    let mut stamp = VisitStamp::new(n);
    let mut indptr2 = Vec::with_capacity(n + 1);
    indptr2.push(0u32);
    let mut indices2 = Vec::new();
    let mut row2 = Vec::new();

    for u in 0..n {
        stamp.clear();
        stamp.mark(u);
        let n1 = adj1.neighbors(u);
        for &w in n1 {
            stamp.mark(w as usize);
        }

        row2.clear();
        for &w in n1 {
            for &v in adj1.neighbors(w as usize) {
                if stamp.visit(v as usize) {
                    row2.push(v);
                }
            }
        }
        row2.sort_unstable();
        indices2.extend_from_slice(&row2);
        let ptr = u32::try_from(indices2.len())
            .map_err(|_| CoreError::invalid_csr("2-hop CSR nnz exceeds u32::MAX"))?;
        indptr2.push(ptr);
    }

    let adj2 = CsrGraph::from_csr(&indptr2, &indices2, None, false, false)?;
    Ok((adj1, adj2))
}

/// Builds a variable-length CSR pure 1+2-hop [`KnnGraph`] with exact `{1.0, 2.0}` discrete
/// distances, per-node 2-hop neighborhood sizes `k2_deg`, and summary [`KnnStats`].
///
/// Neighbors within each row are merged in strictly ascending vertex index order without any
/// `>= 3`-hop fill.
pub fn build_pure_2hop_knn<W>(graph: &CsrGraph<W>) -> CoreResult<(KnnGraph, Vec<u32>, KnnStats)> {
    let (adj1, adj2) = extract_1hop_2hop_csr(graph)?;
    let n = adj1.num_nodes();
    let n_1hop = adj1.nnz();
    let n_2hop = adj2.nnz();
    let total_nnz = n_1hop + n_2hop;

    if n == 0 || total_nnz == 0 {
        let knn = KnnGraph::empty(n)?;
        let k2_deg = vec![0u32; n];
        let stats = KnnStats {
            n_pure_2hop_rows: n,
            ..KnnStats::default()
        };
        return Ok((knn, k2_deg, stats));
    }

    if u32::try_from(total_nnz).is_err() {
        return Err(CoreError::invalid_csr("pure 1+2-hop nnz exceeds u32::MAX"));
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut distances = Vec::with_capacity(total_nnz);
    let mut k2_deg = Vec::with_capacity(n);

    let mut min_row_k = usize::MAX;
    let mut max_row_k = 0usize;

    for u in 0..n {
        let r1 = adj1.neighbors(u);
        let r2 = adj2.neighbors(u);
        let mut i1 = 0usize;
        let mut i2 = 0usize;
        while i1 < r1.len() && i2 < r2.len() {
            if r1[i1] < r2[i2] {
                indices.push(r1[i1]);
                distances.push(1.0);
                i1 += 1;
            } else {
                indices.push(r2[i2]);
                distances.push(2.0);
                i2 += 1;
            }
        }
        while i1 < r1.len() {
            indices.push(r1[i1]);
            distances.push(1.0);
            i1 += 1;
        }
        while i2 < r2.len() {
            indices.push(r2[i2]);
            distances.push(2.0);
            i2 += 1;
        }

        let row_k = r1.len() + r2.len();
        k2_deg.push(row_k as u32);
        min_row_k = min_row_k.min(row_k);
        max_row_k = max_row_k.max(row_k);
        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("pure 1+2-hop nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    let knn = KnnGraph::new_validated(n, indptr, indices, distances)?;
    let n_f64 = n as f64;
    let avg_deg1 = (n_1hop as f64) / n_f64;
    let avg_deg2 = (total_nnz as f64) / n_f64;
    let stats = KnnStats {
        avg_deg1,
        avg_deg2,
        avg_row_k: avg_deg2,
        min_row_k,
        max_row_k,
        nnz: total_nnz,
        n_1hop,
        n_2hop,
        n_truncated_1hop: 0,
        n_truncated_2hop: 0,
        n_filled_residual: 0,
        n_pure_2hop_rows: n,
    };
    Ok((knn, k2_deg, stats))
}
