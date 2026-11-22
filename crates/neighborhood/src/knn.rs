//! Compressed Sparse Row $k$-NN storage (`KnnGraph`), single-pass `VisitStamp` 1+2-hop
//! extraction, hub-safe pair sampling, and 1+2-hop priority CSR construction.

use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, VisitStamp};

/// Variable-length Compressed Sparse Row (CSR) $k$-nearest-neighbor graph in `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnGraph {
    n_nodes: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    distances: Vec<f64>,
}

impl KnnGraph {
    /// Constructs a [`KnnGraph`] after validating CSR offsets, vertex bounds, and finite
    /// non-negative distances.
    pub fn new_validated(
        n_nodes: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        distances: Vec<f64>,
    ) -> CoreResult<Self> {
        if u32::try_from(n_nodes).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if indptr.len() != n_nodes + 1 {
            return Err(CoreError::dimension_mismatch(
                n_nodes + 1,
                indptr.len(),
                "KnnGraph::new_validated::indptr",
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
        let nnz = indptr[n_nodes];
        if indices.len() != nnz {
            return Err(CoreError::dimension_mismatch(
                nnz,
                indices.len(),
                "KnnGraph::new_validated::indices",
            ));
        }
        if distances.len() != nnz {
            return Err(CoreError::dimension_mismatch(
                nnz,
                distances.len(),
                "KnnGraph::new_validated::distances",
            ));
        }
        for (&idx, &dist) in indices.iter().zip(distances.iter()) {
            if (idx as usize) >= n_nodes {
                return Err(CoreError::index_out_of_bounds(
                    idx as usize,
                    n_nodes,
                    "KnnGraph::new_validated::indices",
                ));
            }
            if !dist.is_finite() {
                return Err(CoreError::non_finite("KnnGraph::new_validated::distances"));
            }
            if dist < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "KnnGraph distances must be non-negative, got {dist}"
                )));
            }
        }
        Ok(Self {
            n_nodes,
            indptr,
            indices,
            distances,
        })
    }

    /// Constructs an empty [`KnnGraph`] with `n_nodes` isolated rows.
    pub fn empty(n_nodes: usize) -> CoreResult<Self> {
        if u32::try_from(n_nodes).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        Ok(Self {
            n_nodes,
            indptr: vec![0; n_nodes + 1],
            indices: Vec::new(),
            distances: Vec::new(),
        })
    }

    /// Returns the number of vertices in the $k$-NN graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns the total number of stored neighbor entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns the `(neighbor_indices, distances)` slices for node `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_nodes()`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        (&self.indices[start..end], &self.distances[start..end])
    }

    /// Fallible variant of [`KnnGraph::row`] returning [`CoreError::IndexOutOfBounds`].
    #[inline]
    pub fn try_row(&self, i: usize) -> CoreResult<(&[u32], &[f64])> {
        if i >= self.n_nodes {
            return Err(CoreError::index_out_of_bounds(
                i,
                self.n_nodes,
                "KnnGraph::try_row",
            ));
        }
        Ok(self.row(i))
    }

    /// Returns the CSR row pointer slice of length `num_nodes() + 1`.
    #[inline]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Returns the flat neighbor index slice of length `nnz()`.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Returns the flat distance slice aligned with `indices()`.
    #[inline]
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }
}

/// Summary statistics emitted by 1+2-hop and hybrid $k$-NN builders.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KnnStats {
    pub avg_deg1: f64,
    pub avg_deg2: f64,
    pub avg_row_k: f64,
    pub min_row_k: usize,
    pub max_row_k: usize,
    pub nnz: usize,
    pub n_1hop: usize,
    pub n_2hop: usize,
    pub n_truncated_1hop: usize,
    pub n_truncated_2hop: usize,
    pub n_filled_residual: usize,
    pub n_pure_2hop_rows: usize,
}

/// Canonical undirected unweighted 1-hop and hub-safe 2-hop node-pair lists (`u < v`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sparse12HopPairs {
    pub u1: Vec<u32>,
    pub v1: Vec<u32>,
    pub u2: Vec<u32>,
    pub v2: Vec<u32>,
}

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

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
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
        indptr.push(indices.len());
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

/// Extracts canonical undirected 1-hop (`u < v`) and hub-safe capped 2-hop (`u < v`) node pairs
/// without $O(N^2)$ pair explosion on star hubs.
///
/// Degree-1 neighbors are skipped when expanding 2-hop paths (they only point back to `u`),
/// and neighbors `w` with `deg(w) > max_2hop_per_node` are deterministically subsampled via
/// `DeterministicRng::from_seed_u64(seed)`.
pub fn extract_sparse_1hop_2hop_pairs<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    seed: u64,
) -> CoreResult<Sparse12HopPairs> {
    let adj1 = canonical_unweighted_1hop(graph)?;
    let n = adj1.num_nodes();
    if n <= 1 || adj1.nnz() == 0 {
        return Ok(Sparse12HopPairs::default());
    }

    let mut u1 = Vec::with_capacity(adj1.num_edges());
    let mut v1 = Vec::with_capacity(adj1.num_edges());
    for u in 0..n {
        let u_u32 = u as u32;
        for &v in adj1.neighbors(u) {
            if u_u32 < v {
                u1.push(u_u32);
                v1.push(v);
            }
        }
    }

    if max_2hop_per_node == 0 || n <= 2 {
        return Ok(Sparse12HopPairs {
            u1,
            v1,
            u2: Vec::new(),
            v2: Vec::new(),
        });
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut stamp = VisitStamp::new(n);
    let mut raw_2hop: Vec<(u32, u32)> = Vec::new();

    for u in 0..n {
        let n1_u = adj1.neighbors(u);
        if n1_u.is_empty() {
            continue;
        }
        stamp.clear();
        stamp.mark(u);
        for &w in n1_u {
            stamp.mark(w as usize);
        }

        let u_u32 = u as u32;
        let mut collected = 0usize;
        for &w in n1_u {
            let n1_w = adj1.neighbors(w as usize);
            let deg_w = n1_w.len();
            if deg_w <= 1 {
                continue;
            }
            if deg_w <= max_2hop_per_node {
                for &v in n1_w {
                    if stamp.visit(v as usize) {
                        raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        collected += 1;
                        if collected >= max_2hop_per_node {
                            break;
                        }
                    }
                }
            } else {
                let n_draws = max_2hop_per_node.saturating_mul(2).min(deg_w);
                for _ in 0..n_draws {
                    let v = n1_w[rng.gen_range_usize(deg_w)];
                    if stamp.visit(v as usize) {
                        raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        collected += 1;
                        if collected >= max_2hop_per_node {
                            break;
                        }
                    }
                }
            }
            if collected >= max_2hop_per_node {
                break;
            }
        }
    }

    raw_2hop.sort_unstable();
    raw_2hop.dedup();
    let mut u2 = Vec::with_capacity(raw_2hop.len());
    let mut v2 = Vec::with_capacity(raw_2hop.len());
    for (a, b) in raw_2hop {
        u2.push(a);
        v2.push(b);
    }

    Ok(Sparse12HopPairs { u1, v1, u2, v2 })
}

/// Builds a sorted symmetric CSR [`KnnGraph`] from 1-hop and 2-hop pair distances while strictly
/// enforcing $d_1 < \mathrm{tier2\_margin} \cdot d_2$ and $d_1 < d_2$ over incident 1-hop maxima.
#[allow(clippy::too_many_arguments)]
pub fn build_gumap_12hop_priority_csr(
    n_nodes: usize,
    u1: &[u32],
    v1: &[u32],
    d1: &[f64],
    u2: &[u32],
    v2: &[u32],
    d2: &[f64],
    tier2_margin: f64,
) -> CoreResult<KnnGraph> {
    if u32::try_from(n_nodes).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if !tier2_margin.is_finite() {
        return Err(CoreError::non_finite(
            "build_gumap_12hop_priority_csr::tier2_margin",
        ));
    }
    if tier2_margin <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "tier2_margin must be > 0, got {tier2_margin}"
        )));
    }
    if u1.len() != v1.len() || u1.len() != d1.len() {
        return Err(CoreError::dimension_mismatch(
            u1.len(),
            v1.len().min(d1.len()),
            "build_gumap_12hop_priority_csr::tier1",
        ));
    }
    if u2.len() != v2.len() || u2.len() != d2.len() {
        return Err(CoreError::dimension_mismatch(
            u2.len(),
            v2.len().min(d2.len()),
            "build_gumap_12hop_priority_csr::tier2",
        ));
    }

    for (&u, (&v, &d)) in u1.iter().zip(v1.iter().zip(d1.iter())) {
        if (u as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::u1",
            ));
        }
        if (v as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::v1",
            ));
        }
        if !d.is_finite() {
            return Err(CoreError::non_finite("build_gumap_12hop_priority_csr::d1"));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input("d1 distances must be >= 0.0"));
        }
    }
    for (&u, (&v, &d)) in u2.iter().zip(v2.iter().zip(d2.iter())) {
        if (u as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::u2",
            ));
        }
        if (v as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::v2",
            ));
        }
        if !d.is_finite() {
            return Err(CoreError::non_finite("build_gumap_12hop_priority_csr::d2"));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input("d2 distances must be >= 0.0"));
        }
    }

    if n_nodes <= 1 || (u1.is_empty() && u2.is_empty()) {
        return KnnGraph::empty(n_nodes);
    }

    let mut d1_max = vec![0.0_f64; n_nodes];
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_nodes];

    for (&u, (&v, &d_raw)) in u1.iter().zip(v1.iter().zip(d1.iter())) {
        if u == v {
            continue;
        }
        let d = d_raw.max(1e-12);
        d1_max[u as usize] = d1_max[u as usize].max(d);
        d1_max[v as usize] = d1_max[v as usize].max(d);
        rows[u as usize].push((v, d));
        rows[v as usize].push((u, d));
    }

    for (&u, (&v, &d_raw)) in u2.iter().zip(v2.iter().zip(d2.iter())) {
        if u == v {
            continue;
        }
        let d = d_raw.max(1e-12);
        let d1_pair = d1_max[u as usize].max(d1_max[v as usize]);
        // Guarantee both d1 < tier2_margin * d2 and strict 1-hop priority d1 < d2 even when
        // tier2_margin <= 1.0 or incident 1-hop edges saturate floating-point precision.
        let d2_floor = if d1_pair > 0.0 {
            if tier2_margin >= 1.0 {
                (tier2_margin * d1_pair).max(d1_pair.next_up())
            } else {
                (d1_pair / tier2_margin).next_up()
            }
        } else {
            1e-12
        };
        let d_adj = d.max(d2_floor);
        rows[u as usize].push((v, d_adj));
        rows[v as usize].push((u, d_adj));
    }

    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0usize);
    let mut indices = Vec::new();
    let mut distances = Vec::new();

    for row in &mut rows {
        row.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        let mut prev_col: Option<u32> = None;
        for &(col, dist) in row.iter() {
            if prev_col == Some(col) {
                continue;
            }
            prev_col = Some(col);
            indices.push(col);
            distances.push(dist);
        }
        indptr.push(indices.len());
    }

    KnnGraph::new_validated(n_nodes, indptr, indices, distances)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn cmp_dist_then_id(a: &(f64, u32), b: &(f64, u32)) -> Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_1hop_2hop_and_pure_2hop_knn_exact_disjoint_and_asymmetric() {
        // Star hub 0 connected to 1..=6, plus chain 6-7-8-9 and isolated node 10.
        let mut edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)];
        edges.extend([(6, 7), (7, 8), (8, 9)]);
        let g = CsrGraph::<f64>::from_edges(11, &edges, None, false, false).unwrap();

        let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
        assert_eq!(adj1.neighbors(0), &[1, 2, 3, 4, 5, 6]);
        // Pure 2-hop of node 0 is {7} only (>=3-hop {8,9} and isolated {10} excluded).
        assert_eq!(adj2.neighbors(0), &[7]);
        // Pure 2-hop of node 1 is {2, 3, 4, 5, 6}.
        assert_eq!(adj2.neighbors(1), &[2, 3, 4, 5, 6]);
        assert!(adj1.neighbors(10).is_empty());
        assert!(adj2.neighbors(10).is_empty());

        let (knn, k2_deg, stats) = build_pure_2hop_knn(&g).unwrap();
        assert_eq!(k2_deg[0], 7);
        assert_eq!(k2_deg[10], 0);
        assert_eq!(stats.n_1hop, adj1.nnz());
        assert_eq!(stats.n_2hop, adj2.nnz());
        assert_eq!(stats.nnz, adj1.nnz() + adj2.nnz());
        assert_eq!(stats.n_truncated_1hop, 0);
        assert_eq!(stats.n_truncated_2hop, 0);
        assert_eq!(stats.n_filled_residual, 0);
        assert_eq!(stats.n_pure_2hop_rows, 11);

        let (row0_idx, row0_dist) = knn.row(0);
        assert_eq!(row0_idx, &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(row0_dist, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]);

        // Directed asymmetric CSR must be symmetrized cleanly without self-loops.
        let dg =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, true, false).unwrap();
        let (d1, d2) = extract_1hop_2hop_csr(&dg).unwrap();
        assert_eq!(d1.neighbors(1), &[0, 2]);
        assert_eq!(d2.neighbors(0), &[2]);
        assert_eq!(d2.neighbors(2), &[0]);
    }

    #[test]
    fn test_extract_sparse_12hop_pairs_hub_safe_and_deterministic() {
        // Star graph on 500 nodes: naive 2-hop has 499*498/2 = 124,251 pairs; hub-safe caps it.
        let n = 500;
        let edges: Vec<(u32, u32)> = (1..n as u32).map(|i| (0, i)).collect();
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

        let p1 = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
        let p2 = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.u1.len(), n - 1);
        assert!(p1.u2.len() <= n * 8);
        for (&a, &b) in p1.u2.iter().zip(p1.v2.iter()) {
            assert!(a < b);
            assert_ne!(a, 0);
        }
    }

    #[test]
    fn test_gumap_12hop_priority_csr_and_validation() {
        let knn = build_gumap_12hop_priority_csr(
            4,
            &[0, 1],
            &[1, 2],
            &[2.0, 3.0],
            &[0],
            &[2],
            &[0.5],
            1.2,
        )
        .unwrap();
        // Incident 1-hop max for (0, 2) is max(2.0, 3.0) = 3.0 -> d2 clamped >= 1.2 * 3.0 = 3.6.
        let (r0_idx, r0_dist) = knn.row(0);
        assert_eq!(r0_idx, &[1, 2]);
        assert!((r0_dist[0] - 2.0).abs() < 1e-12);
        assert!(r0_dist[1] >= 3.6 - 1e-12);
        assert!(r0_dist[0] < r0_dist[1]);

        // Test tier2_margin < 1.0 still strictly satisfies d1 < margin * d2 and d1 < d2.
        let knn_lt = build_gumap_12hop_priority_csr(
            3,
            &[0, 1],
            &[1, 2],
            &[2.0, 3.0],
            &[0],
            &[2],
            &[1.0],
            0.5,
        )
        .unwrap();
        let d2_lt = knn_lt.row(0).1[1];
        assert!(0.5 * d2_lt > 3.0);

        // Error checks on KnnGraph and priority builder.
        assert!(KnnGraph::new_validated(2, vec![0, 1], vec![1], vec![1.0]).is_err());
        assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![1], vec![-0.1]).is_err());
        assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![1], vec![f64::NAN]).is_err());
        assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![5], vec![1.0]).is_err());
        assert!(build_gumap_12hop_priority_csr(3, &[0], &[1], &[1.0], &[], &[], &[], 0.0).is_err());
    }

    #[test]
    fn test_knn_empty_and_singleton_non_panic() {
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let (e1, e2) = extract_1hop_2hop_csr(&empty).unwrap();
        assert_eq!(e1.num_nodes(), 0);
        assert_eq!(e2.num_nodes(), 0);
        let (ke, deg_e, se) = build_pure_2hop_knn(&empty).unwrap();
        assert_eq!(ke.num_nodes(), 0);
        assert!(deg_e.is_empty());
        assert_eq!(se.nnz, 0);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let (ks, deg_s, ss) = build_pure_2hop_knn(&single).unwrap();
        assert_eq!(ks.num_nodes(), 1);
        assert_eq!(deg_s, vec![0]);
        assert_eq!(ss.n_pure_2hop_rows, 1);
    }
}
