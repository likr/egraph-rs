//! Validated CSR $k$-NN storage (`KnnGraph`), builder statistics (`KnnStats`), and sparse
//! 1+2-hop node-pair buffers (`Sparse12HopPairs`).

use egraph_core::{CoreError, CoreResult};

/// Variable-length Compressed Sparse Row (CSR) $k$-nearest-neighbor graph in `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnGraph {
    n_nodes: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    distances: Vec<f64>,
}

impl KnnGraph {
    /// Constructs a [`KnnGraph`] after validating CSR offsets, vertex bounds, and finite
    /// non-negative distances.
    pub fn new_validated(
        n_nodes: usize,
        indptr: Vec<u32>,
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
        let nnz = indptr[n_nodes] as usize;
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
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
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
    pub fn indptr(&self) -> &[u32] {
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
