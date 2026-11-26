//! Perplexity configuration constants, symmetrization strategies, and canonical symmetric CSR
//! joint probability storage (`SparseProbabilities`).

use egraph_core::{CoreError, CoreResult};

/// Default multiplier $c$ for node-wise 2-hop adaptive perplexity (`c = 0.9`).
pub const DEFAULT_NODEWISE_PERPLEXITY_C: f64 = 0.9;
/// Default lower clamp for node-wise 2-hop adaptive perplexity (`5.0`).
pub const DEFAULT_NODEWISE_MIN_PERPLEXITY: f64 = 5.0;
/// Default multiplier $c$ for median 2-hop adaptive perplexity (`c = 1.5`).
pub const DEFAULT_MEDIAN_PERPLEXITY_C: f64 = 1.5;
/// Default lower clamp for median 2-hop adaptive perplexity (`15.0`).
pub const DEFAULT_MEDIAN_MIN_PERPLEXITY: f64 = 15.0;
/// Default upper clamp for 2-hop adaptive perplexity (`1200.0`).
pub const DEFAULT_MAX_PERPLEXITY: f64 = 1200.0;

/// Symmetrization strategy applied to directed conditional probabilities or graph affinities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SymmetrizationMode {
    /// Standard t-SNE joint normalization: $P_{ij} = (p_{j|i} + p_{i|j}) / (2 N)$.
    #[default]
    StandardTsne,
    /// Explicit row-normalized symmetrization: $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$.
    RowNormalized,
}

/// Configuration for perplexity-based sparse probability computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerplexityConfig {
    /// Power exponent applied to normalized distances `(d / max_d)^power` (`default: 2.0`).
    pub power: f64,
    /// Maximum binary-search iterations for conditional entropy matching (`default: 50`).
    pub sigma_iters: usize,
    /// Entropy convergence tolerance (`default: 1e-5`).
    pub sigma_tolerance: f64,
    /// Symmetrization strategy (`default: SymmetrizationMode::StandardTsne`).
    pub symmetrization: SymmetrizationMode,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            power: 2.0,
            sigma_iters: 50,
            sigma_tolerance: 1e-5,
            symmetrization: SymmetrizationMode::StandardTsne,
        }
    }
}

/// Canonical sorted symmetric Compressed Sparse Row joint probability matrix $P$.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseProbabilities {
    pub n_nodes: usize,
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl SparseProbabilities {
    /// Constructs a [`SparseProbabilities`] after validating CSR offsets, vertex bounds, and
    /// finite non-negative probability values.
    pub fn from_csr(
        n_nodes: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        values: Vec<f64>,
    ) -> CoreResult<Self> {
        if u32::try_from(n_nodes).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if indptr.len() != n_nodes + 1 {
            return Err(CoreError::dimension_mismatch(
                n_nodes + 1,
                indptr.len(),
                "SparseProbabilities::from_csr::indptr",
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
                "SparseProbabilities::from_csr::indices",
            ));
        }
        if values.len() != nnz {
            return Err(CoreError::dimension_mismatch(
                nnz,
                values.len(),
                "SparseProbabilities::from_csr::values",
            ));
        }
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if (idx as usize) >= n_nodes {
                return Err(CoreError::index_out_of_bounds(
                    idx as usize,
                    n_nodes,
                    "SparseProbabilities::from_csr::indices",
                ));
            }
            if !val.is_finite() {
                return Err(CoreError::non_finite(
                    "SparseProbabilities::from_csr::values",
                ));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "SparseProbabilities values must be non-negative, got {val}"
                )));
            }
        }
        for u in 0..n_nodes {
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            let row = &indices[start..end];
            if !row.windows(2).all(|w| w[0] < w[1]) {
                return Err(CoreError::invalid_csr(format!(
                    "SparseProbabilities row {u} indices must be strictly sorted and duplicate-free"
                )));
            }
        }
        for u in 0..n_nodes {
            let u32_u = u as u32;
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for pos in start..end {
                let v = indices[pos] as usize;
                let v_start = indptr[v] as usize;
                let v_end = indptr[v + 1] as usize;
                let Ok(rev_offset) = indices[v_start..v_end].binary_search(&u32_u) else {
                    return Err(CoreError::invalid_csr(format!(
                        "SparseProbabilities must be symmetric: missing reverse entry ({v}, {u})"
                    )));
                };
                let p_uv = values[pos];
                let p_vu = values[v_start + rev_offset];
                let tol = 1e-12 * 1.0_f64.max(p_uv).max(p_vu);
                if (p_uv - p_vu).abs() > tol {
                    return Err(CoreError::invalid_csr(format!(
                        "SparseProbabilities asymmetric probability for ({u}, {v}): {p_uv} != {p_vu}"
                    )));
                }
            }
        }
        Ok(Self {
            n_nodes,
            indptr,
            indices,
            values,
        })
    }

    /// Creates an empty [`SparseProbabilities`] with `n_nodes` zero-degree rows.
    pub fn empty(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            indptr: vec![0; n_nodes + 1],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns the total number of stored probability entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
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

    /// Returns the flat probability slice aligned with `indices()`.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns `(neighbor_indices, probabilities)` for row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n_nodes`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        (&self.indices[start..end], &self.values[start..end])
    }
}
