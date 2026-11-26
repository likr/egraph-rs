//! Perplexity configuration constants, symmetrization strategies, and canonical symmetric CSR
//! joint probability storage (`SparseProbabilities`).

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
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl SparseProbabilities {
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

    /// Returns `(neighbor_indices, probabilities)` for row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n_nodes`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        (&self.indices[start..end], &self.values[start..end])
    }
}
