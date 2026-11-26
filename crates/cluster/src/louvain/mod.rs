//! Pure-Rust two-phase multi-level Louvain community detection using flat CSR and
//! `VisitStamp` accumulators (zero `HashMap` in inner loops or aggregation).

mod level_graph;
mod solver;

#[cfg(test)]
mod tests;

pub use solver::louvain;

/// Configuration for [`louvain`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LouvainConfig {
    /// Maximum number of multi-level Louvain passes (local moving + aggregation).
    pub max_passes: usize,
    /// Modularity resolution parameter $\gamma \ge 0.0$ (`1.0` for standard modularity).
    pub resolution: f64,
    /// Minimum modularity gain required to commit a pass and aggregate communities.
    pub min_modularity_gain: f64,
    /// Deterministic PRNG seed controlling vertex visit order in Phase 1 sweeps.
    pub seed: u64,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_passes: 100,
            resolution: 1.0,
            min_modularity_gain: 1e-7,
            seed: 42,
        }
    }
}

/// Output of [`louvain`].
#[derive(Debug, Clone, PartialEq)]
pub struct LouvainResult {
    /// Canonical community assignment in `0..num_communities` for each vertex.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty communities detected.
    pub num_communities: usize,
    /// Modularity $Q_\gamma$ of the returned partition.
    pub modularity: f64,
    /// Number of multi-level passes (local moving + aggregation) committed.
    pub passes: usize,
}
