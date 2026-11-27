//! Pure-Rust two-phase multi-level Louvain community detection using flat CSR and
//! `VisitStamp` accumulators (zero `HashMap` in inner loops or aggregation).

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use crate::clusterer::{ClusterResult, CommunityDetector};

mod level_graph;
mod solver;

#[cfg(test)]
mod tests;

pub use solver::louvain;

/// Configuration for [`Louvain`] and [`louvain`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

impl LouvainConfig {
    /// Creates a new [`LouvainConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of multi-level Louvain passes `max_passes`.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        self.max_passes = max_passes;
        self
    }

    /// Sets the modularity resolution parameter `resolution`.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Sets the minimum modularity gain threshold `min_modularity_gain`.
    pub fn with_min_modularity_gain(mut self, min_modularity_gain: f64) -> Self {
        self.min_modularity_gain = min_modularity_gain;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all Louvain hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.max_passes == 0 {
            return Err(CoreError::invalid_input("max_passes must be >= 1"));
        }
        if !self.resolution.is_finite() {
            return Err(CoreError::non_finite("LouvainConfig::resolution"));
        }
        if self.resolution < 0.0 {
            return Err(CoreError::invalid_input("resolution must be >= 0.0"));
        }
        if !self.min_modularity_gain.is_finite() {
            return Err(CoreError::non_finite("LouvainConfig::min_modularity_gain"));
        }
        if self.min_modularity_gain < 0.0 {
            return Err(CoreError::invalid_input(
                "min_modularity_gain must be >= 0.0",
            ));
        }
        Ok(())
    }
}

/// Output of [`Louvain::run`] and [`louvain`].
#[non_exhaustive]
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
    /// Whether Louvain converged before exhausting `max_passes`.
    pub converged: bool,
}

impl From<LouvainResult> for ClusterResult {
    fn from(res: LouvainResult) -> Self {
        Self::new(
            res.labels,
            res.num_communities,
            Some(res.modularity),
            res.passes,
            res.converged,
        )
    }
}

/// Two-phase multi-level Louvain community detector implementing [`CommunityDetector`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Louvain {
    config: LouvainConfig,
}

impl Louvain {
    /// Creates a new [`Louvain`] community detector with the given configuration.
    pub fn new(config: LouvainConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the underlying [`LouvainConfig`].
    #[inline]
    pub fn config(&self) -> &LouvainConfig {
        &self.config
    }

    /// Replaces the configuration with `config`.
    pub fn with_config(mut self, config: LouvainConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the maximum number of multi-level Louvain passes `max_passes`.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        self.config.max_passes = max_passes;
        self
    }

    /// Sets the modularity resolution parameter `resolution`.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.config.resolution = resolution;
        self
    }

    /// Sets the minimum modularity gain threshold `min_modularity_gain`.
    pub fn with_min_modularity_gain(mut self, min_modularity_gain: f64) -> Self {
        self.config.min_modularity_gain = min_modularity_gain;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Runs Louvain community detection and returns the specialized [`LouvainResult`].
    pub fn run<W: Scalar>(&self, graph: &CsrGraph<W>) -> CoreResult<LouvainResult> {
        louvain(graph, &self.config)
    }
}

impl<W: Scalar> CommunityDetector<W> for Louvain {
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult> {
        self.run(graph).map(Into::into)
    }
}
