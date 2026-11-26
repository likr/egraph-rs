use egraph_core::{CoreError, CoreResult, CsrGraph};

/// Configuration for multi-level Louvain coarsening up to a coarse-node budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoarseLouvainConfig {
    /// Target upper bound on the number of coarsest super-nodes (`default: 10_000`).
    ///
    /// Coarsening stops once `num_clusters <= max_coarse_nodes`, or earlier at the
    /// connected-component floor if the graph has more than `max_coarse_nodes` disconnected
    /// components.
    pub max_coarse_nodes: usize,
    /// Initial Louvain modularity resolution $\gamma \ge 0.0$ (`default: 1.0`).
    pub resolution: f64,
    /// Multiplicative resolution decay factor in `(0.0, 1.0)` applied when a round stalls
    /// above `max_coarse_nodes` (`default: 0.5`).
    pub resolution_decay: f64,
    /// Maximum number of inner Louvain passes per outer coarsening round (`default: 100`).
    pub max_passes: usize,
    /// Minimum modularity gain threshold for inner Louvain passes (`default: 1e-7`).
    pub min_modularity_gain: f64,
    /// Maximum number of outer Louvain contraction rounds (`default: 16`).
    pub max_outer_rounds: usize,
    /// Explicit deterministic PRNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for CoarseLouvainConfig {
    fn default() -> Self {
        Self {
            max_coarse_nodes: 10_000,
            resolution: 1.0,
            resolution_decay: 0.5,
            max_passes: 100,
            min_modularity_gain: 1e-7,
            max_outer_rounds: 16,
            seed: 42,
        }
    }
}

impl CoarseLouvainConfig {
    /// Validates all multi-level Louvain coarsening hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.max_coarse_nodes == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_coarse_nodes must be >= 1",
            ));
        }
        if !self.resolution.is_finite() {
            return Err(CoreError::non_finite("CoarseLouvainConfig::resolution"));
        }
        if self.resolution < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::resolution must be >= 0.0, got {}",
                self.resolution
            )));
        }
        if !self.resolution_decay.is_finite() {
            return Err(CoreError::non_finite(
                "CoarseLouvainConfig::resolution_decay",
            ));
        }
        if self.resolution_decay <= 0.0 || self.resolution_decay >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::resolution_decay must be in (0.0, 1.0), got {}",
                self.resolution_decay
            )));
        }
        if self.max_passes == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_passes must be >= 1",
            ));
        }
        if !self.min_modularity_gain.is_finite() {
            return Err(CoreError::non_finite(
                "CoarseLouvainConfig::min_modularity_gain",
            ));
        }
        if self.min_modularity_gain < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::min_modularity_gain must be >= 0.0, got {}",
                self.min_modularity_gain
            )));
        }
        if self.max_outer_rounds == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_outer_rounds must be >= 1",
            ));
        }
        Ok(())
    }
}

/// Single contraction level recorded by [`louvain_coarsen`](super::louvain_coarsen).
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedHierarchyLevel {
    /// Contracted super-graph produced at this round.
    pub coarse: CoarsenedGraph,
    /// Louvain resolution $\gamma$ used at this round.
    pub resolution: f64,
    /// Modularity $Q_\gamma$ achieved at this round.
    pub modularity: f64,
}

/// Multi-level Louvain coarsening hierarchy from fine (`levels[0]`) to coarsest (`coarsest`).
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedHierarchy {
    /// Ordered outer contraction levels (`0..num_levels`).
    pub levels: Vec<CoarsenedHierarchyLevel>,
    /// Coarsest super-graph with direct original-to-coarsest mapping and cumulative sizes.
    pub coarsest: CoarsenedGraph,
}

/// Result of contracting vertex clusters via [`coarsen_graph`](super::coarsen_graph).
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedGraph {
    /// Weighted undirected super-graph over `0..num_clusters`.
    pub graph: CsrGraph<f64>,
    /// Compacted cluster ID in `0..num_clusters` for each original vertex.
    pub node_to_coarse: Vec<u32>,
    /// Number of original vertices belonging to each coarse cluster.
    pub coarse_sizes: Vec<u32>,
    /// Number of distinct compacted clusters.
    pub num_clusters: usize,
}
