mod barrier;
mod landmark_barrier;
mod pivot;
mod pivoted_wrappers;
mod two_hop;

use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_algorithm_shortest_path::{multi_source_dijkstra, PivotedDistanceMatrix};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use pivot::max_min_random_sp;
use rand::prelude::*;

/// Symmetrization rule used for the 2-hop barrier radius `R_pair(u, v)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierSymmetry {
    /// Use `max(R_2(u), R_2(v))`.
    SymMax,
    /// Use `0.5 * (R_2(u) + R_2(v))`.
    SymAvg,
}

/// Summary statistics emitted by [`SparseSgd::build_barrier_from_landmark_tables`].
#[derive(Clone, Debug, PartialEq)]
pub struct BarrierBuildStats {
    pub n_pairs_1hop: usize,
    pub n_pairs_2hop: usize,
    pub n_pairs_non2: usize,
    pub n_pairs_hinge: usize,
    pub n_pairs_two_sided: usize,
    pub perplexity_val: f64,
    pub per_node_neg_used: usize,
    pub perplexity_source: &'static str,
    pub d_floor: f64,
}

/// Builder for creating Sparse SGD instances.
///
/// This builder uses a sparse approximation technique that selects a subset of
/// pivot nodes and computes shortest-path distances only from these pivot nodes to all
/// other nodes. This approach significantly reduces the computational complexity for
/// large graphs compared to the full SGD algorithm, while still producing good layouts.
///
/// Opt-in builder fields additionally enable 1-hop anchor calibration, hub-safe
/// `O(|E| * max_2hop_per_node)` 2-hop pair registration, relative-margin barrier spacing,
/// and one-sided hinge repulsion on non-2-hop pairs.
pub struct SparseSgd {
    /// Number of pivot nodes to use (controls sparsity)
    h: usize,
    /// Whether to calibrate 1-hop edge target lengths against the distance oracle.
    calibrate_one_hop: bool,
    /// Maximum 2-hop pairs registered per node (0 disables 2-hop registration).
    max_2hop_per_node: usize,
    /// Whether Tier 3 non-2-hop barrier pairs use one-sided hinge repulsion.
    hinge_repulsion: bool,
    /// Relative margin multiplier `delta_rel >= 0` for the 2-hop barrier.
    delta_rel: f64,
    /// Symmetrization rule for `R_pair(u, v)`.
    barrier_sym: BarrierSymmetry,
    /// Tier 1 (1-hop) spring weight multiplier.
    omega_1: f64,
    /// Tier 2 (2-hop) spring weight multiplier.
    omega_2: f64,
    /// Tier 3 (non-2-hop) weight multiplier.
    gamma_non2: f64,
    /// Relative distance floor multiplier `min_ratio * median(d1_raw > 0)`.
    min_ratio: f64,
    /// Absolute minimum target distance clamp `> 0`.
    eps_dist: f64,
    /// Deterministic RNG seed for hub-safe 2-hop subsampling and non-2-hop sampling.
    seed: u64,
}

impl Default for SparseSgd {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSgd {
    /// Creates a new `SparseSgd` builder with legacy backward-compatible defaults.
    pub fn new() -> Self {
        Self {
            h: 200,
            calibrate_one_hop: false,
            max_2hop_per_node: 0,
            hinge_repulsion: false,
            delta_rel: 0.0,
            barrier_sym: BarrierSymmetry::SymMax,
            omega_1: 1.0,
            omega_2: 1.0,
            gamma_non2: 1.0,
            min_ratio: 0.0,
            eps_dist: 1e-6,
            seed: 42,
        }
    }

    /// Creates a `SparseSgd` preset configured for 1-hop calibration, hub-safe 2-hop
    /// registration, relative-margin barrier spacing, and one-sided hinge repulsion.
    pub fn barrier() -> Self {
        Self {
            h: 64,
            calibrate_one_hop: true,
            max_2hop_per_node: 32,
            hinge_repulsion: true,
            delta_rel: 0.25,
            barrier_sym: BarrierSymmetry::SymMax,
            omega_1: 1.0,
            omega_2: 1.0,
            gamma_non2: 1.0,
            min_ratio: 0.5,
            eps_dist: 1e-6,
            seed: 42,
        }
    }

    /// Sets the number of pivot nodes.
    pub fn h(&mut self, h: usize) -> &mut Self {
        self.h = h;
        self
    }

    /// Enables or disables 1-hop anchor calibration against the distance oracle.
    pub fn calibrate_one_hop(&mut self, enabled: bool) -> &mut Self {
        self.calibrate_one_hop = enabled;
        self
    }

    /// Sets the hub-safe per-node cap on registered 2-hop pairs (`0` disables 2-hop pairs).
    pub fn max_2hop_per_node(&mut self, cap: usize) -> &mut Self {
        self.max_2hop_per_node = cap;
        self
    }

    /// Enables or disables one-sided hinge repulsion on non-2-hop barrier pairs.
    pub fn hinge_repulsion(&mut self, enabled: bool) -> &mut Self {
        self.hinge_repulsion = enabled;
        self
    }

    /// Sets the relative margin barrier multiplier `delta_rel >= 0`.
    pub fn delta_rel(&mut self, delta_rel: f64) -> &mut Self {
        self.delta_rel = delta_rel;
        self
    }

    /// Sets the symmetrization rule for `R_pair(u, v)`.
    pub fn barrier_sym(&mut self, barrier_sym: BarrierSymmetry) -> &mut Self {
        self.barrier_sym = barrier_sym;
        self
    }

    /// Sets the Tier 1 (1-hop) spring weight multiplier `omega_1 >= 0`.
    pub fn omega_1(&mut self, omega_1: f64) -> &mut Self {
        self.omega_1 = omega_1;
        self
    }

    /// Sets the Tier 2 (2-hop) spring weight multiplier `omega_2 >= 0`.
    pub fn omega_2(&mut self, omega_2: f64) -> &mut Self {
        self.omega_2 = omega_2;
        self
    }

    /// Sets the Tier 3 (non-2-hop) weight multiplier `gamma_non2 >= 0`.
    pub fn gamma_non2(&mut self, gamma_non2: f64) -> &mut Self {
        self.gamma_non2 = gamma_non2;
        self
    }

    /// Sets the relative distance floor multiplier `min_ratio >= 0`.
    pub fn min_ratio(&mut self, min_ratio: f64) -> &mut Self {
        self.min_ratio = min_ratio;
        self
    }

    /// Sets the minimum pairwise target distance clamp `eps_dist > 0`.
    pub fn eps_dist(&mut self, eps_dist: f64) -> &mut Self {
        self.eps_dist = eps_dist;
        self
    }

    /// Sets the deterministic RNG seed used for 2-hop subsampling and non-2-hop sampling.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Creates a new SGD instance with a provided random number generator.
    ///
    /// This variant allows using a specific random number generator
    /// for deterministic behavior in testing scenarios.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - The random number generator to use for selecting pivots
    ///
    /// # Returns
    /// A new SGD instance configured with appropriate node pairs
    pub fn build<G, F, R, S>(&self, graph: G, length: F, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        let mut length = length;
        let n = graph.node_count();
        let h = self.h.min(n);
        if n <= 1 || h == 0 {
            return Sgd::new(vec![]);
        }
        let (pivot, d) = Self::choose_pivot(graph, &mut length, h, rng);
        self.build_with_pivot_and_distance_matrix(graph, length, &pivot, &d)
    }

    /// Creates a new SGD instance with pre-selected pivot nodes.
    ///
    /// This method allows using a specific set of pivot nodes instead of
    /// selecting them randomly, which can be useful when you have domain knowledge
    /// about which nodes might make good pivots.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `pivot` - A slice of node IDs to use as pivot nodes
    ///
    /// # Returns
    /// A new SGD instance configured with the specified pivot nodes
    pub fn build_with_pivot<G, F, S>(&self, graph: G, mut length: F, pivot: &[G::NodeId]) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = multi_source_dijkstra(graph, &mut length, pivot);
        self.build_with_pivot_and_distance_matrix(graph, &mut length, pivot, &d)
    }

    /// Selects pivot nodes using a max-min randomized algorithm.
    ///
    /// This function implements a maximal-minimal distance pivot selection strategy.
    /// It first selects a random node as the first pivot, then iteratively selects
    /// nodes that maximize the minimum distance to all previously selected pivots.
    ///
    /// # Parameters
    /// * `graph` - The input graph to select pivots from
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `h` - The number of pivot nodes to select
    /// * `rng` - The random number generator to use
    ///
    /// # Returns
    /// A tuple containing:
    /// - A vector of selected pivot node IDs
    /// - A SubDistanceMatrix containing distances from pivots to all nodes
    pub fn choose_pivot<G, F, R, S>(
        graph: G,
        length: F,
        h: usize,
        rng: &mut R,
    ) -> (Vec<G::NodeId>, PivotedDistanceMatrix<G::NodeId, S>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        max_min_random_sp(graph, length, h, rng)
    }
}

#[cfg(test)]
mod tests;
