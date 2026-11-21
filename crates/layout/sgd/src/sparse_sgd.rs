use crate::Sgd;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_algorithm_shortest_path::{
    dijkstra_with_distance_matrix, multi_source_dijkstra, PivotedDistanceMatrix,
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

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

    /// Creates a new SGD instance with pre-selected pivot nodes and a pre-computed distance matrix.
    ///
    /// This is the most low-level method, useful when you have already computed
    /// the distance matrix or want to provide a custom distance matrix.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `pivot` - A slice of node IDs to use as pivot nodes
    /// * `distance_matrix` - A pre-computed distance matrix from pivots to all nodes
    ///
    /// # Returns
    /// A new SGD instance configured with the specified pivot nodes and distances
    pub fn build_with_pivoted_neg_log_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogDistance<G::NodeId, S, K>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.build_with_pivoted_kernel_distance(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix,
        )
    }

    pub fn build_with_pivoted_neg_log_sim_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogSimDistance<G::NodeId, S, K>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.build_with_pivoted_kernel_distance(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix,
        )
    }

    fn build_with_pivoted_kernel_distance<G, F, S>(
        &self,
        graph: G,
        length: F,
        pivots: &[usize],
        distance_matrix: &dyn Distance<G::NodeId, S>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let pivot_nodes = pivots.iter().map(|&idx| nodes[idx]).collect::<Vec<_>>();
        self.build_with_pivot_and_distance_matrix(graph, length, &pivot_nodes, distance_matrix)
    }

    pub fn build_with_pivoted_distance_matrix<G, F, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &PivotedDistanceMatrix<G::NodeId, S>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        self.build_with_pivot_and_distance_matrix(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix,
        )
    }

    pub fn build_with_pivot_and_distance_matrix<G, F, S>(
        &self,
        graph: G,
        mut length: F,
        pivot: &[G::NodeId],
        distance_matrix: &dyn Distance<G::NodeId, S>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let h = pivot.len();
        let is_legacy = !self.calibrate_one_hop
            && self.max_2hop_per_node == 0
            && !self.hinge_repulsion
            && self.delta_rel == 0.0
            && self.min_ratio == 0.0
            && self.omega_1 == 1.0
            && self.gamma_non2 == 1.0;

        if is_legacy {
            let mut node_pairs = vec![];
            let mut edges = HashSet::new();
            for edge in graph.edge_references() {
                let i = indices[&edge.source()];
                let j = indices[&edge.target()];
                let dij = length(edge);
                let wij = S::one() / (dij * dij);
                node_pairs.push((i, j, dij, dij, wij, wij));
                edges.insert((i, j));
                edges.insert((j, i));
            }
            if h == 0 || n <= 1 {
                return Sgd::new(node_pairs);
            }

            // Cache h x N pivot distances once so nearest-pivot assignment and Voronoi counting
            // avoid O(N^2) distance queries.
            let mut piv_dists = vec![S::infinity(); h * n];
            for (k, &u) in pivot.iter().enumerate() {
                for j in 0..n {
                    piv_dists[k * n + j] =
                        distance_matrix.get(u, nodes[j]).unwrap_or(S::infinity());
                }
            }

            let mut r_dists = vec![vec![]; h];
            for j in 0..n {
                let best_k = (0..h)
                    .min_by_key(|&k| OrderedFloat(piv_dists[k * n + j]))
                    .unwrap();
                r_dists[best_k].push(piv_dists[best_k * n + j]);
            }
            for dists_k in &mut r_dists {
                dists_k.sort_by_key(|&d| OrderedFloat(d));
            }

            let two = S::from_usize(2).unwrap();
            for (k, &u) in pivot.iter().enumerate() {
                let p = indices[&u];
                for i in 0..n {
                    if edges.contains(&(p, i)) || p == i {
                        continue;
                    }
                    let dpi = piv_dists[k * n + i];
                    if !dpi.is_finite() || dpi <= S::zero() {
                        continue;
                    }
                    let wpi = S::one() / (dpi * dpi);
                    let count =
                        r_dists[k].partition_point(|&dpj| dpj.is_finite() && two * dpj <= dpi);
                    let spi = S::from_usize(count).unwrap();
                    node_pairs.push((p, i, dpi, dpi, S::zero(), spi * wpi));
                }
            }
            return Sgd::new(node_pairs);
        }

        if n <= 1 {
            return Sgd::new(vec![]);
        }

        // Step A: Canonical undirected 1-hop edges and sorted CSR-like adjacency lists.
        let (edges_1hop, adj) = extract_canonical_edges_and_adj(graph, &indices, &mut length);

        // Step B: Cache h x N pivot distances in O(N h) memory without O(N^2) allocations.
        let mut pivot_slot: Vec<Option<usize>> = vec![None; n];
        let mut pivot_node_indices = Vec::with_capacity(h);
        let mut piv_dists = vec![S::infinity(); h * n];
        for (k, &u) in pivot.iter().enumerate() {
            let p = indices[&u];
            pivot_slot[p] = Some(k);
            pivot_node_indices.push(p);
            for j in 0..n {
                piv_dists[k * n + j] = distance_matrix
                    .get(u, nodes[j])
                    .or_else(|| distance_matrix.get(nodes[j], u))
                    .unwrap_or(S::infinity());
            }
        }

        let resolve_pair_distance = |u: usize, v: usize| -> Option<S> {
            if let Some(k) = pivot_slot[u] {
                let d = piv_dists[k * n + v];
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            if let Some(k) = pivot_slot[v] {
                let d = piv_dists[k * n + u];
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            if let Some(d) = distance_matrix.get(nodes[u], nodes[v]) {
                if d.is_finite() && d > S::zero() {
                    return Some(d);
                }
            }
            // Landmark ALT lower bound across cached pivot rows for non-pivot pairs.
            let mut best = S::zero();
            for k in 0..h {
                let du = piv_dists[k * n + u];
                let dv = piv_dists[k * n + v];
                if du.is_finite() && dv.is_finite() {
                    let diff = (du - dv).abs();
                    if diff > best {
                        best = diff;
                    }
                }
            }
            if best > S::zero() {
                Some(best)
            } else {
                None
            }
        };

        // Step C: 1-hop anchor calibration and relative distance floor.
        let eps_s = S::from_f64(self.eps_dist.max(1e-12)).unwrap();
        let mut d1_raw: Vec<S> = Vec::with_capacity(edges_1hop.len());
        let mut scale_1hop_f64 = 1.0_f64;

        if self.calibrate_one_hop {
            let mut ratios = Vec::with_capacity(edges_1hop.len());
            let mut metrics_only = Vec::with_capacity(edges_1hop.len());
            let mut d_metrics = Vec::with_capacity(edges_1hop.len());
            for &(u, v, raw_len) in &edges_1hop {
                let dm = resolve_pair_distance(u, v);
                if let Some(d_val) = dm {
                    let df = d_val.to_f64().unwrap_or(0.0);
                    if df.is_finite() && df > 0.0 {
                        metrics_only.push(df);
                        let rl = raw_len.to_f64().unwrap_or(0.0);
                        if rl.is_finite() && rl > 0.0 {
                            ratios.push(df / rl);
                        }
                    }
                }
                d_metrics.push(dm);
            }
            scale_1hop_f64 = compute_median_f64(&mut ratios)
                .or_else(|| compute_median_f64(&mut metrics_only))
                .unwrap_or(1.0);
            let scale_s = S::from_f64(scale_1hop_f64).unwrap();
            for (idx, &(_, _, raw_len)) in edges_1hop.iter().enumerate() {
                let val = match d_metrics[idx] {
                    Some(dm) if dm.is_finite() && dm > S::zero() => dm,
                    _ => {
                        let rl = if raw_len.is_finite() && raw_len > S::zero() {
                            raw_len
                        } else {
                            S::one()
                        };
                        scale_s * rl
                    }
                };
                d1_raw.push(val);
            }
        } else {
            for &(_, _, raw_len) in &edges_1hop {
                let val = if raw_len.is_finite() && raw_len > S::zero() {
                    raw_len
                } else {
                    eps_s
                };
                d1_raw.push(val);
            }
        }

        let mut pos_d1: Vec<f64> = d1_raw
            .iter()
            .filter_map(|d| {
                let df = d.to_f64()?;
                if df.is_finite() && df > 0.0 {
                    Some(df)
                } else {
                    None
                }
            })
            .collect();
        let ref_med = compute_median_f64(&mut pos_d1).unwrap_or(1.0);
        let d_floor_f64 = self.eps_dist.max(self.min_ratio.max(0.0) * ref_med);
        let d_floor = S::from_f64(d_floor_f64).unwrap();

        let omega_1_s = S::from_f64(self.omega_1.max(0.0)).unwrap();
        let omega_2_s = S::from_f64(self.omega_2.max(0.0)).unwrap();
        let gamma_non2_s = S::from_f64(self.gamma_non2.max(0.0)).unwrap();

        let mut node_pairs = Vec::with_capacity(edges_1hop.len());
        let mut r2 = vec![S::zero(); n];
        for (idx, &(u, v, _)) in edges_1hop.iter().enumerate() {
            let d1 = d1_raw[idx].max(d_floor);
            if d1 > r2[u] {
                r2[u] = d1;
            }
            if d1 > r2[v] {
                r2[v] = d1;
            }
            if omega_1_s > S::zero() {
                let w1 = omega_1_s / (d1 * d1);
                node_pairs.push((u, v, d1, d1, w1, w1));
            }
        }

        // Step D: Hub-safe O(|E| * max_2hop_per_node) 2-hop registration.
        let mut pivot_2hop = vec![vec![]; h];
        if self.max_2hop_per_node > 0 && n > 2 {
            let pairs_2hop = extract_hub_safe_2hop(&adj, self.max_2hop_per_node, self.seed);
            let fallback_2hop = S::from_f64(2.0 * scale_1hop_f64).unwrap().max(d_floor);
            for (u, v) in pairs_2hop {
                if let Some(k) = pivot_slot[u] {
                    pivot_2hop[k].push(v);
                }
                if let Some(k) = pivot_slot[v] {
                    pivot_2hop[k].push(u);
                }
                let d2 = resolve_pair_distance(u, v)
                    .unwrap_or(fallback_2hop)
                    .max(d_floor);
                if d2 > r2[u] {
                    r2[u] = d2;
                }
                if d2 > r2[v] {
                    r2[v] = d2;
                }
                if omega_2_s > S::zero() {
                    let w2 = omega_2_s / (d2 * d2);
                    node_pairs.push((u, v, d2, d2, w2, w2));
                }
            }
        }

        if h == 0 {
            return Sgd::new(node_pairs);
        }

        // Step E: O(N h log N) Voronoi counting + relative-margin barrier + optional hinge repulsion.
        let mut r_dists = vec![vec![]; h];
        for j in 0..n {
            let best_k = (0..h)
                .min_by_key(|&k| OrderedFloat(piv_dists[k * n + j]))
                .unwrap();
            r_dists[best_k].push(piv_dists[best_k * n + j]);
        }
        for dists_k in &mut r_dists {
            dists_k.sort_by_key(|&d| OrderedFloat(d));
        }

        let two = S::from_usize(2).unwrap();
        let half = S::from_f64(0.5).unwrap();
        let barrier_mult = S::from_f64(1.0 + self.delta_rel.max(0.0)).unwrap();
        let mut p_1hop_stamp = vec![0usize; n];
        let mut p_2hop_stamp = vec![0usize; n];
        let mut hinge_pairs = Vec::new();
        let cap = self.max_2hop_per_node;

        for (k, &p) in pivot_node_indices.iter().enumerate() {
            let stamp_val = k + 1;
            p_1hop_stamp[p] = stamp_val;
            for &w in &adj[p] {
                p_1hop_stamp[w] = stamp_val;
            }
            for &v2 in &pivot_2hop[k] {
                p_2hop_stamp[v2] = stamp_val;
            }
            let is_hub_pivot =
                cap > 0 && (adj[p].len() > cap || adj[p].iter().any(|&w| adj[w].len() > cap));

            for i in 0..n {
                if p_1hop_stamp[i] == stamp_val || p_2hop_stamp[i] == stamp_val {
                    continue;
                }
                if is_hub_pivot && adj[i].iter().any(|&w| p_1hop_stamp[w] == stamp_val) {
                    continue;
                }
                let dpi = piv_dists[k * n + i];
                if !dpi.is_finite() || dpi <= S::zero() {
                    continue;
                }
                let r_pair = match self.barrier_sym {
                    BarrierSymmetry::SymMax => r2[p].max(r2[i]),
                    BarrierSymmetry::SymAvg => half * (r2[p] + r2[i]),
                };
                let dpi_bar = dpi.max(barrier_mult * r_pair).max(d_floor);
                if gamma_non2_s <= S::zero() {
                    continue;
                }
                let wpi = gamma_non2_s / (dpi_bar * dpi_bar);
                let count = r_dists[k].partition_point(|&dpj| dpj.is_finite() && two * dpj <= dpi);
                let spi = S::from_usize(count).unwrap();
                let pair = (p, i, dpi_bar, dpi_bar, S::zero(), spi * wpi);
                if self.hinge_repulsion {
                    hinge_pairs.push(pair);
                } else {
                    node_pairs.push(pair);
                }
            }
        }

        Sgd::new_with_hinge_pairs(node_pairs, hinge_pairs)
    }

    /// Builds a 3-tier Sparse Barrier SGD schedule directly from precomputed `(N, k_lm)`
    /// multi-scale Varadhan landmark tables `(d_ms, d_coarse)` in `O(|E| + N k_lm)` time.
    ///
    /// Executes canonical 1-hop calibration, hub-safe `O(|E| * max_2hop_per_node)` 2-hop
    /// registration, adaptive or explicit non-2-hop repulsion sampling, relative-margin
    /// `R_2` barrier enforcement, and native one-sided hinge repulsion on Tier 3 pairs.
    pub fn build_barrier_from_landmark_tables<G>(
        &self,
        graph: G,
        d_ms: ArrayView2<'_, f64>,
        d_coarse: Option<ArrayView2<'_, f64>>,
        pivots: &[usize],
        neg_per_node: Option<usize>,
    ) -> Result<(Sgd<f64>, BarrierBuildStats), String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
    {
        if !self.delta_rel.is_finite() || self.delta_rel < 0.0 {
            return Err(format!(
                "delta_rel must be finite and >= 0, got {}",
                self.delta_rel
            ));
        }
        if !self.omega_1.is_finite() || self.omega_1 < 0.0 {
            return Err(format!(
                "omega_1 must be finite and >= 0, got {}",
                self.omega_1
            ));
        }
        if !self.omega_2.is_finite() || self.omega_2 < 0.0 {
            return Err(format!(
                "omega_2 must be finite and >= 0, got {}",
                self.omega_2
            ));
        }
        if !self.gamma_non2.is_finite() || self.gamma_non2 < 0.0 {
            return Err(format!(
                "gamma_non2 must be finite and >= 0, got {}",
                self.gamma_non2
            ));
        }
        if !self.min_ratio.is_finite() || self.min_ratio < 0.0 {
            return Err(format!(
                "min_ratio must be finite and >= 0, got {}",
                self.min_ratio
            ));
        }
        if !self.eps_dist.is_finite() || self.eps_dist <= 0.0 {
            return Err(format!(
                "eps_dist must be finite and > 0, got {}",
                self.eps_dist
            ));
        }

        let n = graph.node_count();
        if n > 0 {
            if d_ms.nrows() != n {
                return Err(format!(
                    "d_ms row count ({}) must match node count ({n})",
                    d_ms.nrows()
                ));
            }
            if n > 1 && d_ms.ncols() == 0 {
                return Err("d_ms must have at least 1 landmark column when n > 1".to_string());
            }
            if d_ms.iter().any(|&v| !v.is_finite()) {
                return Err("d_ms must contain only finite float64 values".to_string());
            }
            if let Some(dc) = d_coarse {
                if dc.nrows() != n {
                    return Err(format!(
                        "d_coarse row count ({}) must match node count ({n})",
                        dc.nrows()
                    ));
                }
                if n > 1 && dc.ncols() == 0 {
                    return Err(
                        "d_coarse must have at least 1 landmark column when n > 1".to_string()
                    );
                }
                if dc.iter().any(|&v| !v.is_finite()) {
                    return Err("d_coarse must contain only finite float64 values".to_string());
                }
            }
        }
        for &p in pivots {
            if p >= n {
                return Err(format!(
                    "pivot index {p} is out of bounds for graph with {n} nodes"
                ));
            }
        }

        if n <= 1 {
            let (perplexity_val, per_node_neg_used, perplexity_source) = match neg_per_node {
                Some(k) => (k as f64, k, "explicit"),
                None => (1.0, 1, "adaptive_2hop"),
            };
            return Ok((
                Sgd::new(vec![]),
                BarrierBuildStats {
                    n_pairs_1hop: 0,
                    n_pairs_2hop: 0,
                    n_pairs_non2: 0,
                    n_pairs_hinge: 0,
                    n_pairs_two_sided: 0,
                    perplexity_val,
                    per_node_neg_used,
                    perplexity_source,
                    d_floor: self.eps_dist,
                },
            ));
        }

        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let (edges_1hop, adj) = extract_canonical_edges_and_adj(graph, &indices, &mut |_| 1.0_f64);
        let pairs_1hop: Vec<(usize, usize)> = edges_1hop.iter().map(|&(u, v, _)| (u, v)).collect();
        let pairs_2hop = if self.max_2hop_per_node > 0 && n > 2 {
            extract_hub_safe_2hop(&adj, self.max_2hop_per_node, self.seed)
        } else {
            Vec::new()
        };

        let (perplexity_val, per_node_neg_used, perplexity_source) = match neg_per_node {
            Some(k) => (k as f64, k, "explicit"),
            None => {
                let p_val = compute_adaptive_perplexity_from_pairs(n, &pairs_1hop, &pairs_2hop);
                let neg = if n > 3 {
                    round_half_to_even(p_val).max(1)
                } else {
                    1
                };
                (p_val, neg, "adaptive_2hop")
            }
        };

        let pairs_non2 = if n > 2 && per_node_neg_used > 0 {
            sample_non2hop_pairs(
                &adj,
                pivots,
                &pairs_1hop,
                &pairs_2hop,
                per_node_neg_used,
                self.seed,
            )
        } else {
            Vec::new()
        };

        let d1_raw: Vec<f64> = pairs_1hop
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();
        let d2_raw: Vec<f64> = pairs_2hop
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();
        let d3_raw: Vec<f64> = pairs_non2
            .iter()
            .map(|&(u, v)| alt_max_pair_distance(d_ms, d_coarse, u, v))
            .collect();

        let mut pos_ref: Vec<f64> = d1_raw.iter().copied().filter(|&d| d > 0.0).collect();
        if pos_ref.is_empty() {
            pos_ref = d2_raw.iter().copied().filter(|&d| d > 0.0).collect();
        }
        if pos_ref.is_empty() {
            pos_ref = d3_raw.iter().copied().filter(|&d| d > 0.0).collect();
        }
        let ref_med = compute_median_f64(&mut pos_ref).unwrap_or(1.0);
        let d_floor = self.eps_dist.max(self.min_ratio * ref_med);

        let mut r2 = vec![0.0_f64; n];
        let mut node_pairs = Vec::with_capacity(pairs_1hop.len() + pairs_2hop.len());
        let mut hinge_pairs = Vec::new();

        for (idx, &(u, v)) in pairs_1hop.iter().enumerate() {
            let d1 = d1_raw[idx].max(d_floor);
            if d1 > r2[u] {
                r2[u] = d1;
            }
            if d1 > r2[v] {
                r2[v] = d1;
            }
            if self.omega_1 > 0.0 {
                let w1 = self.omega_1 / (d1 * d1);
                node_pairs.push((u, v, d1, d1, w1, w1));
            }
        }

        for (idx, &(u, v)) in pairs_2hop.iter().enumerate() {
            let d2 = d2_raw[idx].max(d_floor);
            if d2 > r2[u] {
                r2[u] = d2;
            }
            if d2 > r2[v] {
                r2[v] = d2;
            }
            if self.omega_2 > 0.0 {
                let w2 = self.omega_2 / (d2 * d2);
                node_pairs.push((u, v, d2, d2, w2, w2));
            }
        }

        let barrier_mult = 1.0 + self.delta_rel;
        for (idx, &(u, v)) in pairs_non2.iter().enumerate() {
            let d3 = d3_raw[idx].max(d_floor);
            let r_pair = match self.barrier_sym {
                BarrierSymmetry::SymMax => r2[u].max(r2[v]),
                BarrierSymmetry::SymAvg => 0.5 * (r2[u] + r2[v]),
            };
            let d3_bar = d3.max(barrier_mult * r_pair).max(self.eps_dist);
            if self.gamma_non2 > 0.0 {
                let w3 = self.gamma_non2 / (d3_bar * d3_bar);
                let pair = (u, v, d3_bar, d3_bar, w3, w3);
                if self.hinge_repulsion {
                    hinge_pairs.push(pair);
                } else {
                    node_pairs.push(pair);
                }
            }
        }

        let n_pairs_two_sided = node_pairs.len();
        let n_pairs_hinge = hinge_pairs.len();
        let sgd = Sgd::new_with_hinge_pairs(node_pairs, hinge_pairs);
        let stats = BarrierBuildStats {
            n_pairs_1hop: pairs_1hop.len(),
            n_pairs_2hop: pairs_2hop.len(),
            n_pairs_non2: pairs_non2.len(),
            n_pairs_hinge,
            n_pairs_two_sided,
            perplexity_val,
            per_node_neg_used,
            perplexity_source,
            d_floor,
        };
        Ok((sgd, stats))
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

/// Implementation of the maximal-minimal distance pivot selection strategy.
///
/// This function selects pivot nodes that are maximally distant from each other,
/// which helps ensure good coverage of the graph for sparse layouts.
///
/// The strategy works as follows:
/// 1. Select a random node as the first pivot
/// 2. Compute shortest paths from this pivot to all other nodes
/// 3. Select the next pivot as the node that maximizes its minimum distance to all existing pivots
/// 4. Repeat steps 2-3 until the desired number of pivots is selected
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
/// - A PivotedDistanceMatrix containing distances from pivots to all nodes
fn max_min_random_sp<G, F, R, S>(
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
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let nodes = graph.node_identifiers().collect::<Vec<_>>();
    let mut length = length;
    let n = indices.len();
    let mut pivot = vec![];
    pivot.push(nodes[rng.gen_range(0..n)]);
    let mut distance_matrix = PivotedDistanceMatrix::empty(graph);
    distance_matrix.push(pivot[0]);
    dijkstra_with_distance_matrix(graph, &mut length, pivot[0], &mut distance_matrix);
    let mut min_d = Array1::from_elem(n, S::infinity());
    for k in 1..h {
        for j in 0..n {
            min_d[j] = min_d[j].min(distance_matrix.get_by_index(k - 1, j));
        }
        pivot.push(nodes[proportional_sampling(&min_d, rng)]);
        distance_matrix.push(pivot[k]);
        dijkstra_with_distance_matrix(graph, &mut length, pivot[k], &mut distance_matrix);
    }
    (pivot, distance_matrix)
}

type CanonicalEdgesAndAdj<S> = (Vec<(usize, usize, S)>, Vec<Vec<usize>>);

fn extract_canonical_edges_and_adj<G, F, S>(
    graph: G,
    indices: &HashMap<G::NodeId, usize>,
    length: &mut F,
) -> CanonicalEdgesAndAdj<S>
where
    G: IntoEdges,
    G::NodeId: Eq + std::hash::Hash,
    F: FnMut(G::EdgeRef) -> S,
    S: DrawingValue,
{
    let n = indices.len();
    let mut adj = vec![Vec::new(); n];
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut edges_1hop = Vec::new();
    for edge in graph.edge_references() {
        let i = indices[&edge.source()];
        let j = indices[&edge.target()];
        if i == j {
            continue;
        }
        let (u, v) = if i < j { (i, j) } else { (j, i) };
        let len = length(edge);
        if seen.insert((u, v)) {
            edges_1hop.push((u, v, len));
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    edges_1hop.sort_unstable_by_key(|&(u, v, _)| (u, v));
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    (edges_1hop, adj)
}

fn extract_hub_safe_2hop(adj: &[Vec<usize>], cap: usize, seed: u64) -> Vec<(usize, usize)> {
    let n = adj.len();
    if cap == 0 || n <= 2 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stamp = vec![0usize; n];
    let mut raw_2hop = Vec::new();

    for u in 0..n {
        if adj[u].is_empty() {
            continue;
        }
        let stamp_val = u + 1;
        stamp[u] = stamp_val;
        for &w in &adj[u] {
            stamp[w] = stamp_val;
        }

        let mut collected = 0usize;
        for &w in &adj[u] {
            // Skipping degree-1 neighbors prevents O(N) scans at star hubs without losing 2-hop pairs.
            let deg_w = adj[w].len();
            if deg_w <= 1 {
                continue;
            }
            if deg_w <= cap {
                for &v in &adj[w] {
                    if stamp[v] != stamp_val {
                        stamp[v] = stamp_val;
                        raw_2hop.push(if u < v { (u, v) } else { (v, u) });
                        collected += 1;
                        if collected >= cap {
                            break;
                        }
                    }
                }
            } else {
                let n_draws = deg_w.min(cap * 2);
                for _ in 0..n_draws {
                    let v = adj[w][rng.gen_range(0..deg_w)];
                    if stamp[v] != stamp_val {
                        stamp[v] = stamp_val;
                        raw_2hop.push(if u < v { (u, v) } else { (v, u) });
                        collected += 1;
                        if collected >= cap {
                            break;
                        }
                    }
                }
            }
            if collected >= cap {
                break;
            }
        }
    }

    raw_2hop.sort_unstable();
    raw_2hop.dedup();
    raw_2hop
}

fn compute_median_f64(vals: &mut [f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by_key(|&v| OrderedFloat(v));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some(0.5 * (vals[mid - 1] + vals[mid]))
    }
}

fn round_half_to_even(x: f64) -> usize {
    x.round_ties_even().max(0.0) as usize
}

fn compute_adaptive_perplexity_from_pairs(
    n_nodes: usize,
    pairs_1hop: &[(usize, usize)],
    pairs_2hop: &[(usize, usize)],
) -> f64 {
    if n_nodes <= 3 {
        return 1.0;
    }
    let upper = 32.0_f64.min((n_nodes - 1) as f64 / 3.0);
    if upper < 1.0 {
        return 1.0;
    }
    let lower = 8.0_f64.min(upper);
    let mut deg2_counts = vec![0usize; n_nodes];
    for &(u, v) in pairs_1hop.iter().chain(pairs_2hop.iter()) {
        deg2_counts[u] += 1;
        deg2_counts[v] += 1;
    }
    deg2_counts.sort_unstable();
    let mid = n_nodes / 2;
    let med = if n_nodes % 2 == 1 {
        deg2_counts[mid] as f64
    } else {
        0.5 * (deg2_counts[mid - 1] + deg2_counts[mid]) as f64
    };
    (1.5 * med).clamp(lower, upper)
}

fn has_common_neighbor(n1_u: &[usize], n1_v: &[usize], stamp: &[usize], stamp_val: usize) -> bool {
    if n1_v.is_empty() || n1_u.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| stamp[x] == stamp_val);
    }
    n1_u.iter().any(|x| n1_v.binary_search(x).is_ok())
}

fn sample_non2hop_pairs(
    adj: &[Vec<usize>],
    pivots: &[usize],
    pairs_1hop: &[(usize, usize)],
    pairs_2hop: &[(usize, usize)],
    budget: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let n = adj.len();
    if n <= 2 || budget == 0 {
        return Vec::new();
    }
    let n_piv = pivots.len();
    let piv_quota = if n_piv > 0 {
        (budget / 2).max(1).min(n_piv)
    } else {
        0
    };

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(101));
    let mut stamp = vec![0usize; n];
    let mut neg_stamp = vec![0usize; n];
    let mut raw_non2 = Vec::new();
    let max_trials = 8usize.max(3 * budget);

    for u in 0..n {
        let stamp_val = u + 1;
        stamp[u] = stamp_val;
        for &w in &adj[u] {
            stamp[w] = stamp_val;
        }

        let mut collected = 0usize;
        if piv_quota > 0 {
            let start_off = if n_piv > piv_quota {
                (u * 7) % n_piv
            } else {
                0
            };
            let mut checked = 0usize;
            let mut idx_p = start_off;
            while checked < n_piv && collected < piv_quota {
                let v = pivots[idx_p];
                idx_p = (idx_p + 1) % n_piv;
                checked += 1;
                if stamp[v] == stamp_val || neg_stamp[v] == stamp_val {
                    continue;
                }
                if has_common_neighbor(&adj[u], &adj[v], &stamp, stamp_val) {
                    continue;
                }
                neg_stamp[v] = stamp_val;
                raw_non2.push(if u < v { (u, v) } else { (v, u) });
                collected += 1;
            }
        }

        if collected < budget {
            for _ in 0..max_trials {
                let v = rng.gen_range(0..n);
                if stamp[v] == stamp_val || neg_stamp[v] == stamp_val {
                    continue;
                }
                if has_common_neighbor(&adj[u], &adj[v], &stamp, stamp_val) {
                    continue;
                }
                neg_stamp[v] = stamp_val;
                raw_non2.push(if u < v { (u, v) } else { (v, u) });
                collected += 1;
                if collected >= budget {
                    break;
                }
            }
        }
    }

    raw_non2.sort_unstable();
    raw_non2.dedup();
    raw_non2.retain(|pair| {
        pairs_1hop.binary_search(pair).is_err() && pairs_2hop.binary_search(pair).is_err()
    });
    raw_non2
}

#[inline]
fn alt_max_pair_distance(
    d_ms: ArrayView2<'_, f64>,
    d_coarse: Option<ArrayView2<'_, f64>>,
    u: usize,
    v: usize,
) -> f64 {
    let mut d_val = 0.0_f64;
    let row_u = d_ms.row(u);
    let row_v = d_ms.row(v);
    for (&au, &av) in row_u.iter().zip(row_v.iter()) {
        let diff = (au - av).abs();
        if diff > d_val {
            d_val = diff;
        }
    }
    if let Some(dc) = d_coarse {
        let cu = dc.row(u);
        let cv = dc.row(v);
        for (&bu, &bv) in cu.iter().zip(cv.iter()) {
            let diff = (bu - bv).abs();
            if diff > d_val {
                d_val = diff;
            }
        }
    }
    d_val
}

/// Selects a node index with probability proportional to its distance value.
pub(crate) fn proportional_sampling<R, S>(values: &Array1<S>, rng: &mut R) -> usize
where
    R: Rng,
    S: DrawingValue,
{
    let n = values.len();
    let mut s = 0.0;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
    }
    if s == 0.0 {
        panic!("could not choose pivot");
    }
    let x = rng.gen_range(0.0..s);
    s = 0.0;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
        if x < s {
            return i;
        }
    }
    n - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use petgraph::graph::UnGraph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_linalg_kernel::{
        PivotedBandPassDiffusionKernelBuilder, PivotedDiffusionKernelBuilder,
        PivotedNegLogSimDistanceBuilder, SparseSymmetricMatrix,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn cycle_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 0..n {
            g.add_edge(nodes[i], nodes[(i + 1) % n], ());
        }
        g
    }

    fn star_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 1..n {
            g.add_edge(nodes[0], nodes[i], ());
        }
        g
    }

    #[test]
    fn test_sparse_sgd_with_pivoted_neg_log_sim_distance() {
        let g = cycle_graph(8);
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let pivots = vec![0usize, 4usize];

        let mut rng = StdRng::seed_from_u64(42);
        let bp_kernel = PivotedBandPassDiffusionKernelBuilder::dog(
            &lap,
            0.5_f64,
            4.0_f64,
            0.5_f64,
            0.05_f64,
            20,
            pivots.clone(),
        )
        .num_hutchinson(Some(16))
        .build(&mut rng)
        .unwrap();
        let bp_dist = PivotedNegLogSimDistanceBuilder::new()
            .build(&g, bp_kernel)
            .unwrap();
        let bp_sgd = SparseSgd::new()
            .h(2)
            .build_with_pivoted_neg_log_sim_distance(&g, |_| 1.0_f64, &bp_dist);
        assert!(!bp_sgd.node_pairs().is_empty());
        for &(_, _, d1, d2, w1, w2) in bp_sgd.node_pairs() {
            assert!(d1.is_finite() && d2.is_finite());
            assert!(w1.is_finite() && w2.is_finite());
            assert!(d1 > 0.0 && d2 > 0.0);
        }

        let diff_kernel = PivotedDiffusionKernelBuilder::new(&lap, 2.0_f64, 20, pivots)
            .num_hutchinson(Some(16))
            .build(&mut rng)
            .unwrap();
        let diff_dist = PivotedNegLogSimDistanceBuilder::new()
            .build(&g, diff_kernel)
            .unwrap();
        let diff_sgd = SparseSgd::new()
            .h(2)
            .build_with_pivoted_neg_log_sim_distance(&g, |_| 1.0_f64, &diff_dist);
        assert!(!diff_sgd.node_pairs().is_empty());
        for &(_, _, d1, d2, w1, w2) in diff_sgd.node_pairs() {
            assert!(d1.is_finite() && d2.is_finite());
            assert!(w1.is_finite() && w2.is_finite());
        }
    }

    #[test]
    fn test_sparse_sgd_legacy_empty_and_zero_h_guards() {
        let empty = UnGraph::<(), ()>::new_undirected();
        let mut rng = StdRng::seed_from_u64(42);
        let sgd_empty = SparseSgd::new().h(10).build(&empty, |_| 1.0_f64, &mut rng);
        assert_eq!(sgd_empty.total_pair_count(), 0);

        let g = cycle_graph(6);
        let sgd_zero_h = SparseSgd::new().h(0).build(&g, |_| 1.0_f64, &mut rng);
        assert_eq!(sgd_zero_h.total_pair_count(), 0);

        // Legacy mode emits no hinge pairs and matches standard Dijkstra pivot structure
        let sgd_legacy = SparseSgd::new().h(3).build(&g, |_| 1.0_f64, &mut rng);
        assert!(!sgd_legacy.node_pairs().is_empty());
        assert!(sgd_legacy.hinge_pairs().is_empty());
    }

    #[test]
    fn test_sparse_sgd_barrier_1hop_calibration_2hop_and_hinge_on_cycle() {
        let g = cycle_graph(12);
        let dm = all_sources_dijkstra(&g, |_| 3.0_f64);
        let nodes: Vec<_> = g.node_indices().collect();
        let pivots = vec![nodes[0], nodes[4], nodes[8]];

        let sgd = SparseSgd::barrier()
            .h(3)
            .max_2hop_per_node(4)
            .delta_rel(0.25)
            .build_with_pivot_and_distance_matrix(&g, |_| 1.0_f64, &pivots, &dm);

        assert!(!sgd.node_pairs().is_empty());
        assert!(!sgd.hinge_pairs().is_empty());
        // 1-hop pairs should be calibrated to 3.0 (not raw 1.0)
        let has_calibrated_1hop = sgd.node_pairs().iter().any(|&(u, v, d, _, _, _)| {
            (u == 0 && v == 1 || u == 1 && v == 0) && (d - 3.0).abs() < 1e-9
        });
        assert!(has_calibrated_1hop);
        // 2-hop pairs (e.g. distance 6.0) should also be registered as two-sided springs
        let has_2hop = sgd
            .node_pairs()
            .iter()
            .any(|&(_, _, d, _, _, _)| (d - 6.0).abs() < 1e-9);
        assert!(has_2hop);
        // Hinge pairs must enforce barrier >= (1 + delta_rel) * r_2 = 1.25 * 6.0 = 7.5
        for &(_, _, d_bar, _, wij, wji) in sgd.hinge_pairs() {
            assert!(d_bar >= 7.5 - 1e-9);
            assert_eq!(wij, 0.0);
            assert!(wji > 0.0 && wji.is_finite());
        }
    }

    #[test]
    fn test_sparse_sgd_barrier_star_2000_hub_safe_zero_o_n2() {
        let n = 2000;
        let cap = 16;
        let g = star_graph(n);
        let mut rng = StdRng::seed_from_u64(42);
        let sgd =
            SparseSgd::barrier()
                .h(16)
                .max_2hop_per_node(cap)
                .build(&g, |_| 1.0_f64, &mut rng);

        // Total 1-hop + 2-hop pairs must be O(N * cap), nowhere near N*(N-1)/2 (~2e6)
        assert!(sgd.node_pairs().len() <= (n - 1) + n * cap);
        assert!(sgd.node_pairs().len() < 40_000);
        // In a star graph every pair of leaves is 2-hop; no non-2-hop pairs exist
        assert!(sgd.hinge_pairs().is_empty());

        let mut seen = HashSet::new();
        for &(u, v, d, _, w1, w2) in sgd.node_pairs() {
            assert_ne!(u, v);
            assert!(d.is_finite() && d > 0.0);
            assert!(w1.is_finite() && w2.is_finite());
            let key = if u < v { (u, v) } else { (v, u) };
            assert!(seen.insert(key), "duplicate pair {:?}", key);
        }
    }

    #[test]
    fn test_build_barrier_from_landmark_tables_parity_and_validation() {
        // 5-node path 0-1-2-3-4 matching tests/test_sparse_barrier_sgd.py
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
        for i in 0..4 {
            g.add_edge(nodes[i], nodes[i + 1], ());
        }
        // Construct 1D landmark coordinate table D[u, 0] = u -> exact path distance |u - v|
        let d_ms = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
        let (sgd_hinge, stats) = SparseSgd::barrier()
            .max_2hop_per_node(32)
            .delta_rel(0.5)
            .barrier_sym(BarrierSymmetry::SymMax)
            .omega_1(2.0)
            .omega_2(0.5)
            .gamma_non2(3.0)
            .min_ratio(0.5)
            .build_barrier_from_landmark_tables(&g, d_ms.view(), None, &[0, 4], Some(8))
            .unwrap();

        assert_eq!(stats.n_pairs_1hop, 4);
        assert_eq!(stats.n_pairs_2hop, 3);
        assert_eq!(stats.n_pairs_non2, 3);
        assert_eq!(stats.n_pairs_two_sided, 7);
        assert_eq!(stats.n_pairs_hinge, 3);
        assert!((stats.d_floor - 0.5).abs() < 1e-12);

        // Verify barrier target distances match 1.5 * max(R2[u], R2[v])
        for &(u, v, d_bar, _, w_u, w_v) in sgd_hinge.hinge_pairs() {
            // In path 0-1-2-3-4 with d_ms=[0,1,2,3,4], R2 = [2, 2, 2, 2, 2] -> 1.5 * 2 = 3.0 for (0,3),(1,4), and max(4, 3)=4.0 for (0,4)
            let expected = if (u == 0 && v == 4) || (u == 4 && v == 0) {
                4.0
            } else {
                3.0
            };
            assert!((d_bar - expected).abs() < 1e-12);
            assert!((w_u - 3.0 / (expected * expected)).abs() < 1e-12);
            assert!((w_v - 3.0 / (expected * expected)).abs() < 1e-12);
        }

        // Validation errors on mismatched row count or out-of-range pivot
        let bad_d = Array2::zeros((3, 1));
        assert!(SparseSgd::barrier()
            .build_barrier_from_landmark_tables(&g, bad_d.view(), None, &[0], Some(4))
            .is_err());
        assert!(SparseSgd::barrier()
            .build_barrier_from_landmark_tables(&g, d_ms.view(), None, &[10], Some(4))
            .is_err());
    }
}
