use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight, RiemannianLayout, Scalar};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats};

use super::config::{
    SgdScheduleConfig, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
};
use super::graph_helpers::{
    build_symmetric_csr_from_sorted_pairs, extract_canonical_undirected_edges,
};
use super::ortmann::{
    OrtmannSparseProblem, build_ortmann_sparse_problem, ortmann_problem_into_storage,
};
use super::pairs::{
    PackedSymPair, compute_eta_bounds, validate_graph_weights, validate_schedule_config,
};
use super::solver::{SgdSolver, StreamingState};
use super::sparse_barrier::{build_sparse_barrier_dog, build_sparse_barrier_from_landmarks};
use super::variant_config::OrtmannSparseStressConfig;

impl<L: RiemannianLayout> SgdSolver<L> {
    /// Constructs a 3-tier Sparse Barrier SGD solver directly from precomputed regularized DoG
    /// Varadhan landmark tables.
    pub fn sparse_barrier_from_landmarks<W>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        pivots: Option<&[u32]>,
        config: &SparseBarrierConfig,
    ) -> CoreResult<(Self, SparseBarrierStats)> {
        let built = build_sparse_barrier_from_landmarks(graph, landmarks, pivots, config)?;
        let solver = Self::from_packed_symmetric(
            graph.num_nodes(),
            built.pairs,
            built.n_two_sided,
            built.n_hinge,
            config.schedule.clone(),
            config.seed,
        )?;
        Ok((solver, built.stats))
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a 3-tier Sparse
    /// Barrier SGD solver in one call.
    pub fn sparse_barrier_dog<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        dog_config: &DogVaradhanConfig,
        barrier_config: &SparseBarrierConfig,
    ) -> CoreResult<(Self, SparseBarrierStats, DogVaradhanStats)> {
        let (built, dog_stats) = build_sparse_barrier_dog(graph, dog_config, barrier_config)?;
        let solver = Self::from_packed_symmetric(
            graph.num_nodes(),
            built.pairs,
            built.n_two_sided,
            built.n_hinge,
            barrier_config.schedule.clone(),
            barrier_config.seed,
        )?;
        Ok((solver, built.stats, dog_stats))
    }

    /// Constructs a minimal-RAM SGD solver storing only 1-hop edges in flat packed pairs and
    /// sampling $O(|V| \cdot \mathrm{negatives\_per\_node})$ non-edge negative updates on-the-fly
    /// each step without preallocating negative pair buffers.
    pub fn streaming_negatives<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        neg_config: StreamingNegativeConfig,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        if !neg_config.target_distance.is_finite() {
            return Err(CoreError::non_finite(
                "StreamingNegativeConfig::target_distance",
            ));
        }
        if neg_config.target_distance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "target_distance must be > 0, got {}",
                neg_config.target_distance
            )));
        }
        if !neg_config.weight.is_finite() {
            return Err(CoreError::non_finite("StreamingNegativeConfig::weight"));
        }
        if neg_config.weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "weight must be >= 0, got {}",
                neg_config.weight
            )));
        }
        validate_graph_weights(graph, "SgdSolver::streaming_negatives")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let target_s = L::Scalar::from_f64(neg_config.target_distance);
        let weight_s = L::Scalar::from_f64(neg_config.weight);
        if !target_s.is_finite() || target_s <= L::Scalar::ZERO || !weight_s.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "StreamingNegativeConfig parameters overflowed scalar precision",
            ));
        }

        let canonical_edges = extract_canonical_undirected_edges(graph);
        let m = canonical_edges.len();
        let mut pairs = Vec::with_capacity(m);
        let mut u1 = Vec::with_capacity(m);
        let mut v1 = Vec::with_capacity(m);

        for &(u, v, edge_len) in &canonical_edges {
            u1.push(u);
            v1.push(v);
            if edge_len.is_finite() && edge_len > 0.0 {
                let d_s = L::Scalar::from_f64(edge_len);
                let w_s = L::Scalar::from_f64(1.0 / (edge_len * edge_len));
                if !d_s.is_finite()
                    || d_s <= L::Scalar::ZERO
                    || !w_s.is_finite()
                    || w_s <= L::Scalar::ZERO
                {
                    return Err(CoreError::invalid_input(format!(
                        "edge ({u}, {v}) distance {edge_len} produces non-finite or non-positive spring weight"
                    )));
                }
                pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
            }
        }

        let n_springs = pairs.len();
        let mut solver = Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed)?;
        if neg_config.negatives_per_node > 0 && weight_s > L::Scalar::ZERO {
            let (adj_indptr, adj_indices) = build_symmetric_csr_from_sorted_pairs(n, &u1, &v1);
            let node_order: Vec<u32> = (0..n as u32).collect();
            solver.streaming = Some(StreamingState {
                negatives_per_node: neg_config.negatives_per_node,
                target_distance: target_s,
                weight: weight_s,
                hinge_only: neg_config.hinge_only,
                adj_indptr,
                adj_indices,
                node_order,
            });
            let (eta_min, eta_max) =
                compute_eta_bounds(&solver.pairs, Some(weight_s), solver.schedule_cfg.eps);
            solver.eta_min = eta_min;
            solver.eta_max = eta_max;
        }

        Ok(solver)
    }

    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver using graph Voronoi
    /// partitioning $R(p)$ (breaking distance ties in favor of the smallest current $|R(p)|$)
    /// and directional region-adapted weights $w'_{ip} = |\{j \in R(p) : 2 d_{jp} \le d_{ip}\}| / d_{ip}^2$.
    pub fn ortmann_sparse_stress<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        config: OrtmannSparseStressConfig,
    ) -> CoreResult<Self> {
        let problem = build_ortmann_sparse_problem::<W, L::Scalar>(graph, config.clone())?;
        Self::ortmann_sparse_stress_from_problem(&problem, config)
    }

    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver directly from a prebuilt
    /// [`OrtmannSparseProblem`] IR, applying `config`'s `seed`, `eta_max`, `eta_min`, and `eps`
    /// overrides.
    pub fn ortmann_sparse_stress_from_problem(
        problem: &OrtmannSparseProblem<L::Scalar>,
        config: OrtmannSparseStressConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        problem.schedule_cfg.validate()?;
        let seed = config.seed;
        let eta_max_ovr = config.eta_max;
        let eta_min_ovr = config.eta_min;
        let eps = config.eps;
        let storage = ortmann_problem_into_storage(problem)?;
        let n_springs = storage.len();
        let mut solver = Self::from_pair_storage(
            problem.num_nodes,
            storage,
            n_springs,
            0,
            problem.schedule_cfg.clone(),
            seed,
        )?;
        if solver.eta_max > 0.0 {
            if let Some(emax) = eta_max_ovr {
                solver.eta_max = emax;
            }
            if let Some(emin) = eta_min_ovr {
                solver.eta_min = emin;
            }
            if solver.eta_min > solver.eta_max {
                if eta_max_ovr.is_some() && eta_min_ovr.is_none() {
                    solver.eta_min = solver.eta_max * eps.min(1.0);
                } else {
                    solver.eta_max = solver.eta_min;
                }
            }
        } else {
            solver.eta_min = 0.0;
            solver.eta_max = 0.0;
        }
        Ok(solver)
    }

    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver from preselected `pivots` and
    /// a precomputed row-major `(pivots.len(), num_nodes)` pivot-to-all-nodes distance slice
    /// (`pivot_distances_kn[k * num_nodes + j]`).
    pub fn ortmann_sparse_stress_from_pivots<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        pivots: &[u32],
        pivot_distances_kn: &[f64],
        config: OrtmannSparseStressConfig,
    ) -> CoreResult<Self> {
        let problem = OrtmannSparseProblem::<L::Scalar>::from_pivots(
            graph,
            pivots,
            pivot_distances_kn,
            config.clone(),
        )?;
        Self::ortmann_sparse_stress_from_problem(&problem, config)
    }
}
