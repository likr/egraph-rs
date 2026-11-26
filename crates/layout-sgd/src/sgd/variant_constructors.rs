use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats};

use crate::coarse_backbone::{CoarseBackboneConfig, CoarseBackboneResult, coarse_backbone_init};

use super::config::{
    PacmapGraphConfig, Sgd2Config, SgdScheduleConfig, SgdScheduleKind, SnapTFdpConfig,
    SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
};
use super::drgraph_coarsen::build_drgraph_state;
use super::graph_cne::build_graph_cne_state;
use super::graph_sampling::{
    build_symmetric_csr_from_sorted_pairs, extract_canonical_undirected_edges,
};
use super::largevis::build_largevis_state;
use super::ortmann::build_ortmann_storage;
use super::pacmap::{PacmapExternalLandmarks, build_pacmap_state};
use super::pairs::{
    PackedSymPair, PairStorage, compute_eta_bounds, validate_graph_weights,
    validate_schedule_config,
};
use super::sgd2::build_sgd2_state;
use super::snap_t_fdp::build_snap_t_fdp_state;
use super::solver::{SgdSolver, StreamingState};
use super::sparse_barrier::{build_sparse_barrier_dog, build_sparse_barrier_from_landmarks};
#[allow(unused_imports)]
use super::variant_config::GraphCneObjective;
use super::variant_config::{
    DrGraphConfig, GraphCneConfig, LargeVisConfig, OrtmannSparseStressConfig,
};

impl<S: Scalar> SgdSolver<S> {
    /// Constructs a Track B 3-tier Sparse Barrier SGD solver directly from precomputed
    /// regularized DoG Varadhan landmark tables.
    pub fn sparse_barrier_from_landmarks<W: Scalar>(
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
            config.schedule,
            config.seed,
        )?;
        Ok((solver, built.stats))
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a Track B
    /// 3-tier Sparse Barrier SGD solver in one call.
    pub fn sparse_barrier_dog<W: Scalar>(
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
            barrier_config.schedule,
            barrier_config.seed,
        )?;
        Ok((solver, built.stats, dog_stats))
    }

    /// Constructs a minimal-RAM SGD solver storing only 1-hop edges in flat packed pairs and
    /// sampling $O(|V| \cdot \mathrm{negatives\_per\_node})$ non-edge negative updates on-the-fly
    /// each step without preallocating negative pair buffers.
    pub fn streaming_negatives<W: Scalar>(
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

        let target_s = S::from_f64(neg_config.target_distance);
        let weight_s = S::from_f64(neg_config.weight);
        if !target_s.is_finite() || target_s <= S::ZERO || !weight_s.is_finite() {
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
                let d_s = S::from_f64(edge_len);
                let w_s = S::from_f64(1.0 / (edge_len * edge_len));
                if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                    pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                }
            }
        }

        let n_springs = pairs.len();
        let mut solver = Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed)?;
        if neg_config.negatives_per_node > 0 && weight_s > S::ZERO {
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

    /// Constructs a compact-memory `SNAP-tFDP` solver storing only canonical undirected 1-hop
    /// edges (`edges_u`, `edges_v`, optional `f32` weights, and `edge_order`) and drawing
    /// `negatives_per_edge` uniform random negative vertices on-the-fly per edge.
    pub fn snap_t_fdp<W: Scalar>(graph: &CsrGraph<W>, config: SnapTFdpConfig) -> CoreResult<Self> {
        let built = build_snap_t_fdp_state(graph, config)?;
        Ok(Self {
            num_nodes: built.num_nodes,
            pairs: PairStorage::Symmetric(Vec::new()),
            num_springs: built.num_springs,
            num_hinges: 0,
            block_order: built.block_order,
            streaming: None,
            snap_t_fdp: Some(built.state),
            pacmap: None,
            sgd2: None,
            largevis: None,
            drgraph: None,
            graph_cne: None,
            schedule_cfg: built.schedule_cfg,
            eta_min: 0.0,
            eta_max: config.learning_rate,
            iter: 0,
            status: built.status,
            rng: built.rng,
        })
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver (`NB` 1+2-hop, `MN` mid-near, `FP` further)
    /// with 3-phase dynamic loss weights and zero-allocation Adam updates.
    pub fn pacmap_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: PacmapGraphConfig,
    ) -> CoreResult<Self> {
        Self::pacmap_graph_inner(graph, config, None)
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver using precomputed [`DogVaradhanLandmarks`]
    /// for Mid-Near (`MN`) proximity ranking instead of internal BFS landmarks.
    pub fn pacmap_graph_with_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        config: PacmapGraphConfig,
    ) -> CoreResult<Self> {
        Self::pacmap_graph_inner(graph, config, Some(PacmapExternalLandmarks::Dog(landmarks)))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver using an externally supplied row-major
    /// `(num_nodes, num_landmarks)` landmark distance table for Mid-Near (`MN`) ranking.
    pub fn pacmap_graph_with_landmark_distances<W: Scalar>(
        graph: &CsrGraph<W>,
        landmark_distances: &[f64],
        num_landmarks: usize,
        config: PacmapGraphConfig,
    ) -> CoreResult<Self> {
        Self::pacmap_graph_inner(
            graph,
            config,
            Some(PacmapExternalLandmarks::Distances {
                data: landmark_distances,
                num_landmarks,
            }),
        )
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver coupled with a coarse Louvain area-reserved
    /// backbone initialization and `num_landmarks`-BFS Mid-Near (`MN`) landmarks, returning
    /// `(solver, initial_layout, backbone_result)`.
    ///
    /// Replacing unweighted random-BFS/PivotMDS initialization with a globally unfolded
    /// coarse backbone resolves folding on large-diameter FEM meshes (`bcsstk31`, `venkat50`,
    /// `troll`, `ship_003`) while `num_landmarks`-BFS `MN` landmarks (default `16`) provide
    /// vertex-level Mid-Near distances without fine-graph `RHGH` overhead, remaining
    /// deterministic and cooperative via `step()`/`run_steps()`.
    pub fn pacmap_graph_with_backbone<W: Scalar>(
        graph: &CsrGraph<W>,
        pacmap_cfg: PacmapGraphConfig,
        backbone_cfg: &CoarseBackboneConfig,
    ) -> CoreResult<(Self, Layout2d<S>, CoarseBackboneResult<S>)> {
        pacmap_cfg.validate()?;
        backbone_cfg.validate()?;
        validate_graph_weights(graph, "SgdSolver::pacmap_graph_with_backbone")?;

        let backbone = coarse_backbone_init::<W, S>(graph, backbone_cfg)?;
        let solver = Self::pacmap_graph(graph, pacmap_cfg)?;
        let init_layout = backbone.fine_layout.clone();
        Ok((solver, init_layout, backbone))
    }

    fn pacmap_graph_inner<W: Scalar>(
        graph: &CsrGraph<W>,
        config: PacmapGraphConfig,
        ext_landmarks: Option<PacmapExternalLandmarks<'_>>,
    ) -> CoreResult<Self> {
        let built = build_pacmap_state(graph, config, ext_landmarks)?;
        Ok(Self {
            num_nodes: built.num_nodes,
            pairs: PairStorage::Symmetric(Vec::new()),
            num_springs: built.num_springs,
            num_hinges: built.num_hinges,
            block_order: Vec::new(),
            streaming: None,
            snap_t_fdp: None,
            pacmap: Some(built.state),
            sgd2: None,
            largevis: None,
            drgraph: None,
            graph_cne: None,
            schedule_cfg: built.schedule_cfg,
            eta_min: config.learning_rate,
            eta_max: config.learning_rate,
            iter: 0,
            status: built.status,
            rng: built.rng,
        })
    }

    /// Constructs a doubly-stochastic mini-batch diagonal-preconditioned stress SGD solver
    /// using [`SgdScheduleKind::Squared`] and zero per-step heap allocations.
    ///
    /// Note: This implements diagonal-preconditioned mini-batch stress SGD and is distinct
    /// from Ahmed et al. (2021) multi-criteria `(SGD)^2`.
    pub fn sgd2<W: Scalar>(graph: &CsrGraph<W>, config: Sgd2Config) -> CoreResult<Self> {
        config.validate()?;
        validate_graph_weights(graph, "SgdSolver::sgd2")?;

        let schedule = SgdScheduleConfig {
            iterations: config.iterations,
            eps: config.eps,
            kind: SgdScheduleKind::Squared,
            step_scale: config.step_scale,
        };
        let n = graph.num_nodes();
        let mut solver = if n <= 256 {
            Self::full(graph, schedule, config.seed)?
        } else {
            Self::random_pair(graph, 16, schedule, config.seed)?
        };
        solver.sgd2 = Some(build_sgd2_state(n, config));
        Ok(solver)
    }

    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver using graph Voronoi
    /// partitioning $R(p)$ (breaking distance ties in favor of the smallest current $|R(p)|$)
    /// and directional region-adapted weights $w'_{ip} = |\{j \in R(p) : 2 d_{jp} \le d_{ip}\}| / d_{ip}^2$.
    pub fn ortmann_sparse_stress<W: Scalar>(
        graph: &CsrGraph<W>,
        config: OrtmannSparseStressConfig,
    ) -> CoreResult<Self> {
        let built = build_ortmann_storage(graph, config)?;
        let n_springs = built.storage.len();
        let mut solver = Self::from_pair_storage(
            built.num_nodes,
            built.storage,
            n_springs,
            0,
            built.schedule_cfg,
            config.seed,
        )?;
        if solver.eta_max > 0.0 {
            if let Some(emax) = config.eta_max {
                solver.eta_max = emax;
            }
            if let Some(emin) = config.eta_min {
                solver.eta_min = emin;
            }
            if solver.eta_min > solver.eta_max {
                if config.eta_max.is_some() && config.eta_min.is_none() {
                    solver.eta_min = solver.eta_max * config.eps.min(1.0);
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

    /// Constructs a single-threaded deterministic `LargeVis` solver (`Tang et al., 2016`)
    /// using Vose alias tables over positive edges and $d_j^{\alpha}$-weighted vertices.
    pub fn largevis<W: Scalar>(graph: &CsrGraph<W>, config: LargeVisConfig) -> CoreResult<Self> {
        let built = build_largevis_state(graph, config)?;
        Ok(Self {
            num_nodes: built.num_nodes,
            pairs: PairStorage::Symmetric(Vec::new()),
            num_springs: built.num_springs,
            num_hinges: 0,
            block_order: Vec::new(),
            streaming: None,
            snap_t_fdp: None,
            pacmap: None,
            sgd2: None,
            largevis: Some(built.state),
            drgraph: None,
            graph_cne: None,
            schedule_cfg: built.schedule_cfg,
            eta_min: config.min_lr,
            eta_max: config.initial_lr,
            iter: 0,
            status: built.status,
            rng: built.rng,
        })
    }

    /// Constructs a multilevel negative-sampling `DRGraph` solver (`Zhu et al., 2020`)
    /// with random-merge coarsening, $b$-parameterized Cauchy proximity, and cooperative
    /// coarse-to-fine stepping.
    pub fn drgraph<W: Scalar>(graph: &CsrGraph<W>, config: DrGraphConfig) -> CoreResult<Self> {
        let built = build_drgraph_state(graph, config)?;
        Ok(Self {
            num_nodes: built.num_nodes,
            pairs: PairStorage::Symmetric(Vec::new()),
            num_springs: built.num_springs,
            num_hinges: 0,
            block_order: Vec::new(),
            streaming: None,
            snap_t_fdp: None,
            pacmap: None,
            sgd2: None,
            largevis: None,
            drgraph: Some(built.state),
            graph_cne: None,
            schedule_cfg: built.schedule_cfg,
            eta_min: config.min_lr,
            eta_max: config.initial_lr,
            iter: 0,
            status: built.status,
            rng: built.rng,
        })
    }

    /// Constructs a `Graph-CNE` contrastive neighbor embedding solver (`Böhm et al., 2025`)
    /// supporting [`GraphCneObjective::InfoNce`] and [`GraphCneObjective::NegTSne`].
    pub fn graph_cne<W: Scalar>(graph: &CsrGraph<W>, config: GraphCneConfig) -> CoreResult<Self> {
        let built = build_graph_cne_state(graph, config)?;
        Ok(Self {
            num_nodes: built.num_nodes,
            pairs: PairStorage::Symmetric(Vec::new()),
            num_springs: built.num_springs,
            num_hinges: 0,
            block_order: Vec::new(),
            streaming: None,
            snap_t_fdp: None,
            pacmap: None,
            sgd2: None,
            largevis: None,
            drgraph: None,
            graph_cne: Some(built.state),
            schedule_cfg: built.schedule_cfg,
            eta_min: config.min_lr,
            eta_max: config.initial_lr,
            iter: 0,
            status: built.status,
            rng: built.rng,
        })
    }
}
