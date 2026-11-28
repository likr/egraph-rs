use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, Sparse12HopPairs, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks,
};
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};

use crate::validation::{compute_12hop_alt_distances, validate_sparse_probabilities};

use super::config::{FieldFftStats, FieldRepulsionBackend};
use super::ib_umap_config::IbUmapConfig;
use super::ib_umap_step::step_inner;
use super::quadtree::FieldQuadtree;

/// Default per-node 2-hop cap used across 1+2-hop priority CSR builders (`32`).
pub const DEFAULT_MAX_2HOP_PER_NODE: usize = 32;

const DEFAULT_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOL: f64 = 1e-5;
const DEFAULT_TIER2_MARGIN: f64 = 1.1;

/// Cooperative, zero-per-step-allocation `ibUMAP` (interpolation-based batch Graph UMAP)
/// solver targeting [`Layout2d`].
#[derive(Debug, Clone)]
pub struct IbUmapSolver<S: Scalar = f64> {
    pub(super) config: IbUmapConfig,
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) edge_weights: Vec<f32>,
    pub(super) hub_eta: Vec<f64>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) phi_1: Vec<S>,
    pub(super) rep_x: Vec<S>,
    pub(super) rep_y: Vec<S>,
    pub(super) fft_ws: Fft2dWorkspace,
    pub(super) ibfft: Option<IbFft2d<S>>,
    pub(super) last_fft_stats: Option<FieldFftStats>,
    pub(super) field_quadtree: FieldQuadtree,
    pub(super) rng: DeterministicRng,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) status: SolverStatus,
}

impl<S: Scalar> IbUmapSolver<S> {
    fn finish_from_pairs<W>(
        graph: &CsrGraph<W>,
        pairs: Sparse12HopPairs,
        d1: Vec<f64>,
        d2: Vec<f64>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let n = graph.num_nodes();
        let knn = build_gumap_12hop_priority_csr(
            n,
            pairs.u1(),
            pairs.v1(),
            &d1,
            pairs.u2(),
            pairs.v2(),
            &d2,
            DEFAULT_TIER2_MARGIN,
        )?;
        drop(pairs);
        drop(d1);
        drop(d2);
        let probs = compute_fuzzy_simplicial_set(&knn, DEFAULT_SIGMA_ITERS, DEFAULT_SIGMA_TOL)?;
        drop(knn);
        let mut degrees = vec![0u32; n];
        if !graph.is_directed() {
            for (u, deg_out) in degrees.iter_mut().enumerate() {
                *deg_out = graph.degree(u) as u32;
            }
        } else {
            for u in 0..n {
                for &v in graph.neighbors(u) {
                    let v_usize = v as usize;
                    if v_usize != u && v_usize < n {
                        degrees[u] = degrees[u].saturating_add(1);
                        degrees[v_usize] = degrees[v_usize].saturating_add(1);
                    }
                }
            }
        }
        Self::from_fuzzy_set(probs, Some(&degrees), config)
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using default regularized DoG
    /// Varadhan parameters and [`DEFAULT_MAX_2HOP_PER_NODE`].
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: IbUmapConfig) -> CoreResult<Self> {
        Self::from_graph_dog(
            graph,
            DEFAULT_MAX_2HOP_PER_NODE,
            &DogVaradhanConfig::default(),
            config,
        )
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using $O(|E|)$ hub-safe 1+2-hop pairs, landmark ALT
    /// distances, priority CSR construction, and fuzzy simplicial sets.
    pub fn from_landmarks<W>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let (pairs, d1, d2) = compute_12hop_alt_distances(
            graph,
            landmarks,
            max_2hop_per_node,
            config.seed,
            "IbUmapSolver::from_landmarks::landmarks",
        )?;
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), Some(&degs), config);
        }
        Self::finish_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using $O(|E|)$ 1+2-hop pair
    /// sampling, regularized DoG Varadhan landmark ALT distances, priority CSR construction,
    /// and fuzzy simplicial set computation.
    pub fn from_graph_dog<W: Scalar>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_cfg: &DogVaradhanConfig,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        let (landmarks, _dog_stats) = compute_regularized_dog_varadhan_landmarks(graph, dog_cfg)?;
        let (pairs, d1, d2) = compute_12hop_alt_distances(
            graph,
            &landmarks,
            max_2hop_per_node,
            config.seed,
            "IbUmapSolver::from_graph_dog::landmarks",
        )?;
        drop(landmarks);
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), Some(&degs), config);
        }
        Self::finish_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an [`IbUmapSolver`] from a validated [`KnnGraph`] via
    /// [`compute_fuzzy_simplicial_set`].
    pub fn from_knn(
        knn: &KnnGraph,
        sigma_iters: usize,
        sigma_tolerance: f64,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let probs = compute_fuzzy_simplicial_set(knn, sigma_iters, sigma_tolerance)?;
        Self::from_fuzzy_set(probs, None, config)
    }

    fn build_inner(
        n: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let (a, b) = config.resolve_ab()?;
        let ibfft = if config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft.clone())?)
        } else {
            None
        };

        let mut hub_eta = vec![1.0_f64; n];
        if config.hub_damping {
            for (i, eta_out) in hub_eta.iter_mut().enumerate() {
                let deg_i = match degrees {
                    Some(deg) => deg[i],
                    None => indptr[i + 1] - indptr[i],
                };
                *eta_out = 1.0 / (deg_i.max(1) as f64);
            }
        }

        let total_iterations = config.iterations;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let buf_len = if config.backend == FieldRepulsionBackend::IbFft {
            n
        } else {
            0
        };
        let seed = config.seed;

        Ok(Self {
            config,
            a,
            b,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta,
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            phi_1: vec![S::ZERO; buf_len],
            rep_x: vec![S::ZERO; n],
            rep_y: vec![S::ZERO; n],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            last_fft_stats: None,
            field_quadtree: FieldQuadtree::new(),
            rng: DeterministicRng::from_seed_u64(seed),
            iteration: 0,
            total_iterations,
            status,
        })
    }

    /// Constructs an [`IbUmapSolver`] directly from a precomputed fuzzy simplicial set
    /// [`SparseProbabilities`] and optional per-node degrees for inverse-degree hub damping.
    pub fn from_fuzzy_set(
        probs: SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;
        let n = probs.num_nodes();
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "IbUmapSolver::from_fuzzy_set::degrees",
            ));
        }
        let (_n, indptr, indices, values) = probs.into_parts();
        let edge_weights: Vec<f32> = values.into_iter().map(|w| w as f32).collect();
        Self::build_inner(n, indptr, indices, edge_weights, degrees, config)
    }

    /// Constructs an [`IbUmapSolver`] from a borrowed [`SparseProbabilities`] without cloning its
    /// `f64` values vector.
    pub fn from_fuzzy_set_ref(
        probs: &SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(probs)?;
        let n = probs.num_nodes();
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "IbUmapSolver::from_fuzzy_set_ref::degrees",
            ));
        }
        let indptr = probs.indptr().to_vec();
        let indices = probs.indices().to_vec();
        let edge_weights: Vec<f32> = probs.values().iter().map(|&w| w as f32).collect();
        Self::build_inner(n, indptr, indices, edge_weights, degrees, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &IbUmapConfig {
        &self.config
    }

    /// Returns the number of vertices expected in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the active repulsion evaluation backend.
    #[inline]
    pub fn backend(&self) -> FieldRepulsionBackend {
        self.config.backend
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.fft_ws.fft_2d_call_count()
    }

    /// Returns telemetry counters from the most recent `IbFft` step, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FieldFftStats> {
        self.last_fft_stats.as_ref()
    }

    /// Returns the resolved UMAP low-dimensional curve parameters `(a, b)`.
    #[inline]
    pub fn ab(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of completed iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Returns the total scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.total_iterations)
    }
}

impl<S: Scalar> IterativeSolver for IbUmapSolver<S> {
    type Target = Layout2d<S>;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                target.len(),
                "IbUmapSolver::validate_target",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if let Err(err) = step_inner(self, target) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.iteration += 1;
        self.status = if self.iteration >= self.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.total_iterations)
    }
}
