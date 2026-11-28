use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, EdgeWeight, EuclideanLayout, SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, Sparse12HopPairs, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks,
};

use crate::validation::{compute_12hop_alt_distances, validate_sparse_probabilities};

use super::ib_umap::{DEFAULT_MAX_2HOP_PER_NODE, IbUmapSolver};
use super::ib_umap_config::IbUmapConfig;
use super::ib_umap_repulsion::{DefaultIbUmapBackend, IbUmapCore, IbUmapRepulsion};

const DEFAULT_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOL: f64 = 1e-5;
const DEFAULT_TIER2_MARGIN: f64 = 1.1;

impl<L, B> IbUmapSolver<L, B>
where
    L: EuclideanLayout + DefaultIbUmapBackend,
    B: IbUmapRepulsion<L>,
{
    fn finish_from_pairs<W>(
        graph: &CsrGraph<W>,
        pairs: Sparse12HopPairs,
        d1: Vec<f64>,
        d2: Vec<f64>,
        config: IbUmapConfig,
        backend: B,
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
        Self::from_fuzzy_set_with_backend(probs, Some(&degrees), config, backend)
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using default regularized DoG
    /// Varadhan parameters, [`DEFAULT_MAX_2HOP_PER_NODE`], and an explicit repulsion `backend`.
    #[inline]
    pub fn new_with_backend<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_dog_with_backend(
            graph,
            DEFAULT_MAX_2HOP_PER_NODE,
            &DogVaradhanConfig::default(),
            config,
            backend,
        )
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`], precomputed [`DogVaradhanLandmarks`],
    /// and an explicit repulsion `backend`.
    pub fn from_landmarks_with_backend<W>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let n = graph.num_nodes();
        let (pairs, d1, d2) = compute_12hop_alt_distances(
            graph,
            landmarks,
            max_2hop_per_node,
            config.seed,
            "IbUmapSolver::from_landmarks_with_backend::landmarks",
        )?;
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::from_fuzzy_set_with_backend(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
                backend,
            );
        }
        Self::finish_from_pairs(graph, pairs, d1, d2, config, backend)
    }

    /// Constructs an [`IbUmapSolver`] from a [`CsrGraph`] using $O(|E|)$ 1+2-hop pair
    /// sampling, regularized DoG Varadhan landmark ALT distances, and an explicit repulsion
    /// `backend`.
    pub fn from_graph_dog_with_backend<W: EdgeWeight>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_cfg: &DogVaradhanConfig,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
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
            "IbUmapSolver::from_graph_dog_with_backend::landmarks",
        )?;
        drop(landmarks);
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::from_fuzzy_set_with_backend(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
                backend,
            );
        }
        Self::finish_from_pairs(graph, pairs, d1, d2, config, backend)
    }

    /// Constructs an [`IbUmapSolver`] from a validated [`KnnGraph`] via
    /// [`compute_fuzzy_simplicial_set`] with an explicit repulsion `backend`.
    pub fn from_knn_with_backend(
        knn: &KnnGraph,
        sigma_iters: usize,
        sigma_tolerance: f64,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let probs = compute_fuzzy_simplicial_set(knn, sigma_iters, sigma_tolerance)?;
        Self::from_fuzzy_set_with_backend(probs, None, config, backend)
    }

    /// Constructs an [`IbUmapSolver`] directly from a precomputed fuzzy simplicial set
    /// [`SparseProbabilities`], optional per-node degrees, and an explicit repulsion `backend`.
    pub fn from_fuzzy_set_with_backend(
        probs: SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        validate_sparse_probabilities(&probs)?;
        let n = probs.num_nodes();
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "IbUmapSolver::from_fuzzy_set_with_backend::degrees",
            ));
        }
        let (_n, indptr, indices, values) = probs.into_parts();
        let mut edge_weights = Vec::with_capacity(values.len());
        for w in values {
            let wf = w as f32;
            if !wf.is_finite() {
                return Err(CoreError::non_finite(
                    "IbUmapSolver::from_fuzzy_set_with_backend::edge_weights",
                ));
            }
            edge_weights.push(wf);
        }
        Self::build_inner(n, indptr, indices, edge_weights, degrees, config, backend)
    }

    /// Constructs an [`IbUmapSolver`] from a borrowed [`SparseProbabilities`] without cloning
    /// its `f64` values vector, using an explicit repulsion `backend`.
    pub fn from_fuzzy_set_ref_with_backend(
        probs: &SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        validate_sparse_probabilities(probs)?;
        let n = probs.num_nodes();
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "IbUmapSolver::from_fuzzy_set_ref_with_backend::degrees",
            ));
        }
        let indptr = probs.indptr().to_vec();
        let indices = probs.indices().to_vec();
        let mut edge_weights = Vec::with_capacity(probs.values().len());
        for &w in probs.values() {
            let wf = w as f32;
            if !wf.is_finite() {
                return Err(CoreError::non_finite(
                    "IbUmapSolver::from_fuzzy_set_ref_with_backend::edge_weights",
                ));
            }
            edge_weights.push(wf);
        }
        Self::build_inner(n, indptr, indices, edge_weights, degrees, config, backend)
    }

    fn build_inner(
        n: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
        backend: B,
    ) -> CoreResult<Self> {
        let (a, b) = config.resolve_ab()?;
        let backend_state = backend.init_state(n)?;
        let space = L::init_ib_umap_workspace(n);

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
        let seed = config.seed;

        let core = IbUmapCore {
            config,
            a,
            b,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta,
            rng: DeterministicRng::from_seed_u64(seed),
            iteration: 0,
            total_iterations,
            status,
        };

        Ok(Self {
            core,
            space,
            backend,
            backend_state,
            _target: PhantomData,
        })
    }
}
