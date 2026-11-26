use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, SolverStatus};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, Sparse12HopPairs, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks,
};
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};
use egraph_spatial_trees::ArenaQuadtree2d;

use crate::validation::{compute_12hop_alt_distances, validate_sparse_probabilities};

use super::config::{FieldRepulsionBackend, IbUmapConfig};
use super::quadtree::FieldQuadtree;
use super::solver::{FieldLayoutSolver, FieldSolverMode};

const DEFAULT_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOL: f64 = 1e-5;
const DEFAULT_TIER2_MARGIN: f64 = 1.1;

impl<S: Scalar> FieldLayoutSolver<S> {
    fn finish_ib_umap_from_pairs<W: Scalar>(
        graph: &CsrGraph<W>,
        pairs: Sparse12HopPairs,
        d1: Vec<f64>,
        d2: Vec<f64>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let n = graph.num_nodes();
        let knn = build_gumap_12hop_priority_csr(
            n,
            &pairs.u1,
            &pairs.v1,
            &d1,
            &pairs.u2,
            &pairs.v2,
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
        Self::ib_umap_from_fuzzy_set(probs, Some(&degrees), config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using $O(|E|)$ hub-safe 1+2-hop pairs, landmark ALT
    /// distances, priority CSR construction, and fuzzy simplicial sets.
    pub fn ib_umap_from_landmarks<W: Scalar>(
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
            "FieldLayoutSolver::ib_umap_from_landmarks::landmarks",
        )?;
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::ib_umap_from_fuzzy_set(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
            );
        }
        Self::finish_ib_umap_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a [`CsrGraph`] using $O(|E|)$
    /// 1+2-hop pair sampling, regularized DoG Varadhan landmark ALT distances, priority CSR
    /// construction, and fuzzy simplicial set computation.
    pub fn ib_umap<W: Scalar>(
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
            "FieldLayoutSolver::ib_umap_from_landmarks::landmarks",
        )?;
        drop(landmarks);
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::ib_umap_from_fuzzy_set(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
            );
        }
        Self::finish_ib_umap_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a validated [`KnnGraph`] via
    /// [`compute_fuzzy_simplicial_set`].
    pub fn ib_umap_from_knn(
        knn: &KnnGraph,
        sigma_iters: usize,
        sigma_tol: f64,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let probs = compute_fuzzy_simplicial_set(knn, sigma_iters, sigma_tol)?;
        Self::ib_umap_from_fuzzy_set(probs, None, config)
    }

    fn build_ib_umap_inner(
        n: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let (a, b) = config.resolve_ab()?;
        let ibfft = if config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft)?)
        } else {
            None
        };

        let mut hub_eta = vec![1.0_f64; n];
        if config.hub_damping {
            for (i, eta_out) in hub_eta.iter_mut().enumerate() {
                let deg_i = match degrees {
                    Some(deg) => deg[i],
                    None => (indptr[i + 1] - indptr[i]) as u32,
                };
                *eta_out = 1.0 / (deg_i.max(1) as f64);
            }
        }

        let total_iterations = config.n_epochs;
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

        Ok(Self {
            mode: FieldSolverMode::IbUmap { config, a, b },
            backend: config.backend,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta,
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            phi_1: vec![S::ZERO; buf_len],
            rep_x: vec![S::ZERO; n],
            rep_y: vec![S::ZERO; n],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            quadtree: ArenaQuadtree2d::new(),
            field_quadtree: FieldQuadtree::new(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            iteration: 0,
            total_iterations,
            status,
        })
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] directly from a precomputed fuzzy simplicial
    /// set [`SparseProbabilities`] and optional per-node degrees for inverse-degree hub damping.
    pub fn ib_umap_from_fuzzy_set(
        probs: SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;
        let n = probs.n_nodes;
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "FieldLayoutSolver::ib_umap_from_fuzzy_set::degrees",
            ));
        }
        let SparseProbabilities {
            n_nodes: _,
            indptr,
            indices,
            values,
        } = probs;
        let edge_weights: Vec<f32> = values.into_iter().map(|w| w as f32).collect();
        Self::build_ib_umap_inner(n, indptr, indices, edge_weights, degrees, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a borrowed [`SparseProbabilities`]
    /// without cloning its `f64` values vector.
    pub fn ib_umap_from_fuzzy_set_ref(
        probs: &SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(probs)?;
        let n = probs.n_nodes;
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "FieldLayoutSolver::ib_umap_from_fuzzy_set::degrees",
            ));
        }
        let indptr = probs.indptr.clone();
        let indices = probs.indices.clone();
        let edge_weights: Vec<f32> = probs.values.iter().map(|&w| w as f32).collect();
        Self::build_ib_umap_inner(n, indptr, indices, edge_weights, degrees, config)
    }
}
