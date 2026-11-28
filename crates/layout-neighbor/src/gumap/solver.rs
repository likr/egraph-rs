use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, Sparse12HopPairs, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks,
};

use crate::field_layout::DEFAULT_MAX_2HOP_PER_NODE;
use crate::validation::{compute_12hop_alt_distances, validate_sparse_probabilities};

use super::config::GuMapConfig;
use super::target::{GuMapCore, GuMapTarget, GuMapWorkspace};

const DEFAULT_TIER2_MARGIN: f64 = 1.1;

/// Cooperative, zero-per-step-allocation Graph UMAP (`GuMap`) layout solver supporting any
/// [`GuMapTarget`] ([`Layout2d`], [`Layout3d`](egraph_core::Layout3d),
/// [`LayoutNd`](egraph_core::LayoutNd), [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d), and
/// external [`RiemannianLayout`](egraph_core::RiemannianLayout) targets).
///
/// Uses the `dist_sq.powf(b)` identity ($d^{2b} = (d^2)^b$, $d^{2b-2} = (d^2)^b / d^2$)
/// to avoid per-pair square roots in both attractive and repulsive gradient evaluations.
pub struct GuMapSolver<L: GuMapTarget = Layout2d<f64>> {
    pub(super) core: GuMapCore,
    space: GuMapWorkspace,
    _target: PhantomData<fn() -> L>,
}

impl<L: GuMapTarget> Clone for GuMapSolver<L> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            space: self.space.clone(),
            _target: PhantomData,
        }
    }
}

impl<L: GuMapTarget> fmt::Debug for GuMapSolver<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuMapSolver")
            .field("core", &self.core)
            .field("space", &self.space)
            .finish()
    }
}

impl<L: GuMapTarget> GuMapSolver<L> {
    fn from_fuzzy_set_parts(
        n: usize,
        mut bucket_offsets: Vec<usize>,
        mut edge_src: Vec<u32>,
        mut edge_tgt: Vec<u32>,
        mut edge_weight: Vec<f32>,
        config: GuMapConfig,
        ab: (f64, f64),
    ) -> Self {
        if !edge_src.is_empty() {
            // Sort each canonical bucket `u = min(s, t)` by `(other, s > t)` using one reusable
            // O(max_deg) scratch buffer so `(u, v)` and `(v, u)` (`u < v`) remain interleaved.
            let mut bucket_buf: Vec<(u32, bool, f32)> = Vec::new();
            for u in 0..n {
                let start = bucket_offsets[u];
                let end = bucket_offsets[u + 1];
                if end - start <= 1 {
                    continue;
                }
                let u_u32 = u as u32;
                bucket_buf.clear();
                for pos in start..end {
                    let s = edge_src[pos];
                    let t = edge_tgt[pos];
                    let is_rev = s > t;
                    let other = if is_rev { s } else { t };
                    bucket_buf.push((other, is_rev, edge_weight[pos]));
                }
                bucket_buf.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
                for (offset, &(other, is_rev, w)) in bucket_buf.iter().enumerate() {
                    let pos = start + offset;
                    if is_rev {
                        edge_src[pos] = other;
                        edge_tgt[pos] = u_u32;
                    } else {
                        edge_src[pos] = u_u32;
                        edge_tgt[pos] = other;
                    }
                    edge_weight[pos] = w;
                }
            }
            drop(bucket_buf);
        }
        bucket_offsets.clear();
        drop(bucket_offsets);

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let seed = config.seed;

        let core = GuMapCore {
            config,
            num_nodes: n,
            a: ab.0,
            b: ab.1,
            edge_src,
            edge_tgt,
            edge_weight,
            epoch: 0,
            status,
            rng: DeterministicRng::from_seed_u64(seed),
        };
        let space = L::init_gumap_workspace();

        Self {
            core,
            space,
            _target: PhantomData,
        }
    }

    fn scatter_fuzzy_buckets(
        probs: &SparseProbabilities,
    ) -> (usize, Vec<usize>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let n = probs.num_nodes();
        let mut bucket_offsets = vec![0usize; n + 1];
        for i in 0..n {
            let i_u32 = i as u32;
            let (cols, vals) = probs.row(i);
            for (&j, &w) in cols.iter().zip(vals.iter()) {
                if j != i_u32 && w > 0.0 {
                    let u = i_u32.min(j) as usize;
                    bucket_offsets[u + 1] += 1;
                }
            }
        }
        for u in 0..n {
            bucket_offsets[u + 1] += bucket_offsets[u];
        }
        let m = bucket_offsets[n];
        if m == 0 {
            return (n, bucket_offsets, Vec::new(), Vec::new(), Vec::new());
        }

        let mut edge_src = vec![0u32; m];
        let mut edge_tgt = vec![0u32; m];
        let mut edge_weight = vec![0.0_f32; m];
        let mut cursor = bucket_offsets[..n].to_vec();
        for i in 0..n {
            let i_u32 = i as u32;
            let (cols, vals) = probs.row(i);
            for (&j, &w) in cols.iter().zip(vals.iter()) {
                if j != i_u32 && w > 0.0 {
                    let u = i_u32.min(j) as usize;
                    let pos = cursor[u];
                    edge_src[pos] = i_u32;
                    edge_tgt[pos] = j;
                    edge_weight[pos] = w as f32;
                    cursor[u] = pos + 1;
                }
            }
        }
        drop(cursor);
        (n, bucket_offsets, edge_src, edge_tgt, edge_weight)
    }

    /// Constructs a [`GuMapSolver`] directly from a [`CsrGraph`] using default regularized DoG
    /// Varadhan landmark ALT distances and `DEFAULT_MAX_2HOP_PER_NODE`.
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: GuMapConfig) -> CoreResult<Self> {
        Self::from_graph_dog(
            graph,
            DEFAULT_MAX_2HOP_PER_NODE,
            &DogVaradhanConfig::default(),
            config,
        )
    }

    /// Constructs a [`GuMapSolver`] directly from a precomputed fuzzy simplicial set [`SparseProbabilities`].
    pub fn from_fuzzy_set(probs: SparseProbabilities, config: GuMapConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;
        let ab = config.resolve_ab()?;
        let (n, bucket_offsets, edge_src, edge_tgt, edge_weight) =
            Self::scatter_fuzzy_buckets(&probs);
        drop(probs);
        Ok(Self::from_fuzzy_set_parts(
            n,
            bucket_offsets,
            edge_src,
            edge_tgt,
            edge_weight,
            config,
            ab,
        ))
    }

    /// Constructs a [`GuMapSolver`] from a borrowed [`SparseProbabilities`] without cloning it.
    pub fn from_fuzzy_set_ref(
        probs: &SparseProbabilities,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(probs)?;
        let ab = config.resolve_ab()?;
        let (n, bucket_offsets, edge_src, edge_tgt, edge_weight) =
            Self::scatter_fuzzy_buckets(probs);
        Ok(Self::from_fuzzy_set_parts(
            n,
            bucket_offsets,
            edge_src,
            edge_tgt,
            edge_weight,
            config,
            ab,
        ))
    }

    /// Constructs a [`GuMapSolver`] from a validated [`KnnGraph`] via [`compute_fuzzy_simplicial_set`].
    pub fn from_knn(knn: &KnnGraph, config: GuMapConfig) -> CoreResult<Self> {
        config.validate()?;
        let probs = compute_fuzzy_simplicial_set(knn, config.sigma_iters, config.sigma_tolerance)?;
        Self::from_fuzzy_set(probs, config)
    }

    fn finish_from_pairs(
        n: usize,
        pairs: Sparse12HopPairs,
        d1: Vec<f64>,
        d2: Vec<f64>,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
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
        let probs = compute_fuzzy_simplicial_set(&knn, config.sigma_iters, config.sigma_tolerance)?;
        drop(knn);
        Self::from_fuzzy_set(probs, config)
    }

    /// Constructs a [`GuMapSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using $O(|E|)$ hub-safe 1+2-hop pairs, landmark ALT
    /// distances, `eps_dist` flooring, 1+2-hop priority CSR, and fuzzy simplicial sets.
    pub fn from_landmarks<W>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let (pairs, d1, d2) = compute_12hop_alt_distances(
            graph,
            landmarks,
            max_2hop_per_node,
            config.seed,
            "GuMapSolver::from_landmarks::landmarks",
        )?;
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), config);
        }
        Self::finish_from_pairs(n, pairs, d1, d2, config)
    }

    /// Constructs a [`GuMapSolver`] from a [`CsrGraph`] using $O(|E|)$ hub-safe 1+2-hop pairs,
    /// regularized Chebyshev DoG Varadhan landmark ALT distances, `eps_dist` distance flooring,
    /// 1+2-hop priority CSR construction, and fuzzy simplicial set computation.
    pub fn from_graph_dog<W: Scalar>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_config: &DogVaradhanConfig,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if n <= 1 {
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), config);
        }

        let (landmarks, _dog_stats) =
            compute_regularized_dog_varadhan_landmarks(graph, dog_config)?;
        let (pairs, d1, d2) = compute_12hop_alt_distances(
            graph,
            &landmarks,
            max_2hop_per_node,
            config.seed,
            "GuMapSolver::from_graph_dog::landmarks",
        )?;
        drop(landmarks);
        Self::finish_from_pairs(n, pairs, d1, d2, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GuMapConfig {
        &self.core.config
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.core.num_nodes
    }

    /// Returns the number of stored directed positive probability edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.core.edge_src.len()
    }

    /// Returns the resolved UMAP low-dimensional curve parameters `(a, b)`.
    #[inline]
    pub fn ab(&self) -> (f64, f64) {
        (self.core.a, self.core.b)
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[inline]
    pub fn a(&self) -> f64 {
        self.core.a
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[inline]
    pub fn b(&self) -> f64 {
        self.core.b
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.core.status
    }

    /// Returns the number of completed optimization iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.core.epoch
    }

    /// Returns the total configured optimization iterations.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.core.config.iterations)
    }
}

impl<L: GuMapTarget> IterativeSolver for GuMapSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        target.validate_gumap_target(&self.core, &self.space)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.core.status.is_finished() {
            return Ok(self.core.status);
        }

        let n = self.core.num_nodes;
        if n < 2 || self.core.edge_src.is_empty() {
            self.core.status = SolverStatus::Converged;
            return Ok(self.core.status);
        }

        if let Err(err) = target.step_gumap_target(&mut self.core, &mut self.space) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.core.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.core.epoch += 1;
        self.core.status = if self.core.epoch >= self.core.config.iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.core.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.core.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.core.epoch
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.core.config.iterations)
    }
}
