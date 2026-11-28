use std::collections::VecDeque;
use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_linalg::CsrMatrix;
use egraph_linalg_amg::{AmgPcgConfig, AmgPcgSolver, AmgPcgWorkspace};

use super::config::MaxentStressConfig;
use super::graph_prep::{
    MaxentEdge, extract_canonical_positive_edges, extract_canonical_unweighted_edges,
};
use super::repulsion::{DefaultMaxentBackend, MaxentRepulsion};
use super::step::{step_euclidean, validate_euclidean_target};

#[derive(Debug, Clone)]
pub(super) struct MaxentCoreState {
    pub(super) config: MaxentStressConfig,
    pub(super) num_nodes: usize,
    pub(super) effective_q: f64,
    pub(super) alpha: f64,
    pub(super) edges: Vec<MaxentEdge>,
    pub(super) all_unit_edges: bool,
    pub(super) comp_id: Vec<u32>,
    pub(super) comp_sizes: Vec<u32>,
    pub(super) comp_sum: Vec<f64>,
    pub(super) amg: AmgPcgSolver,
    pub(super) amg_ws: AmgPcgWorkspace,
    pub(super) cur: Vec<f64>,
    pub(super) next: Vec<f64>,
    pub(super) stress_rhs: Vec<f64>,
    pub(super) rep: Vec<f64>,
    pub(super) rhs_axis: Vec<f64>,
    pub(super) next_axis: Vec<f64>,
    pub(super) locked_dims: Option<usize>,
    pub(super) stage_idx: usize,
    pub(super) stage_step: usize,
    pub(super) total_iter_executed: usize,
    pub(super) total_iterations_budget: usize,
    pub(super) status: SolverStatus,
}

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver implementing [`IterativeSolver`]
/// for [`EuclideanLayout`] targets (`Layout2d`, `Layout3d`, `LayoutNd`).
///
/// The repulsion backend `B: MaxentRepulsion<L>` is resolved at compile time via
/// [`DefaultMaxentBackend`]:
/// - [`Layout2d`](egraph_core::Layout2d) defaults to [`BarnesHut`](super::BarnesHut)
///   (`ArenaQuadtree2d`) and also supports [`Exact`](super::Exact).
/// - [`Layout3d`](egraph_core::Layout3d) defaults to [`BarnesHut`](super::BarnesHut)
///   (`ArenaOctree3d`) and also supports [`Exact`](super::Exact).
/// - [`LayoutNd`](egraph_core::LayoutNd) defaults to [`Exact`](super::Exact).
///
/// Attempting to pair [`LayoutNd`](egraph_core::LayoutNd) with [`BarnesHut`](super::BarnesHut)
/// fails at compile time:
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_stress::{BarnesHut, MaxentStressConfig, MaxentStressSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _solver = MaxentStressSolver::<LayoutNd<f64>, BarnesHut>::new(
///     &g,
///     MaxentStressConfig::default(),
/// );
/// ```
///
/// Using [`LayoutNd`](egraph_core::LayoutNd) with its default backend automatically selects
/// [`Exact`](super::Exact):
///
/// ```
/// use egraph_core::{CsrGraph, IterativeSolver, LayoutNd};
/// use egraph_layout_stress::{MaxentStressConfig, MaxentStressSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let mut solver =
///     MaxentStressSolver::<LayoutNd<f64>>::new(&g, MaxentStressConfig::default()).unwrap();
/// let mut layout = LayoutNd::<f64>::zeros(2, 4).unwrap();
/// assert!(solver.run(&mut layout).unwrap().is_finished());
/// ```
pub struct MaxentStressSolver<
    L: EuclideanLayout + DefaultMaxentBackend = Layout2d<f64>,
    B: MaxentRepulsion<L> = <L as DefaultMaxentBackend>::Backend,
> {
    core: MaxentCoreState,
    backend: B,
    evaluator: B::Evaluator,
    _layout: PhantomData<fn() -> L>,
}

impl<L, B> Clone for MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            backend: self.backend.clone(),
            evaluator: self.evaluator.clone(),
            _layout: PhantomData,
        }
    }
}

impl<L, B> fmt::Debug for MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L> + fmt::Debug,
    B::Evaluator: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaxentStressSolver")
            .field("core", &self.core)
            .field("backend", &self.backend)
            .field("evaluator", &self.evaluator)
            .finish()
    }
}

impl<L, B> MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L> + Default,
{
    /// Constructs a [`MaxentStressSolver`] for `graph` using `config` and `B::default()`.
    ///
    /// Weighted graphs use positive finite edge weights as target lengths $d_{ij}$ and spring
    /// weights $w_{ij} = d_{ij}^{-2}$. Zero-weight edges and self-loops are skipped; directed
    /// reciprocal edges are canonicalized using the minimum positive edge length.
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: MaxentStressConfig) -> CoreResult<Self> {
        Self::from_graph_with_backend(graph, config, B::default())
    }

    /// Constructs a [`MaxentStressSolver`] for `graph` using `config` and `B::default()`.
    #[inline]
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: MaxentStressConfig,
    ) -> CoreResult<Self> {
        Self::from_graph_with_backend(graph, config, B::default())
    }

    /// Constructs a [`MaxentStressSolver`] from an unweighted [`CsrGraph`] (ignoring edge
    /// weights and accepting `CsrGraph<()>` directly) using `B::default()`.
    #[inline]
    pub fn from_unweighted_graph<W>(
        graph: &CsrGraph<W>,
        config: MaxentStressConfig,
    ) -> CoreResult<Self> {
        Self::from_unweighted_graph_with_backend(graph, config, B::default())
    }
}

impl<L, B> MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L>,
{
    /// Constructs a [`MaxentStressSolver`] for `graph` with an explicit repulsion backend.
    #[inline]
    pub fn new_with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: MaxentStressConfig,
        backend: B,
    ) -> CoreResult<Self> {
        Self::from_graph_with_backend(graph, config, backend)
    }

    /// Constructs a [`MaxentStressSolver`] for `graph` with an explicit repulsion backend.
    pub fn from_graph_with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: MaxentStressConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr(
                "MaxentStressSolver node count exceeds u32::MAX",
            ));
        }

        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("MaxentStressSolver::graph::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "MaxentStressSolver requires non-negative edge weights",
                    ));
                }
            }
        }

        let raw_edges = extract_canonical_positive_edges(graph)?;
        Self::build_from_raw_edges(n, raw_edges, config, backend)
    }

    /// Constructs a [`MaxentStressSolver`] from an unweighted [`CsrGraph`] with an explicit
    /// repulsion backend.
    pub fn from_unweighted_graph_with_backend<W>(
        graph: &CsrGraph<W>,
        config: MaxentStressConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr(
                "MaxentStressSolver node count exceeds u32::MAX",
            ));
        }
        let raw_edges = extract_canonical_unweighted_edges(graph)?;
        Self::build_from_raw_edges(n, raw_edges, config, backend)
    }

    fn build_from_raw_edges(
        n: usize,
        raw_edges: Vec<(u32, u32, f64)>,
        config: MaxentStressConfig,
        backend: B,
    ) -> CoreResult<Self> {
        let mut edges = Vec::with_capacity(raw_edges.len());
        let mut all_unit_edges = true;
        let mut deg = vec![0usize; n];
        let mut adj_entries: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut diag_sums = vec![0.0_f64; n];

        for (u, v, d) in raw_edges {
            let w = 1.0 / (d * d);
            if !w.is_finite() {
                return Err(CoreError::invalid_input(format!(
                    "edge ({u}, {v}) distance {d} produces non-finite spring weight"
                )));
            }
            if (d - 1.0).abs() > 0.0 {
                all_unit_edges = false;
            }
            edges.push(MaxentEdge { u, v, d, w });
            let ui = u as usize;
            let vi = v as usize;
            deg[ui] += 1;
            deg[vi] += 1;
            diag_sums[ui] += w;
            diag_sums[vi] += w;
            adj_entries[ui].push((v, -w));
            adj_entries[vi].push((u, -w));
        }

        let effective_q = match config.q {
            Some(qv) => qv,
            None => {
                let deg1_count = deg.iter().filter(|&&d| d == 1).count();
                if n > 0 && (deg1_count as f64) > 0.30 * (n as f64) {
                    0.8
                } else {
                    0.0
                }
            }
        };

        let evaluator = backend.build_evaluator(n, effective_q, config.eps_sq)?;

        // Build weighted combinatorial Laplacian L_w with strictly ascending column indices per row.
        let mut indptr = Vec::with_capacity(n + 1);
        let total_nnz = edges.len().saturating_mul(2).saturating_add(n);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut values = Vec::with_capacity(total_nnz);
        indptr.push(0u32);
        for u in 0..n {
            if !adj_entries[u].is_empty() {
                let row = &mut adj_entries[u];
                row.push((u as u32, diag_sums[u]));
                row.sort_unstable_by_key(|&(col, _)| col);
                for &(col, val) in row.iter() {
                    indices.push(col);
                    values.push(val);
                }
            }
            let nnz_u32 = u32::try_from(indices.len()).map_err(|_| {
                CoreError::invalid_csr("MaxentStress Laplacian nnz exceeds u32::MAX")
            })?;
            indptr.push(nnz_u32);
        }
        let l_w = CsrMatrix::from_csr(n, indptr, indices, values)?;

        let amg_cfg = AmgPcgConfig::new()
            .with_rel_tol(config.cg_tol)
            .with_max_iterations(config.cg_max_iters)
            .with_seed(config.seed);
        let amg = AmgPcgSolver::new(&l_w, amg_cfg)?;
        let amg_ws = AmgPcgWorkspace::for_solver(&amg);

        // Compute connected components of S once for per-component zero-sum RHS centering.
        let mut comp_id = vec![u32::MAX; n];
        let mut comp_sizes = Vec::new();
        let mut queue = VecDeque::new();
        for start in 0..n {
            if comp_id[start] != u32::MAX {
                continue;
            }
            let cid = comp_sizes.len() as u32;
            comp_id[start] = cid;
            queue.push_back(start);
            let mut size = 0u32;
            while let Some(u) = queue.pop_front() {
                size += 1;
                for &(v_u32, _) in &adj_entries[u] {
                    let v = v_u32 as usize;
                    if v != u && comp_id[v] == u32::MAX {
                        comp_id[v] = cid;
                        queue.push_back(v);
                    }
                }
            }
            comp_sizes.push(size);
        }

        let total_iterations_budget = config.stages.saturating_mul(config.steps_per_stage);
        let status = if total_iterations_budget == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let core = MaxentCoreState {
            alpha: config.alpha_init,
            config,
            num_nodes: n,
            effective_q,
            edges,
            all_unit_edges,
            comp_id,
            comp_sizes,
            comp_sum: Vec::new(),
            amg,
            amg_ws,
            cur: Vec::new(),
            next: Vec::new(),
            stress_rhs: Vec::new(),
            rep: Vec::new(),
            rhs_axis: vec![0.0; n],
            next_axis: vec![0.0; n],
            locked_dims: None,
            stage_idx: 0,
            stage_step: 0,
            total_iter_executed: 0,
            total_iterations_budget,
            status,
        };

        Ok(Self {
            core,
            backend,
            evaluator,
            _layout: PhantomData,
        })
    }
}

impl<L, B> MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L>,
{
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &MaxentStressConfig {
        &self.core.config
    }

    /// Returns the configured repulsion backend.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.core.num_nodes
    }

    /// Returns the active annealing repulsion weight $\alpha$.
    #[inline]
    pub fn current_alpha(&self) -> f64 {
        self.core.alpha
    }

    /// Returns the effective repulsion exponent $q$ used by the solver.
    #[inline]
    pub fn effective_q(&self) -> f64 {
        self.core.effective_q
    }

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.core.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.core.total_iter_executed
    }

    /// Returns the configured maximum iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations_budget)
    }
}

impl<L, B> IterativeSolver for MaxentStressSolver<L, B>
where
    L: EuclideanLayout + DefaultMaxentBackend,
    B: MaxentRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_euclidean_target(&self.core, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.core.status.is_finished() {
            return Ok(self.core.status);
        }

        if let Err(err) = step_euclidean(&mut self.core, &mut self.evaluator, target) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.core.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        Ok(self.core.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.core.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.core.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations_budget)
    }
}
