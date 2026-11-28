//! Quadrilateral Simmelian Backbone extraction and layout solver (Nocaj, Ortmann & Brandes, 2015).
//!
//! Implements the hairball-untangling graph simplification pipeline from Nocaj et al. (2015):
//! 1. **Chiba–Nishizeki $O(m\,\alpha(G))$ quadrangle counting**: counts 4-cycles $q(u, v)$ containing
//!    each canonical undirected simple edge $(u, v)$.
//! 2. **Normalized quadrilateral embeddedness**:
//!    $$Q(u, v) = \frac{q(u, v)}{\sqrt{q(u)\,q(v)}}, \qquad q(v) = \sum_{w \in N(v)} q(v, w).$$
//! 3. **Tie-invariant Simmelian prefix-Jaccard reweighting**: ranks each vertex's neighbors by
//!    $Q(u, \cdot)$ descending (expanding ties symmetrically) and sets $w(u, v)$ to the maximum
//!    prefix Jaccard similarity.
//! 4. **Bucketed Union of Maximum Spanning Trees (`UMST`)**: processes equal-weight edge buckets
//!    before committing Union-Find merges so all edges belonging to any MST are preserved.
//! 5. **Backbone assembly & layout**: combines $E_{\text{UMST}}$ with top-`retain_ratio` edges and
//!    runs cooperative [`StressMajorizationSolver`] or [`SgdSolver`] via [`IterativeSolver`].

mod simmelian;
#[cfg(test)]
mod tests_simmelian;
#[cfg(test)]
mod tests_solver;
mod umst;

pub use umst::{extract_quadrilateral_backbone, extract_quadrilateral_backbone_unweighted};

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_layout_sgd::{SgdPairBuffer, SgdScheduleConfig, SgdSolver};
use egraph_layout_stress::{EuclideanPcg, StressMajorizationConfig, StressMajorizationSolver};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

/// Layout backend used by [`BackboneSolver`] on the extracted backbone graph.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackboneLayoutMode {
    /// Gansner-Koren-North (2004) Stress Majorization (Nocaj et al. 2015 default).
    #[default]
    StressMajorization,
    /// Full-APSP Stochastic Gradient Descent (`SgdSolver::full`).
    Sgd,
}

/// Configuration for Quadrilateral Simmelian Backbone extraction and layout.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneConfig {
    /// Fraction of top-ranked edges retained beyond `E_UMST`, in `[0.0, 1.0]` (`default: 0.2`).
    pub retain_ratio: f64,
    /// Layout edge weight scale assigned to `E_UMST` edges, finite `>= 0.0` (`default: 1.0`).
    pub umst_weight_scale: f64,
    /// Layout edge weight scale assigned to non-backbone edges, finite `>= 0.0` (`default: 0.0`,
    /// omitting non-backbone edges as in Nocaj et al. 2015 §3.2).
    pub non_backbone_weight_scale: f64,
    /// When `true` and the input graph carries edge weights, incorporate input edge weights into
    /// Simmelian ranking, UMST priority, and output backbone weights (`default: true`).
    pub use_weights: bool,
    /// Layout optimization backend (`default: BackboneLayoutMode::StressMajorization`).
    pub layout_mode: BackboneLayoutMode,
    /// Maximum layout optimization iterations (`default: 100`).
    pub iterations: usize,
    /// Deterministic RNG seed used when `layout_mode == BackboneLayoutMode::Sgd` (`default: 42`).
    pub seed: u64,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            retain_ratio: 0.2,
            umst_weight_scale: 1.0,
            non_backbone_weight_scale: 0.0,
            use_weights: true,
            layout_mode: BackboneLayoutMode::StressMajorization,
            iterations: 100,
            seed: 42,
        }
    }
}

impl BackboneConfig {
    /// Creates a new [`BackboneConfig`] initialized with [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the fraction of top-ranked edges retained beyond `E_UMST` in `[0.0, 1.0]`.
    #[inline]
    pub fn with_retain_ratio(mut self, retain_ratio: f64) -> Self {
        self.retain_ratio = retain_ratio;
        self
    }

    /// Sets the layout edge weight scale assigned to `E_UMST` edges.
    #[inline]
    pub fn with_umst_weight_scale(mut self, umst_weight_scale: f64) -> Self {
        self.umst_weight_scale = umst_weight_scale;
        self
    }

    /// Sets the layout edge weight scale assigned to non-backbone edges.
    #[inline]
    pub fn with_non_backbone_weight_scale(mut self, non_backbone_weight_scale: f64) -> Self {
        self.non_backbone_weight_scale = non_backbone_weight_scale;
        self
    }

    /// Sets whether input edge weights are incorporated into Simmelian ranking and UMST priority.
    #[inline]
    pub fn with_use_weights(mut self, use_weights: bool) -> Self {
        self.use_weights = use_weights;
        self
    }

    /// Sets the layout optimization backend.
    #[inline]
    pub fn with_layout_mode(mut self, layout_mode: BackboneLayoutMode) -> Self {
        self.layout_mode = layout_mode;
        self
    }

    /// Sets the maximum layout optimization iterations.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the deterministic RNG seed used when `layout_mode == BackboneLayoutMode::Sgd`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all backbone extraction and layout hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.retain_ratio.is_finite() {
            return Err(CoreError::non_finite("BackboneConfig::retain_ratio"));
        }
        if !(0.0..=1.0).contains(&self.retain_ratio) {
            return Err(CoreError::invalid_input(format!(
                "retain_ratio must be in [0.0, 1.0], got {}",
                self.retain_ratio
            )));
        }
        if !self.umst_weight_scale.is_finite() {
            return Err(CoreError::non_finite("BackboneConfig::umst_weight_scale"));
        }
        if self.umst_weight_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "umst_weight_scale must be >= 0.0, got {}",
                self.umst_weight_scale
            )));
        }
        if !self.non_backbone_weight_scale.is_finite() {
            return Err(CoreError::non_finite(
                "BackboneConfig::non_backbone_weight_scale",
            ));
        }
        if self.non_backbone_weight_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "non_backbone_weight_scale must be >= 0.0, got {}",
                self.non_backbone_weight_scale
            )));
        }
        Ok(())
    }
}

/// Output of [`extract_quadrilateral_backbone`].
///
/// Canonical edge order: `simmelian_weights` and `edge_weights` are indexed identically to the
/// graph's unique undirected simple edges `(u, v)` with `u < v` sorted lexicographically by
/// `(u, v)`. Both `umst_edges` and `backbone_edges` follow the same `(u, v)` (`u < v`) order.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BackboneResult {
    /// Simmelian prefix-Jaccard weights $w(u, v) \in [0, 1]$ aligned with canonical undirected
    /// simple edges `(u, v)` (`u < v`) sorted lexicographically.
    pub simmelian_weights: Vec<f64>,
    /// Canonical undirected edges `(u, v)` (`u < v`) belonging to the Union of Maximum Spanning Trees.
    pub umst_edges: Vec<(u32, u32)>,
    /// Canonical undirected edges `(u, v)` (`u < v`) in $E_{\text{UMST}} \cup E_{\text{threshold}}$.
    pub backbone_edges: Vec<(u32, u32)>,
    /// Assigned layout edge weights aligned with `simmelian_weights`: `umst_weight_scale` for
    /// `E_UMST` edges, `1.0` for non-UMST backbone edges, and `non_backbone_weight_scale` for
    /// remaining edges.
    pub edge_weights: Vec<f64>,
    /// Undirected CSR representation of the active backbone graph (unweighted when all active
    /// edges have unit weight `1.0`, weighted otherwise; omits zero-weight edges).
    pub backbone_csr: CsrGraph<f64>,
}

impl BackboneResult {
    /// Replaces `simmelian_weights` on this [`BackboneResult`].
    #[inline]
    pub fn with_simmelian_weights(mut self, simmelian_weights: Vec<f64>) -> Self {
        self.simmelian_weights = simmelian_weights;
        self
    }

    /// Replaces `umst_edges` on this [`BackboneResult`].
    #[inline]
    pub fn with_umst_edges(mut self, umst_edges: Vec<(u32, u32)>) -> Self {
        self.umst_edges = umst_edges;
        self
    }

    /// Replaces `backbone_edges` on this [`BackboneResult`].
    #[inline]
    pub fn with_backbone_edges(mut self, backbone_edges: Vec<(u32, u32)>) -> Self {
        self.backbone_edges = backbone_edges;
        self
    }

    /// Replaces `edge_weights` on this [`BackboneResult`].
    #[inline]
    pub fn with_edge_weights(mut self, edge_weights: Vec<f64>) -> Self {
        self.edge_weights = edge_weights;
        self
    }

    /// Replaces `backbone_csr` on this [`BackboneResult`].
    #[inline]
    pub fn with_backbone_csr(mut self, backbone_csr: CsrGraph<f64>) -> Self {
        self.backbone_csr = backbone_csr;
        self
    }
}

#[derive(Debug, Clone)]
enum BackboneInnerSolver<S: Scalar> {
    StressMajorization(Box<StressMajorizationSolver<Layout2d<S>, EuclideanPcg>>),
    Sgd(Box<SgdSolver<Layout2d<S>>>),
}

/// Cooperative layout solver running Stress Majorization or SGD on the Quadrilateral Simmelian
/// Backbone extracted from a graph.
#[derive(Debug, Clone)]
pub struct BackboneSolver<S: Scalar = f64> {
    config: BackboneConfig,
    num_nodes: usize,
    backbone: BackboneResult,
    inner: BackboneInnerSolver<S>,
}

fn build_apsp_distance_matrix(backbone_csr: &CsrGraph<f64>) -> CoreResult<Vec<f64>> {
    let n = backbone_csr.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
    let mut dists = vec![f64::INFINITY; total];
    if n == 0 {
        return Ok(dists);
    }

    if !backbone_csr.is_weighted() {
        let mut bfs = BfsWorkspace::new(n);
        for u in 0..n {
            bfs.bfs_from(backbone_csr, u, &mut dists[u * n..(u + 1) * n])?;
        }
        return Ok(dists);
    }

    // Convert edge strengths w > 0 into shortest-path edge lengths len = 1 / w.
    let orig_w = backbone_csr.weights().unwrap_or(&[]);
    let mut inv_w = Vec::with_capacity(orig_w.len());
    for &w in orig_w {
        let len = 1.0 / w;
        if !len.is_finite() || len <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "backbone edge weight {w} produces non-finite inverse edge length"
            )));
        }
        inv_w.push(len);
    }
    let length_graph = CsrGraph::from_csr(
        backbone_csr.indptr(),
        backbone_csr.indices(),
        Some(&inv_w),
        false,
        false,
    )?;
    let mut dijk = DijkstraWorkspace::new(n);
    for u in 0..n {
        dijk.dijkstra_from(&length_graph, u, &mut dists[u * n..(u + 1) * n])?;
    }
    Ok(dists)
}

impl<S: Scalar> BackboneSolver<S> {
    /// Extracts the quadrilateral Simmelian backbone from `graph` and initializes the layout solver.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: BackboneConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let backbone = extract_quadrilateral_backbone(graph, &config)?;
        Self::from_backbone(n, backbone, config)
    }

    /// Extracts the quadrilateral Simmelian backbone from an unweighted [`CsrGraph<W>`] (or
    /// ignoring stored edge weights) without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: BackboneConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let backbone = extract_quadrilateral_backbone_unweighted(graph, &config)?;
        Self::from_backbone(n, backbone, config)
    }

    fn from_backbone(
        n: usize,
        backbone: BackboneResult,
        config: BackboneConfig,
    ) -> CoreResult<Self> {
        let inner = match config.layout_mode {
            BackboneLayoutMode::StressMajorization => {
                let sm_cfg = StressMajorizationConfig::new().with_max_iterations(config.iterations);
                let solver = if !backbone.backbone_csr.is_weighted() {
                    StressMajorizationSolver::<Layout2d<S>, EuclideanPcg>::from_unweighted_graph(
                        &backbone.backbone_csr,
                        sm_cfg,
                    )?
                } else {
                    let dists = build_apsp_distance_matrix(&backbone.backbone_csr)?;
                    StressMajorizationSolver::<Layout2d<S>, EuclideanPcg>::from_distance_matrix(
                        n, &dists, sm_cfg,
                    )?
                };
                BackboneInnerSolver::StressMajorization(Box::new(solver))
            }
            BackboneLayoutMode::Sgd => {
                let sched = SgdScheduleConfig::new(config.iterations);
                let solver = if !backbone.backbone_csr.is_weighted() {
                    SgdSolver::<Layout2d<S>>::full(&backbone.backbone_csr, sched, config.seed)?
                } else {
                    let dists = build_apsp_distance_matrix(&backbone.backbone_csr)?;
                    let mut buf = SgdPairBuffer::<S>::with_capacity(
                        n.saturating_mul(n.saturating_sub(1)) / 2,
                    );
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let d = dists[i * n + j];
                            if d.is_finite() && d > 0.0 {
                                let w = 1.0 / (d * d);
                                let d_s = S::from_f64(d);
                                let w_s = S::from_f64(w);
                                if d_s.is_finite()
                                    && d_s > S::ZERO
                                    && w_s.is_finite()
                                    && w_s > S::ZERO
                                {
                                    buf.push_spring(i as u32, j as u32, d_s, w_s)?;
                                }
                            }
                        }
                    }
                    SgdSolver::<Layout2d<S>>::from_pair_buffer(n, buf, sched, config.seed)?
                };
                BackboneInnerSolver::Sgd(Box::new(solver))
            }
        };

        Ok(Self {
            config,
            num_nodes: n,
            backbone,
            inner,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &BackboneConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the extracted [`BackboneResult`].
    #[inline]
    pub fn backbone_result(&self) -> &BackboneResult {
        &self.backbone
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.status(),
            BackboneInnerSolver::Sgd(s) => s.status(),
        }
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.iteration(),
            BackboneInnerSolver::Sgd(s) => s.iteration(),
        }
    }

    /// Returns the scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.total_iterations(),
            BackboneInnerSolver::Sgd(s) => s.total_iterations(),
        }
    }
}

impl<S: Scalar> IterativeSolver for BackboneSolver<S> {
    type Target = Layout2d<S>;

    #[inline]
    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.validate_target(target),
            BackboneInnerSolver::Sgd(s) => s.validate_target(target),
        }
    }

    #[inline]
    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        match &mut self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.step(target),
            BackboneInnerSolver::Sgd(s) => s.step(target),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status()
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration()
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        self.total_iterations()
    }
}
