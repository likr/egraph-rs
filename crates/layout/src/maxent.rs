//! Gansner et al. (2013) Maxent-Stress graph layout solver.
//!
//! Implements the force-augmented stress majorization model from
//! *"A Maxent-Stress Model for Graph Layout"* (Gansner, Hu & North, IEEE TVCG 2013):
//!
//! $$L_w x^{(t+1)} = L_{w,d}(x^{(t)}) x^{(t)} + \alpha\, b(x^{(t)})$$
//!
//! where $L_w$ is the weighted graph Laplacian over 1-hop edges $S = E$ ($w_{ij} = d_{ij}^{-2}$),
//! $b(x)$ is the generalized $q$-power repulsive force over non-adjacent vertex pairs
//! ($\{i,j\} \notin S$) evaluated via [`ArenaQuadtree2d::compute_maxent_repulsion`] (or exact
//! pairwise summation), and $L_w x = \mathrm{rhs}$ is solved per coordinate via [`AmgPcgSolver`].
//!
//! On disconnected or isolated-vertex graphs, $b(x)$ is zeroed on isolated vertices and centered
//! per connected component so the right-hand side lies strictly in $\mathrm{range}(L_w)$.

use std::collections::VecDeque;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_linalg::amg_pcg::{AmgPcgConfig, AmgPcgSolver, AmgPcgWorkspace};
use egraph_linalg::csr_matrix::CsrMatrix;
use egraph_linalg::mds::{PivotMdsConfig, pivot_mds};
use egraph_spatial::ArenaQuadtree2d;

const MIN_NORM_SQ_SUM: f64 = 1e-12;
const COLLAPSED_SPAN_EPS: f64 = 1e-12;

/// Hyperparameters for [`MaxentStressSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxentStressConfig {
    /// Optional explicit repulsion exponent $q > -2.0$ (`default: None`).
    ///
    /// When `None`, defaults to `0.0` (logarithmic entropy), or automatically switches to `0.8`
    /// when strictly more than `30%` of vertices have 1-hop degree `1` (Gansner et al. 2013 §4.2).
    pub q: Option<f64>,
    /// Initial annealing repulsion weight $\alpha_0 \ge 0.0$ (`default: 1.0`).
    pub alpha_init: f64,
    /// Multiplicative decay factor in `(0.0, 1.0]` applied between annealing stages (`default: 0.3`).
    pub alpha_decay: f64,
    /// Minimum floor for $\alpha$ across annealing stages (`default: 0.008`).
    pub alpha_min: f64,
    /// Number of outer annealing stages (`default: 5`).
    pub stages: usize,
    /// Maximum number of majorization steps per annealing stage (`default: 50`).
    pub steps_per_stage: usize,
    /// Relative position change convergence tolerance per stage (`default: 1e-3`).
    pub tol: f64,
    /// Relative residual tolerance passed to [`AmgPcgSolver`] (`default: 0.1`).
    pub cg_tol: f64,
    /// Maximum PCG iterations per linear solve (`default: 10`).
    pub cg_max_iters: usize,
    /// Barnes-Hut opening angle $\theta \ge 0.0$ for [`ArenaQuadtree2d`] (`default: 1.2`).
    pub theta: f64,
    /// Whether to use Barnes-Hut quadtree approximation (`true`) or exact $O(N^2)$ repulsion (`false`) (`default: true`).
    pub use_barnes_hut: bool,
    /// Softening squared distance $\varepsilon^2 > 0.0$ in repulsive denominator (`default: 1e-6`).
    pub eps_sq: f64,
    /// Deterministic PRNG seed for AMG tie-breaking and collapsed-layout initialization (`default: 42`).
    pub seed: u64,
}

impl Default for MaxentStressConfig {
    fn default() -> Self {
        Self {
            q: None,
            alpha_init: 1.0,
            alpha_decay: 0.3,
            alpha_min: 0.008,
            stages: 5,
            steps_per_stage: 50,
            tol: 1e-3,
            cg_tol: 0.1,
            cg_max_iters: 10,
            theta: 1.2,
            use_barnes_hut: true,
            eps_sq: 1e-6,
            seed: 42,
        }
    }
}

impl MaxentStressConfig {
    /// Validates all Maxent-Stress configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if let Some(qv) = self.q {
            if !qv.is_finite() {
                return Err(CoreError::non_finite("MaxentStressConfig::q"));
            }
            if qv <= -2.0 {
                return Err(CoreError::invalid_input(format!(
                    "MaxentStressConfig::q must be > -2.0, got {qv}"
                )));
            }
        }
        if !self.alpha_init.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_init"));
        }
        if self.alpha_init < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_init must be >= 0.0, got {}",
                self.alpha_init
            )));
        }
        if !self.alpha_decay.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_decay"));
        }
        if self.alpha_decay <= 0.0 || self.alpha_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_decay must be in (0.0, 1.0], got {}",
                self.alpha_decay
            )));
        }
        if !self.alpha_min.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_min"));
        }
        if self.alpha_min < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_min must be >= 0.0, got {}",
                self.alpha_min
            )));
        }
        if !self.tol.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::tol"));
        }
        if self.tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::tol must be >= 0.0, got {}",
                self.tol
            )));
        }
        if !self.cg_tol.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::cg_tol"));
        }
        if self.cg_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::cg_tol must be >= 0.0, got {}",
                self.cg_tol
            )));
        }
        if self.cg_max_iters == 0 {
            return Err(CoreError::invalid_input(
                "MaxentStressConfig::cg_max_iters must be >= 1",
            ));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        if !self.eps_sq.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::eps_sq"));
        }
        if self.eps_sq <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::eps_sq must be > 0.0, got {}",
                self.eps_sq
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct MaxentEdge {
    u: u32,
    v: u32,
    d: f64,
    w: f64,
}

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver.
#[derive(Debug, Clone)]
pub struct MaxentStressSolver<S: Scalar = f64> {
    config: MaxentStressConfig,
    num_nodes: usize,
    effective_q: f64,
    alpha: f64,
    edges: Vec<MaxentEdge>,
    all_unit_edges: bool,
    comp_id: Vec<u32>,
    comp_sizes: Vec<u32>,
    comp_sum_x: Vec<f64>,
    comp_sum_y: Vec<f64>,
    amg: AmgPcgSolver,
    amg_ws: AmgPcgWorkspace,
    quadtree: ArenaQuadtree2d<f64>,
    x_cur: Vec<f64>,
    y_cur: Vec<f64>,
    next_x: Vec<f64>,
    next_y: Vec<f64>,
    stress_rhs_x: Vec<f64>,
    stress_rhs_y: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    rhs_x: Vec<f64>,
    rhs_y: Vec<f64>,
    stage_idx: usize,
    stage_step: usize,
    total_iter_executed: usize,
    total_iterations_budget: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> MaxentStressSolver<S> {
    /// Constructs a [`MaxentStressSolver`] for `graph` using [`MaxentStressConfig`].
    ///
    /// Weighted graphs use positive finite edge weights as target lengths $d_{ij}$ and spring
    /// weights $w_{ij} = d_{ij}^{-2}$. Zero-weight edges and self-loops are skipped; directed
    /// reciprocal edges are canonicalized using the minimum positive edge length.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: MaxentStressConfig) -> CoreResult<Self> {
        config.validate()?;
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

        let amg_cfg = AmgPcgConfig {
            rel_tol: config.cg_tol,
            max_iterations: config.cg_max_iters,
            seed: config.seed,
            ..Default::default()
        };
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

        let num_comps = comp_sizes.len();
        let total_iterations_budget = config.stages.saturating_mul(config.steps_per_stage);
        let status = if total_iterations_budget == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            effective_q,
            alpha: config.alpha_init,
            edges,
            all_unit_edges,
            comp_id,
            comp_sizes,
            comp_sum_x: vec![0.0; num_comps],
            comp_sum_y: vec![0.0; num_comps],
            amg,
            amg_ws,
            quadtree: ArenaQuadtree2d::new(),
            x_cur: vec![0.0; n],
            y_cur: vec![0.0; n],
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            stress_rhs_x: vec![0.0; n],
            stress_rhs_y: vec![0.0; n],
            rep_x: vec![0.0; n],
            rep_y: vec![0.0; n],
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            stage_idx: 0,
            stage_step: 0,
            total_iter_executed: 0,
            total_iterations_budget,
            status,
            _marker: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &MaxentStressConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the active annealing repulsion weight $\alpha$.
    #[inline]
    pub fn current_alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the effective repulsion exponent $q$ used by the solver.
    #[inline]
    pub fn effective_q(&self) -> f64 {
        self.effective_q
    }

    fn initialize_collapsed_layout(&mut self, layout: &mut Layout2d<S>) -> CoreResult<()> {
        let n = self.num_nodes;
        let edge_pairs: Vec<(u32, u32)> = self.edges.iter().map(|e| (e.u, e.v)).collect();
        let edge_dists: Vec<f64> = self.edges.iter().map(|e| e.d).collect();
        let weights_opt = if self.all_unit_edges {
            None
        } else {
            Some(edge_dists.as_slice())
        };
        let init_graph = CsrGraph::from_edges(n, &edge_pairs, weights_opt, false, false)?;
        let pmds_cfg = PivotMdsConfig {
            seed: self.config.seed,
            ..Default::default()
        };

        if let Ok(pmds_layout) = pivot_mds::<f64, S>(&init_graph, &pmds_cfg)
            && !is_layout_collapsed(&pmds_layout)
        {
            let (dst_x, dst_y) = layout.xy_mut();
            dst_x.copy_from_slice(pmds_layout.x());
            dst_y.copy_from_slice(pmds_layout.y());
            return Ok(());
        }

        // Deterministic fallback if PivotMDS produces a degenerate/collapsed configuration.
        let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
        let (dst_x, dst_y) = layout.xy_mut();
        for i in 0..n {
            dst_x[i] = S::from_f64(rng.next_f64() - 0.5);
            dst_y[i] = S::from_f64(rng.next_f64() - 0.5);
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S> for MaxentStressSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "MaxentStressSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.total_iter_executed == 0 && is_layout_collapsed(layout) {
            self.initialize_collapsed_layout(layout)?;
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.x_cur[i] = xs[i].to_f64();
                self.y_cur[i] = ys[i].to_f64();
            }
        }

        // 1. Majorizing stress RHS: L_{w,d}(x^{(t)}) x^{(t)}.
        self.stress_rhs_x.fill(0.0);
        self.stress_rhs_y.fill(0.0);
        for edge in &self.edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let dx = self.x_cur[u] - self.x_cur[v];
            let dy = self.y_cur[u] - self.y_cur[v];
            let norm = dx.hypot(dy);
            if norm > 1e-12 && norm.is_finite() {
                let f = edge.w * edge.d / norm;
                let fx = f * dx;
                let fy = f * dy;
                self.stress_rhs_x[u] += fx;
                self.stress_rhs_y[u] += fy;
                self.stress_rhs_x[v] -= fx;
                self.stress_rhs_y[v] -= fy;
            }
        }

        // 2. Repulsive force b(x) over non-adjacent vertex pairs {i, j} \notin S.
        let eps_sq = self.config.eps_sq;
        let q = self.effective_q;
        let is_q_zero = q == 0.0;
        let exp = 0.5 * (q + 2.0);

        if self.alpha > 0.0 {
            if self.config.use_barnes_hut {
                self.quadtree.build_xy(&self.x_cur, &self.y_cur)?;
                self.quadtree.compute_maxent_repulsion(
                    self.config.theta,
                    q,
                    eps_sq,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            } else {
                self.rep_x.fill(0.0);
                self.rep_y.fill(0.0);
                for i in 0..n {
                    let xi = self.x_cur[i];
                    let yi = self.y_cur[i];
                    for j in (i + 1)..n {
                        let dx = xi - self.x_cur[j];
                        let dy = yi - self.y_cur[j];
                        let d2 = dx * dx + dy * dy + eps_sq;
                        let mult = if is_q_zero {
                            1.0 / d2
                        } else {
                            1.0 / d2.powf(exp)
                        };
                        let fx = mult * dx;
                        let fy = mult * dy;
                        self.rep_x[i] += fx;
                        self.rep_y[i] += fy;
                        self.rep_x[j] -= fx;
                        self.rep_y[j] -= fy;
                    }
                }
            }

            // Subtract 1-hop S-edge pairs so b(x) sums strictly over {i, j} \notin S.
            for edge in &self.edges {
                let u = edge.u as usize;
                let v = edge.v as usize;
                let dx = self.x_cur[u] - self.x_cur[v];
                let dy = self.y_cur[u] - self.y_cur[v];
                let d2 = dx * dx + dy * dy + eps_sq;
                let mult = if is_q_zero {
                    1.0 / d2
                } else {
                    1.0 / d2.powf(exp)
                };
                let fx = mult * dx;
                let fy = mult * dy;
                self.rep_x[u] -= fx;
                self.rep_y[u] -= fy;
                self.rep_x[v] += fx;
                self.rep_y[v] += fy;
            }

            // Zero isolated vertices and center b(x) per connected component so
            // sum_{i in C_k} b_i == 0 lies strictly in range(L_w).
            self.comp_sum_x.fill(0.0);
            self.comp_sum_y.fill(0.0);
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                if self.comp_sizes[cid] <= 1 {
                    self.rep_x[i] = 0.0;
                    self.rep_y[i] = 0.0;
                } else {
                    self.comp_sum_x[cid] += self.rep_x[i];
                    self.comp_sum_y[cid] += self.rep_y[i];
                }
            }
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                let sz = self.comp_sizes[cid];
                if sz > 1 {
                    let inv_sz = 1.0 / (sz as f64);
                    self.rep_x[i] -= self.comp_sum_x[cid] * inv_sz;
                    self.rep_y[i] -= self.comp_sum_y[cid] * inv_sz;
                }
            }

            let norm_stress = norm2_pair(&self.stress_rhs_x, &self.stress_rhs_y);
            let norm_rep = norm2_pair(&self.rep_x, &self.rep_y);
            if norm_rep > 1e-30 {
                let scale = if norm_stress > 0.0 {
                    norm_stress / norm_rep
                } else {
                    1.0
                };
                let coeff = self.alpha * scale;
                for i in 0..n {
                    self.rhs_x[i] = self.stress_rhs_x[i] + coeff * self.rep_x[i];
                    self.rhs_y[i] = self.stress_rhs_y[i] + coeff * self.rep_y[i];
                }
            } else {
                self.rhs_x.copy_from_slice(&self.stress_rhs_x);
                self.rhs_y.copy_from_slice(&self.stress_rhs_y);
            }
        } else {
            self.rhs_x.copy_from_slice(&self.stress_rhs_x);
            self.rhs_y.copy_from_slice(&self.stress_rhs_y);
        }

        // 3. Solve L_w x^{(t+1)} = rhs warm-started from x^{(t)}.
        self.next_x.copy_from_slice(&self.x_cur);
        self.next_y.copy_from_slice(&self.y_cur);
        self.amg
            .solve(&self.rhs_x, &mut self.next_x, &mut self.amg_ws)?;
        self.amg
            .solve(&self.rhs_y, &mut self.next_y, &mut self.amg_ws)?;

        for i in 0..n {
            if self.comp_sizes[self.comp_id[i] as usize] <= 1
                || !self.next_x[i].is_finite()
                || !self.next_y[i].is_finite()
            {
                self.next_x[i] = self.x_cur[i];
                self.next_y[i] = self.y_cur[i];
            }
        }

        let mut diff_sq = 0.0_f64;
        let mut cur_sq = 0.0_f64;
        for i in 0..n {
            let dx = self.next_x[i] - self.x_cur[i];
            let dy = self.next_y[i] - self.y_cur[i];
            diff_sq += dx * dx + dy * dy;
            cur_sq += self.x_cur[i] * self.x_cur[i] + self.y_cur[i] * self.y_cur[i];
        }
        let rel_change = diff_sq.sqrt() / cur_sq.sqrt().max(MIN_NORM_SQ_SUM);

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.next_x[i]);
                let ny = S::from_f64(self.next_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        }

        self.stage_step += 1;
        self.total_iter_executed += 1;

        let stage_converged = rel_change < self.config.tol;
        if stage_converged || self.stage_step >= self.config.steps_per_stage {
            self.stage_idx += 1;
            self.stage_step = 0;
            if self.stage_idx >= self.config.stages {
                self.status = if stage_converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Completed
                };
            } else {
                self.alpha = (self.alpha * self.config.alpha_decay).max(self.config.alpha_min);
                self.status = SolverStatus::Running;
            }
        } else {
            self.status = SolverStatus::Running;
        }

        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations_budget
    }
}

#[inline]
fn norm2_pair(xs: &[f64], ys: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        sum += x * x + y * y;
    }
    sum.sqrt()
}

fn is_layout_collapsed<S: Scalar>(layout: &Layout2d<S>) -> bool {
    let n = layout.len();
    if n < 2 {
        return false;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = xs[0].to_f64();
    let mut max_x = min_x;
    let mut min_y = ys[0].to_f64();
    let mut max_y = min_y;
    for i in 1..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x) < COLLAPSED_SPAN_EPS && (max_y - min_y) < COLLAPSED_SPAN_EPS
}

fn extract_canonical_positive_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<Vec<(u32, u32, f64)>> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Ok(Vec::new());
    }
    let weights_opt = graph.weights();
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    if w > 0.0 {
                        edges.push((u32_u, v, w));
                    }
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return Ok(edges);
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            if w <= 0.0 {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            raw.push((a, b, w));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 = last.2.min(w);
        } else {
            out.push((a, b, w));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn path_graph(n: usize) -> CsrGraph<f64> {
        let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1))
            .map(|i| (i as u32, (i + 1) as u32))
            .collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn star_graph(leaves: usize) -> CsrGraph<f64> {
        let n = leaves + 1;
        let edges: Vec<(u32, u32)> = (1..n).map(|i| (0u32, i as u32)).collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_maxent_config_validation() {
        assert!(MaxentStressConfig::default().validate().is_ok());
        assert!(matches!(
            MaxentStressConfig {
                q: Some(-2.0),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            MaxentStressConfig {
                q: Some(f64::NAN),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            MaxentStressConfig {
                alpha_decay: 0.0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            MaxentStressConfig {
                alpha_decay: 1.5,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            MaxentStressConfig {
                cg_max_iters: 0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            MaxentStressConfig {
                eps_sq: 0.0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_maxent_adaptive_q_and_annealing() {
        // Grid 4x4 has 4 degree-2 corners and 0 degree-1 vertices -> q = 0.0.
        let grid = grid_graph(4, 4);
        let solver_grid =
            MaxentStressSolver::<f64>::new(&grid, MaxentStressConfig::default()).unwrap();
        assert_eq!(solver_grid.effective_q(), 0.0);

        // Star graph has > 30% degree-1 leaves -> q = 0.8.
        let star = star_graph(6);
        let solver_star =
            MaxentStressSolver::<f64>::new(&star, MaxentStressConfig::default()).unwrap();
        assert_eq!(solver_star.effective_q(), 0.8);

        // Explicit q overrides automatic selection.
        let solver_override = MaxentStressSolver::<f64>::new(
            &star,
            MaxentStressConfig {
                q: Some(-0.5),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(solver_override.effective_q(), -0.5);

        // Annealing alpha decays cleanly from 1.0 -> 0.3 -> 0.09 -> 0.027 -> 0.0081.
        let mut s = MaxentStressSolver::<f64>::new(
            &grid,
            MaxentStressConfig {
                stages: 5,
                steps_per_stage: 2,
                tol: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut layout = Layout2d::zeros(grid.num_nodes());
        assert_eq!(s.current_alpha(), 1.0);
        s.run_steps(&mut layout, 2).unwrap();
        assert!((s.current_alpha() - 0.3).abs() < 1e-12);
        s.run(&mut layout).unwrap();
        assert!(s.current_alpha() >= 0.008);
        assert!(layout.is_all_finite());
    }

    #[test]
    fn test_maxent_step_run_bit_exact_f64_and_f32() {
        let g = grid_graph(3, 4);

        // f64 bit-exact step() == run(), both Barnes-Hut and exact O(N^2).
        for &use_bh in &[true, false] {
            let cfg = MaxentStressConfig {
                use_barnes_hut: use_bh,
                stages: 3,
                steps_per_stage: 10,
                ..Default::default()
            };
            let mut s1 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
            let mut s2 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
            let mut l1 = Layout2d::zeros(g.num_nodes());
            let mut l2 = Layout2d::zeros(g.num_nodes());

            let st1 = s1.run(&mut l1).unwrap();
            let mut st2 = SolverStatus::Initialized;
            while !st2.is_finished() {
                st2 = s2.step(&mut l2).unwrap();
            }
            assert_eq!(st1, st2);
            assert_eq!(s1.iteration(), s2.iteration());
            assert_eq!(
                l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }

        // f32 bit-exact step() == run().
        let cfg32 = MaxentStressConfig {
            stages: 3,
            steps_per_stage: 8,
            ..Default::default()
        };
        let mut s1 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
        let mut s2 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
        let mut l1 = Layout2d::<f32>::zeros(g.num_nodes());
        let mut l2 = Layout2d::<f32>::zeros(g.num_nodes());
        let st1 = s1.run(&mut l1).unwrap();
        let mut st2 = SolverStatus::Initialized;
        while !st2.is_finished() {
            st2 = s2.step(&mut l2).unwrap();
        }
        assert_eq!(st1, st2);
        assert!(l1.is_all_finite());
        assert_eq!(
            l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_maxent_edge_cases_disconnected_isolated_weighted_directed() {
        // N = 0 and N = 1.
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = MaxentStressSolver::<f64>::new(&g0, MaxentStressConfig::default()).unwrap();
        let mut l0 = Layout2d::zeros(0);
        assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = MaxentStressSolver::<f64>::new(&g1, MaxentStressConfig::default()).unwrap();
        let mut l1 = Layout2d::zeros(1);
        assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);

        // Edgeless graph (N = 4).
        let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let mut s_edgeless =
            MaxentStressSolver::<f64>::new(&g_edgeless, MaxentStressConfig::default()).unwrap();
        let mut l_edgeless = Layout2d::zeros(4);
        s_edgeless.run(&mut l_edgeless).unwrap();
        assert!(l_edgeless.is_all_finite());

        // Disconnected graph: two 3-paths + 2 isolated vertices (N = 8).
        let g_disc =
            CsrGraph::<f64>::from_edges(8, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
                .unwrap();
        let mut s_disc =
            MaxentStressSolver::<f64>::new(&g_disc, MaxentStressConfig::default()).unwrap();
        let mut l_disc = Layout2d::from_xy(
            vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, -3.0, -4.0],
            vec![0.0, 0.5, 0.0, 2.0, 2.5, 2.0, 8.0, -8.0],
        )
        .unwrap();
        let st_disc = s_disc.run(&mut l_disc).unwrap();
        assert!(st_disc.is_finished());
        assert!(l_disc.is_all_finite());
        // Isolated vertices (6, 7) must remain untouched.
        assert_eq!(l_disc.x()[6], -3.0);
        assert_eq!(l_disc.y()[6], 8.0);
        assert_eq!(l_disc.x()[7], -4.0);
        assert_eq!(l_disc.y()[7], -8.0);

        // Weighted + directed graph with reciprocal edges.
        let g_dir = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
            Some(&[2.0, 1.5, 1.0, 3.0, 2.0]),
            true,
            false,
        )
        .unwrap();
        let mut s_dir =
            MaxentStressSolver::<f64>::new(&g_dir, MaxentStressConfig::default()).unwrap();
        let mut l_dir = Layout2d::zeros(4);
        s_dir.run(&mut l_dir).unwrap();
        assert!(l_dir.is_all_finite());

        // Caller-provided non-collapsed layout is refined rather than overwritten by PivotMDS.
        let g_path = path_graph(5);
        let mut s_custom = MaxentStressSolver::<f64>::new(
            &g_path,
            MaxentStressConfig {
                stages: 1,
                steps_per_stage: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l_custom = Layout2d::from_xy(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(s_custom.total_iterations(), 0);
        s_custom.run(&mut l_custom).unwrap();
        assert_eq!(l_custom.x(), &[10.0, 20.0, 30.0, 40.0, 50.0]);
    }
}
