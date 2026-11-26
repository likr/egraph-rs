use std::collections::VecDeque;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, LayoutNd, Scalar, SolverStatus,
};
use egraph_linalg::csr_matrix::CsrMatrix;
use egraph_linalg_amg::amg_pcg::{AmgPcgConfig, AmgPcgSolver, AmgPcgWorkspace};
use egraph_linalg_mds::mds::{HdeConfig, PivotMdsConfig, hde_nd, pivot_mds};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use super::config::MaxentStressConfig;
use super::graph_prep::{
    MaxentEdge, extract_canonical_positive_edges, is_layout_collapsed, is_layout_collapsed_nd,
};

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver.
#[derive(Debug, Clone)]
pub struct MaxentStressSolver<S: Scalar = f64> {
    pub(super) config: MaxentStressConfig,
    pub(super) num_nodes: usize,
    pub(super) effective_q: f64,
    pub(super) alpha: f64,
    pub(super) edges: Vec<MaxentEdge>,
    all_unit_edges: bool,
    pub(super) comp_id: Vec<u32>,
    pub(super) comp_sizes: Vec<u32>,
    pub(super) comp_sum_x: Vec<f64>,
    pub(super) comp_sum_y: Vec<f64>,
    pub(super) amg: AmgPcgSolver,
    pub(super) amg_ws: AmgPcgWorkspace,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) octree: ArenaOctree3d<f64>,
    pub(super) x_cur: Vec<f64>,
    pub(super) y_cur: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) stress_rhs_x: Vec<f64>,
    pub(super) stress_rhs_y: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) rhs_x: Vec<f64>,
    pub(super) rhs_y: Vec<f64>,
    pub(super) nd_dims: Option<usize>,
    pub(super) nd_cur: Vec<f64>,
    pub(super) nd_next: Vec<f64>,
    pub(super) nd_stress_rhs: Vec<f64>,
    pub(super) nd_rep: Vec<f64>,
    pub(super) nd_comp_sum: Vec<f64>,
    pub(super) nd_rhs_axis: Vec<f64>,
    pub(super) nd_next_axis: Vec<f64>,
    pub(super) stage_idx: usize,
    pub(super) stage_step: usize,
    pub(super) total_iter_executed: usize,
    pub(super) total_iterations_budget: usize,
    pub(super) status: SolverStatus,
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
            octree: ArenaOctree3d::new(),
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
            nd_dims: None,
            nd_cur: Vec::new(),
            nd_next: Vec::new(),
            nd_stress_rhs: Vec::new(),
            nd_rep: Vec::new(),
            nd_comp_sum: Vec::new(),
            nd_rhs_axis: Vec::new(),
            nd_next_axis: Vec::new(),
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

    /// Returns the current lifecycle status of the solver.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.total_iter_executed
    }

    /// Returns the configured maximum iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.total_iterations_budget
    }

    pub(super) fn initialize_collapsed_layout(
        &mut self,
        layout: &mut Layout2d<S>,
    ) -> CoreResult<()> {
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

    pub(super) fn initialize_collapsed_layout_nd(
        &mut self,
        layout: &mut LayoutNd<S>,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        let d = layout.dims();
        let edge_pairs: Vec<(u32, u32)> = self.edges.iter().map(|e| (e.u, e.v)).collect();
        let edge_dists: Vec<f64> = self.edges.iter().map(|e| e.d).collect();
        let weights_opt = if self.all_unit_edges {
            None
        } else {
            Some(edge_dists.as_slice())
        };
        let init_graph = CsrGraph::from_edges(n, &edge_pairs, weights_opt, false, false)?;
        let hde_cfg = HdeConfig {
            output_dims: d,
            seed: self.config.seed,
            ..Default::default()
        };

        if let Ok(hde_layout) = hde_nd::<f64, S>(&init_graph, &hde_cfg)
            && !is_layout_collapsed_nd(&hde_layout)
        {
            layout.coords_mut().copy_from_slice(hde_layout.coords());
            return Ok(());
        }

        // Deterministic fallback if HDE produces a degenerate/collapsed configuration.
        let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
        for v in layout.coords_mut().iter_mut() {
            *v = S::from_f64(rng.next_f64() - 0.5);
        }
        Ok(())
    }
}
