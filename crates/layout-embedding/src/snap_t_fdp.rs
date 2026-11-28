use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, RiemannianLayout,
    Scalar, SolverStatus,
};

use crate::config_pacmap_snap::SnapTFdpConfig;
use crate::sampling_utils::{
    BLOCK_SHUFFLE_THRESHOLD, BLOCK_SIZE, MAX_PACKED_NODES, init_pair_blocking,
    validate_graph_weights, validate_target_riemannian,
};
use crate::snap_t_fdp_edges::extract_snap_edges;

/// Compact-memory edge-incident negative sampling `SNAP-tFDP` solver.
///
/// Stores only canonical undirected 1-hop edges (`edges_u`, `edges_v`, `edge_order`, and
/// optional `f32` weights / degree scales) and draws `negatives_per_edge` uniform random
/// negative vertices on-the-fly per edge.
#[derive(Debug, Clone)]
pub struct SnapTFdpSolver<L: RiemannianLayout = Layout2d<f64>> {
    config: SnapTFdpConfig,
    num_nodes: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edge_weights: Option<Vec<f32>>,
    deg_scales: Option<Vec<f32>>,
    edge_order: Vec<u32>,
    block_order: Vec<u32>,
    log_uv: Vec<f64>,
    log_vu: Vec<f64>,
    delta_u: Vec<f64>,
    delta_v: Vec<f64>,
    iter: usize,
    status: SolverStatus,
    rng: DeterministicRng,
    locked_tangent_dim: Option<usize>,
    _target: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout> SnapTFdpSolver<L> {
    /// Constructs a compact-memory `SnapTFdpSolver` from `graph` and `config`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: SnapTFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_graph_weights(graph, "SnapTFdpSolver::new")?;

        let n = graph.num_nodes();
        if n > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }

        let (edges_u, edges_v, edge_weights) = extract_snap_edges(graph)?;
        let m = edges_u.len();

        let mut edge_order: Vec<u32> = (0..m as u32).collect();
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let block_order = init_pair_blocking(&mut edge_order, &mut rng);

        // Allocate optional 4N-byte degree attenuation factors only when degree_beta > 0.
        let deg_scales = if config.degree_beta > 0.0 && n > 1 && !edges_u.is_empty() {
            let mut degs = vec![0u32; n];
            for (&u, &v) in edges_u.iter().zip(edges_v.iter()) {
                degs[u as usize] += 1;
                degs[v as usize] += 1;
            }
            let beta = config.degree_beta;
            Some(
                degs.into_iter()
                    .map(|d| (d.max(1) as f64).powf(-beta) as f32)
                    .collect(),
            )
        } else {
            None
        };

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edge_weights,
            deg_scales,
            edge_order,
            block_order,
            log_uv: Vec::new(),
            log_vu: Vec::new(),
            delta_u: Vec::new(),
            delta_v: Vec::new(),
            iter: 0,
            status,
            rng,
            locked_tangent_dim: None,
            _target: PhantomData,
        })
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of canonical undirected 1-hop edges stored.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.edges_u.len()
    }

    /// Returns a reference to the active [`SnapTFdpConfig`].
    #[inline]
    pub fn config(&self) -> &SnapTFdpConfig {
        &self.config
    }

    /// Returns the canonical edge source endpoint slice.
    #[inline]
    pub fn edges_u(&self) -> &[u32] {
        &self.edges_u
    }

    /// Returns the canonical edge target endpoint slice.
    #[inline]
    pub fn edges_v(&self) -> &[u32] {
        &self.edges_v
    }

    /// Returns the edge permutation buffer.
    #[inline]
    pub fn edge_order(&self) -> &[u32] {
        &self.edge_order
    }

    /// Returns the block-shuffle order slice (non-empty when edge count exceeds threshold).
    #[inline]
    pub fn block_order(&self) -> &[u32] {
        &self.block_order
    }

    /// Returns the optional `f32` edge weight slice (present only for weighted input graphs).
    #[inline]
    pub fn edge_weights(&self) -> Option<&[f32]> {
        self.edge_weights.as_deref()
    }

    /// Returns the optional `f32` per-vertex degree attenuation factors.
    #[inline]
    pub fn deg_scales(&self) -> Option<&[f32]> {
        self.deg_scales.as_deref()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn edge_buffer_bytes(&self) -> usize {
        (self.edges_u.capacity() + self.edges_v.capacity() + self.edge_order.capacity())
            * std::mem::size_of::<u32>()
    }

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` if finished).
    #[inline]
    pub fn current_eta(&self) -> f64 {
        let total = self.config.iterations;
        if total == 0 || self.iter >= total {
            return 0.0;
        }
        self.config.learning_rate * (1.0 - (self.iter as f64) / (total as f64))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_edge_slice(
        layout: &mut L,
        dim: usize,
        order_slice: &[u32],
        edges_u: &[u32],
        edges_v: &[u32],
        edge_weights: Option<&[f32]>,
        deg_scales: Option<&[f32]>,
        bounded_attraction: bool,
        num_nodes: usize,
        negatives_per_edge: usize,
        base_attr: f64,
        base_rep: f64,
        rng: &mut DeterministicRng,
        log_uv: &mut [f64],
        log_vu: &mut [f64],
        delta_u: &mut [f64],
        delta_v: &mut [f64],
    ) -> CoreResult<()> {
        const MAX_ATTR: f64 = 0.49;
        const MAX_REP: f64 = 0.5;
        for (pos, &edge_idx) in order_slice.iter().enumerate() {
            let idx = edge_idx as usize;
            let u = edges_u[idx] as usize;
            let v = edges_v[idx] as usize;
            let w_e = edge_weights.map_or(1.0, |ws| ws[idx] as f64);
            let deg_factor = deg_scales.map_or(1.0, |ds| (ds[u] as f64) * (ds[v] as f64));

            if w_e > 0.0 && base_attr > 0.0 && deg_factor > 0.0 {
                layout.log_map_into(u, v, log_uv)?;
                let mut d2 = 0.0_f64;
                for &val in &log_uv[..dim] {
                    d2 += val * val;
                }
                let inv_1_d2 = if d2.is_finite() {
                    1.0 / (1.0 + d2)
                } else {
                    0.0
                };
                let kernel = if bounded_attraction {
                    8.0 * inv_1_d2
                } else {
                    1.0 + 8.0 * inv_1_d2
                };
                let attr = base_attr * w_e * deg_factor * kernel;
                if attr.is_finite() && attr > 0.0 {
                    let coeff = attr.min(MAX_ATTR);
                    layout.log_map_into(v, u, log_vu)?;
                    let mut ok = true;
                    for a in 0..dim {
                        let su = log_uv[a] * coeff;
                        let sv = log_vu[a] * coeff;
                        if !su.is_finite() || !sv.is_finite() {
                            ok = false;
                            break;
                        }
                        delta_u[a] = su;
                        delta_v[a] = sv;
                    }
                    if ok {
                        layout.exp_map(u, &delta_u[..dim])?;
                        layout.exp_map(v, &delta_v[..dim])?;
                    }
                }
            }

            if base_rep > 0.0 {
                for neg_idx in 0..negatives_per_edge {
                    let anchor = if ((pos + neg_idx) & 1) == 0 { u } else { v };
                    let w = rng.gen_range_usize(num_nodes);
                    if w == anchor {
                        continue;
                    }
                    layout.log_map_into(anchor, w, log_uv)?;
                    let mut rd2 = 0.0_f64;
                    for &val in &log_uv[..dim] {
                        rd2 += val * val;
                    }
                    if rd2.is_finite() {
                        let denom = (1.0 + rd2) * (1.0 + rd2);
                        if denom.is_finite() && denom > 0.0 {
                            let rep = (base_rep / denom).min(MAX_REP);
                            if rep.is_finite() && rep > 0.0 {
                                layout.log_map_into(w, anchor, log_vu)?;
                                let mut ok = true;
                                for a in 0..dim {
                                    let sa = -log_uv[a] * rep;
                                    let sw = -log_vu[a] * rep;
                                    if !sa.is_finite() || !sw.is_finite() {
                                        ok = false;
                                        break;
                                    }
                                    delta_u[a] = sa;
                                    delta_v[a] = sw;
                                }
                                if ok {
                                    layout.exp_map(anchor, &delta_u[..dim])?;
                                    layout.exp_map(w, &delta_v[..dim])?;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn step_inner(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        let total = self.config.iterations;
        if self.num_nodes <= 1 || self.edges_u.is_empty() || total == 0 || self.iter >= total {
            return Ok(());
        }
        let eta = self.current_eta();
        if !eta.is_finite() || eta <= 0.0 {
            return Ok(());
        }
        if self.log_uv.len() != dim {
            self.log_uv.resize(dim, 0.0);
            self.log_vu.resize(dim, 0.0);
            self.delta_u.resize(dim, 0.0);
            self.delta_v.resize(dim, 0.0);
        }

        let progress = if total <= 1 {
            0.0
        } else {
            (self.iter as f64) / ((total - 1) as f64)
        };
        let rho = self.config.rho_start
            + (self.config.rho_end - self.config.rho_start) * progress.powf(self.config.rho_power);
        let base_attr = eta * 0.1 * self.config.alpha;
        let base_rep = eta * rho * self.config.repulsion_scale;
        let len = self.edge_order.len();

        if len <= BLOCK_SHUFFLE_THRESHOLD {
            self.rng.shuffle(&mut self.edge_order);
            Self::apply_edge_slice(
                layout,
                dim,
                &self.edge_order,
                &self.edges_u,
                &self.edges_v,
                self.edge_weights.as_deref(),
                self.deg_scales.as_deref(),
                self.config.bounded_attraction,
                self.num_nodes,
                self.config.negatives_per_edge,
                base_attr,
                base_rep,
                &mut self.rng,
                &mut self.log_uv,
                &mut self.log_vu,
                &mut self.delta_u,
                &mut self.delta_v,
            )?;
        } else {
            self.rng.shuffle(&mut self.block_order);
            for &b in &self.block_order {
                let start = (b as usize) * BLOCK_SIZE;
                let end = (start + BLOCK_SIZE).min(len);
                let chunk = &mut self.edge_order[start..end];
                self.rng.shuffle(chunk);
                Self::apply_edge_slice(
                    layout,
                    dim,
                    chunk,
                    &self.edges_u,
                    &self.edges_v,
                    self.edge_weights.as_deref(),
                    self.deg_scales.as_deref(),
                    self.config.bounded_attraction,
                    self.num_nodes,
                    self.config.negatives_per_edge,
                    base_attr,
                    base_rep,
                    &mut self.rng,
                    &mut self.log_uv,
                    &mut self.log_vu,
                    &mut self.delta_u,
                    &mut self.delta_v,
                )?;
            }
        }
        Ok(())
    }
}

impl<L: RiemannianLayout> IterativeSolver for SnapTFdpSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "SnapTFdpSolver::validate_target",
        )?;
        Ok(())
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        let dim = validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "SnapTFdpSolver::step",
        )?;
        if self.status.is_finished() {
            return Ok(self.status);
        }
        self.locked_tangent_dim.get_or_insert(dim);
        if let Err(err) = self.step_inner(target, dim) {
            self.status = SolverStatus::Failed;
            return Err(err);
        }
        self.iter += 1;
        self.status = if self.iter >= self.config.iterations {
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
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.config.iterations)
    }
}
