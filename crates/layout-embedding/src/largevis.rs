use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, RiemannianLayout,
    Scalar, SolverStatus,
};

use crate::config_negative::LargeVisConfig;
use crate::sampling_utils::{
    AliasTable, MAX_PACKED_NODES, extract_canonical_undirected_edges,
    scatter_collapsed_riemannian_if_needed, validate_graph_weights, validate_target_riemannian,
};

/// Single-threaded deterministic `LargeVis` graph embedding solver (`Tang et al., 2016`).
#[derive(Debug, Clone)]
pub struct LargeVisSolver<L: RiemannianLayout = Layout2d<f64>> {
    config: LargeVisConfig,
    num_nodes: usize,
    num_springs: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edge_alias: Option<AliasTable>,
    neg_alias: Option<AliasTable>,
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

impl<L: RiemannianLayout> LargeVisSolver<L> {
    /// Constructs a `LargeVisSolver` using Vose alias tables over positive edges and
    /// $d_j^{\alpha}$-weighted vertices.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: LargeVisConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_graph_weights(graph, "LargeVisSolver::new")?;

        let n = graph.num_nodes();
        if n > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let mut edges_u = Vec::new();
        let mut edges_v = Vec::new();
        let mut edge_weights = Vec::new();
        let mut deg_w = vec![0.0_f64; n];

        if n > 1 && graph.nnz() > 0 {
            for (u, v, w) in extract_canonical_undirected_edges(graph) {
                if w.is_finite() && w > 0.0 {
                    edges_u.push(u);
                    edges_v.push(v);
                    edge_weights.push(w);
                    deg_w[u as usize] += w;
                    deg_w[v as usize] += w;
                }
            }
        }

        let edge_alias = AliasTable::from_weights(&edge_weights);
        let neg_weights: Vec<f64> = deg_w
            .into_iter()
            .map(|dw| {
                if dw > 0.0 {
                    dw.powf(config.degree_exponent)
                } else {
                    0.0
                }
            })
            .collect();
        let neg_alias = AliasTable::from_weights(&neg_weights);
        let m = edges_u.len();

        Ok(Self {
            rng: DeterministicRng::from_seed_u64(config.seed),
            config,
            num_nodes: n,
            num_springs: m,
            edges_u,
            edges_v,
            edge_alias,
            neg_alias,
            log_uv: Vec::new(),
            log_vu: Vec::new(),
            delta_u: Vec::new(),
            delta_v: Vec::new(),
            iter: 0,
            status,
            locked_tangent_dim: None,
            _target: PhantomData,
        })
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of positive undirected edges indexed by the alias table.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.num_springs
    }

    /// Returns a reference to the active [`LargeVisConfig`].
    #[inline]
    pub fn config(&self) -> &LargeVisConfig {
        &self.config
    }

    /// Returns the learning rate for the upcoming iteration (`0.0` if finished).
    #[inline]
    pub fn current_lr(&self) -> f64 {
        let total = self.config.iterations;
        if total == 0 || self.iter >= total {
            return 0.0;
        }
        (self.config.initial_lr * (1.0 - (self.iter as f64) / (total as f64)))
            .max(self.config.min_lr)
    }

    fn step_inner(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        let total = self.config.iterations;
        if self.num_nodes <= 1 || total == 0 || self.iter >= total {
            return Ok(());
        }
        if self.log_uv.len() != dim {
            self.log_uv.resize(dim, 0.0);
            self.log_vu.resize(dim, 0.0);
            self.delta_u.resize(dim, 0.0);
            self.delta_v.resize(dim, 0.0);
        }
        if self.iter == 0 {
            scatter_collapsed_riemannian_if_needed(layout, 0.02, &mut self.rng, &mut self.delta_u)?;
        }
        let Some(edge_alias) = &self.edge_alias else {
            return Ok(());
        };

        let lr = self.current_lr();
        if !lr.is_finite() || lr <= 0.0 {
            return Ok(());
        }

        let m = self.edges_u.len();
        let gamma = self.config.gamma;
        let neg_samples = self.config.negative_samples;

        for _ in 0..m {
            let e_idx = edge_alias.sample(&mut self.rng);
            let u_raw = self.edges_u[e_idx] as usize;
            let v_raw = self.edges_v[e_idx] as usize;
            let (u, v) = if (self.rng.next_u32() & 1) == 0 {
                (u_raw, v_raw)
            } else {
                (v_raw, u_raw)
            };

            layout.log_map_into(u, v, &mut self.log_uv)?;
            layout.log_map_into(v, u, &mut self.log_vu)?;
            let mut d2 = 0.0_f64;
            for &val in &self.log_uv[..dim] {
                d2 += val * val;
            }
            if d2.is_finite() {
                let pos_coeff = (2.0 * lr) / (1.0 + d2);
                let mut ok = true;
                for a in 0..dim {
                    let su = (pos_coeff * self.log_uv[a]).clamp(-4.0, 4.0);
                    let sv = (pos_coeff * self.log_vu[a]).clamp(-4.0, 4.0);
                    if !su.is_finite() || !sv.is_finite() {
                        ok = false;
                        break;
                    }
                    self.delta_u[a] = su;
                    self.delta_v[a] = sv;
                }
                if ok {
                    layout.exp_map(u, &self.delta_u[..dim])?;
                    layout.exp_map(v, &self.delta_v[..dim])?;
                }
            }

            if let Some(neg_alias) = &self.neg_alias
                && gamma > 0.0
            {
                for _ in 0..neg_samples {
                    let k = neg_alias.sample(&mut self.rng);
                    if k == u {
                        continue;
                    }
                    layout.log_map_into(u, k, &mut self.log_uv)?;
                    let mut rd2 = 0.0_f64;
                    for &val in &self.log_uv[..dim] {
                        rd2 += val * val;
                    }
                    if rd2.is_finite() && rd2 > 0.0 {
                        let grad = (2.0 * gamma) / ((rd2 + 1e-6) * (1.0 + rd2));
                        let mut ok = true;
                        for a in 0..dim {
                            let su = (-lr * grad * self.log_uv[a]).clamp(-4.0, 4.0);
                            if !su.is_finite() {
                                ok = false;
                                break;
                            }
                            self.delta_u[a] = su;
                        }
                        if ok {
                            layout.exp_map(u, &self.delta_u[..dim])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<L: RiemannianLayout> IterativeSolver for LargeVisSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "LargeVisSolver::validate_target",
        )?;
        Ok(())
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        let dim = validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "LargeVisSolver::step",
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
