use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, RiemannianLayout,
    Scalar, SolverStatus,
};
use egraph_neighborhood::graph_row_normalized_probabilities;

use crate::config_negative::{GraphCneConfig, GraphCneObjective};
use crate::sampling_utils::{
    AliasTable, MAX_PACKED_NODES, extract_canonical_undirected_edges,
    scatter_collapsed_riemannian_if_needed, validate_graph_weights, validate_target_riemannian,
};

/// `Graph-CNE` contrastive neighbor embedding solver (`Böhm et al., 2025`) supporting
/// [`GraphCneObjective::InfoNce`] and [`GraphCneObjective::NegTSne`].
#[derive(Debug, Clone)]
pub struct GraphCneSolver<L: RiemannianLayout = Layout2d<f64>> {
    config: GraphCneConfig,
    num_nodes: usize,
    num_springs: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edge_alias: Option<AliasTable>,
    neg_scratch: Vec<u32>,
    neg_tangents: Vec<f64>,
    neg_w: Vec<f64>,
    acc_rep: Vec<f64>,
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

impl<L: RiemannianLayout> GraphCneSolver<L> {
    /// Constructs a `GraphCneSolver` from `graph` and `config`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: GraphCneConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_graph_weights(graph, "GraphCneSolver::new")?;

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

        if n > 1 && graph.nnz() > 0 {
            if config.row_normalize {
                let probs = graph_row_normalized_probabilities(graph)?;
                for u in 0..n {
                    let u_u32 = u as u32;
                    let (nbrs, vals) = probs.row(u);
                    for (&v, &p_uv) in nbrs.iter().zip(vals.iter()) {
                        if u_u32 < v && p_uv.is_finite() && p_uv > 0.0 {
                            edges_u.push(u_u32);
                            edges_v.push(v);
                            edge_weights.push(p_uv);
                        }
                    }
                }
            } else {
                for (u, v, w) in extract_canonical_undirected_edges(graph) {
                    if w.is_finite() && w > 0.0 {
                        edges_u.push(u);
                        edges_v.push(v);
                        edge_weights.push(w);
                    }
                }
            }
        }

        let edge_alias = AliasTable::from_weights(&edge_weights);
        let m = edges_u.len();
        let neg_cap = config.negative_samples;

        Ok(Self {
            rng: DeterministicRng::from_seed_u64(config.seed),
            config,
            num_nodes: n,
            num_springs: m,
            edges_u,
            edges_v,
            edge_alias,
            neg_scratch: Vec::with_capacity(neg_cap),
            neg_tangents: Vec::new(),
            neg_w: Vec::with_capacity(neg_cap),
            acc_rep: Vec::new(),
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

    /// Returns a reference to the active [`GraphCneConfig`].
    #[inline]
    pub fn config(&self) -> &GraphCneConfig {
        &self.config
    }

    /// Returns the positive undirected edge endpoint slices `(edges_u, edges_v)`.
    #[inline]
    pub fn edges(&self) -> (&[u32], &[u32]) {
        (&self.edges_u, &self.edges_v)
    }

    /// Returns the Vose edge alias table `(prob, alias)` if positive edges exist.
    #[inline]
    pub fn edge_alias_table(&self) -> Option<(&[f64], &[u32])> {
        self.edge_alias.as_ref().map(AliasTable::as_slices)
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
            self.acc_rep.resize(dim, 0.0);
            self.neg_tangents
                .resize(self.config.negative_samples * dim, 0.0);
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
        let neg_samples = self.config.negative_samples;
        let rho = self.config.repulsion_weight;
        let objective = self.config.objective;

        for _ in 0..m {
            let e_idx = edge_alias.sample(&mut self.rng);
            let u_raw = self.edges_u[e_idx] as usize;
            let v_raw = self.edges_v[e_idx] as usize;
            let (u, v) = if (self.rng.next_u32() & 1) == 0 {
                (u_raw, v_raw)
            } else {
                (v_raw, u_raw)
            };

            self.neg_scratch.clear();
            for _ in 0..neg_samples {
                let k = self.rng.gen_range_usize(self.num_nodes);
                if k != u {
                    self.neg_scratch.push(k as u32);
                }
            }

            layout.log_map_into(u, v, &mut self.log_uv)?;
            layout.log_map_into(v, u, &mut self.log_vu)?;
            let mut d2_uv = 0.0_f64;
            for &val in &self.log_uv[..dim] {
                d2_uv += val * val;
            }
            if !d2_uv.is_finite() {
                continue;
            }
            let w_uv = 1.0 / (1.0 + d2_uv);

            self.neg_w.clear();
            let mut sum_neg_w = 0.0_f64;
            if rho > 0.0 {
                let needed = self.neg_scratch.len() * dim;
                if self.neg_tangents.len() < needed {
                    self.neg_tangents.resize(needed, 0.0);
                }
                for (slot, &k_u32) in self.neg_scratch.iter().enumerate() {
                    let t_slice = &mut self.neg_tangents[slot * dim..(slot + 1) * dim];
                    layout.log_map_into(u, k_u32 as usize, t_slice)?;
                    let mut d2 = 0.0_f64;
                    for &val in t_slice.iter() {
                        d2 += val * val;
                    }
                    let w_uk = if d2.is_finite() {
                        1.0 / (1.0 + d2)
                    } else {
                        0.0
                    };
                    self.neg_w.push(w_uk);
                    sum_neg_w += w_uk;
                }
            }

            self.acc_rep.fill(0.0);
            let attr_coeff = match objective {
                GraphCneObjective::InfoNce => {
                    let denom = w_uv + rho * sum_neg_w;
                    if !denom.is_finite() || denom <= 0.0 {
                        continue;
                    }
                    if rho > 0.0 {
                        for (slot, &w_uk) in self.neg_w.iter().enumerate() {
                            if w_uk > 0.0 {
                                let t_slice = &self.neg_tangents[slot * dim..(slot + 1) * dim];
                                let rep_coeff = 2.0 * lr * ((rho * w_uk) / denom) * w_uk;
                                for a in 0..dim {
                                    self.acc_rep[a] += (-rep_coeff * t_slice[a]).clamp(-4.0, 4.0);
                                }
                            }
                        }
                    }
                    2.0 * lr * (1.0 - w_uv / denom) * w_uv
                }
                GraphCneObjective::NegTSne => {
                    if rho > 0.0 && !self.neg_w.is_empty() {
                        let rep_scale = rho / (self.neg_w.len() as f64);
                        for (slot, &w_uk) in self.neg_w.iter().enumerate() {
                            if w_uk > 0.0 {
                                let t_slice = &self.neg_tangents[slot * dim..(slot + 1) * dim];
                                let rep_coeff = 2.0 * lr * rep_scale * w_uk * w_uk;
                                for a in 0..dim {
                                    self.acc_rep[a] += (-rep_coeff * t_slice[a]).clamp(-4.0, 4.0);
                                }
                            }
                        }
                    }
                    2.0 * lr * w_uv
                }
            };

            let mut ok = true;
            for a in 0..dim {
                let pos_u = (attr_coeff * self.log_uv[a]).clamp(-4.0, 4.0);
                let pos_v = (attr_coeff * self.log_vu[a]).clamp(-4.0, 4.0);
                let rep_a = if self.acc_rep[a].is_finite() {
                    self.acc_rep[a]
                } else {
                    0.0
                };
                let du = pos_u + rep_a;
                if !du.is_finite() || !pos_v.is_finite() {
                    ok = false;
                    break;
                }
                self.delta_u[a] = du;
                self.delta_v[a] = pos_v;
            }
            if ok {
                layout.exp_map(u, &self.delta_u[..dim])?;
                layout.exp_map(v, &self.delta_v[..dim])?;
            }
        }
        Ok(())
    }
}

impl<L: RiemannianLayout> IterativeSolver for GraphCneSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "GraphCneSolver::validate_target",
        )?;
        Ok(())
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        let dim = validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "GraphCneSolver::step",
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
