//! Structural Entropy Layout (`SEL`) solver.
//!
//! Implements Xue et al. (2025), "Structural Entropy Based Visualization of Social Networks"
//! (*Computer Graphics Forum / Pacific Graphics 2025*) and the reference defaults from
//! `IDEAS-Laboratory/SEL` (`master`):
//! - $k$-dimensional coding tree (`mode = 'v2'`, default `tree_depth = 5`, `+1` Laplace smoothing)
//!   with virtual global root handling for disconnected graphs and isolated vertices.
//! - Algorithm 1 Minimum Entropy Priority Queue (`pq_max_pops = 10`, unassigned sentinel `100.0`)
//!   assigning cross-partition structural entropy scores $W[i, j]$ to edges.
//! - Elbow threshold selection bounded by $\mathrm{SE}_{\mathrm{target}} \le 0.5 H_{\max}$.
//! - Five edge-weight modulation mappings (`Sigmoid`, `Linear`, `Tanh`, `CustomLogistic`, `Binary`).
//! - Cooperative layout optimization delegating to [`FruchtermanReingoldSolver`] (`Fr`) or
//!   [`FieldLayoutSolver::t_fdp`] (`TFdp`).

mod coding_tree;
mod entropy;
mod k_tree;
#[cfg(test)]
mod tests;
mod tree_expand;

use coding_tree::compute_sel_weights;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_layout_force::{FruchtermanReingoldConfig, FruchtermanReingoldSolver};
use egraph_layout_neighbor::{FieldLayoutSolver, TFdpConfig};

/// Edge-weight modulation function applied to structural-entropy edge scores $W[i, j]$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelWeightMapping {
    /// $\sigma(1 - \mathrm{SE}_{\mathrm{target}} / W)$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0` (`default`).
    #[default]
    Sigmoid,
    /// Linear $1 - \mathrm{SE}_{\mathrm{target}} / W$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Linear,
    /// $(\tanh(1 - \mathrm{SE}_{\mathrm{target}} / W) + 1) / 2$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Tanh,
    /// Decaying logistic $(1 + \exp(10(z - 0.8)))^{-1}$ matching `IDEAS-Laboratory/SEL` `custom_logistic`.
    CustomLogistic,
    /// Binary `1.0` if $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Binary,
}

/// Underlying force-directed layout backend driven by SEL-modulated edge weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelLayoutBackend {
    /// Classic Fruchterman–Reingold solver ([`FruchtermanReingoldSolver`], `default`).
    #[default]
    Fr,
    /// Bounded Student-$t$ force-directed placement ([`FieldLayoutSolver::t_fdp`]).
    TFdp,
}

/// Hyperparameters for [`SelLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelConfig {
    /// Target height of the $k$-dimensional coding tree (`default: 5`).
    pub tree_depth: usize,
    /// Maximum priority-queue pops in Algorithm 1 (`default: 10`).
    pub pq_max_pops: usize,
    /// Edge-weight mapping function (`default: SelWeightMapping::Sigmoid`).
    pub mapping: SelWeightMapping,
    /// Force-directed layout backend (`default: SelLayoutBackend::Fr`).
    pub layout_backend: SelLayoutBackend,
    /// Number of iterations forwarded to the backend solver (`default: 200`).
    pub iterations: usize,
    /// Explicit deterministic RNG seed forwarded to the backend solver (`default: 42`).
    pub seed: u64,
}

impl Default for SelConfig {
    fn default() -> Self {
        Self {
            tree_depth: 5,
            pq_max_pops: 10,
            mapping: SelWeightMapping::Sigmoid,
            layout_backend: SelLayoutBackend::Fr,
            iterations: 200,
            seed: 42,
        }
    }
}

impl SelConfig {
    /// Validates all SEL hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.tree_depth == 0 {
            return Err(CoreError::invalid_input("tree_depth must be >= 1"));
        }
        if self.pq_max_pops == 0 {
            return Err(CoreError::invalid_input("pq_max_pops must be >= 1"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum SelBackendSolver<S: Scalar> {
    Fr(Box<FruchtermanReingoldSolver<S>>),
    TFdp(Box<FieldLayoutSolver<S>>),
}

/// Cooperative Structural Entropy Layout (`SEL`) solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct SelLayoutSolver<S: Scalar = f64> {
    config: SelConfig,
    num_nodes: usize,
    modulated_weights: Vec<f64>,
    se_target: f64,
    backend: SelBackendSolver<S>,
}

impl<S: Scalar> SelLayoutSolver<S> {
    /// Constructs a [`SelLayoutSolver`] from a [`CsrGraph`] and [`SelConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: SelConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_uv, orig_weights) = extract_canonical_undirected_edges(graph)?;
        let m = edges_uv.len();

        let (se_target, modulated_weights) = if n <= 1 || m == 0 {
            (0.0, vec![0.0; m])
        } else {
            let mut node_vol = vec![0.0_f64; n];
            let mut total_vol = 0.0_f64;
            for (&(u, v), &w) in edges_uv.iter().zip(orig_weights.iter()) {
                node_vol[u as usize] += w;
                node_vol[v as usize] += w;
                total_vol += 2.0 * w;
            }
            if total_vol <= 0.0 {
                (0.0, vec![0.0; m])
            } else {
                compute_sel_weights(n, &edges_uv, &orig_weights, &node_vol, total_vol, &config)
            }
        };

        let mod_graph = CsrGraph::from_edges(n, &edges_uv, Some(&modulated_weights), false, false)?;

        let backend = match config.layout_backend {
            SelLayoutBackend::Fr => {
                let fr_cfg = FruchtermanReingoldConfig {
                    iterations: config.iterations,
                    seed: config.seed,
                    ..FruchtermanReingoldConfig::default()
                };
                SelBackendSolver::Fr(Box::new(FruchtermanReingoldSolver::new(
                    &mod_graph, fr_cfg,
                )?))
            }
            SelLayoutBackend::TFdp => {
                let tfdp_cfg = TFdpConfig {
                    total_iterations: config.iterations,
                    seed: config.seed,
                    ..TFdpConfig::default()
                };
                SelBackendSolver::TFdp(Box::new(FieldLayoutSolver::t_fdp(&mod_graph, tfdp_cfg)?))
            }
        };

        Ok(Self {
            config,
            num_nodes: n,
            modulated_weights,
            se_target,
            backend,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &SelConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the SEL-modulated edge weights aligned with canonical sorted undirected edges `(u, v)` (`u < v`).
    #[inline]
    pub fn modulated_weights(&self) -> &[f64] {
        &self.modulated_weights
    }

    /// Returns the elbow-selected structural entropy threshold $\mathrm{SE}_{\mathrm{target}}$.
    #[inline]
    pub fn se_target(&self) -> f64 {
        self.se_target
    }
}

impl<S: Scalar> IterativeSolver<S> for SelLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SelLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        match &mut self.backend {
            SelBackendSolver::Fr(solver) => solver.step(layout),
            SelBackendSolver::TFdp(solver) => solver.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.status(),
            SelBackendSolver::TFdp(solver) => solver.status(),
        }
    }

    #[inline]
    fn iteration(&self) -> usize {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.iteration(),
            SelBackendSolver::TFdp(solver) => solver.iteration(),
        }
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.total_iterations(),
            SelBackendSolver::TFdp(solver) => solver.total_iterations(),
        }
    }
}

#[inline]
pub(super) fn map_entropy_weight(entropy: f64, se_target: f64, mapping: SelWeightMapping) -> f64 {
    if !entropy.is_finite() || entropy <= se_target || entropy <= 0.0 {
        return 0.0;
    }
    let z = 1.0 - se_target / entropy;
    match mapping {
        SelWeightMapping::Sigmoid => 1.0 / (1.0 + (-z).exp()),
        SelWeightMapping::Linear => z,
        SelWeightMapping::Tanh => 0.5 * (z.tanh() + 1.0),
        SelWeightMapping::CustomLogistic => {
            if z >= 1.0 {
                0.0
            } else if z <= 0.0 {
                1.0
            } else {
                1.0 / (1.0 + (10.0 * (z - 0.8)).exp())
            }
        }
        SelWeightMapping::Binary => 1.0,
    }
}

#[allow(clippy::type_complexity)]
fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<(u32, u32)>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("SelLayoutSolver::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let nbrs = graph.neighbors(u);
        let ws = graph.neighbor_weights(u);
        for (pos, &v_u32) in nbrs.iter().enumerate() {
            if u32_u == v_u32 {
                continue;
            }
            let (a, b) = if u32_u < v_u32 {
                (u32_u, v_u32)
            } else {
                (v_u32, u32_u)
            };
            raw.push((a, b, ws.map_or(1.0, |w| w[pos].to_f64())));
        }
    }
    raw.sort_unstable_by_key(|x| (x.0, x.1));

    let weighted = graph.is_weighted();
    let mut edges_uv = Vec::new();
    let mut weights = Vec::new();
    let mut idx = 0usize;
    while idx < raw.len() {
        let (a, b, first_w) = raw[idx];
        let mut sum_w = first_w;
        let mut cnt = 1usize;
        idx += 1;
        while idx < raw.len() && raw[idx].0 == a && raw[idx].1 == b {
            sum_w += raw[idx].2;
            cnt += 1;
            idx += 1;
        }
        let merged_w = if weighted { sum_w / (cnt as f64) } else { 1.0 };
        edges_uv.push((a, b));
        weights.push(merged_w);
    }

    Ok((n, edges_uv, weights))
}
