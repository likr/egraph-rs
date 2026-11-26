//! GRIP (Graph dRawing with Intelligent Placement) multilevel force-directed layout solver.
//!
//! Implements Gajer & Kobourov (2002), "GRIP: Graph Drawing with Intelligent Placement"
//! (*JGAA* 6(3), 203–224):
//! - Maximal Independent Set (`MIS`) vertex filtration $V_0 \supset V_1 \supset \dots \supset V_{k-1}$
//!   constructed via depth-$2^{i-1}$ partial BFS ("create-then-destroy") down to $|V_{k-1}| = 3$.
//! - Triangle initialization of $V_{k-1}$ from exact pairwise graph distances.
//! - Coarse-to-fine intelligent placement of $V_i \setminus V_{i+1}$ from up to 3 nearest
//!   ancestors in $V_{i+1}$ with local refinement.
//! - Localized Kamada–Kawai forces on coarse levels ($i > 0$), localized Fruchterman–Reingold
//!   forces on the finest level ($i = 0$), and per-vertex adaptive local temperature
//!   (`updateLocalTemp`).
//! - Connected-component decomposition with [`crate::group_in_a_box::squarified_treemap`]
//!   packing for disconnected graphs, and guards for `N <= 2`.

mod filtration;
mod forces;
#[cfg(test)]
mod tests;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};

use crate::group_in_a_box::squarified_treemap;
use filtration::{
    GripComponent, build_grip_component, build_simple_undirected_adj, extract_connected_components,
};
use forces::{map_component_into_box, step_grip_component};

/// Hyperparameters for [`GripSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripConfig {
    /// Minimum refinement rounds at the finest level $i = 0$ (`default: 5`).
    pub rounds_min: usize,
    /// Maximum refinement rounds at the coarsest level (`default: 30`).
    pub rounds_max: usize,
    /// Maximum size of each vertex's local neighborhood $|N_i(v)|$ (`default: 20`).
    pub local_neighborhood_size: usize,
    /// Repulsion scaling factor $s$ used in the finest-level ($i = 0$) FR force (`default: 0.05`).
    pub repulsion_scale: f64,
    /// Target unit edge length (`default: 1.0`).
    pub edge_length: f64,
    /// Local temperature adjustment step $r$ in `updateLocalTemp` (`default: 0.15`).
    pub temp_r: f64,
    /// Local temperature oscillation/acceleration multiplier $s$ (`default: 3.0`).
    pub temp_s: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GripConfig {
    fn default() -> Self {
        Self {
            rounds_min: 5,
            rounds_max: 30,
            local_neighborhood_size: 20,
            repulsion_scale: 0.05,
            edge_length: 1.0,
            temp_r: 0.15,
            temp_s: 3.0,
            seed: 42,
        }
    }
}

impl GripConfig {
    /// Validates all GRIP hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.rounds_min > self.rounds_max {
            return Err(CoreError::invalid_input(format!(
                "rounds_min ({}) must be <= rounds_max ({})",
                self.rounds_min, self.rounds_max
            )));
        }
        if self.local_neighborhood_size == 0 {
            return Err(CoreError::invalid_input(
                "local_neighborhood_size must be >= 1",
            ));
        }
        if !self.repulsion_scale.is_finite() {
            return Err(CoreError::non_finite("GripConfig::repulsion_scale"));
        }
        if self.repulsion_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "repulsion_scale must be >= 0.0, got {}",
                self.repulsion_scale
            )));
        }
        if !self.edge_length.is_finite() {
            return Err(CoreError::non_finite("GripConfig::edge_length"));
        }
        if self.edge_length <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_length must be > 0.0, got {}",
                self.edge_length
            )));
        }
        if !self.temp_r.is_finite() {
            return Err(CoreError::non_finite("GripConfig::temp_r"));
        }
        if self.temp_r < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "temp_r must be >= 0.0, got {}",
                self.temp_r
            )));
        }
        if !self.temp_s.is_finite() {
            return Err(CoreError::non_finite("GripConfig::temp_s"));
        }
        if self.temp_s < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "temp_s must be >= 0.0, got {}",
                self.temp_s
            )));
        }
        Ok(())
    }
}

/// Cooperative GRIP (Gajer & Kobourov, 2002) multilevel layout solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct GripSolver<S: Scalar = f64> {
    config: GripConfig,
    num_nodes: usize,
    num_levels: usize,
    total_iterations: usize,
    components: Vec<GripComponent>,
    component_boxes: Vec<[f64; 4]>,
    rng: DeterministicRng,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> GripSolver<S> {
    /// Constructs a [`GripSolver`] from a [`CsrGraph`] and [`GripConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: GripConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("GripSolver::graph_weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "graph edge weights must be non-negative",
                    ));
                }
            }
        }

        let mut rng = DeterministicRng::from_seed_u64(config.seed);

        if n <= 1 {
            return Ok(Self {
                config,
                num_nodes: n,
                num_levels: n,
                total_iterations: 0,
                components: Vec::new(),
                component_boxes: Vec::new(),
                rng,
                iter: 0,
                status: SolverStatus::Initialized,
                _marker: std::marker::PhantomData,
            });
        }

        let (adj_indptr, adj_indices) = build_simple_undirected_adj(graph);
        let raw_components = extract_connected_components(n, &adj_indptr, &adj_indices);

        let mut components = Vec::with_capacity(raw_components.len());
        let mut max_levels = 0usize;
        let mut max_rounds_sum = 0usize;

        for comp_nodes in raw_components {
            let comp =
                build_grip_component(comp_nodes, &adj_indptr, &adj_indices, &config, &mut rng);
            max_levels = max_levels.max(comp.levels.len());
            let rounds_sum: usize = comp.levels.iter().map(|lvl| lvl.rounds).sum();
            max_rounds_sum = max_rounds_sum.max(rounds_sum);
            components.push(comp);
        }

        let component_boxes = if components.len() > 1 {
            let weights: Vec<f64> = components
                .iter()
                .map(|c| c.global_nodes.len() as f64)
                .collect();
            let side = (n as f64).sqrt() * config.edge_length * 2.5;
            squarified_treemap(&weights, -0.5 * side, -0.5 * side, side, side)?
        } else {
            Vec::new()
        };

        // Ensure `run()` executes at least one step for `n >= 2` so disconnected tiny
        // components or `rounds_max == 0` configs still emit valid initial/packed coordinates.
        let total_iterations = max_rounds_sum.max(1);

        Ok(Self {
            config,
            num_nodes: n,
            num_levels: max_levels,
            total_iterations,
            components,
            component_boxes,
            rng,
            iter: 0,
            status: SolverStatus::Initialized,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GripConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the maximum number of MIS filtration levels across connected components.
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn write_to_layout(&self, layout: &mut Layout2d<S>) {
        let (out_x, out_y) = layout.xy_mut();
        if self.components.len() == 1 {
            let comp = &self.components[0];
            for (local_idx, &global_u) in comp.global_nodes.iter().enumerate() {
                let ui = global_u as usize;
                let nx = S::from_f64(comp.pos_x[local_idx]);
                let ny = S::from_f64(comp.pos_y[local_idx]);
                if nx.is_finite() {
                    out_x[ui] = nx;
                }
                if ny.is_finite() {
                    out_y[ui] = ny;
                }
            }
            return;
        }

        for (comp, &[x_min, x_max, y_min, y_max]) in
            self.components.iter().zip(self.component_boxes.iter())
        {
            map_component_into_box(comp, x_min, x_max, y_min, y_max, out_x, out_y);
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for GripSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GripSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let mut any_active = false;
        for comp in &mut self.components {
            if !comp.finished {
                step_grip_component(comp, &self.config, &mut self.rng);
                if !comp.finished {
                    any_active = true;
                }
            }
        }

        self.write_to_layout(layout);
        self.iter += 1;

        self.status = if !any_active || self.iter >= self.total_iterations {
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
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}
