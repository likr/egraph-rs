//! Cooperative Screened Poisson unilateral upper-bound 2D density field refiner bindings (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize};
use crate::js_options::{
    check_options_arg, opt_bool_prop, opt_f64_prop, opt_seed_prop, opt_usize_prop,
    solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::screened_poisson::{JsScreenedPoissonResult, parse_screened_poisson_config};
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn parse_screened_poisson_refiner_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ScreenedPoissonRefinerConfig, JsError> {
    let mut cfg = ScreenedPoissonRefinerConfig {
        field: parse_screened_poisson_config(options, ctx)?,
        ..ScreenedPoissonRefinerConfig::default()
    };
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.step_size = opt_f64_prop(options, "stepSize", cfg.step_size)?;
        cfg.momentum = opt_f64_prop(options, "momentum", cfg.momentum)?;
        cfg.max_step_ratio = opt_f64_prop(options, "maxStepRatio", cfg.max_step_ratio)?;
        cfg.anchor_weight = opt_f64_prop(options, "anchorWeight", cfg.anchor_weight)?;
        cfg.gamma_tether = opt_f64_prop(options, "gammaTether", cfg.gamma_tether)?;
        cfg.recenter = opt_bool_prop(options, "recenter", cfg.recenter)?;
        cfg.energy_tol = opt_f64_prop(options, "energyTol", cfg.energy_tol)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// Cooperative Screened Poisson unilateral upper-bound density field refiner (`f32`-backed).
#[wasm_bindgen(js_name = ScreenedPoissonRefiner)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonRefiner {
    inner: ScreenedPoissonRefiner<f32>,
}

#[wasm_bindgen(js_class = ScreenedPoissonRefiner)]
impl JsScreenedPoissonRefiner {
    /// Creates a standalone `ScreenedPoissonRefiner` for `numNodes` vertices.
    #[wasm_bindgen(constructor)]
    pub fn new(num_nodes: f64, options: &JsValue) -> Result<JsScreenedPoissonRefiner, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_screened_poisson_refiner_config(options, "ScreenedPoissonRefiner")?;
        let inner = ScreenedPoissonRefiner::<f32>::new(n, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ScreenedPoissonRefiner` retaining 1-hop edges from `graph` for optional anchors.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsScreenedPoissonRefiner, JsError> {
        let cfg =
            parse_screened_poisson_refiner_config(options, "ScreenedPoissonRefiner.fromGraph")?;
        let inner = ScreenedPoissonRefiner::<f32>::with_graph(graph.inner(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Attaches or replaces a coarse-level cluster centroid tether (`alignClusteredCentroids`).
    #[wasm_bindgen(js_name = setCoarseTether)]
    pub fn set_coarse_tether(
        &mut self,
        coarse_layout: &JsLayout2d,
        node_to_coarse: &[u32],
    ) -> Result<(), JsError> {
        let coarse_soa = coarse_layout.inner().to_layout2d();
        self.inner
            .set_coarse_tether(coarse_soa, node_to_coarse.to_vec())
            .map_err(core_err_to_js)
    }

    /// Executes one density refinement step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` density refinement steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ScreenedPoissonRefiner::run_steps",
        )
    }

    /// Runs density refinement to completion or early convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ScreenedPoissonRefiner::run",
        )
    }

    /// Executes a single decoupled epoch-boundary density step on `layout` in place and returns stats.
    #[wasm_bindgen(js_name = applyEpochStep)]
    pub fn apply_epoch_step(
        &mut self,
        layout: &mut JsLayout2d,
    ) -> Result<JsScreenedPoissonResult, JsError> {
        let mut soa = layout.inner().to_layout2d();
        let stats = self
            .inner
            .apply_epoch_step(&mut soa)
            .map_err(core_err_to_js)?;
        layout
            .inner_mut()
            .copy_from_layout2d(&soa)
            .map_err(ffi_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_stats(stats))
    }

    /// Returns telemetry from the most recent density field evaluation.
    #[wasm_bindgen(js_name = lastStats)]
    pub fn last_stats(&self) -> JsScreenedPoissonResult {
        JsScreenedPoissonResult::from_stats(*self.inner.last_stats())
    }

    /// Most recently evaluated variational density energy.
    #[wasm_bindgen(getter, js_name = lastEnergy)]
    pub fn last_energy(&self) -> f64 {
        self.inner.last_energy()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of refinement iterations completed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the configured maximum number of refinement iterations.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver reached `"converged"` or `"completed"`.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected number of vertices in `Layout2d`.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns total heap bytes allocated by the underlying density field workspace.
    #[wasm_bindgen(js_name = workspaceMemoryBytes)]
    pub fn workspace_memory_bytes(&self) -> usize {
        self.inner.workspace_memory_bytes()
    }
}
