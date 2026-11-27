//! Cooperative `StressMajorizationSolver` (`f32`) WebAssembly bindings across 2D, N-D, and Manifolds.

use egraph::layout::{StressMajorizationConfig, StressMajorizationSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
};
use crate::js_options::{
    is_plain_js_object, opt_f64_prop, opt_maybe_usize_prop, opt_usize_prop,
    reject_extra_args_with_options, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_manifold_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_manifold_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_manifold_in_place, step_nd_in_place,
};

fn parse_stress_majorization_config(
    epsilon_or_options: &JsValue,
    max_iterations: Option<f64>,
    pcg_tolerance: Option<f64>,
    pcg_max_iterations: Option<f64>,
    min_norm: Option<f64>,
) -> Result<StressMajorizationConfig, JsError> {
    let mut cfg = StressMajorizationConfig::default();
    if is_plain_js_object(epsilon_or_options) {
        let has_extra = max_iterations.is_some()
            || pcg_tolerance.is_some()
            || pcg_max_iterations.is_some()
            || min_norm.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.epsilon = opt_f64_prop(epsilon_or_options, "epsilon", cfg.epsilon)?;
        cfg.max_iterations =
            opt_usize_prop(epsilon_or_options, "maxIterations", cfg.max_iterations)?;
        cfg.pcg_tolerance = opt_f64_prop(epsilon_or_options, "pcgTolerance", cfg.pcg_tolerance)?;
        cfg.pcg_max_iterations = opt_maybe_usize_prop(epsilon_or_options, "pcgMaxIterations")?;
        cfg.min_norm = opt_f64_prop(epsilon_or_options, "minNorm", cfg.min_norm)?;
        return Ok(cfg);
    }

    if !epsilon_or_options.is_undefined() && !epsilon_or_options.is_null() {
        cfg.epsilon = epsilon_or_options.as_f64().ok_or_else(|| {
            JsError::new("epsilonOrOptions must be a number, options object, null, or undefined")
        })?;
    }
    cfg.max_iterations = parse_optional_usize(max_iterations, cfg.max_iterations, "maxIterations")?;
    cfg.pcg_tolerance = pcg_tolerance.unwrap_or(cfg.pcg_tolerance);
    cfg.pcg_max_iterations = parse_opt_usize_param(pcg_max_iterations, "pcgMaxIterations")?;
    cfg.min_norm = min_norm.unwrap_or(cfg.min_norm);
    Ok(cfg)
}

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver (`f32`)
/// across `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.
#[wasm_bindgen(js_name = StressMajorizationSolver)]
#[derive(Debug, Clone)]
pub struct JsStressMajorizationSolver {
    inner: StressMajorizationSolver<f32>,
}

#[wasm_bindgen(js_class = StressMajorizationSolver)]
impl JsStressMajorizationSolver {
    /// Constructs a `StressMajorizationSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsStressMajorizationSolver, JsError> {
        Self::from_graph(graph, options, None, None, None, None)
    }

    /// Constructs a `StressMajorizationSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        epsilon_or_options: &JsValue,
        max_iterations: Option<f64>,
        pcg_tolerance: Option<f64>,
        pcg_max_iterations: Option<f64>,
        min_norm: Option<f64>,
    ) -> Result<JsStressMajorizationSolver, JsError> {
        let cfg = parse_stress_majorization_config(
            epsilon_or_options,
            max_iterations,
            pcg_tolerance,
            pcg_max_iterations,
            min_norm,
        )?;
        let inner = StressMajorizationSolver::<f32>::from_graph(graph.inner(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `StressMajorizationSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        epsilon_or_options: &JsValue,
        max_iterations: Option<f64>,
        pcg_tolerance: Option<f64>,
        pcg_max_iterations: Option<f64>,
        min_norm: Option<f64>,
    ) -> Result<JsStressMajorizationSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_stress_majorization_config(
            epsilon_or_options,
            max_iterations,
            pcg_tolerance,
            pcg_max_iterations,
            min_norm,
        )?;
        let inner = StressMajorizationSolver::<f32>::from_distance_matrix(n, dists_flat, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one majorization step on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` majorization steps on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "StressMajorizationSolver::run_steps",
        )
    }

    /// Runs majorization to completion on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "StressMajorizationSolver::run",
        )
    }

    /// Executes one majorization step on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` majorization steps on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        run_steps_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "StressMajorizationSolver::run_steps_nd",
        )
    }

    /// Runs majorization to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        run_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "StressMajorizationSolver::run_nd",
        )
    }

    /// Executes one geodesic majorization step on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        step_manifold_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` geodesic majorization steps on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsError> {
        run_steps_manifold_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "StressMajorizationSolver::run_steps_manifold",
        )
    }

    /// Runs geodesic majorization to completion on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        run_manifold_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "StressMajorizationSolver::run_manifold",
        )
    }

    /// Evaluates weighted stress for a `Layout2d`.
    #[wasm_bindgen(js_name = _computeStress2d)]
    pub fn compute_stress_2d(&self, layout: &JsLayout2d) -> Result<f64, JsError> {
        let soa = layout.inner().to_layout2d();
        self.inner.compute_stress(&soa).map_err(core_err_to_js)
    }

    /// Evaluates weighted stress for a `LayoutNd`.
    #[wasm_bindgen(js_name = _computeStressNd)]
    pub fn compute_stress_nd(&self, layout: &JsLayoutNd) -> Result<f64, JsError> {
        self.inner
            .compute_stress_nd(layout.inner())
            .map_err(core_err_to_js)
    }

    /// Evaluates weighted geodesic stress for a `ManifoldLayout2d`.
    #[wasm_bindgen(js_name = _computeStressManifold)]
    pub fn compute_stress_manifold(&self, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
        self.inner
            .compute_stress_manifold(layout.inner())
            .map_err(core_err_to_js)
    }

    /// Returns the stress recorded after the most recent step, or `undefined`.
    #[wasm_bindgen(getter, js_name = lastStress)]
    pub fn last_stress(&self) -> Option<f64> {
        self.inner.last_stress()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled maximum iteration count.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}
