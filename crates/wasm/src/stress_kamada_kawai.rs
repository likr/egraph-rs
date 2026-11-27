//! Cooperative Kamada-Kawai (1989) spring layout solver (`f32`) WebAssembly bindings.

use egraph::layout::{KamadaKawaiConfig, KamadaKawaiSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_optional_usize};
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

fn parse_kamada_kawai_config(
    eps_or_options: &JsValue,
    max_inner_iters: Option<f64>,
    max_iterations: Option<f64>,
) -> Result<KamadaKawaiConfig, JsError> {
    let mut cfg = KamadaKawaiConfig::default();
    if is_plain_js_object(eps_or_options) {
        let has_extra = max_inner_iters.is_some() || max_iterations.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.eps = opt_f64_prop(eps_or_options, "eps", cfg.eps)?;
        cfg.max_inner_iters = opt_usize_prop(eps_or_options, "maxInnerIters", cfg.max_inner_iters)?;
        cfg.max_iterations = opt_maybe_usize_prop(eps_or_options, "maxIterations")?;
        return Ok(cfg);
    }

    if !eps_or_options.is_undefined() && !eps_or_options.is_null() {
        cfg.eps = eps_or_options.as_f64().ok_or_else(|| {
            JsError::new("epsOrOptions must be a number, options object, null, or undefined")
        })?;
    }
    cfg.max_inner_iters =
        parse_optional_usize(max_inner_iters, cfg.max_inner_iters, "maxInnerIters")?;
    cfg.max_iterations = max_iterations
        .map(|v| parse_non_negative_usize(v, "maxIterations"))
        .transpose()?;
    Ok(cfg)
}

/// Cooperative Kamada-Kawai (1989) spring layout solver (`f32`) across 2D, N-D, and Manifolds.
#[wasm_bindgen(js_name = KamadaKawaiSolver)]
#[derive(Debug, Clone)]
pub struct JsKamadaKawaiSolver {
    inner: KamadaKawaiSolver<f32>,
}

#[wasm_bindgen(js_class = KamadaKawaiSolver)]
impl JsKamadaKawaiSolver {
    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsKamadaKawaiSolver, JsError> {
        Self::from_graph(graph, options, None, None)
    }

    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        eps_or_options: &JsValue,
        max_inner_iters: Option<f64>,
        max_iterations: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_iterations)?;
        let inner =
            KamadaKawaiSolver::<f32>::from_graph(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `KamadaKawaiSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        eps_or_options: &JsValue,
        max_inner_iters: Option<f64>,
        max_iterations: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_iterations)?;
        let inner = KamadaKawaiSolver::<f32>::from_distance_matrix(n, dists_flat, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one outer node-selection Newton-Raphson step on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` outer steps on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "KamadaKawaiSolver::run_steps",
        )
    }

    /// Runs Kamada-Kawai optimization to convergence or `maxIterations` on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "KamadaKawaiSolver::run",
        )
    }

    /// Executes one outer node-selection Newton-Raphson step on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` outer steps on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        run_steps_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "KamadaKawaiSolver::run_steps_nd",
        )
    }

    /// Runs Kamada-Kawai optimization to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        run_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "KamadaKawaiSolver::run_nd",
        )
    }

    /// Executes one outer node-selection geodesic step on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        step_manifold_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` outer geodesic steps on `ManifoldLayout2d` in place.
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
            "KamadaKawaiSolver::run_steps_manifold",
        )
    }

    /// Runs Kamada-Kawai optimization to completion on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        run_manifold_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "KamadaKawaiSolver::run_manifold",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of outer steps executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the outer step cap.
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
