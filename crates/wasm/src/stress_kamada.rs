//! Cooperative `StressMajorizationSolver` and `KamadaKawaiSolver` (`f32`) WebAssembly bindings.

use egraph::layout::{
    KamadaKawaiConfig, KamadaKawaiSolver, StressMajorizationConfig, StressMajorizationSolver,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
};
use crate::js_options::{
    is_plain_js_object, opt_f64_prop, opt_maybe_usize_prop, opt_usize_prop,
    reject_extra_args_with_options, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

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

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver (`f32`).
#[wasm_bindgen(js_name = StressMajorizationSolver)]
#[derive(Debug, Clone)]
pub struct JsStressMajorizationSolver {
    inner: StressMajorizationSolver<f32>,
}

#[wasm_bindgen(js_class = StressMajorizationSolver)]
impl JsStressMajorizationSolver {
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

    /// Executes one majorization step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` majorization steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "StressMajorizationSolver::run_steps",
        )
    }

    /// Runs majorization to completion or relative stress convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "StressMajorizationSolver::run",
        )
    }

    /// Evaluates weighted stress $\sum_{i < j} w_{ij} (\|x_i - x_j\| - d_{ij})^2$ for `layout`.
    #[wasm_bindgen(js_name = computeStress)]
    pub fn compute_stress(&self, layout: &JsLayout2d) -> Result<f64, JsError> {
        let soa = layout.inner().to_layout2d();
        self.inner.compute_stress(&soa).map_err(core_err_to_js)
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

fn parse_kamada_kawai_config(
    eps_or_options: &JsValue,
    max_inner_iters: Option<f64>,
    max_outer_iters: Option<f64>,
) -> Result<KamadaKawaiConfig, JsError> {
    let mut cfg = KamadaKawaiConfig::default();
    if is_plain_js_object(eps_or_options) {
        let has_extra = max_inner_iters.is_some() || max_outer_iters.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.eps = opt_f64_prop(eps_or_options, "eps", cfg.eps)?;
        cfg.max_inner_iters = opt_usize_prop(eps_or_options, "maxInnerIters", cfg.max_inner_iters)?;
        cfg.max_iterations = opt_maybe_usize_prop(eps_or_options, "maxOuterIters")?;
        return Ok(cfg);
    }

    if !eps_or_options.is_undefined() && !eps_or_options.is_null() {
        cfg.eps = eps_or_options.as_f64().ok_or_else(|| {
            JsError::new("epsOrOptions must be a number, options object, null, or undefined")
        })?;
    }
    cfg.max_inner_iters =
        parse_optional_usize(max_inner_iters, cfg.max_inner_iters, "maxInnerIters")?;
    cfg.max_iterations = parse_opt_usize_param(max_outer_iters, "maxOuterIters")?;
    Ok(cfg)
}

/// Cooperative Kamada-Kawai (1989) spring layout solver (`f32`).
#[wasm_bindgen(js_name = KamadaKawaiSolver)]
#[derive(Debug, Clone)]
pub struct JsKamadaKawaiSolver {
    inner: KamadaKawaiSolver<f32>,
}

#[wasm_bindgen(js_class = KamadaKawaiSolver)]
impl JsKamadaKawaiSolver {
    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        eps_or_options: &JsValue,
        max_inner_iters: Option<f64>,
        max_outer_iters: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_outer_iters)?;
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
        max_outer_iters: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_outer_iters)?;
        let inner = KamadaKawaiSolver::<f32>::from_distance_matrix(n, dists_flat, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one outer node-selection Newton-Raphson step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` outer steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "KamadaKawaiSolver::run_steps",
        )
    }

    /// Runs Kamada-Kawai optimization to convergence or `maxOuterIters` on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "KamadaKawaiSolver::run",
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
