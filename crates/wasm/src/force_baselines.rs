//! WebAssembly bindings for GEM and Structural Entropy Layout (`SEL`) baseline solvers (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{
    GemConfig, GemSolver, SelConfig, SelLayoutBackend, SelLayoutSolver, SelWeightMapping,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_maybe_usize_prop, opt_seed_prop, opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

/// Cooperative Frick et al. (1995) GEM adaptive local-temperature layout solver (`f32`).
#[wasm_bindgen(js_name = GemSolver)]
#[derive(Debug, Clone)]
pub struct JsGemSolver {
    inner: GemSolver<f32>,
}

#[wasm_bindgen(js_class = GemSolver)]
impl JsGemSolver {
    /// Constructs a `GemSolver` from `graph` and optional `GemOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGemSolver, JsError> {
        let mut cfg = GemConfig::default();
        if check_options_arg(options, "GemSolver")? {
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.t_init = opt_f64_prop(options, "tInit", cfg.t_init)?;
            cfg.t_max = opt_f64_prop(options, "tMax", cfg.t_max)?;
            cfg.t_min = opt_f64_prop(options, "tMin", cfg.t_min)?;
            cfg.max_rounds = opt_maybe_usize_prop(options, "maxRounds")?;
            cfg.oscillation_angle =
                opt_f64_prop(options, "oscillationAngle", cfg.oscillation_angle)?;
            cfg.rotation_angle = opt_f64_prop(options, "rotationAngle", cfg.rotation_angle)?;
            cfg.oscillation_sensitivity = opt_f64_prop(
                options,
                "oscillationSensitivity",
                cfg.oscillation_sensitivity,
            )?;
            cfg.rotation_sensitivity = opt_maybe_f64_prop(options, "rotationSensitivity")?;
            cfg.perturbation = opt_f64_prop(options, "perturbation", cfg.perturbation)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GemSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GEM round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GEM rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GemSolver::run_steps",
        )
    }

    /// Runs GEM layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GemSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled rounds (`maxRounds`).
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

fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, JsError> {
    match normalize_enum_token(raw).as_str() {
        "sigmoid" => Ok(SelWeightMapping::Sigmoid),
        "linear" => Ok(SelWeightMapping::Linear),
        "tanh" => Ok(SelWeightMapping::Tanh),
        "customlogistic" => Ok(SelWeightMapping::CustomLogistic),
        "binary" => Ok(SelWeightMapping::Binary),
        _ => Err(JsError::new(&format!(
            "unknown SelWeightMapping {raw:?}; expected 'sigmoid', 'linear', 'tanh', 'custom_logistic', or 'binary'"
        ))),
    }
}

fn parse_sel_layout_backend(raw: &str) -> Result<SelLayoutBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(SelLayoutBackend::Fr),
        "tfdp" => Ok(SelLayoutBackend::TFdp),
        _ => Err(JsError::new(&format!(
            "unknown SelLayoutBackend {raw:?}; expected 'fr' or 't_fdp'"
        ))),
    }
}

/// Cooperative Xue et al. (2025) Structural Entropy Layout (`SEL`) solver (`f32`).
#[wasm_bindgen(js_name = SelLayoutSolver)]
#[derive(Debug, Clone)]
pub struct JsSelLayoutSolver {
    inner: SelLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = SelLayoutSolver)]
impl JsSelLayoutSolver {
    /// Constructs a `SelLayoutSolver` from `graph` and optional `SelOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSelLayoutSolver, JsError> {
        let mut cfg = SelConfig::default();
        if check_options_arg(options, "SelLayoutSolver")? {
            cfg.tree_depth = opt_usize_prop(options, "treeDepth", cfg.tree_depth)?;
            cfg.pq_max_pops = opt_usize_prop(options, "pqMaxPops", cfg.pq_max_pops)?;
            if let Some(m_str) = opt_string_prop(options, "mapping")? {
                cfg.mapping = parse_sel_weight_mapping(&m_str)?;
            }
            if let Some(b_str) = opt_string_prop(options, "layoutBackend")? {
                cfg.layout_backend = parse_sel_layout_backend(&b_str)?;
            }
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SelLayoutSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one SEL layout iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` SEL layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SelLayoutSolver::run_steps",
        )
    }

    /// Runs SEL layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SelLayoutSolver::run",
        )
    }

    /// Returns the SEL-modulated edge weights aligned with canonical undirected edges `(u < v)` (`Float64Array`).
    #[wasm_bindgen(js_name = modulatedWeights)]
    pub fn modulated_weights(&self) -> Vec<f64> {
        self.inner.modulated_weights().to_vec()
    }

    /// Returns the elbow-selected structural entropy threshold $\mathrm{SE}_{\mathrm{target}}$.
    #[wasm_bindgen(getter, js_name = seTarget)]
    pub fn se_target(&self) -> f64 {
        self.inner.se_target()
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

    /// Returns the scheduled iteration count.
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
