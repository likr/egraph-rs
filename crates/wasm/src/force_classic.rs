//! WebAssembly bindings for classic force-directed layout solvers (Noack LinLog and
//! Fruchterman-Reingold; `f32`-backed layouts with `f64` internal solver math).

use egraph::layout::{
    FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver, LinLogConfig, LinLogRepulsion,
    LinLogSolver,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_manifold_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_manifold_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_manifold_in_place, step_nd_in_place,
};

fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, JsError> {
    match normalize_enum_token(raw).as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(JsError::new(&format!(
            "unknown LinLogRepulsion {raw:?}; expected 'edge_repulsion' or 'node_repulsion'"
        ))),
    }
}

/// Cooperative Noack (2007/2009) LinLog / $(a, r)$-PolyLog force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = LinLogSolver)]
#[derive(Debug, Clone)]
pub struct JsLinLogSolver {
    inner: LinLogSolver<f32>,
}

#[wasm_bindgen(js_class = LinLogSolver)]
impl JsLinLogSolver {
    /// Constructs a `LinLogSolver` from `graph` and optional `LinLogOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLinLogSolver, JsError> {
        let mut cfg = LinLogConfig::default();
        if check_options_arg(options, "LinLogSolver")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            if let Some(rep_str) = opt_string_prop(options, "repulsionModel")? {
                cfg.repulsion_model = parse_linlog_repulsion(&rep_str)?;
            }
            cfg.attraction_exponent =
                opt_f64_prop(options, "attractionExponent", cfg.attraction_exponent)?;
            cfg.repulsion_exponent =
                opt_f64_prop(options, "repulsionExponent", cfg.repulsion_exponent)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.max_displacement = opt_maybe_f64_prop(options, "maxDisplacement")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = LinLogSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one LinLog iteration on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` LinLog iterations on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "LinLogSolver::run_steps",
        )
    }

    /// Runs LinLog to completion on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "LinLogSolver::run",
        )
    }

    /// Executes one LinLog iteration on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` LinLog iterations on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        run_steps_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "LinLogSolver::run_steps_nd",
        )
    }

    /// Runs LinLog to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        run_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "LinLogSolver::run_nd",
        )
    }

    /// Evaluates the exact LinLog energy of `Layout2d`.
    #[wasm_bindgen(js_name = _energy2d)]
    pub fn energy_2d(&self, layout: &JsLayout2d) -> Result<f64, JsError> {
        let soa = layout.inner().to_layout2d();
        self.inner.energy(&soa).map_err(core_err_to_js)
    }

    /// Evaluates the exact LinLog energy of `LayoutNd`.
    #[wasm_bindgen(js_name = _energyNd)]
    pub fn energy_nd(&self, layout: &JsLayoutNd) -> Result<f64, JsError> {
        self.inner.energy_nd(layout.inner()).map_err(core_err_to_js)
    }

    /// Returns the per-node repulsive masses (`Float64Array`).
    pub fn masses(&self) -> Vec<f64> {
        self.inner.masses().to_vec()
    }

    /// Returns the effective per-step vertex displacement cap.
    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        self.inner.max_displacement()
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

fn parse_fr_backend(raw: &str) -> Result<FrBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exact" => Ok(FrBackend::Exact),
        "barneshut" | "bh" => Ok(FrBackend::BarnesHut),
        "grid" => Ok(FrBackend::Grid),
        _ => Err(JsError::new(&format!(
            "unknown FrBackend {raw:?}; expected 'exact', 'barnes_hut', or 'grid'"
        ))),
    }
}

pub(crate) fn parse_fr_config(
    options: &JsValue,
    ctx: &str,
) -> Result<FruchtermanReingoldConfig, JsError> {
    let mut cfg = FruchtermanReingoldConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        if let Some(b_str) = opt_string_prop(options, "backend")? {
            cfg.backend = parse_fr_backend(&b_str)?;
        }
        cfg.k = opt_maybe_f64_prop(options, "k")?;
        cfg.area = opt_maybe_f64_prop(options, "area")?;
        cfg.initial_temperature = opt_maybe_f64_prop(options, "initialTemperature")?;
        cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Fruchterman-Reingold (1991) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = FruchtermanReingoldSolver)]
#[derive(Debug, Clone)]
pub struct JsFruchtermanReingoldSolver {
    inner: FruchtermanReingoldSolver<f32>,
}

#[wasm_bindgen(js_class = FruchtermanReingoldSolver)]
impl JsFruchtermanReingoldSolver {
    /// Constructs a `FruchtermanReingoldSolver` from `graph` and optional `FruchtermanReingoldOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsFruchtermanReingoldSolver, JsError> {
        let cfg = parse_fr_config(options, "FruchtermanReingoldSolver")?;
        let inner =
            FruchtermanReingoldSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Fruchterman-Reingold iteration on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Fruchterman-Reingold iterations on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "FruchtermanReingoldSolver::run_steps",
        )
    }

    /// Runs Fruchterman-Reingold to completion on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FruchtermanReingoldSolver::run",
        )
    }

    /// Executes one Fruchterman-Reingold iteration on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Fruchterman-Reingold iterations on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        run_steps_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "FruchtermanReingoldSolver::run_steps_nd",
        )
    }

    /// Runs Fruchterman-Reingold to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        run_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FruchtermanReingoldSolver::run_nd",
        )
    }

    /// Executes one Fruchterman-Reingold geodesic step on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        step_manifold_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Fruchterman-Reingold geodesic steps on `ManifoldLayout2d` in place.
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
            "FruchtermanReingoldSolver::run_steps_manifold",
        )
    }

    /// Runs Fruchterman-Reingold to completion on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        run_manifold_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FruchtermanReingoldSolver::run_manifold",
        )
    }

    /// Returns the effective optimal pairwise distance $k$.
    #[wasm_bindgen(getter, js_name = optimalDistance)]
    pub fn optimal_distance(&self) -> f64 {
        self.inner.optimal_distance()
    }

    /// Returns the current cooling temperature.
    #[wasm_bindgen(getter, js_name = currentTemperature)]
    pub fn current_temperature(&self) -> f64 {
        self.inner.current_temperature()
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
