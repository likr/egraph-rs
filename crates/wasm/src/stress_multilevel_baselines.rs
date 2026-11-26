//! WebAssembly bindings for Maxent-Stress and GRIP multilevel layout solvers (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{GripConfig, GripSolver, MaxentStressConfig, MaxentStressSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver (`f32`).
#[wasm_bindgen(js_name = MaxentStressSolver)]
#[derive(Debug, Clone)]
pub struct JsMaxentStressSolver {
    inner: MaxentStressSolver<f32>,
}

#[wasm_bindgen(js_class = MaxentStressSolver)]
impl JsMaxentStressSolver {
    /// Constructs a `MaxentStressSolver` from `graph` and optional `MaxentStressOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsMaxentStressSolver, JsError> {
        let mut cfg = MaxentStressConfig::default();
        if check_options_arg(options, "MaxentStressSolver")? {
            cfg.q = opt_maybe_f64_prop(options, "q")?;
            cfg.alpha_init = opt_f64_prop(options, "alphaInit", cfg.alpha_init)?;
            cfg.alpha_decay = opt_f64_prop(options, "alphaDecay", cfg.alpha_decay)?;
            cfg.alpha_min = opt_f64_prop(options, "alphaMin", cfg.alpha_min)?;
            cfg.stages = opt_usize_prop(options, "stages", cfg.stages)?;
            cfg.steps_per_stage = opt_usize_prop(options, "stepsPerStage", cfg.steps_per_stage)?;
            cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
            cfg.cg_tol = opt_f64_prop(options, "cgTol", cfg.cg_tol)?;
            cfg.cg_max_iters = opt_usize_prop(options, "cgMaxIters", cfg.cg_max_iters)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.eps_sq = opt_f64_prop(options, "epsSq", cfg.eps_sq)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = MaxentStressSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Maxent-Stress iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Maxent-Stress iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "MaxentStressSolver::run_steps",
        )
    }

    /// Runs Maxent-Stress optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "MaxentStressSolver::run",
        )
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

    /// Returns the scheduled iteration count (`stages * stepsPerStage`).
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

    /// Returns the current annealed entropy-repulsion weight $\alpha$.
    #[wasm_bindgen(getter, js_name = currentAlpha)]
    pub fn current_alpha(&self) -> f64 {
        self.inner.current_alpha()
    }

    /// Returns the resolved repulsion exponent $q$.
    #[wasm_bindgen(getter, js_name = effectiveQ)]
    pub fn effective_q(&self) -> f64 {
        self.inner.effective_q()
    }
}

/// Cooperative Gajer & Kobourov (2002) GRIP multilevel force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = GripSolver)]
#[derive(Debug, Clone)]
pub struct JsGripSolver {
    inner: GripSolver<f32>,
}

#[wasm_bindgen(js_class = GripSolver)]
impl JsGripSolver {
    /// Constructs a `GripSolver` from `graph` and optional `GripOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGripSolver, JsError> {
        let mut cfg = GripConfig::default();
        if check_options_arg(options, "GripSolver")? {
            cfg.rounds_min = opt_usize_prop(options, "roundsMin", cfg.rounds_min)?;
            cfg.rounds_max = opt_usize_prop(options, "roundsMax", cfg.rounds_max)?;
            cfg.local_neighborhood_size = opt_usize_prop(
                options,
                "localNeighborhoodSize",
                cfg.local_neighborhood_size,
            )?;
            cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.temp_r = opt_f64_prop(options, "tempR", cfg.temp_r)?;
            cfg.temp_s = opt_f64_prop(options, "tempS", cfg.temp_s)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GripSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GRIP refinement step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GRIP refinement steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GripSolver::run_steps",
        )
    }

    /// Runs GRIP layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GripSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of steps executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled step count across filtration levels.
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

    /// Returns the maximum number of MIS filtration levels across connected components.
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }
}
