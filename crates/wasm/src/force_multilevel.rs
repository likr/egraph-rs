//! Cooperative ForceAtlas2 and multilevel Yifan Hu WebAssembly bindings (`f32`).

use egraph::layout::{ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_optional_f64_sequence};
use crate::js_options::{
    check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

/// Cooperative ForceAtlas2 (Jacomy et al., 2014) force-directed layout solver (`f32`).
///
/// Note: `linlogMode` enables ForceAtlas2 logarithmic attraction
/// ($w \frac{\ln(1 + d)}{d} (p_v - p_u)$, decaying linearly to `0` as $d \to 0$) and is
/// distinct from Noack (2007) true LinLog (`LinLogSolver`).
#[wasm_bindgen(js_name = ForceAtlas2Solver)]
#[derive(Debug, Clone)]
pub struct JsForceAtlas2Solver {
    inner: ForceAtlas2Solver<f32>,
}

#[wasm_bindgen(js_class = ForceAtlas2Solver)]
impl JsForceAtlas2Solver {
    /// Constructs a `ForceAtlas2Solver` from `graph` and optional `ForceAtlas2Options`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsForceAtlas2Solver, JsError> {
        let mut cfg = ForceAtlas2Config::default();
        let mut node_radii: Option<Vec<f64>> = None;
        if check_options_arg(options, "ForceAtlas2Solver")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.linlog_mode = opt_bool_prop(options, "linlogMode", cfg.linlog_mode)?;
            cfg.strong_gravity = opt_bool_prop(options, "strongGravity", cfg.strong_gravity)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.scaling_ratio = opt_f64_prop(options, "scalingRatio", cfg.scaling_ratio)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.outbound_attraction_distribution = opt_bool_prop(
                options,
                "outboundAttractionDistribution",
                cfg.outbound_attraction_distribution,
            )?;
            cfg.jitter_tolerance = opt_f64_prop(options, "jitterTolerance", cfg.jitter_tolerance)?;
            cfg.max_displacement = opt_maybe_f64_prop(options, "maxDisplacement")?;
            cfg.prevent_overlap = opt_bool_prop(options, "preventOverlap", cfg.prevent_overlap)?;
            cfg.default_node_radius =
                opt_f64_prop(options, "defaultNodeRadius", cfg.default_node_radius)?;
            node_radii =
                parse_optional_f64_sequence(&get_prop(options, "nodeRadii")?, "nodeRadii")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match node_radii {
            Some(radii) => ForceAtlas2Solver::<f32>::with_node_radii(graph.inner(), cfg, radii),
            None => ForceAtlas2Solver::<f32>::new(graph.inner(), cfg),
        }
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one ForceAtlas2 iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` ForceAtlas2 iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ForceAtlas2Solver::run_steps",
        )
    }

    /// Runs ForceAtlas2 to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ForceAtlas2Solver::run",
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

    /// Returns the per-node collision radii (`Float64Array`).
    #[wasm_bindgen(js_name = nodeRadii)]
    pub fn node_radii(&self) -> Vec<f64> {
        self.inner.node_radii().to_vec()
    }

    /// Returns the effective per-step vertex displacement cap.
    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        self.inner.max_displacement()
    }
}

/// Cooperative multilevel Yifan Hu (2005) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = YifanHuSolver)]
#[derive(Debug, Clone)]
pub struct JsYifanHuSolver {
    inner: YifanHuSolver<f32>,
}

#[wasm_bindgen(js_class = YifanHuSolver)]
impl JsYifanHuSolver {
    /// Constructs a `YifanHuSolver` from `graph` and optional `YifanHuOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsYifanHuSolver, JsError> {
        let mut cfg = YifanHuConfig::default();
        if check_options_arg(options, "YifanHuSolver")? {
            cfg.max_iterations_per_level = opt_usize_prop(
                options,
                "maxIterationsPerLevel",
                cfg.max_iterations_per_level,
            )?;
            cfg.step_size = opt_f64_prop(options, "stepSize", cfg.step_size)?;
            cfg.step_ratio = opt_f64_prop(options, "stepRatio", cfg.step_ratio)?;
            cfg.relative_strength =
                opt_f64_prop(options, "relativeStrength", cfg.relative_strength)?;
            cfg.optimal_distance = opt_maybe_f64_prop(options, "optimalDistance")?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.min_coarsen_size = opt_usize_prop(options, "minCoarsenSize", cfg.min_coarsen_size)?;
            cfg.min_coarsen_ratio =
                opt_f64_prop(options, "minCoarsenRatio", cfg.min_coarsen_ratio)?;
            cfg.convergence_tolerance =
                opt_f64_prop(options, "convergenceTolerance", cfg.convergence_tolerance)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = YifanHuSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Yifan Hu iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Yifan Hu iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "YifanHuSolver::run_steps",
        )
    }

    /// Runs Yifan Hu multilevel optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "YifanHuSolver::run",
        )
    }

    /// Returns the number of levels in the `EC + MIVS` coarsening hierarchy (`>= 1`).
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Returns the currently active level index (`numLevels - 1` coarsest down to `0` finest).
    #[wasm_bindgen(getter, js_name = currentLevel)]
    pub fn current_level(&self) -> usize {
        self.inner.current_level()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the total iterations executed so far across all levels.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled iterations (`numLevels * maxIterationsPerLevel`).
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
