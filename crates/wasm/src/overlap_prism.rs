//! Cooperative Gansner-Hu (2010) PRISM proximity-stress overlap removal WebAssembly bindings (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{PrismConfig, PrismOverlapRemoval};
use wasm_bindgen::prelude::*;

use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, opt_f64_prop, opt_seed_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn parse_prism_config(options: &JsValue) -> Result<PrismConfig, JsError> {
    let mut cfg = PrismConfig::default();
    if check_options_arg(options, "PrismOverlapRemoval")? {
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.max_scale_factor = opt_f64_prop(options, "maxScaleFactor", cfg.max_scale_factor)?;
        cfg.cg_max_iters = opt_usize_prop(options, "cgMaxIters", cfg.cg_max_iters)?;
        cfg.cg_tolerance = opt_f64_prop(options, "cgTolerance", cfg.cg_tolerance)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress overlap removal solver (`f32`).
///
/// Uses `Delaunay2d` when compiled with `delaunay` and falls back automatically to
/// `StaticKdTree` $k$-NN proximity + `RectangleOverlapSweep` under `--no-default-features`.
#[wasm_bindgen(js_name = PrismOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsPrismOverlapRemoval {
    inner: PrismOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = PrismOverlapRemoval)]
impl JsPrismOverlapRemoval {
    /// Constructs a `PrismOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsPrismOverlapRemoval, JsError> {
        let cfg = parse_prism_config(options)?;
        let inner =
            PrismOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `PrismOverlapRemoval` solver for circular/square nodes with equal
    /// half-width and half-height given by `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsPrismOverlapRemoval, JsError> {
        let cfg = parse_prism_config(options)?;
        let inner =
            PrismOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one PRISM proximity-stress round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` PRISM rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "PrismOverlapRemoval::run_steps",
        )
    }

    /// Runs PRISM overlap removal until `0` overlaps remain or `maxIterations` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "PrismOverlapRemoval::run",
        )
    }

    /// Returns the number of overlapping rectangle pairs remaining after the most recent step.
    #[wasm_bindgen(getter, js_name = lastOverlapCount)]
    pub fn last_overlap_count(&self) -> usize {
        self.inner.last_overlap_count()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of PRISM rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled PRISM rounds (`maxIterations`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged (`0` overlaps) or completed.
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
