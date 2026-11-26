//! WebAssembly bindings for GTree and FORBID node overlap removal solvers (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{
    ForbidConfig, ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn parse_gtree_config(options: &JsValue, ctx: &str) -> Result<GTreeConfig, JsError> {
    let mut cfg = GTreeConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.compact_t_max = opt_maybe_f64_prop(options, "compactTMax")?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Nachmanson et al. (2016) GTree overlap removal solver (`f32`).
///
/// Uses `Delaunay2d` when compiled with `delaunay` and falls back automatically to
/// `StaticKdTree` proximity under `--no-default-features`.
#[wasm_bindgen(js_name = GTreeOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsGTreeOverlapRemoval {
    inner: GTreeOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = GTreeOverlapRemoval)]
impl JsGTreeOverlapRemoval {
    /// Constructs a `GTreeOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    ///
    /// When `options.compactTMax` is set, it must satisfy `compactTMax > 1.0` (`null`/`undefined`
    /// disables compact expansion capping; values `<= 1.0` throw `Error`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsGTreeOverlapRemoval, JsError> {
        let cfg = parse_gtree_config(options, "GTreeOverlapRemoval")?;
        let inner =
            GTreeOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GTreeOverlapRemoval` solver with uniform square `radii`.
    ///
    /// When `options.compactTMax` is set, it must satisfy `compactTMax > 1.0` (`null`/`undefined`
    /// disables compact expansion capping; values `<= 1.0` throw `Error`).
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsGTreeOverlapRemoval, JsError> {
        let cfg = parse_gtree_config(options, "GTreeOverlapRemoval.fromRadii")?;
        let inner =
            GTreeOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GTree overlap removal pass on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GTree overlap removal passes on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GTreeOverlapRemoval::run_steps",
        )
    }

    /// Runs GTree overlap removal until `0` overlaps remain or `maxIterations` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GTreeOverlapRemoval::run",
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

    /// Returns the number of passes executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled passes (`maxIterations`).
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

fn parse_forbid_variant(raw: &str) -> Result<ForbidVariant, JsError> {
    match normalize_enum_token(raw).as_str() {
        "forbid" => Ok(ForbidVariant::Forbid),
        "forbidprime" => Ok(ForbidVariant::ForbidPrime),
        "sparseforbid" => Ok(ForbidVariant::SparseForbid),
        _ => Err(JsError::new(&format!(
            "unknown ForbidVariant {raw:?}; expected 'forbid', 'forbid_prime', or 'sparse_forbid'"
        ))),
    }
}

fn parse_forbid_config(options: &JsValue, ctx: &str) -> Result<ForbidConfig, JsError> {
    let mut cfg = ForbidConfig::default();
    if check_options_arg(options, ctx)? {
        if let Some(v_str) = opt_string_prop(options, "variant")? {
            cfg.variant = parse_forbid_variant(&v_str)?;
        }
        cfg.max_scale_iters = opt_usize_prop(options, "maxScaleIters", cfg.max_scale_iters)?;
        cfg.sgd_iterations = opt_usize_prop(options, "sgdIterations", cfg.sgd_iterations)?;
        cfg.sparse_k = opt_usize_prop(options, "sparseK", cfg.sparse_k)?;
        cfg.overlap_weight_k = opt_f64_prop(options, "overlapWeightK", cfg.overlap_weight_k)?;
        cfg.scale_step = opt_f64_prop(options, "scaleStep", cfg.scale_step)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative FORBID / FORBID' / SparseFORBID overlap removal solver (`f32`).
#[wasm_bindgen(js_name = ForbidOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsForbidOverlapRemoval {
    inner: ForbidOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = ForbidOverlapRemoval)]
impl JsForbidOverlapRemoval {
    /// Constructs a `ForbidOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsForbidOverlapRemoval, JsError> {
        let cfg = parse_forbid_config(options, "ForbidOverlapRemoval")?;
        let inner =
            ForbidOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `ForbidOverlapRemoval` solver with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsForbidOverlapRemoval, JsError> {
        let cfg = parse_forbid_config(options, "ForbidOverlapRemoval.fromRadii")?;
        let inner =
            ForbidOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one FORBID scaling + SGD round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` FORBID rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ForbidOverlapRemoval::run_steps",
        )
    }

    /// Runs FORBID overlap removal until `0` overlaps remain or `maxScaleIters` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ForbidOverlapRemoval::run",
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

    /// Returns the number of scaling rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled scaling rounds (`maxScaleIters`).
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
