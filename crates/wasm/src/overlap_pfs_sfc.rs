//! WebAssembly bindings for `PFS'` node overlap removal and canonical community-ordered
//! Space-Filling Curve (`SFC`) layout solvers (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{
    PfsPrimeConfig, PfsPrimeOverlapRemoval, SfcConfig, SfcCurveKind, SfcLayoutSolver,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_u32};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_f64_prop, opt_seed_prop,
    opt_string_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn opt_maybe_u32_prop(obj: &JsValue, key: &str) -> Result<Option<u32>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_u32(num, key).map(Some)
}

fn parse_pfs_prime_config(options: &JsValue, ctx: &str) -> Result<PfsPrimeConfig, JsError> {
    let mut cfg = PfsPrimeConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal solver (`f32`).
#[wasm_bindgen(js_name = PfsPrimeOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsPfsPrimeOverlapRemoval {
    inner: PfsPrimeOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = PfsPrimeOverlapRemoval)]
impl JsPfsPrimeOverlapRemoval {
    /// Constructs a `PfsPrimeOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsError> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval")?;
        let inner =
            PfsPrimeOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `PfsPrimeOverlapRemoval` solver with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(
        radii: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsError> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval.fromRadii")?;
        let inner = PfsPrimeOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes the single two-pass `PFS'` overlap removal step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `PFS'` passes on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "PfsPrimeOverlapRemoval::run_steps",
        )
    }

    /// Runs `PFS'` overlap removal to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "PfsPrimeOverlapRemoval::run",
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

    /// Returns the scheduled pass count (`1`).
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

fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(JsError::new(&format!(
            "unknown SfcCurveKind {raw:?}; expected 'hilbert' or 'morton'"
        ))),
    }
}

/// Canonical community-ordered Space-Filling Curve (`SFC`) layout solver (`f32`).
///
/// Note: `muelder2008_space_filling` (Muelder & Ma, IEEE TVCG 2008) is currently marked
/// `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
/// post-refinement parameters from that paper cannot be verified or claimed. This solver
/// exposes a canonical community-ordered Hilbert/Morton space-filling curve baseline using
/// `louvain_coarsen` and strictly injective 2D curve mapping (`totalIterations == 1`).
#[wasm_bindgen(js_name = SfcLayoutSolver)]
#[derive(Debug, Clone)]
pub struct JsSfcLayoutSolver {
    inner: SfcLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = SfcLayoutSolver)]
impl JsSfcLayoutSolver {
    /// Constructs an `SfcLayoutSolver` from `graph` and optional `SfcOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSfcLayoutSolver, JsError> {
        let mut cfg = SfcConfig::default();
        if check_options_arg(options, "SfcLayoutSolver")? {
            if let Some(c_str) = opt_string_prop(options, "curve")? {
                cfg.curve = parse_sfc_curve_kind(&c_str)?;
            }
            cfg.scale = opt_f64_prop(options, "scale", cfg.scale)?;
            cfg.resolution_order = opt_maybe_u32_prop(options, "resolutionOrder")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SfcLayoutSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes the single `SFC` coordinate assignment step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `SFC` steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SfcLayoutSolver::run_steps",
        )
    }

    /// Runs `SFC` coordinate assignment to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SfcLayoutSolver::run",
        )
    }

    /// Returns the deterministic community-ordered vertex permutation (`Uint32Array`).
    #[wasm_bindgen(js_name = vertexOrder)]
    pub fn vertex_order(&self) -> Vec<u32> {
        self.inner.vertex_order().to_vec()
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

    /// Returns the scheduled step count (`1`).
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
