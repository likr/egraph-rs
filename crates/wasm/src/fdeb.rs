//! Cooperative Force-Directed Edge Bundling (`FDEB`) WebAssembly bindings (`f32`)
//! supporting polymorphic `Layout2d` and `LayoutNd` (including 3D) bundling.

use egraph::layout::{FdebConfig, FdebSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_usize_prop, solver_status_str};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::solver_runner::{
    run_layout_in_place, run_nd_in_place, run_steps_layout_in_place, run_steps_nd_in_place,
    step_layout_in_place, step_nd_in_place,
};

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver (`f32`) storing flat edge
/// polyline buffers across `Layout2d` and `LayoutNd`.
#[wasm_bindgen(js_name = FdebSolver)]
#[derive(Debug, Clone)]
pub struct JsFdebSolver {
    inner: FdebSolver<f32>,
}

#[wasm_bindgen(js_class = FdebSolver)]
impl JsFdebSolver {
    /// Constructs an `FdebSolver` from `graph` and optional `FdebOptions` (`stepsPerCycle`).
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsFdebSolver, JsError> {
        let mut cfg = FdebConfig::default();
        if check_options_arg(options, "FdebSolver")? {
            cfg.cycles = opt_usize_prop(options, "cycles", cfg.cycles)?;
            cfg.s0 = opt_f64_prop(options, "s0", cfg.s0)?;
            cfg.steps_per_cycle = opt_usize_prop(options, "stepsPerCycle", cfg.steps_per_cycle)?;
            cfg.s_step = opt_f64_prop(options, "sStep", cfg.s_step)?;
            cfg.i_step = opt_f64_prop(options, "iStep", cfg.i_step)?;
            cfg.spring_k = opt_f64_prop(options, "springK", cfg.spring_k)?;
            cfg.minimum_edge_compatibility = opt_f64_prop(
                options,
                "minimumEdgeCompatibility",
                cfg.minimum_edge_compatibility,
            )?;
        }
        let inner = FdebSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one FDEB iteration using `Layout2d`.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` FDEB iterations using `Layout2d`.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "FdebSolver::run_steps",
        )
    }

    /// Runs FDEB edge bundling to completion using `Layout2d`.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FdebSolver::run",
        )
    }

    /// Executes one FDEB iteration using `LayoutNd`.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` FDEB iterations using `LayoutNd`.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        run_steps_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "FdebSolver::run_steps_nd",
        )
    }

    /// Runs FDEB edge bundling to completion using `LayoutNd`.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        run_nd_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FdebSolver::run_nd",
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

    /// Returns the total scheduled iterations across all subdivision cycles.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when edge bundling has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of edges being bundled.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the coordinate dimensionality of bundled polylines (`0` before first `step`).
    #[wasm_bindgen(getter, js_name = pointDims)]
    pub fn point_dims(&self) -> usize {
        self.inner.point_dims()
    }

    /// Returns CSR-style polyline point offsets of length `numEdges + 1` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeOffsets)]
    pub fn edge_offsets(&self) -> Vec<u32> {
        self.inner.edge_offsets().to_vec()
    }

    /// Returns all bundled polyline control points packed as flat interleaved `[x0, y0, ...]` (`Float32Array`).
    pub fn points(&self) -> Vec<f32> {
        self.inner.to_interleaved_points()
    }

    /// Returns flat row-major N-D coordinates of all edge control points (`Float32Array`).
    #[wasm_bindgen(js_name = pointsNd)]
    pub fn points_nd(&self) -> Vec<f32> {
        self.inner.to_interleaved_points_nd()
    }

    /// Extracts a single coordinate `axis` (`0..pointDims`) across all bundled edge control points.
    #[wasm_bindgen(js_name = pointsAxis)]
    pub fn points_axis(&self, axis: f64) -> Result<Vec<f32>, JsError> {
        let ax = parse_non_negative_usize(axis, "axis")?;
        self.inner.points_axis(ax).map_err(core_err_to_js)
    }

    /// Extracts the `z` coordinate (`axis === 2`) across all 3D/N-D bundled edge control points.
    #[wasm_bindgen(js_name = pointsZ)]
    pub fn points_z(&self) -> Result<Vec<f32>, JsError> {
        self.inner.points_z().map_err(core_err_to_js)
    }

    /// Returns all edge polyline points packed as row-major interleaved coordinates (`Float32Array`).
    #[wasm_bindgen(js_name = toInterleavedPointsNd)]
    pub fn to_interleaved_points_nd(&self) -> Vec<f32> {
        self.inner.to_interleaved_points_nd()
    }
}
