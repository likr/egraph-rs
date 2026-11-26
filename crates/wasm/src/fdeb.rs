//! Reference-passing cooperative Force-Directed Edge Bundling (`FDEB`) WebAssembly bindings (`f32`).

use egraph::core::{CoreError, IterativeSolver, Layout2d};
use egraph::layout::{FdebConfig, FdebSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_usize_prop, solver_status_str};
use crate::layout_buffers::JsLayout2d;

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver (`f32`) storing flat edge
/// polyline buffers without `serde` or consuming input `Layout2d` handles.
#[wasm_bindgen(js_name = FdebSolver)]
#[derive(Debug, Clone)]
pub struct JsFdebSolver {
    inner: FdebSolver<f32>,
    layout: Layout2d<f32>,
}

#[wasm_bindgen(js_class = FdebSolver)]
impl JsFdebSolver {
    /// Constructs an `FdebSolver` from `graph` and `layout` by shared reference.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        layout: &JsLayout2d,
        options: &JsValue,
    ) -> Result<JsFdebSolver, JsError> {
        let mut cfg = FdebConfig::default();
        if check_options_arg(options, "FdebSolver")? {
            cfg.cycles = opt_usize_prop(options, "cycles", cfg.cycles)?;
            cfg.s0 = opt_f64_prop(options, "s0", cfg.s0)?;
            cfg.steps_per_cycle = opt_usize_prop(options, "i0", cfg.steps_per_cycle)?;
            cfg.s_step = opt_f64_prop(options, "sStep", cfg.s_step)?;
            cfg.i_step = opt_f64_prop(options, "iStep", cfg.i_step)?;
            cfg.spring_k = opt_f64_prop(options, "springK", cfg.spring_k)?;
            cfg.minimum_edge_compatibility = opt_f64_prop(
                options,
                "minimumEdgeCompatibility",
                cfg.minimum_edge_compatibility,
            )?;
        }
        let soa = layout.inner().to_layout2d();
        soa.validate_finite().map_err(core_err_to_js)?;
        if soa.len() != graph.inner().num_nodes() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                graph.inner().num_nodes(),
                soa.len(),
                "FdebSolver::new",
            )));
        }
        let inner = FdebSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner, layout: soa })
    }

    /// Executes one FDEB iteration and returns the updated solver status string.
    pub fn step(&mut self) -> Result<String, JsError> {
        let status = self.inner.step(&mut self.layout).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` FDEB iterations and returns the updated solver status string.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, steps: f64) -> Result<String, JsError> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        let status = self
            .inner
            .run_steps(&mut self.layout, n_steps)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs FDEB edge bundling to completion and returns the final solver status string.
    pub fn run(&mut self) -> Result<String, JsError> {
        let status = self.inner.run(&mut self.layout).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
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

    /// Returns the number of edges being bundled.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
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
}
