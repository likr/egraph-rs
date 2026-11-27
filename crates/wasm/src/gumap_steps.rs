//! Cooperative stepping, execution, and getters for `GuMapSolver` WebAssembly bindings.

use wasm_bindgen::prelude::*;

use crate::gumap::JsGuMapSolver;
use crate::js_options::solver_status_str;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_manifold_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_manifold_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_manifold_in_place, step_nd_in_place,
};

#[wasm_bindgen(js_class = GuMapSolver)]
impl JsGuMapSolver {
    /// Executes one epoch on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` epochs on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_steps_layout_in_place(n, self.inner_mut(), layout, steps, "GuMapSolver::run_steps")
    }

    /// Runs GuMap optimization to completion on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_layout_in_place(n, self.inner_mut(), layout, "GuMapSolver::run")
    }

    /// Executes one epoch on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` epochs on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_steps_nd_in_place(
            n,
            self.inner_mut(),
            layout,
            steps,
            "GuMapSolver::run_steps_nd",
        )
    }

    /// Runs GuMap optimization to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_nd_in_place(n, self.inner_mut(), layout, "GuMapSolver::run_nd")
    }

    /// Executes one epoch on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        step_manifold_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` epochs on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_steps_manifold_in_place(
            n,
            self.inner_mut(),
            layout,
            steps,
            "GuMapSolver::run_steps_manifold",
        )
    }

    /// Runs GuMap optimization to completion on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        let n = self.inner().num_nodes();
        run_manifold_in_place(n, self.inner_mut(), layout, "GuMapSolver::run_manifold")
    }

    /// Returns the number of directed positive attractive edges stored.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner().num_edges()
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner().ab().0
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.inner().ab().1
    }

    /// Returns the resolved UMAP curve parameters `[a, b]`.
    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = self.inner().ab();
        vec![a, b]
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner().status()).to_owned()
    }

    /// Returns the number of epochs executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner().iteration()
    }

    /// Returns the total scheduled epoch count (`iterations`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner().total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner().status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner().num_nodes()
    }
}
