//! Cooperative stepping, execution, and getters for `GuMapSolver` WebAssembly bindings.

use wasm_bindgen::prelude::*;

use crate::gumap::{GuMapInner, JsGuMapSolver};
use crate::js_convert::unsupported_err;
use crate::js_options::solver_status_str;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_manifold_in_place, run_nd_in_place,
    run_steps_layout_in_place, run_steps_layout3d_in_place, run_steps_manifold_in_place,
    run_steps_nd_in_place, step_layout_in_place, step_layout3d_in_place, step_manifold_in_place,
    step_nd_in_place,
};

#[wasm_bindgen(js_class = GuMapSolver)]
impl JsGuMapSolver {
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim2(s) => {
                run_steps_layout_in_place(s.num_nodes(), s, layout, steps, "GuMapSolver::run_steps")
            }
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "GuMapSolver::run")
            }
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "GuMapSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "GuMapSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::DimN(s) => {
                run_steps_nd_in_place(s.num_nodes(), s, layout, steps, "GuMapSolver::run_steps_nd")
            }
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match self.inner_mut() {
            GuMapInner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "GuMapSolver::run_nd"),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            GuMapInner::Manifold(s) => {
                let n = s.num_nodes();
                step_manifold_in_place(s, layout, &mut locked, n, "GuMapSolver::step_manifold")
            }
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }

    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            GuMapInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut locked,
                "GuMapSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            GuMapInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut locked,
                "GuMapSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "GuMapSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match self.inner() {
            GuMapInner::Dim2(_) => "layout2d",
            GuMapInner::Dim3(_) => "layout3d",
            GuMapInner::DimN(_) => "layoutNd",
            GuMapInner::Manifold(_) => "manifold2d",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        match self.inner() {
            GuMapInner::Dim2(s) => s.num_edges(),
            GuMapInner::Dim3(s) => s.num_edges(),
            GuMapInner::DimN(s) => s.num_edges(),
            GuMapInner::Manifold(s) => s.num_edges(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        match self.inner() {
            GuMapInner::Dim2(s) => s.ab().0,
            GuMapInner::Dim3(s) => s.ab().0,
            GuMapInner::DimN(s) => s.ab().0,
            GuMapInner::Manifold(s) => s.ab().0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        match self.inner() {
            GuMapInner::Dim2(s) => s.ab().1,
            GuMapInner::Dim3(s) => s.ab().1,
            GuMapInner::DimN(s) => s.ab().1,
            GuMapInner::Manifold(s) => s.ab().1,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = match self.inner() {
            GuMapInner::Dim2(s) => s.ab(),
            GuMapInner::Dim3(s) => s.ab(),
            GuMapInner::DimN(s) => s.ab(),
            GuMapInner::Manifold(s) => s.ab(),
        };
        vec![a, b]
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match self.inner() {
            GuMapInner::Dim2(s) => s.status(),
            GuMapInner::Dim3(s) => s.status(),
            GuMapInner::DimN(s) => s.status(),
            GuMapInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match self.inner() {
            GuMapInner::Dim2(s) => s.iteration(),
            GuMapInner::Dim3(s) => s.iteration(),
            GuMapInner::DimN(s) => s.iteration(),
            GuMapInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match self.inner() {
            GuMapInner::Dim2(s) => s.total_iterations(),
            GuMapInner::Dim3(s) => s.total_iterations(),
            GuMapInner::DimN(s) => s.total_iterations(),
            GuMapInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match self.inner() {
            GuMapInner::Dim2(s) => s.status(),
            GuMapInner::Dim3(s) => s.status(),
            GuMapInner::DimN(s) => s.status(),
            GuMapInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match self.inner() {
            GuMapInner::Dim2(s) => s.num_nodes(),
            GuMapInner::Dim3(s) => s.num_nodes(),
            GuMapInner::DimN(s) => s.num_nodes(),
            GuMapInner::Manifold(s) => s.num_nodes(),
        }
    }
}
