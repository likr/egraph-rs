//! Cooperative `StressMajorizationSolver` (`f32`) WebAssembly bindings across
//! `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::ManifoldKind;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, unsupported_err};
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
use crate::stress_majorization_dispatch::{
    StressMajInner, build_stress_from_distances, build_stress_from_graph, parse_stress_options,
};

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver (`f32`)
/// across `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.
#[wasm_bindgen(js_name = StressMajorizationSolver)]
#[derive(Debug, Clone)]
pub struct JsStressMajorizationSolver {
    inner: StressMajInner,
    locked_manifold: Option<ManifoldKind>,
}

impl JsStressMajorizationSolver {
    fn from_graph_inner(graph: &JsCsrGraph, options: &JsValue, ctx: &str) -> Result<Self, JsValue> {
        let (cfg, target, backend) = parse_stress_options(options, ctx)?;
        let inner = build_stress_from_graph(graph, cfg, target, backend)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }
}

#[wasm_bindgen(js_class = StressMajorizationSolver)]
impl JsStressMajorizationSolver {
    /// Constructs a `StressMajorizationSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsStressMajorizationSolver, JsValue> {
        Self::from_graph_inner(graph, options, "StressMajorizationSolver")
    }

    /// Constructs a `StressMajorizationSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsStressMajorizationSolver, JsValue> {
        Self::from_graph_inner(graph, options, "StressMajorizationSolver.fromGraph")
    }

    /// Constructs a `StressMajorizationSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        options: &JsValue,
    ) -> Result<JsStressMajorizationSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (cfg, target, backend) =
            parse_stress_options(options, "StressMajorizationSolver.fromDistanceMatrix")?;
        let inner = build_stress_from_distances(n, dists_flat, cfg, target, backend)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim2Pcg(s) => step_layout_in_place(s, layout),
            StressMajInner::Dim2Geo(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim2Pcg(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps",
            ),
            StressMajInner::Dim2Geo(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim2Pcg(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run")
            }
            StressMajInner::Dim2Geo(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run")
            }
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim3Pcg(s) => step_layout3d_in_place(s, layout),
            StressMajInner::Dim3Geo(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim3Pcg(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps_3d",
            ),
            StressMajInner::Dim3Geo(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Dim3Pcg(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run_3d")
            }
            StressMajInner::Dim3Geo(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::DimNPcg(s) => step_nd_in_place(s, layout),
            StressMajInner::DimNGeo(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::DimNPcg(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps_nd",
            ),
            StressMajInner::DimNGeo(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "StressMajorizationSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::DimNPcg(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run_nd")
            }
            StressMajInner::DimNGeo(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "StressMajorizationSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Manifold(s) => {
                let n = s.num_nodes();
                step_manifold_in_place(
                    s,
                    layout,
                    &mut self.locked_manifold,
                    n,
                    "StressMajorizationSolver::step_manifold",
                )
            }
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "StressMajorizationSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            StressMajInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "StressMajorizationSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _computeStress2d)]
    pub fn compute_stress_2d(&self, layout: &JsLayout2d) -> Result<f64, JsValue> {
        let soa = layout.inner().to_layout2d();
        match &self.inner {
            StressMajInner::Dim2Pcg(s) => s.compute_stress(&soa).map_err(core_err_to_js),
            StressMajInner::Dim2Geo(s) => s.compute_stress(&soa).map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _computeStress3d)]
    pub fn compute_stress_3d(&self, layout: &JsLayout3d) -> Result<f64, JsValue> {
        let soa = layout.inner().to_layout3d();
        match &self.inner {
            StressMajInner::Dim3Pcg(s) => s.compute_stress(&soa).map_err(core_err_to_js),
            StressMajInner::Dim3Geo(s) => s.compute_stress(&soa).map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _computeStressNd)]
    pub fn compute_stress_nd(&self, layout: &JsLayoutNd) -> Result<f64, JsValue> {
        match &self.inner {
            StressMajInner::DimNPcg(s) => s.compute_stress(layout.inner()).map_err(core_err_to_js),
            StressMajInner::DimNGeo(s) => s.compute_stress(layout.inner()).map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _computeStressManifold)]
    pub fn compute_stress_manifold(&self, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
        match &self.inner {
            StressMajInner::Manifold(s) => s.compute_stress(layout.inner()).map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "StressMajorizationSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        self.inner.target_str().to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.inner.backend_str().to_owned()
    }

    #[wasm_bindgen(getter, js_name = lastStress)]
    pub fn last_stress(&self) -> Option<f64> {
        self.inner.last_stress()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        self.inner.total_iterations()
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}
