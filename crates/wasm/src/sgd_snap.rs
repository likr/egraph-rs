//! Cooperative compact-memory `SNAP-tFDP` (`SnapTFdpSolver`, `f32`) WebAssembly bindings
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{SnapTFdpConfig, SnapTFdpSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_bool_prop, opt_f64_prop, opt_seed_prop, opt_target_prop,
    opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::sgd::target_kind_str;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_manifold_in_place, run_nd_in_place,
    run_steps_layout_in_place, run_steps_layout3d_in_place, run_steps_manifold_in_place,
    run_steps_nd_in_place, step_layout_in_place, step_layout3d_in_place, step_manifold_in_place,
    step_nd_in_place,
};

#[derive(Debug, Clone)]
enum SnapInner {
    Dim2(SnapTFdpSolver<Layout2d<f32>>),
    Dim3(SnapTFdpSolver<Layout3d<f32>>),
    DimN(SnapTFdpSolver<LayoutNd<f32>>),
    Manifold(SnapTFdpSolver<ManifoldLayout2d<f32>>),
}

fn parse_snap_options(options: &JsValue) -> Result<(SnapTFdpConfig, JsTargetKind), JsValue> {
    let mut cfg = SnapTFdpConfig::default();
    let mut target = JsTargetKind::Layout2d;
    if check_options_arg(options, "SnapTFdpSolver")? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.negatives_per_edge =
            opt_usize_prop(options, "negativesPerEdge", cfg.negatives_per_edge)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
        cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
        cfg.rho_start = opt_f64_prop(options, "rhoStart", cfg.rho_start)?;
        cfg.rho_end = opt_f64_prop(options, "rhoEnd", cfg.rho_end)?;
        cfg.rho_power = opt_f64_prop(options, "rhoPower", cfg.rho_power)?;
        cfg.degree_beta = opt_f64_prop(options, "degreeBeta", cfg.degree_beta)?;
        cfg.bounded_attraction =
            opt_bool_prop(options, "boundedAttraction", cfg.bounded_attraction)?;
        cfg.seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
    }
    Ok((cfg, target))
}

/// Cooperative compact-memory edge-incident negative sampling `SNAP-tFDP` solver (`f32`).
#[wasm_bindgen(js_name = SnapTFdpSolver)]
#[derive(Debug, Clone)]
pub struct JsSnapTFdpSolver {
    inner: SnapInner,
    locked_manifold: Option<ManifoldKind>,
}

#[wasm_bindgen(js_class = SnapTFdpSolver)]
impl JsSnapTFdpSolver {
    /// Constructs a `SnapTFdpSolver` from `graph` and `SnapTFdpOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSnapTFdpSolver, JsValue> {
        let (cfg, target) = parse_snap_options(options)?;
        let inner = match target {
            JsTargetKind::Layout2d => SnapInner::Dim2(
                SnapTFdpSolver::<Layout2d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SnapInner::Dim3(
                SnapTFdpSolver::<Layout3d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SnapInner::DimN(
                SnapTFdpSolver::<LayoutNd<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SnapInner::Manifold(
                SnapTFdpSolver::<ManifoldLayout2d<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "SnapTFdpSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "SnapTFdpSolver::run")
            }
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "SnapTFdpSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "SnapTFdpSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "SnapTFdpSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "SnapTFdpSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "SnapTFdpSolver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'manifold2d'",
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
            SnapInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "SnapTFdpSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SnapInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "SnapTFdpSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "SnapTFdpSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        let kind = match &self.inner {
            SnapInner::Dim2(_) => JsTargetKind::Layout2d,
            SnapInner::Dim3(_) => JsTargetKind::Layout3d,
            SnapInner::DimN(_) => JsTargetKind::LayoutNd,
            SnapInner::Manifold(_) => JsTargetKind::Manifold2d,
        };
        target_kind_str(kind).to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            SnapInner::Dim2(s) => s.num_nodes(),
            SnapInner::Dim3(s) => s.num_nodes(),
            SnapInner::DimN(s) => s.num_nodes(),
            SnapInner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        match &self.inner {
            SnapInner::Dim2(s) => s.num_springs(),
            SnapInner::Dim3(s) => s.num_springs(),
            SnapInner::DimN(s) => s.num_springs(),
            SnapInner::Manifold(s) => s.num_springs(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            SnapInner::Dim2(s) => s.status(),
            SnapInner::Dim3(s) => s.status(),
            SnapInner::DimN(s) => s.status(),
            SnapInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            SnapInner::Dim2(s) => s.iteration(),
            SnapInner::Dim3(s) => s.iteration(),
            SnapInner::DimN(s) => s.iteration(),
            SnapInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            SnapInner::Dim2(s) => s.total_iterations(),
            SnapInner::Dim3(s) => s.total_iterations(),
            SnapInner::DimN(s) => s.total_iterations(),
            SnapInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            SnapInner::Dim2(s) => s.status(),
            SnapInner::Dim3(s) => s.status(),
            SnapInner::DimN(s) => s.status(),
            SnapInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }
}
