//! Cooperative multilevel negative-sampling `DRGraph` (Zhu et al., 2020) bindings (`f32`)
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{DrGraphConfig, DrGraphSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_f64_prop, opt_seed_prop, opt_target_prop, opt_usize_prop,
    solver_status_str,
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
enum DrGraphInner {
    Dim2(DrGraphSolver<Layout2d<f32>>),
    Dim3(DrGraphSolver<Layout3d<f32>>),
    DimN(DrGraphSolver<LayoutNd<f32>>),
    Manifold(DrGraphSolver<ManifoldLayout2d<f32>>),
}

fn parse_drgraph_options(options: &JsValue) -> Result<(DrGraphConfig, JsTargetKind), JsValue> {
    let mut cfg = DrGraphConfig::default();
    let mut target = JsTargetKind::Layout2d;
    if check_options_arg(options, "DrGraphSolver")? {
        cfg.k_order = opt_usize_prop(options, "kOrder", cfg.k_order)?;
        cfg.negative_samples = opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
        cfg.b = opt_f64_prop(options, "b", cfg.b)?;
        cfg.gamma_coarse = opt_f64_prop(options, "gammaCoarse", cfg.gamma_coarse)?;
        cfg.gamma_fine = opt_f64_prop(options, "gammaFine", cfg.gamma_fine)?;
        cfg.coarsen_stop_ratio = opt_f64_prop(options, "coarsenStopRatio", cfg.coarsen_stop_ratio)?;
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
        cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
        cfg.seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
    }
    Ok((cfg, target))
}

/// Cooperative multilevel negative-sampling `DRGraph` (Zhu et al., 2020) layout solver (`f32`).
#[wasm_bindgen(js_name = DrGraphSolver)]
#[derive(Debug, Clone)]
pub struct JsDrGraphSolver {
    inner: DrGraphInner,
    locked_manifold: Option<ManifoldKind>,
}

#[wasm_bindgen(js_class = DrGraphSolver)]
impl JsDrGraphSolver {
    /// Constructs a `DrGraphSolver` from `graph` and `DrGraphOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsDrGraphSolver, JsValue> {
        let (cfg, target) = parse_drgraph_options(options)?;
        let inner = match target {
            JsTargetKind::Layout2d => DrGraphInner::Dim2(
                DrGraphSolver::<Layout2d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => DrGraphInner::Dim3(
                DrGraphSolver::<Layout3d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => DrGraphInner::DimN(
                DrGraphSolver::<LayoutNd<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => DrGraphInner::Manifold(
                DrGraphSolver::<ManifoldLayout2d<f32>>::new(graph.inner(), cfg)
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
            DrGraphInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "DrGraphSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "DrGraphSolver::run")
            }
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "DrGraphSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "DrGraphSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "DrGraphSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "DrGraphSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "DrGraphSolver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'manifold2d'",
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
            DrGraphInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "DrGraphSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            DrGraphInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "DrGraphSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "DrGraphSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        let kind = match &self.inner {
            DrGraphInner::Dim2(_) => JsTargetKind::Layout2d,
            DrGraphInner::Dim3(_) => JsTargetKind::Layout3d,
            DrGraphInner::DimN(_) => JsTargetKind::LayoutNd,
            DrGraphInner::Manifold(_) => JsTargetKind::Manifold2d,
        };
        target_kind_str(kind).to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.num_nodes(),
            DrGraphInner::Dim3(s) => s.num_nodes(),
            DrGraphInner::DimN(s) => s.num_nodes(),
            DrGraphInner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.num_springs(),
            DrGraphInner::Dim3(s) => s.num_springs(),
            DrGraphInner::DimN(s) => s.num_springs(),
            DrGraphInner::Manifold(s) => s.num_springs(),
        }
    }

    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.num_levels(),
            DrGraphInner::Dim3(s) => s.num_levels(),
            DrGraphInner::DimN(s) => s.num_levels(),
            DrGraphInner::Manifold(s) => s.num_levels(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentLr)]
    pub fn current_lr(&self) -> f64 {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.current_lr(),
            DrGraphInner::Dim3(s) => s.current_lr(),
            DrGraphInner::DimN(s) => s.current_lr(),
            DrGraphInner::Manifold(s) => s.current_lr(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            DrGraphInner::Dim2(s) => s.status(),
            DrGraphInner::Dim3(s) => s.status(),
            DrGraphInner::DimN(s) => s.status(),
            DrGraphInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.iteration(),
            DrGraphInner::Dim3(s) => s.iteration(),
            DrGraphInner::DimN(s) => s.iteration(),
            DrGraphInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            DrGraphInner::Dim2(s) => s.total_iterations(),
            DrGraphInner::Dim3(s) => s.total_iterations(),
            DrGraphInner::DimN(s) => s.total_iterations(),
            DrGraphInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            DrGraphInner::Dim2(s) => s.status(),
            DrGraphInner::Dim3(s) => s.status(),
            DrGraphInner::DimN(s) => s.status(),
            DrGraphInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }
}
