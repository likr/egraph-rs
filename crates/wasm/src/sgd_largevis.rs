//! Cooperative `LargeVis` (Tang et al., 2016) WebAssembly bindings (`f32`)
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{LargeVisConfig, LargeVisSolver};
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
enum LargeVisInner {
    Dim2(LargeVisSolver<Layout2d<f32>>),
    Dim3(LargeVisSolver<Layout3d<f32>>),
    DimN(LargeVisSolver<LayoutNd<f32>>),
    Manifold(LargeVisSolver<ManifoldLayout2d<f32>>),
}

fn parse_largevis_options(options: &JsValue) -> Result<(LargeVisConfig, JsTargetKind), JsValue> {
    let mut cfg = LargeVisConfig::default();
    let mut target = JsTargetKind::Layout2d;
    if check_options_arg(options, "LargeVisSolver")? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.negative_samples = opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
        cfg.gamma = opt_f64_prop(options, "gamma", cfg.gamma)?;
        cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
        cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
        cfg.degree_exponent = opt_f64_prop(options, "degreeExponent", cfg.degree_exponent)?;
        cfg.seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
    }
    Ok((cfg, target))
}

/// Cooperative single-threaded deterministic `LargeVis` (Tang et al., 2016) layout solver (`f32`).
#[wasm_bindgen(js_name = LargeVisSolver)]
#[derive(Debug, Clone)]
pub struct JsLargeVisSolver {
    inner: LargeVisInner,
    locked_manifold: Option<ManifoldKind>,
}

#[wasm_bindgen(js_class = LargeVisSolver)]
impl JsLargeVisSolver {
    /// Constructs a `LargeVisSolver` from `graph` and `LargeVisOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLargeVisSolver, JsValue> {
        let (cfg, target) = parse_largevis_options(options)?;
        let inner = match target {
            JsTargetKind::Layout2d => LargeVisInner::Dim2(
                LargeVisSolver::<Layout2d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => LargeVisInner::Dim3(
                LargeVisSolver::<Layout3d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => LargeVisInner::DimN(
                LargeVisSolver::<LayoutNd<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => LargeVisInner::Manifold(
                LargeVisSolver::<ManifoldLayout2d<f32>>::new(graph.inner(), cfg)
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
            LargeVisInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LargeVisSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "LargeVisSolver::run")
            }
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LargeVisSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "LargeVisSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LargeVisSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "LargeVisSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "LargeVisSolver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'manifold2d'",
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
            LargeVisInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "LargeVisSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            LargeVisInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "LargeVisSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "LargeVisSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        let kind = match &self.inner {
            LargeVisInner::Dim2(_) => JsTargetKind::Layout2d,
            LargeVisInner::Dim3(_) => JsTargetKind::Layout3d,
            LargeVisInner::DimN(_) => JsTargetKind::LayoutNd,
            LargeVisInner::Manifold(_) => JsTargetKind::Manifold2d,
        };
        target_kind_str(kind).to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            LargeVisInner::Dim2(s) => s.num_nodes(),
            LargeVisInner::Dim3(s) => s.num_nodes(),
            LargeVisInner::DimN(s) => s.num_nodes(),
            LargeVisInner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        match &self.inner {
            LargeVisInner::Dim2(s) => s.num_springs(),
            LargeVisInner::Dim3(s) => s.num_springs(),
            LargeVisInner::DimN(s) => s.num_springs(),
            LargeVisInner::Manifold(s) => s.num_springs(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentLr)]
    pub fn current_lr(&self) -> f64 {
        match &self.inner {
            LargeVisInner::Dim2(s) => s.current_lr(),
            LargeVisInner::Dim3(s) => s.current_lr(),
            LargeVisInner::DimN(s) => s.current_lr(),
            LargeVisInner::Manifold(s) => s.current_lr(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            LargeVisInner::Dim2(s) => s.status(),
            LargeVisInner::Dim3(s) => s.status(),
            LargeVisInner::DimN(s) => s.status(),
            LargeVisInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            LargeVisInner::Dim2(s) => s.iteration(),
            LargeVisInner::Dim3(s) => s.iteration(),
            LargeVisInner::DimN(s) => s.iteration(),
            LargeVisInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            LargeVisInner::Dim2(s) => s.total_iterations(),
            LargeVisInner::Dim3(s) => s.total_iterations(),
            LargeVisInner::DimN(s) => s.total_iterations(),
            LargeVisInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            LargeVisInner::Dim2(s) => s.status(),
            LargeVisInner::Dim3(s) => s.status(),
            LargeVisInner::DimN(s) => s.status(),
            LargeVisInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }
}
