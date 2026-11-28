//! Cooperative multilevel Yifan Hu (2005) WebAssembly bindings (`f32`)
//! supporting Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, and N-D.

use egraph::core::{Layout2d, Layout3d, LayoutNd};
use egraph::layout::force::Exact;
use egraph::layout::{ForceBackend, YifanHuConfig, YifanHuSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::force_classic::{build_force_backend, parse_force_backend_for_target};
use crate::js_convert::{core_err_to_js, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_target_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_layout3d_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_layout3d_in_place, step_nd_in_place,
};

#[derive(Debug, Clone)]
enum YhInner {
    Dim2(YifanHuSolver<Layout2d<f32>, ForceBackend>),
    Dim3(YifanHuSolver<Layout3d<f32>, ForceBackend>),
    DimN(YifanHuSolver<LayoutNd<f32>, Exact>),
}

/// Cooperative multilevel Yifan Hu (2005) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = YifanHuSolver)]
#[derive(Debug, Clone)]
pub struct JsYifanHuSolver {
    inner: YhInner,
}

#[wasm_bindgen(js_class = YifanHuSolver)]
impl JsYifanHuSolver {
    /// Constructs a `YifanHuSolver` from `graph` and optional `YifanHuOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsYifanHuSolver, JsValue> {
        let mut cfg = YifanHuConfig::default();
        let has_opts = check_options_arg(options, "YifanHuSolver")?;
        let target = if has_opts {
            opt_target_prop(options)?
        } else {
            JsTargetKind::Layout2d
        };
        let (use_bh, theta) =
            parse_force_backend_for_target(target, options, has_opts, "YifanHuSolver")?;
        if has_opts {
            cfg.max_iterations_per_level = opt_usize_prop(
                options,
                "maxIterationsPerLevel",
                cfg.max_iterations_per_level,
            )?;
            cfg.step_size = opt_f64_prop(options, "stepSize", cfg.step_size)?;
            cfg.step_ratio = opt_f64_prop(options, "stepRatio", cfg.step_ratio)?;
            cfg.relative_strength =
                opt_f64_prop(options, "relativeStrength", cfg.relative_strength)?;
            cfg.optimal_distance = opt_maybe_f64_prop(options, "optimalDistance")?;
            cfg.min_coarsen_size = opt_usize_prop(options, "minCoarsenSize", cfg.min_coarsen_size)?;
            cfg.min_coarsen_ratio =
                opt_f64_prop(options, "minCoarsenRatio", cfg.min_coarsen_ratio)?;
            cfg.convergence_tolerance =
                opt_f64_prop(options, "convergenceTolerance", cfg.convergence_tolerance)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match target {
            JsTargetKind::Layout2d => YhInner::Dim2(
                YifanHuSolver::<Layout2d<f32>, ForceBackend>::with_backend(
                    graph.inner(),
                    cfg,
                    build_force_backend(use_bh, theta),
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => YhInner::Dim3(
                YifanHuSolver::<Layout3d<f32>, ForceBackend>::with_backend(
                    graph.inner(),
                    cfg,
                    build_force_backend(use_bh, theta),
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => YhInner::DimN(
                YifanHuSolver::<LayoutNd<f32>, Exact>::with_backend(
                    graph.inner(),
                    cfg,
                    Exact::new(),
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => unreachable!(),
        };
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "YifanHuSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim2(s) => run_layout_in_place(s.num_nodes(), s, layout, "YifanHuSolver::run"),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "YifanHuSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "YifanHuSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "YifanHuSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            YhInner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "YifanHuSolver::run_nd"),
            _ => Err(unsupported_err(
                "YifanHuSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        match &self.inner {
            YhInner::Dim2(s) => s.num_levels(),
            YhInner::Dim3(s) => s.num_levels(),
            YhInner::DimN(s) => s.num_levels(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentLevel)]
    pub fn current_level(&self) -> usize {
        match &self.inner {
            YhInner::Dim2(s) => s.current_level(),
            YhInner::Dim3(s) => s.current_level(),
            YhInner::DimN(s) => s.current_level(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            YhInner::Dim2(_) => "layout2d",
            YhInner::Dim3(_) => "layout3d",
            YhInner::DimN(_) => "layoutNd",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            YhInner::Dim2(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                _ => "barnesHut",
            },
            YhInner::Dim3(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                _ => "barnesHut",
            },
            YhInner::DimN(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            YhInner::Dim2(s) => s.status(),
            YhInner::Dim3(s) => s.status(),
            YhInner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            YhInner::Dim2(s) => s.iteration(),
            YhInner::Dim3(s) => s.iteration(),
            YhInner::DimN(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            YhInner::Dim2(s) => s.total_iterations(),
            YhInner::Dim3(s) => s.total_iterations(),
            YhInner::DimN(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            YhInner::Dim2(s) => s.status(),
            YhInner::Dim3(s) => s.status(),
            YhInner::DimN(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            YhInner::Dim2(s) => s.num_nodes(),
            YhInner::Dim3(s) => s.num_nodes(),
            YhInner::DimN(s) => s.num_nodes(),
        }
    }
}
