//! WebAssembly bindings for Maxent-Stress and GRIP multilevel layout solvers (`f32`).

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd};
use egraph::layout::stress::{BarnesHut as StressBarnesHut, Exact as StressExact};
use egraph::layout::{GripConfig, GripSolver, MaxentStressConfig, MaxentStressSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::force_classic::{DEFAULT_BARNES_HUT_THETA, parse_force_backend_for_target};
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
enum MaxentInner {
    Dim2Bh(MaxentStressSolver<Layout2d<f32>, StressBarnesHut>),
    Dim2Exact(MaxentStressSolver<Layout2d<f32>, StressExact>),
    Dim3Bh(MaxentStressSolver<Layout3d<f32>, StressBarnesHut>),
    Dim3Exact(MaxentStressSolver<Layout3d<f32>, StressExact>),
    DimN(MaxentStressSolver<LayoutNd<f32>, StressExact>),
}

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver (`f32`).
#[wasm_bindgen(js_name = MaxentStressSolver)]
#[derive(Debug, Clone)]
pub struct JsMaxentStressSolver {
    inner: MaxentInner,
}

#[wasm_bindgen(js_class = MaxentStressSolver)]
impl JsMaxentStressSolver {
    /// Constructs a `MaxentStressSolver` from `graph` and optional `MaxentStressOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsMaxentStressSolver, JsValue> {
        let mut cfg = MaxentStressConfig::default();
        let has_opts = check_options_arg(options, "MaxentStressSolver")?;
        let target = if has_opts {
            opt_target_prop(options)?
        } else {
            JsTargetKind::Layout2d
        };
        let (use_bh, theta) =
            parse_force_backend_for_target(target, options, has_opts, "MaxentStressSolver")?;
        if has_opts {
            cfg.q = opt_maybe_f64_prop(options, "q")?;
            cfg.alpha_init = opt_f64_prop(options, "alphaInit", cfg.alpha_init)?;
            cfg.alpha_decay = opt_f64_prop(options, "alphaDecay", cfg.alpha_decay)?;
            cfg.alpha_min = opt_f64_prop(options, "alphaMin", cfg.alpha_min)?;
            cfg.stages = opt_usize_prop(options, "stages", cfg.stages)?;
            cfg.steps_per_stage = opt_usize_prop(options, "stepsPerStage", cfg.steps_per_stage)?;
            cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
            cfg.cg_tol = opt_f64_prop(options, "cgTol", cfg.cg_tol)?;
            cfg.cg_max_iters = opt_usize_prop(options, "cgMaxIters", cfg.cg_max_iters)?;
            cfg.eps_sq = opt_f64_prop(options, "epsSq", cfg.eps_sq)?;
            cfg.seed = opt_seed_prop(options)?;
        } else {
            let _ = DEFAULT_BARNES_HUT_THETA;
        }
        let inner = match (target, use_bh) {
            (JsTargetKind::Layout2d, true) => MaxentInner::Dim2Bh(
                MaxentStressSolver::<Layout2d<f32>, StressBarnesHut>::new_with_backend(
                    graph.inner(),
                    cfg,
                    StressBarnesHut::new(theta),
                )
                .map_err(core_err_to_js)?,
            ),
            (JsTargetKind::Layout2d, false) => MaxentInner::Dim2Exact(
                MaxentStressSolver::<Layout2d<f32>, StressExact>::new_with_backend(
                    graph.inner(),
                    cfg,
                    StressExact::new(),
                )
                .map_err(core_err_to_js)?,
            ),
            (JsTargetKind::Layout3d, true) => MaxentInner::Dim3Bh(
                MaxentStressSolver::<Layout3d<f32>, StressBarnesHut>::new_with_backend(
                    graph.inner(),
                    cfg,
                    StressBarnesHut::new(theta),
                )
                .map_err(core_err_to_js)?,
            ),
            (JsTargetKind::Layout3d, false) => MaxentInner::Dim3Exact(
                MaxentStressSolver::<Layout3d<f32>, StressExact>::new_with_backend(
                    graph.inner(),
                    cfg,
                    StressExact::new(),
                )
                .map_err(core_err_to_js)?,
            ),
            (JsTargetKind::LayoutNd, _) => MaxentInner::DimN(
                MaxentStressSolver::<LayoutNd<f32>, StressExact>::new_with_backend(
                    graph.inner(),
                    cfg,
                    StressExact::new(),
                )
                .map_err(core_err_to_js)?,
            ),
            (JsTargetKind::Manifold2d, _) => unreachable!(),
        };
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim2Bh(s) => step_layout_in_place(s, layout),
            MaxentInner::Dim2Exact(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim2Bh(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "MaxentStressSolver::run_steps",
            ),
            MaxentInner::Dim2Exact(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "MaxentStressSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim2Bh(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "MaxentStressSolver::run")
            }
            MaxentInner::Dim2Exact(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "MaxentStressSolver::run")
            }
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim3Bh(s) => step_layout3d_in_place(s, layout),
            MaxentInner::Dim3Exact(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim3Bh(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "MaxentStressSolver::run_steps_3d",
            ),
            MaxentInner::Dim3Exact(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "MaxentStressSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::Dim3Bh(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "MaxentStressSolver::run_3d")
            }
            MaxentInner::Dim3Exact(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "MaxentStressSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "MaxentStressSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            MaxentInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "MaxentStressSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "MaxentStressSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            MaxentInner::Dim2Bh(_) | MaxentInner::Dim2Exact(_) => "layout2d",
            MaxentInner::Dim3Bh(_) | MaxentInner::Dim3Exact(_) => "layout3d",
            MaxentInner::DimN(_) => "layoutNd",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            MaxentInner::Dim2Bh(_) | MaxentInner::Dim3Bh(_) => "barnesHut",
            MaxentInner::Dim2Exact(_) | MaxentInner::Dim3Exact(_) | MaxentInner::DimN(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            MaxentInner::Dim2Bh(s) => s.status(),
            MaxentInner::Dim2Exact(s) => s.status(),
            MaxentInner::Dim3Bh(s) => s.status(),
            MaxentInner::Dim3Exact(s) => s.status(),
            MaxentInner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            MaxentInner::Dim2Bh(s) => s.iteration(),
            MaxentInner::Dim2Exact(s) => s.iteration(),
            MaxentInner::Dim3Bh(s) => s.iteration(),
            MaxentInner::Dim3Exact(s) => s.iteration(),
            MaxentInner::DimN(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            MaxentInner::Dim2Bh(s) => s.total_iterations(),
            MaxentInner::Dim2Exact(s) => s.total_iterations(),
            MaxentInner::Dim3Bh(s) => s.total_iterations(),
            MaxentInner::Dim3Exact(s) => s.total_iterations(),
            MaxentInner::DimN(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            MaxentInner::Dim2Bh(s) => s.status(),
            MaxentInner::Dim2Exact(s) => s.status(),
            MaxentInner::Dim3Bh(s) => s.status(),
            MaxentInner::Dim3Exact(s) => s.status(),
            MaxentInner::DimN(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            MaxentInner::Dim2Bh(s) => s.num_nodes(),
            MaxentInner::Dim2Exact(s) => s.num_nodes(),
            MaxentInner::Dim3Bh(s) => s.num_nodes(),
            MaxentInner::Dim3Exact(s) => s.num_nodes(),
            MaxentInner::DimN(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentAlpha)]
    pub fn current_alpha(&self) -> f64 {
        match &self.inner {
            MaxentInner::Dim2Bh(s) => s.current_alpha(),
            MaxentInner::Dim2Exact(s) => s.current_alpha(),
            MaxentInner::Dim3Bh(s) => s.current_alpha(),
            MaxentInner::Dim3Exact(s) => s.current_alpha(),
            MaxentInner::DimN(s) => s.current_alpha(),
        }
    }

    #[wasm_bindgen(getter, js_name = effectiveQ)]
    pub fn effective_q(&self) -> f64 {
        match &self.inner {
            MaxentInner::Dim2Bh(s) => s.effective_q(),
            MaxentInner::Dim2Exact(s) => s.effective_q(),
            MaxentInner::Dim3Bh(s) => s.effective_q(),
            MaxentInner::Dim3Exact(s) => s.effective_q(),
            MaxentInner::DimN(s) => s.effective_q(),
        }
    }
}

/// Cooperative Gajer & Kobourov (2002) GRIP multilevel force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = GripSolver)]
#[derive(Debug, Clone)]
pub struct JsGripSolver {
    inner: GripSolver<f32>,
}

#[wasm_bindgen(js_class = GripSolver)]
impl JsGripSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGripSolver, JsValue> {
        let mut cfg = GripConfig::default();
        if check_options_arg(options, "GripSolver")? {
            cfg.rounds_min = opt_usize_prop(options, "roundsMin", cfg.rounds_min)?;
            cfg.rounds_max = opt_usize_prop(options, "roundsMax", cfg.rounds_max)?;
            cfg.local_neighborhood_size = opt_usize_prop(
                options,
                "localNeighborhoodSize",
                cfg.local_neighborhood_size,
            )?;
            cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.temp_r = opt_f64_prop(options, "tempR", cfg.temp_r)?;
            cfg.temp_s = opt_f64_prop(options, "tempS", cfg.temp_s)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GripSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GripSolver::run_steps",
        )
    }

    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GripSolver::run",
        )
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

    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }
}
