//! Cooperative preconditioned mini-batch stress SGD solver (`Sgd2Solver`, `f32`)
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{Sgd2Config, Sgd2Solver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_f64_prop, opt_seed_prop, opt_target_prop, opt_usize_prop,
    solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::sgd::{parse_sgd_schedule_options, target_kind_str};
use crate::solver_runner::{
    check_and_lock_manifold, run_layout_in_place, run_layout3d_in_place, run_manifold_in_place,
    run_nd_in_place, run_steps_layout_in_place, run_steps_layout3d_in_place,
    run_steps_manifold_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_layout3d_in_place, step_manifold_in_place, step_nd_in_place,
};

#[derive(Debug, Clone)]
enum Sgd2Inner {
    Dim2(Sgd2Solver<Layout2d<f32>>),
    Dim3(Sgd2Solver<Layout3d<f32>>),
    DimN(Sgd2Solver<LayoutNd<f32>>),
    Manifold(Sgd2Solver<ManifoldLayout2d<f32>>),
}

fn parse_sgd2_options(options: &JsValue) -> Result<(Sgd2Config, JsTargetKind), JsValue> {
    let mut cfg = Sgd2Config::default();
    let mut target = JsTargetKind::Layout2d;
    if check_options_arg(options, "Sgd2Solver")? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
        cfg.step_scale = opt_f64_prop(options, "stepScale", cfg.step_scale)?;
        cfg.batch_size = opt_usize_prop(options, "batchSize", cfg.batch_size)?;
        cfg.damping = opt_f64_prop(options, "damping", cfg.damping)?;
        cfg.seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
    }
    Ok((cfg, target))
}

/// Cooperative preconditioned mini-batch stress SGD solver (`f32`) across 2D, 3D, N-D, and Manifolds.
#[wasm_bindgen(js_name = Sgd2Solver)]
#[derive(Debug, Clone)]
pub struct JsSgd2Solver {
    inner: Sgd2Inner,
    locked_manifold: Option<ManifoldKind>,
}

#[wasm_bindgen(js_class = Sgd2Solver)]
impl JsSgd2Solver {
    /// Constructs a preconditioned mini-batch stress SGD solver from `graph` and `Sgd2Options`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgd2Solver, JsValue> {
        let (cfg, target) = parse_sgd2_options(options)?;
        let inner = match target {
            JsTargetKind::Layout2d => Sgd2Inner::Dim2(
                Sgd2Solver::<Layout2d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => Sgd2Inner::Dim3(
                Sgd2Solver::<Layout3d<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => Sgd2Inner::DimN(
                Sgd2Solver::<LayoutNd<f32>>::new(graph.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => Sgd2Inner::Manifold(
                Sgd2Solver::<ManifoldLayout2d<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Resets the iteration schedule and RNG without reallocating stored pair buffers.
    #[wasm_bindgen(js_name = resetSchedule)]
    pub fn reset_schedule(&mut self, options: &JsValue) -> Result<(), JsValue> {
        let (sched, seed_u64, _) = parse_sgd_schedule_options(options, "Sgd2Solver.resetSchedule")?;
        match &mut self.inner {
            Sgd2Inner::Dim2(s) => s.reset_schedule(sched, seed_u64),
            Sgd2Inner::Dim3(s) => s.reset_schedule(sched, seed_u64),
            Sgd2Inner::DimN(s) => s.reset_schedule(sched, seed_u64),
            Sgd2Inner::Manifold(s) => s.reset_schedule(sched, seed_u64),
        }
        .map_err(core_err_to_js)
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim2(s) => {
                run_steps_layout_in_place(s.num_nodes(), s, layout, steps, "Sgd2Solver::run_steps")
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim2(s) => run_layout_in_place(s.num_nodes(), s, layout, "Sgd2Solver::run"),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "Sgd2Solver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "Sgd2Solver::run_3d")
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::DimN(s) => {
                run_steps_nd_in_place(s.num_nodes(), s, layout, steps, "Sgd2Solver::run_steps_nd")
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "Sgd2Solver::run_nd"),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "Sgd2Solver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'manifold2d'",
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
            Sgd2Inner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "Sgd2Solver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Sgd2Inner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "Sgd2Solver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runScheduleWindow2d)]
    pub fn run_schedule_window_2d(
        &mut self,
        layout: &mut JsLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match &mut self.inner {
            Sgd2Inner::Dim2(s) => {
                let mut soa = layout.inner().to_layout2d();
                let status = s
                    .run_schedule_window(&mut soa, start, cnt)
                    .map_err(core_err_to_js)?;
                layout
                    .inner_mut()
                    .copy_from_layout2d(&soa)
                    .map_err(ffi_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runScheduleWindow3d)]
    pub fn run_schedule_window_3d(
        &mut self,
        layout: &mut JsLayout3d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match &mut self.inner {
            Sgd2Inner::Dim3(s) => {
                let mut soa = layout.inner().to_layout3d();
                let status = s
                    .run_schedule_window(&mut soa, start, cnt)
                    .map_err(core_err_to_js)?;
                layout
                    .inner_mut()
                    .copy_from_layout3d(&soa)
                    .map_err(ffi_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runScheduleWindowNd)]
    pub fn run_schedule_window_nd(
        &mut self,
        layout: &mut JsLayoutNd,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match &mut self.inner {
            Sgd2Inner::DimN(s) => {
                let status = s
                    .run_schedule_window(layout.inner_mut(), start, cnt)
                    .map_err(core_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "Sgd2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runScheduleWindowManifold)]
    pub fn run_schedule_window_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        let n = self.num_nodes();
        let Sgd2Inner::Manifold(s) = &mut self.inner else {
            return Err(unsupported_err(
                "Sgd2Solver was not configured for target 'manifold2d'",
            ));
        };
        check_and_lock_manifold(
            &mut self.locked_manifold,
            layout,
            n,
            "Sgd2Solver::run_schedule_window_manifold",
        )?;
        let status = s
            .run_schedule_window(layout.inner_mut(), start, cnt)
            .map_err(core_err_to_js)?;
        layout.sync_shadow()?;
        Ok(solver_status_str(status).to_owned())
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        let kind = match &self.inner {
            Sgd2Inner::Dim2(_) => JsTargetKind::Layout2d,
            Sgd2Inner::Dim3(_) => JsTargetKind::Layout3d,
            Sgd2Inner::DimN(_) => JsTargetKind::LayoutNd,
            Sgd2Inner::Manifold(_) => JsTargetKind::Manifold2d,
        };
        target_kind_str(kind).to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            Sgd2Inner::Dim2(s) => s.num_nodes(),
            Sgd2Inner::Dim3(s) => s.num_nodes(),
            Sgd2Inner::DimN(s) => s.num_nodes(),
            Sgd2Inner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        match &self.inner {
            Sgd2Inner::Dim2(s) => s.num_springs(),
            Sgd2Inner::Dim3(s) => s.num_springs(),
            Sgd2Inner::DimN(s) => s.num_springs(),
            Sgd2Inner::Manifold(s) => s.num_springs(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentEta)]
    pub fn current_eta(&self) -> f64 {
        match &self.inner {
            Sgd2Inner::Dim2(s) => s.current_eta(),
            Sgd2Inner::Dim3(s) => s.current_eta(),
            Sgd2Inner::DimN(s) => s.current_eta(),
            Sgd2Inner::Manifold(s) => s.current_eta(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            Sgd2Inner::Dim2(s) => s.status(),
            Sgd2Inner::Dim3(s) => s.status(),
            Sgd2Inner::DimN(s) => s.status(),
            Sgd2Inner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            Sgd2Inner::Dim2(s) => s.iteration(),
            Sgd2Inner::Dim3(s) => s.iteration(),
            Sgd2Inner::DimN(s) => s.iteration(),
            Sgd2Inner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            Sgd2Inner::Dim2(s) => s.total_iterations(),
            Sgd2Inner::Dim3(s) => s.total_iterations(),
            Sgd2Inner::DimN(s) => s.total_iterations(),
            Sgd2Inner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            Sgd2Inner::Dim2(s) => s.status(),
            Sgd2Inner::Dim3(s) => s.status(),
            Sgd2Inner::DimN(s) => s.status(),
            Sgd2Inner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }
}
