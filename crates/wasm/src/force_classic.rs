//! WebAssembly bindings for Noack LinLog layout solver (`f32`-backed layouts with `f64` math)
//! supporting Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, and N-D.

use egraph::core::{Layout2d, Layout3d, LayoutNd};
use egraph::layout::force::{BarnesHut, Exact};
use egraph::layout::{ForceBackend, LinLogConfig, LinLogRepulsion, LinLogSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_bool_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop,
    solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_layout3d_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_layout3d_in_place, step_nd_in_place,
};

pub(crate) const DEFAULT_BARNES_HUT_THETA: f64 = 1.2;

pub(crate) fn parse_force_backend_for_target(
    target: JsTargetKind,
    options: &JsValue,
    has_opts: bool,
    solver_name: &str,
) -> Result<(bool, f64), JsValue> {
    let theta = if has_opts {
        opt_f64_prop(options, "theta", DEFAULT_BARNES_HUT_THETA)?
    } else {
        DEFAULT_BARNES_HUT_THETA
    };
    let backend_str = if has_opts {
        opt_string_prop(options, "backend")?
    } else {
        None
    };
    let use_bh_opt = if has_opts {
        opt_maybe_bool_prop(options, "useBarnesHut")?
    } else {
        None
    };

    if target == JsTargetKind::Manifold2d {
        return Err(unsupported_err(format!(
            "{solver_name} does not support target 'manifold2d'"
        )));
    }

    let parsed_backend = match backend_str {
        Some(ref b) => match normalize_enum_token(b)?.as_str() {
            "barneshut" | "bh" => Some(true),
            "exact" => Some(false),
            _ => {
                let expected = if target == JsTargetKind::LayoutNd {
                    "'exact'"
                } else {
                    "'barnesHut' or 'exact'"
                };
                return Err(invalid_input_err(format!(
                    "unknown {solver_name} backend {b:?}; expected {expected}"
                )));
            }
        },
        None => None,
    };

    let use_bh = match (parsed_backend, use_bh_opt) {
        (Some(false), Some(true)) => {
            return Err(invalid_input_err(format!(
                "conflicting {solver_name} options: backend='exact' cannot be combined with useBarnesHut=true"
            )));
        }
        (Some(true), Some(false)) => {
            return Err(invalid_input_err(format!(
                "conflicting {solver_name} options: backend='barnesHut' cannot be combined with useBarnesHut=false"
            )));
        }
        (Some(b), _) => b,
        (None, Some(b)) => b,
        (None, None) => matches!(target, JsTargetKind::Layout2d | JsTargetKind::Layout3d),
    };

    if target == JsTargetKind::LayoutNd && use_bh {
        return Err(unsupported_err(format!(
            "{solver_name} target 'layoutNd' only supports backend 'exact'"
        )));
    }
    Ok((use_bh, theta))
}

pub(crate) fn build_force_backend(use_bh: bool, theta: f64) -> ForceBackend {
    if use_bh {
        ForceBackend::BarnesHut(BarnesHut::new(theta))
    } else {
        ForceBackend::Exact(Exact::new())
    }
}

fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(invalid_input_err(format!(
            "unknown LinLogRepulsion {raw:?}; expected 'edgeRepulsion' or 'nodeRepulsion'"
        ))),
    }
}

#[derive(Debug, Clone)]
enum LinLogInner {
    Dim2(LinLogSolver<Layout2d<f32>, ForceBackend>),
    Dim3(LinLogSolver<Layout3d<f32>, ForceBackend>),
    DimN(LinLogSolver<LayoutNd<f32>, Exact>),
}

/// Cooperative Noack (2007/2009) LinLog / $(a, r)$-PolyLog force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = LinLogSolver)]
#[derive(Debug, Clone)]
pub struct JsLinLogSolver {
    inner: LinLogInner,
}

#[wasm_bindgen(js_class = LinLogSolver)]
impl JsLinLogSolver {
    /// Constructs a `LinLogSolver` from `graph` and optional `LinLogOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLinLogSolver, JsValue> {
        let mut cfg = LinLogConfig::default();
        let has_opts = check_options_arg(options, "LinLogSolver")?;
        let target = if has_opts {
            opt_target_prop(options)?
        } else {
            JsTargetKind::Layout2d
        };
        let (use_bh, theta) =
            parse_force_backend_for_target(target, options, has_opts, "LinLogSolver")?;
        if has_opts {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            if let Some(rep_str) = opt_string_prop(options, "repulsionModel")? {
                cfg.repulsion_model = parse_linlog_repulsion(&rep_str)?;
            }
            cfg.attraction_exponent =
                opt_f64_prop(options, "attractionExponent", cfg.attraction_exponent)?;
            cfg.repulsion_exponent =
                opt_f64_prop(options, "repulsionExponent", cfg.repulsion_exponent)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.max_displacement = opt_maybe_f64_prop(options, "maxDisplacement")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match target {
            JsTargetKind::Layout2d => LinLogInner::Dim2(
                LinLogSolver::<Layout2d<f32>, ForceBackend>::with_backend(
                    graph.inner(),
                    cfg,
                    build_force_backend(use_bh, theta),
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => LinLogInner::Dim3(
                LinLogSolver::<Layout3d<f32>, ForceBackend>::with_backend(
                    graph.inner(),
                    cfg,
                    build_force_backend(use_bh, theta),
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => LinLogInner::DimN(
                LinLogSolver::<LayoutNd<f32>, Exact>::with_backend(
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
            LinLogInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LinLogSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "LinLogSolver::run")
            }
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LinLogSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "LinLogSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "LinLogSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            LinLogInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "LinLogSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _energy2d)]
    pub fn energy_2d(&self, layout: &JsLayout2d) -> Result<f64, JsValue> {
        match &self.inner {
            LinLogInner::Dim2(s) => s
                .energy(&layout.inner().to_layout2d())
                .map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _energy3d)]
    pub fn energy_3d(&self, layout: &JsLayout3d) -> Result<f64, JsValue> {
        match &self.inner {
            LinLogInner::Dim3(s) => s
                .energy(&layout.inner().to_layout3d())
                .map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _energyNd)]
    pub fn energy_nd(&self, layout: &JsLayoutNd) -> Result<f64, JsValue> {
        match &self.inner {
            LinLogInner::DimN(s) => s.energy(layout.inner()).map_err(core_err_to_js),
            _ => Err(unsupported_err(
                "LinLogSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    pub fn masses(&self) -> Vec<f64> {
        match &self.inner {
            LinLogInner::Dim2(s) => s.masses().to_vec(),
            LinLogInner::Dim3(s) => s.masses().to_vec(),
            LinLogInner::DimN(s) => s.masses().to_vec(),
        }
    }

    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        match &self.inner {
            LinLogInner::Dim2(s) => s.max_displacement(),
            LinLogInner::Dim3(s) => s.max_displacement(),
            LinLogInner::DimN(s) => s.max_displacement(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            LinLogInner::Dim2(_) => "layout2d".to_owned(),
            LinLogInner::Dim3(_) => "layout3d".to_owned(),
            LinLogInner::DimN(_) => "layoutNd".to_owned(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            LinLogInner::Dim2(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                ForceBackend::BarnesHut(_) => "barnesHut",
                _ => "barnesHut",
            },
            LinLogInner::Dim3(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                ForceBackend::BarnesHut(_) => "barnesHut",
                _ => "barnesHut",
            },
            LinLogInner::DimN(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            LinLogInner::Dim2(s) => s.status(),
            LinLogInner::Dim3(s) => s.status(),
            LinLogInner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            LinLogInner::Dim2(s) => s.iteration(),
            LinLogInner::Dim3(s) => s.iteration(),
            LinLogInner::DimN(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            LinLogInner::Dim2(s) => s.total_iterations(),
            LinLogInner::Dim3(s) => s.total_iterations(),
            LinLogInner::DimN(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            LinLogInner::Dim2(s) => s.status(),
            LinLogInner::Dim3(s) => s.status(),
            LinLogInner::DimN(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            LinLogInner::Dim2(s) => s.num_nodes(),
            LinLogInner::Dim3(s) => s.num_nodes(),
            LinLogInner::DimN(s) => s.num_nodes(),
        }
    }
}
