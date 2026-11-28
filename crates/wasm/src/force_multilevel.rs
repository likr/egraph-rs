//! Cooperative ForceAtlas2 WebAssembly bindings (`f32`)
//! supporting Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, and N-D.

use egraph::core::{Layout2d, Layout3d, LayoutNd};
use egraph::layout::force::Exact;
use egraph::layout::{ForceAtlas2Config, ForceAtlas2Solver, ForceBackend};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::force_classic::{build_force_backend, parse_force_backend_for_target};
pub use crate::force_yifan_hu::JsYifanHuSolver;
use crate::js_convert::{core_err_to_js, parse_optional_f64_sequence, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_target_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_layout3d_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_layout3d_in_place, step_nd_in_place,
};

#[derive(Debug, Clone)]
enum Fa2Inner {
    Dim2(ForceAtlas2Solver<Layout2d<f32>, ForceBackend>),
    Dim3(ForceAtlas2Solver<Layout3d<f32>, ForceBackend>),
    DimN(ForceAtlas2Solver<LayoutNd<f32>, Exact>),
}

/// Cooperative ForceAtlas2 (Jacomy et al., 2014) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = ForceAtlas2Solver)]
#[derive(Debug, Clone)]
pub struct JsForceAtlas2Solver {
    inner: Fa2Inner,
}

#[wasm_bindgen(js_class = ForceAtlas2Solver)]
impl JsForceAtlas2Solver {
    /// Constructs a `ForceAtlas2Solver` from `graph` and optional `ForceAtlas2Options`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsForceAtlas2Solver, JsValue> {
        let mut cfg = ForceAtlas2Config::default();
        let mut node_radii: Option<Vec<f64>> = None;
        let has_opts = check_options_arg(options, "ForceAtlas2Solver")?;
        let target = if has_opts {
            opt_target_prop(options)?
        } else {
            JsTargetKind::Layout2d
        };
        let (use_bh, theta) =
            parse_force_backend_for_target(target, options, has_opts, "ForceAtlas2Solver")?;
        if has_opts {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.linlog_mode = opt_bool_prop(options, "linlogMode", cfg.linlog_mode)?;
            cfg.strong_gravity = opt_bool_prop(options, "strongGravity", cfg.strong_gravity)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.scaling_ratio = opt_f64_prop(options, "scalingRatio", cfg.scaling_ratio)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.outbound_attraction_distribution = opt_bool_prop(
                options,
                "outboundAttractionDistribution",
                cfg.outbound_attraction_distribution,
            )?;
            cfg.jitter_tolerance = opt_f64_prop(options, "jitterTolerance", cfg.jitter_tolerance)?;
            cfg.max_displacement = opt_maybe_f64_prop(options, "maxDisplacement")?;
            cfg.prevent_overlap = opt_bool_prop(options, "preventOverlap", cfg.prevent_overlap)?;
            cfg.default_node_radius =
                opt_f64_prop(options, "defaultNodeRadius", cfg.default_node_radius)?;
            node_radii =
                parse_optional_f64_sequence(&get_prop(options, "nodeRadii")?, "nodeRadii")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match target {
            JsTargetKind::Layout2d => {
                let b = build_force_backend(use_bh, theta);
                let s = match node_radii.as_deref() {
                    Some(radii) => ForceAtlas2Solver::<Layout2d<f32>, ForceBackend>::with_node_radii_and_backend(graph.inner(), cfg, radii, b),
                    None => ForceAtlas2Solver::<Layout2d<f32>, ForceBackend>::with_backend(graph.inner(), cfg, b),
                }.map_err(core_err_to_js)?;
                Fa2Inner::Dim2(s)
            }
            JsTargetKind::Layout3d => {
                let b = build_force_backend(use_bh, theta);
                let s = match node_radii.as_deref() {
                    Some(radii) => ForceAtlas2Solver::<Layout3d<f32>, ForceBackend>::with_node_radii_and_backend(graph.inner(), cfg, radii, b),
                    None => ForceAtlas2Solver::<Layout3d<f32>, ForceBackend>::with_backend(graph.inner(), cfg, b),
                }.map_err(core_err_to_js)?;
                Fa2Inner::Dim3(s)
            }
            JsTargetKind::LayoutNd => {
                let s = match node_radii.as_deref() {
                    Some(radii) => {
                        ForceAtlas2Solver::<LayoutNd<f32>, Exact>::with_node_radii_and_backend(
                            graph.inner(),
                            cfg,
                            radii,
                            Exact::new(),
                        )
                    }
                    None => ForceAtlas2Solver::<LayoutNd<f32>, Exact>::with_backend(
                        graph.inner(),
                        cfg,
                        Exact::new(),
                    ),
                }
                .map_err(core_err_to_js)?;
                Fa2Inner::DimN(s)
            }
            JsTargetKind::Manifold2d => unreachable!(),
        };
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "ForceAtlas2Solver::run_steps",
            ),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "ForceAtlas2Solver::run")
            }
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "ForceAtlas2Solver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "ForceAtlas2Solver::run_3d")
            }
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "ForceAtlas2Solver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            Fa2Inner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "ForceAtlas2Solver::run_nd")
            }
            _ => Err(unsupported_err(
                "ForceAtlas2Solver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            Fa2Inner::Dim2(_) => "layout2d",
            Fa2Inner::Dim3(_) => "layout3d",
            Fa2Inner::DimN(_) => "layoutNd",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            Fa2Inner::Dim2(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                _ => "barnesHut",
            },
            Fa2Inner::Dim3(s) => match s.backend() {
                ForceBackend::Exact(_) => "exact",
                _ => "barnesHut",
            },
            Fa2Inner::DimN(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            Fa2Inner::Dim2(s) => s.status(),
            Fa2Inner::Dim3(s) => s.status(),
            Fa2Inner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            Fa2Inner::Dim2(s) => s.iteration(),
            Fa2Inner::Dim3(s) => s.iteration(),
            Fa2Inner::DimN(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            Fa2Inner::Dim2(s) => s.total_iterations(),
            Fa2Inner::Dim3(s) => s.total_iterations(),
            Fa2Inner::DimN(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            Fa2Inner::Dim2(s) => s.status(),
            Fa2Inner::Dim3(s) => s.status(),
            Fa2Inner::DimN(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            Fa2Inner::Dim2(s) => s.num_nodes(),
            Fa2Inner::Dim3(s) => s.num_nodes(),
            Fa2Inner::DimN(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(js_name = nodeRadii)]
    pub fn node_radii(&self) -> Vec<f64> {
        match &self.inner {
            Fa2Inner::Dim2(s) => s.node_radii().to_vec(),
            Fa2Inner::Dim3(s) => s.node_radii().to_vec(),
            Fa2Inner::DimN(s) => s.node_radii().to_vec(),
        }
    }

    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        match &self.inner {
            Fa2Inner::Dim2(s) => s.max_displacement(),
            Fa2Inner::Dim3(s) => s.max_displacement(),
            Fa2Inner::DimN(s) => s.max_displacement(),
        }
    }
}
