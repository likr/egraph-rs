//! WebAssembly bindings for Fruchterman-Reingold (1991) layout solver (`f32`)
//! supporting Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, N-D, and manifolds.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::force::Exact;
use egraph::layout::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::force_classic::DEFAULT_BARNES_HUT_THETA;
use crate::js_convert::{core_err_to_js, invalid_input_err, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_manifold_in_place, run_nd_in_place,
    run_steps_layout_in_place, run_steps_layout3d_in_place, run_steps_manifold_in_place,
    run_steps_nd_in_place, step_layout_in_place, step_layout3d_in_place, step_manifold_in_place,
    step_nd_in_place,
};

fn parse_fr_backend_kind(raw: &str, theta: f64) -> Result<FrBackend, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "exact" => Ok(FrBackend::exact()),
        "barneshut" | "bh" => Ok(FrBackend::barnes_hut(theta)),
        "grid" => Ok(FrBackend::grid()),
        _ => Err(invalid_input_err(format!(
            "unknown FrBackend {raw:?}; expected 'exact', 'barnesHut', or 'grid'"
        ))),
    }
}

pub(crate) fn parse_fr_config(
    options: &JsValue,
    ctx: &str,
) -> Result<FruchtermanReingoldConfig, JsValue> {
    let mut cfg = FruchtermanReingoldConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.k = opt_maybe_f64_prop(options, "k")?;
        cfg.area = opt_maybe_f64_prop(options, "area")?;
        cfg.initial_temperature = opt_maybe_f64_prop(options, "initialTemperature")?;
        cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

#[derive(Debug, Clone)]
enum FrInner {
    Dim2(FruchtermanReingoldSolver<Layout2d<f32>, FrBackend>),
    Dim3(FruchtermanReingoldSolver<Layout3d<f32>, FrBackend>),
    DimN(FruchtermanReingoldSolver<LayoutNd<f32>, Exact>),
    Manifold(FruchtermanReingoldSolver<ManifoldLayout2d<f32>, Exact>),
}

/// Cooperative Fruchterman-Reingold (1991) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = FruchtermanReingoldSolver)]
#[derive(Debug, Clone)]
pub struct JsFruchtermanReingoldSolver {
    inner: FrInner,
    locked_manifold: Option<ManifoldKind>,
}

#[wasm_bindgen(js_class = FruchtermanReingoldSolver)]
impl JsFruchtermanReingoldSolver {
    /// Constructs a `FruchtermanReingoldSolver` from `graph` and optional `FruchtermanReingoldOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsFruchtermanReingoldSolver, JsValue> {
        let cfg = parse_fr_config(options, "FruchtermanReingoldSolver")?;
        let has_opts = check_options_arg(options, "FruchtermanReingoldSolver")?;
        let target = if has_opts {
            opt_target_prop(options)?
        } else {
            JsTargetKind::Layout2d
        };
        let theta = if has_opts {
            opt_f64_prop(options, "theta", DEFAULT_BARNES_HUT_THETA)?
        } else {
            DEFAULT_BARNES_HUT_THETA
        };
        let backend_opt = if has_opts {
            opt_string_prop(options, "backend")?
        } else {
            None
        };

        let inner = match target {
            JsTargetKind::Layout2d => {
                let backend = match backend_opt {
                    Some(ref b) => parse_fr_backend_kind(b, theta)?,
                    None => FrBackend::exact(),
                };
                FrInner::Dim2(
                    FruchtermanReingoldSolver::<Layout2d<f32>, FrBackend>::with_backend(
                        graph.inner(),
                        cfg,
                        backend,
                    )
                    .map_err(core_err_to_js)?,
                )
            }
            JsTargetKind::Layout3d => {
                let backend = match backend_opt {
                    Some(ref b) => parse_fr_backend_kind(b, theta)?,
                    None => FrBackend::exact(),
                };
                FrInner::Dim3(
                    FruchtermanReingoldSolver::<Layout3d<f32>, FrBackend>::with_backend(
                        graph.inner(),
                        cfg,
                        backend,
                    )
                    .map_err(core_err_to_js)?,
                )
            }
            JsTargetKind::LayoutNd => {
                if let Some(ref b) = backend_opt
                    && normalize_enum_token(b)?.as_str() != "exact"
                {
                    return Err(unsupported_err(format!(
                        "FruchtermanReingoldSolver target 'layoutNd' only supports backend 'exact', got {b:?}"
                    )));
                }
                FrInner::DimN(
                    FruchtermanReingoldSolver::<LayoutNd<f32>, Exact>::with_backend(
                        graph.inner(),
                        cfg,
                        Exact::new(),
                    )
                    .map_err(core_err_to_js)?,
                )
            }
            JsTargetKind::Manifold2d => {
                if let Some(ref b) = backend_opt
                    && normalize_enum_token(b)?.as_str() != "exact"
                {
                    return Err(unsupported_err(format!(
                        "FruchtermanReingoldSolver target 'manifold2d' only supports backend 'exact', got {b:?}"
                    )));
                }
                FrInner::Manifold(
                    FruchtermanReingoldSolver::<ManifoldLayout2d<f32>, Exact>::with_backend(
                        graph.inner(),
                        cfg,
                        Exact::new(),
                    )
                    .map_err(core_err_to_js)?,
                )
            }
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "FruchtermanReingoldSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "FruchtermanReingoldSolver::run")
            }
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "FruchtermanReingoldSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Dim3(s) => run_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                "FruchtermanReingoldSolver::run_3d",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "FruchtermanReingoldSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::DimN(s) => run_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                "FruchtermanReingoldSolver::run_nd",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Manifold(s) => {
                let n = s.num_nodes();
                step_manifold_in_place(
                    s,
                    layout,
                    &mut self.locked_manifold,
                    n,
                    "FruchtermanReingoldSolver::step_manifold",
                )
            }
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'manifold2d'",
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
            FrInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "FruchtermanReingoldSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            FrInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "FruchtermanReingoldSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "FruchtermanReingoldSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter, js_name = optimalDistance)]
    pub fn optimal_distance(&self) -> f64 {
        match &self.inner {
            FrInner::Dim2(s) => s.optimal_distance(),
            FrInner::Dim3(s) => s.optimal_distance(),
            FrInner::DimN(s) => s.optimal_distance(),
            FrInner::Manifold(s) => s.optimal_distance(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentTemperature)]
    pub fn current_temperature(&self) -> f64 {
        match &self.inner {
            FrInner::Dim2(s) => s.current_temperature(),
            FrInner::Dim3(s) => s.current_temperature(),
            FrInner::DimN(s) => s.current_temperature(),
            FrInner::Manifold(s) => s.current_temperature(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            FrInner::Dim2(_) => "layout2d",
            FrInner::Dim3(_) => "layout3d",
            FrInner::DimN(_) => "layoutNd",
            FrInner::Manifold(_) => "manifold2d",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            FrInner::Dim2(s) => match s.backend() {
                FrBackend::Exact(_) => "exact",
                FrBackend::BarnesHut(_) => "barnesHut",
                FrBackend::Grid(_) => "grid",
                _ => "exact",
            },
            FrInner::Dim3(s) => match s.backend() {
                FrBackend::Exact(_) => "exact",
                FrBackend::BarnesHut(_) => "barnesHut",
                FrBackend::Grid(_) => "grid",
                _ => "exact",
            },
            FrInner::DimN(_) | FrInner::Manifold(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            FrInner::Dim2(s) => s.status(),
            FrInner::Dim3(s) => s.status(),
            FrInner::DimN(s) => s.status(),
            FrInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            FrInner::Dim2(s) => s.iteration(),
            FrInner::Dim3(s) => s.iteration(),
            FrInner::DimN(s) => s.iteration(),
            FrInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            FrInner::Dim2(s) => s.total_iterations(),
            FrInner::Dim3(s) => s.total_iterations(),
            FrInner::DimN(s) => s.total_iterations(),
            FrInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            FrInner::Dim2(s) => s.status(),
            FrInner::Dim3(s) => s.status(),
            FrInner::DimN(s) => s.status(),
            FrInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            FrInner::Dim2(s) => s.num_nodes(),
            FrInner::Dim3(s) => s.num_nodes(),
            FrInner::DimN(s) => s.num_nodes(),
            FrInner::Manifold(s) => s.num_nodes(),
        }
    }
}
