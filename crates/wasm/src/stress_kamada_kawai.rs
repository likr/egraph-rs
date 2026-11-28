//! Cooperative Kamada-Kawai (1989) spring layout solver (`f32`) WebAssembly bindings
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{KamadaKawaiConfig, KamadaKawaiSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_f64_prop, opt_maybe_usize_prop, opt_target_prop,
    opt_usize_prop, solver_status_str,
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

fn parse_kk_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(KamadaKawaiConfig, JsTargetKind), JsValue> {
    let mut cfg = KamadaKawaiConfig::default();
    let has_opts = check_options_arg(options, ctx)?;
    let target = if has_opts {
        opt_target_prop(options)?
    } else {
        JsTargetKind::Layout2d
    };
    if has_opts {
        cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
        cfg.max_inner_iters = opt_usize_prop(options, "maxInnerIters", cfg.max_inner_iters)?;
        cfg.max_iterations = opt_maybe_usize_prop(options, "maxIterations")?;
    }
    Ok((cfg, target))
}

#[derive(Debug, Clone)]
enum KamadaKawaiInner {
    Dim2(KamadaKawaiSolver<Layout2d<f32>>),
    Dim3(KamadaKawaiSolver<Layout3d<f32>>),
    DimN(KamadaKawaiSolver<LayoutNd<f32>>),
    Manifold(KamadaKawaiSolver<ManifoldLayout2d<f32>>),
}

/// Cooperative Kamada-Kawai (1989) spring layout solver (`f32`) across 2D, 3D, N-D, and Manifolds.
#[wasm_bindgen(js_name = KamadaKawaiSolver)]
#[derive(Debug, Clone)]
pub struct JsKamadaKawaiSolver {
    inner: KamadaKawaiInner,
    locked_manifold: Option<ManifoldKind>,
}

impl JsKamadaKawaiSolver {
    fn from_graph_inner(graph: &JsCsrGraph, options: &JsValue, ctx: &str) -> Result<Self, JsValue> {
        let (cfg, target) = parse_kk_options(options, ctx)?;
        let inner = match target {
            JsTargetKind::Layout2d => KamadaKawaiInner::Dim2(
                KamadaKawaiSolver::<Layout2d<f32>>::from_graph(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => KamadaKawaiInner::Dim3(
                KamadaKawaiSolver::<Layout3d<f32>>::from_graph(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => KamadaKawaiInner::DimN(
                KamadaKawaiSolver::<LayoutNd<f32>>::from_graph(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => KamadaKawaiInner::Manifold(
                KamadaKawaiSolver::<ManifoldLayout2d<f32>>::from_graph(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }
}

#[wasm_bindgen(js_class = KamadaKawaiSolver)]
impl JsKamadaKawaiSolver {
    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsKamadaKawaiSolver, JsValue> {
        Self::from_graph_inner(graph, options, "KamadaKawaiSolver")
    }

    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsKamadaKawaiSolver, JsValue> {
        Self::from_graph_inner(graph, options, "KamadaKawaiSolver.fromGraph")
    }

    /// Constructs a `KamadaKawaiSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        options: &JsValue,
    ) -> Result<JsKamadaKawaiSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (cfg, target) = parse_kk_options(options, "KamadaKawaiSolver.fromDistanceMatrix")?;
        let inner = match target {
            JsTargetKind::Layout2d => KamadaKawaiInner::Dim2(
                KamadaKawaiSolver::<Layout2d<f32>>::from_distance_matrix(n, dists_flat, cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => KamadaKawaiInner::Dim3(
                KamadaKawaiSolver::<Layout3d<f32>>::from_distance_matrix(n, dists_flat, cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => KamadaKawaiInner::DimN(
                KamadaKawaiSolver::<LayoutNd<f32>>::from_distance_matrix(n, dists_flat, cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => KamadaKawaiInner::Manifold(
                KamadaKawaiSolver::<ManifoldLayout2d<f32>>::from_distance_matrix(
                    n, dists_flat, cfg,
                )
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
            KamadaKawaiInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "KamadaKawaiSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "KamadaKawaiSolver::run")
            }
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "KamadaKawaiSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "KamadaKawaiSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "KamadaKawaiSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "KamadaKawaiSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Manifold(s) => {
                let n = s.num_nodes();
                step_manifold_in_place(
                    s,
                    layout,
                    &mut self.locked_manifold,
                    n,
                    "KamadaKawaiSolver::step_manifold",
                )
            }
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'manifold2d'",
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
            KamadaKawaiInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "KamadaKawaiSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            KamadaKawaiInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "KamadaKawaiSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "KamadaKawaiSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            KamadaKawaiInner::Dim2(_) => "layout2d",
            KamadaKawaiInner::Dim3(_) => "layout3d",
            KamadaKawaiInner::DimN(_) => "layoutNd",
            KamadaKawaiInner::Manifold(_) => "manifold2d",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            KamadaKawaiInner::Dim2(s) => s.status(),
            KamadaKawaiInner::Dim3(s) => s.status(),
            KamadaKawaiInner::DimN(s) => s.status(),
            KamadaKawaiInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            KamadaKawaiInner::Dim2(s) => s.iteration(),
            KamadaKawaiInner::Dim3(s) => s.iteration(),
            KamadaKawaiInner::DimN(s) => s.iteration(),
            KamadaKawaiInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            KamadaKawaiInner::Dim2(s) => s.total_iterations(),
            KamadaKawaiInner::Dim3(s) => s.total_iterations(),
            KamadaKawaiInner::DimN(s) => s.total_iterations(),
            KamadaKawaiInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            KamadaKawaiInner::Dim2(s) => s.status(),
            KamadaKawaiInner::Dim3(s) => s.status(),
            KamadaKawaiInner::DimN(s) => s.status(),
            KamadaKawaiInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            KamadaKawaiInner::Dim2(s) => s.num_nodes(),
            KamadaKawaiInner::Dim3(s) => s.num_nodes(),
            KamadaKawaiInner::DimN(s) => s.num_nodes(),
            KamadaKawaiInner::Manifold(s) => s.num_nodes(),
        }
    }
}
