//! Cooperative `PaCMAP-Graph` (`PacmapGraphSolver`) WebAssembly bindings (`f32`)
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, and coarse-backbone init.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{PacmapGraphSolver, pacmap_graph_with_backbone};
use wasm_bindgen::prelude::*;

use crate::coarsen_backbone::{JsCoarseBackboneResult, parse_coarse_backbone_config};
use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_target_prop, parse_pacmap_graph_config, solver_status_str,
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
enum PacmapInner {
    Dim2(PacmapGraphSolver<Layout2d<f32>>),
    Dim3(PacmapGraphSolver<Layout3d<f32>>),
    DimN(PacmapGraphSolver<LayoutNd<f32>>),
    Manifold(PacmapGraphSolver<ManifoldLayout2d<f32>>),
}

fn target_from_options(options: &JsValue, ctx: &str) -> Result<JsTargetKind, JsValue> {
    if check_options_arg(options, ctx)? {
        opt_target_prop(options)
    } else {
        Ok(JsTargetKind::Layout2d)
    }
}

/// Cooperative PaCMAP-Graph solver (`f32`) across 2D, 3D, N-D, and Manifolds.
#[wasm_bindgen(js_name = PacmapGraphSolver)]
#[derive(Debug, Clone)]
pub struct JsPacmapGraphSolver {
    inner: PacmapInner,
    locked_manifold: Option<ManifoldKind>,
}

/// Result of `PacmapGraphSolver.withBackbone`.
#[wasm_bindgen(js_name = PacmapBackboneResult)]
#[derive(Debug, Clone)]
pub struct JsPacmapBackboneResult {
    solver: JsPacmapGraphSolver,
    fine_layout: JsLayout2d,
    backbone: JsCoarseBackboneResult,
}

#[wasm_bindgen(js_class = PacmapBackboneResult)]
impl JsPacmapBackboneResult {
    /// Returns the initialized `PacmapGraphSolver`.
    pub fn solver(&self) -> JsPacmapGraphSolver {
        self.solver.clone()
    }

    /// Returns the area-reserved fine-level initial `Layout2d` (`f32`).
    #[wasm_bindgen(js_name = fineLayout)]
    pub fn fine_layout(&self) -> JsLayout2d {
        self.fine_layout.clone()
    }

    /// Returns the underlying `CoarseBackboneResult` (`coarseLayout`, `coarseGraph`, `stats`, etc.).
    pub fn backbone(&self) -> JsCoarseBackboneResult {
        self.backbone.clone()
    }
}

#[wasm_bindgen(js_class = PacmapGraphSolver)]
impl JsPacmapGraphSolver {
    /// Constructs a `PacmapGraphSolver` from `graph` using internal Chebyshev DoG Varadhan landmarks.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsPacmapGraphSolver, JsValue> {
        Self::from_graph(graph, options)
    }

    /// Constructs a `PacmapGraphSolver` from `graph` using internal Chebyshev DoG Varadhan landmarks.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsPacmapGraphSolver, JsValue> {
        let cfg = parse_pacmap_graph_config(options, "PacmapGraphSolver.fromGraph")?;
        let target = target_from_options(options, "PacmapGraphSolver.fromGraph")?;
        let inner = match target {
            JsTargetKind::Layout2d => PacmapInner::Dim2(
                PacmapGraphSolver::<Layout2d<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => PacmapInner::Dim3(
                PacmapGraphSolver::<Layout3d<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => PacmapInner::DimN(
                PacmapGraphSolver::<LayoutNd<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => PacmapInner::Manifold(
                PacmapGraphSolver::<ManifoldLayout2d<f32>>::new(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Constructs a `PacmapGraphSolver` reusing precomputed `DogVaradhanLandmarks`.
    #[wasm_bindgen(js_name = withLandmarks)]
    pub fn with_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        options: &JsValue,
    ) -> Result<JsPacmapGraphSolver, JsValue> {
        let cfg = parse_pacmap_graph_config(options, "PacmapGraphSolver.withLandmarks")?;
        let target = target_from_options(options, "PacmapGraphSolver.withLandmarks")?;
        let inner = match target {
            JsTargetKind::Layout2d => PacmapInner::Dim2(
                PacmapGraphSolver::<Layout2d<f32>>::with_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => PacmapInner::Dim3(
                PacmapGraphSolver::<Layout3d<f32>>::with_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => PacmapInner::DimN(
                PacmapGraphSolver::<LayoutNd<f32>>::with_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => PacmapInner::Manifold(
                PacmapGraphSolver::<ManifoldLayout2d<f32>>::with_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Constructs a `PacmapGraphSolver` from a flat row-major `(numNodes, numLandmarks)` distance table.
    #[wasm_bindgen(js_name = withLandmarkDistances)]
    pub fn with_landmark_distances(
        graph: &JsCsrGraph,
        landmark_dists: &[f64],
        num_landmarks: f64,
        options: &JsValue,
    ) -> Result<JsPacmapGraphSolver, JsValue> {
        let n_lm = parse_non_negative_usize(num_landmarks, "numLandmarks")?;
        let cfg = parse_pacmap_graph_config(options, "PacmapGraphSolver.withLandmarkDistances")?;
        let target = target_from_options(options, "PacmapGraphSolver.withLandmarkDistances")?;
        let inner = match target {
            JsTargetKind::Layout2d => PacmapInner::Dim2(
                PacmapGraphSolver::<Layout2d<f32>>::with_landmark_distances(
                    graph.inner(),
                    landmark_dists,
                    n_lm,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => PacmapInner::Dim3(
                PacmapGraphSolver::<Layout3d<f32>>::with_landmark_distances(
                    graph.inner(),
                    landmark_dists,
                    n_lm,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => PacmapInner::DimN(
                PacmapGraphSolver::<LayoutNd<f32>>::with_landmark_distances(
                    graph.inner(),
                    landmark_dists,
                    n_lm,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => PacmapInner::Manifold(
                PacmapGraphSolver::<ManifoldLayout2d<f32>>::with_landmark_distances(
                    graph.inner(),
                    landmark_dists,
                    n_lm,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Initializes an area-reserved coarse backbone `Layout2d` (`f32`) together with a configured
    /// `PacmapGraphSolver` for end-to-end multilevel layout.
    #[wasm_bindgen(js_name = withBackbone)]
    pub fn with_backbone(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsPacmapBackboneResult, JsValue> {
        let target = target_from_options(options, "PacmapGraphSolver.withBackbone")?;
        if target != JsTargetKind::Layout2d {
            return Err(unsupported_err(
                "PacmapGraphSolver.withBackbone only supports target 'layout2d'",
            ));
        }
        let pm_cfg = parse_pacmap_graph_config(options, "PacmapGraphSolver.withBackbone")?;
        let bb_cfg = parse_coarse_backbone_config(options, "PacmapGraphSolver.withBackbone")?;
        let (solver, fine_layout, backbone) =
            pacmap_graph_with_backbone::<f32, f32>(graph.inner(), pm_cfg, &bb_cfg)
                .map_err(core_err_to_js)?;
        Ok(JsPacmapBackboneResult {
            solver: Self {
                inner: PacmapInner::Dim2(solver),
                locked_manifold: None,
            },
            fine_layout: JsLayout2d::from_layout2d(&fine_layout),
            backbone: JsCoarseBackboneResult::from_core(backbone)?,
        })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim2(s) => run_steps_layout_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "PacmapGraphSolver::run_steps",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "PacmapGraphSolver::run")
            }
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "PacmapGraphSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "PacmapGraphSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::DimN(s) => run_steps_nd_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "PacmapGraphSolver::run_steps_nd",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::DimN(s) => {
                run_nd_in_place(s.num_nodes(), s, layout, "PacmapGraphSolver::run_nd")
            }
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "PacmapGraphSolver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'manifold2d'",
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
            PacmapInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "PacmapGraphSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            PacmapInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "PacmapGraphSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "PacmapGraphSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        let kind = match &self.inner {
            PacmapInner::Dim2(_) => JsTargetKind::Layout2d,
            PacmapInner::Dim3(_) => JsTargetKind::Layout3d,
            PacmapInner::DimN(_) => JsTargetKind::LayoutNd,
            PacmapInner::Manifold(_) => JsTargetKind::Manifold2d,
        };
        target_kind_str(kind).to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            PacmapInner::Dim2(s) => s.num_nodes(),
            PacmapInner::Dim3(s) => s.num_nodes(),
            PacmapInner::DimN(s) => s.num_nodes(),
            PacmapInner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        match &self.inner {
            PacmapInner::Dim2(s) => s.num_springs(),
            PacmapInner::Dim3(s) => s.num_springs(),
            PacmapInner::DimN(s) => s.num_springs(),
            PacmapInner::Manifold(s) => s.num_springs(),
        }
    }

    #[wasm_bindgen(getter, js_name = numHinges)]
    pub fn num_hinges(&self) -> usize {
        match &self.inner {
            PacmapInner::Dim2(s) => s.num_hinges(),
            PacmapInner::Dim3(s) => s.num_hinges(),
            PacmapInner::DimN(s) => s.num_hinges(),
            PacmapInner::Manifold(s) => s.num_hinges(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            PacmapInner::Dim2(s) => s.status(),
            PacmapInner::Dim3(s) => s.status(),
            PacmapInner::DimN(s) => s.status(),
            PacmapInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            PacmapInner::Dim2(s) => s.iteration(),
            PacmapInner::Dim3(s) => s.iteration(),
            PacmapInner::DimN(s) => s.iteration(),
            PacmapInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            PacmapInner::Dim2(s) => s.total_iterations(),
            PacmapInner::Dim3(s) => s.total_iterations(),
            PacmapInner::DimN(s) => s.total_iterations(),
            PacmapInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            PacmapInner::Dim2(s) => s.status(),
            PacmapInner::Dim3(s) => s.status(),
            PacmapInner::DimN(s) => s.status(),
            PacmapInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }
}
