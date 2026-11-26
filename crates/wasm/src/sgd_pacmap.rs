//! `SgdSolver` constructors for `SNAP-tFDP`, `PaCMAP-Graph`, and diagonal-preconditioned `sgd2`.

use egraph::layout::{Sgd2Config, SgdSolver, SnapTFdpConfig};
use wasm_bindgen::prelude::*;

use crate::coarsen_backbone::{JsCoarseBackboneResult, parse_coarse_backbone_config};
use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use crate::js_options::{
    check_options_arg, opt_bool_prop, opt_f64_prop, opt_seed_prop, opt_usize_prop,
    parse_pacmap_graph_config,
};
use crate::layout_buffers::JsLayout2d;
use crate::sgd::JsSgdSolver;

/// Output of `SgdSolver.pacmapGraphWithBackbone`.
#[wasm_bindgen(js_name = PacmapBackboneResult)]
#[derive(Debug, Clone)]
pub struct JsPacmapBackboneResult {
    solver: JsSgdSolver,
    fine_layout: JsLayout2d,
    backbone: JsCoarseBackboneResult,
}

#[wasm_bindgen(js_class = PacmapBackboneResult)]
impl JsPacmapBackboneResult {
    /// Returns the initialized cooperative `SgdSolver` (`PaCMAP-Graph` mode).
    pub fn solver(&self) -> JsSgdSolver {
        self.solver.clone()
    }

    /// Returns the coarse-backbone initial `Layout2d` (`f32`).
    #[wasm_bindgen(js_name = fineLayout)]
    pub fn fine_layout(&self) -> JsLayout2d {
        self.fine_layout.clone()
    }

    /// Returns the underlying `CoarseBackboneResult` handle.
    pub fn backbone(&self) -> JsCoarseBackboneResult {
        self.backbone.clone()
    }
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a compact-memory edge-incident negative sampling `SNAP-tFDP` solver.
    #[wasm_bindgen(js_name = snapTFdp)]
    pub fn snap_t_fdp(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let mut cfg = SnapTFdpConfig::default();
        if check_options_arg(options, "SgdSolver.snapTFdp")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.negatives_per_edge =
                opt_usize_prop(options, "negativesPerEdge", cfg.negatives_per_edge)?;
            cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
            cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
            cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
            cfg.rho_start = opt_f64_prop(options, "rhoStart", cfg.rho_start)?;
            cfg.rho_end = opt_f64_prop(options, "rhoEnd", cfg.rho_end)?;
            cfg.rho_power = opt_f64_prop(options, "rhoPower", cfg.rho_power)?;
            cfg.degree_beta = opt_f64_prop(options, "degreeBeta", cfg.degree_beta)?;
            cfg.bounded_attraction =
                opt_bool_prop(options, "boundedAttraction", cfg.bounded_attraction)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SgdSolver::<f32>::snap_t_fdp(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) Adam layout solver.
    #[wasm_bindgen(js_name = pacmapGraph)]
    pub fn pacmap_graph(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let cfg = parse_pacmap_graph_config(options, "SgdSolver.pacmapGraph")?;
        let inner = SgdSolver::<f32>::pacmap_graph(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` Adam layout solver using precomputed `DogVaradhanLandmarks`.
    #[wasm_bindgen(js_name = pacmapGraphWithLandmarks)]
    pub fn pacmap_graph_with_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        let cfg = parse_pacmap_graph_config(options, "SgdSolver.pacmapGraphWithLandmarks")?;
        let inner =
            SgdSolver::<f32>::pacmap_graph_with_landmarks(graph.inner(), landmarks.inner(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` Adam layout solver using a flat row-major
    /// `(num_nodes, num_landmarks)` landmark distance table.
    #[wasm_bindgen(js_name = pacmapGraphWithLandmarkDistances)]
    pub fn pacmap_graph_with_landmark_distances(
        graph: &JsCsrGraph,
        landmark_distances: &[f64],
        num_landmarks: f64,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        let k_lm = parse_non_negative_usize(num_landmarks, "numLandmarks")?;
        let cfg = parse_pacmap_graph_config(options, "SgdSolver.pacmapGraphWithLandmarkDistances")?;
        let inner = SgdSolver::<f32>::pacmap_graph_with_landmark_distances(
            graph.inner(),
            landmark_distances,
            k_lm,
            cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver coupled with a coarse Louvain area-reserved
    /// backbone initialization and `num_landmarks`-BFS Mid-Near (`MN`) landmarks.
    #[wasm_bindgen(js_name = pacmapGraphWithBackbone)]
    pub fn pacmap_graph_with_backbone(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsPacmapBackboneResult, JsError> {
        let pm_cfg = parse_pacmap_graph_config(options, "SgdSolver.pacmapGraphWithBackbone")?;
        let bb_cfg = parse_coarse_backbone_config(options, "SgdSolver.pacmapGraphWithBackbone")?;
        let (solver, fine_layout, bb) =
            SgdSolver::<f32>::pacmap_graph_with_backbone(graph.inner(), pm_cfg, &bb_cfg)
                .map_err(core_err_to_js)?;
        Ok(JsPacmapBackboneResult {
            solver: Self::from_inner(solver),
            fine_layout: JsLayout2d::from_layout2d(&fine_layout),
            backbone: JsCoarseBackboneResult::from_core(bb)?,
        })
    }

    /// Constructs a doubly-stochastic mini-batch diagonal-preconditioned stress SGD solver
    /// (`SgdScheduleKind::Squared`, distinct from Ahmed et al. 2021 multi-criteria `(SGD)^2`).
    pub fn sgd2(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let mut cfg = Sgd2Config::default();
        if check_options_arg(options, "SgdSolver.sgd2")? {
            cfg.iterations = opt_usize_prop(options, "tMax", cfg.iterations)?;
            cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
            cfg.step_scale = opt_f64_prop(options, "stepScale", cfg.step_scale)?;
            cfg.batch_size = opt_usize_prop(options, "batchSize", cfg.batch_size)?;
            cfg.damping = opt_f64_prop(options, "damping", cfg.damping)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SgdSolver::<f32>::sgd2(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }
}
