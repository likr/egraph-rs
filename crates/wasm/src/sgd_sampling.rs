//! Sampling-based `SgdSolver` constructors (`randomPair`, `omega`, `streamingNegatives`).

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d};
use egraph::layout::{SgdSolver, StreamingNegativeConfig};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_bool_prop, opt_f64_prop, opt_usize_prop,
};
use crate::sgd::{JsSgdSolver, SgdInner, parse_sgd_schedule_options};

const DEFAULT_RANDOM_PAIRS_PER_NODE: usize = 30;
const DEFAULT_OMEGA_MIN_DIST: f64 = 1e-3;

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a random-pair sparse SGD solver combining 1-hop edges with sampled random pairs.
    #[wasm_bindgen(js_name = randomPair)]
    pub fn random_pair(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsValue> {
        let (sched, seed_u64, target) =
            parse_sgd_schedule_options(options, "SgdSolver.randomPair")?;
        let max_mu = sched.max_mu;
        let pairs_per_node = if check_options_arg(options, "SgdSolver.randomPair")? {
            opt_usize_prop(options, "pairsPerNode", DEFAULT_RANDOM_PAIRS_PER_NODE)?
        } else {
            DEFAULT_RANDOM_PAIRS_PER_NODE
        };
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::random_pair(
                    graph.inner(),
                    pairs_per_node,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::random_pair(
                    graph.inner(),
                    pairs_per_node,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::random_pair(
                    graph.inner(),
                    pairs_per_node,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::random_pair(
                    graph.inner(),
                    pairs_per_node,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Constructs an Omega SGD solver from a flat row-major `(numNodes, dims)` spectral embedding.
    pub fn omega(
        graph: &JsCsrGraph,
        embedding: &[f64],
        dims: f64,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let dims_val = parse_non_negative_usize(dims, "dims")?;
        let (sched, seed_u64, target) = parse_sgd_schedule_options(options, "SgdSolver.omega")?;
        let max_mu = sched.max_mu;
        let (k_val, md_val) = if check_options_arg(options, "SgdSolver.omega")? {
            (
                opt_usize_prop(options, "k", DEFAULT_RANDOM_PAIRS_PER_NODE)?,
                opt_f64_prop(options, "minDist", DEFAULT_OMEGA_MIN_DIST)?,
            )
        } else {
            (DEFAULT_RANDOM_PAIRS_PER_NODE, DEFAULT_OMEGA_MIN_DIST)
        };
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::omega(
                    graph.inner(),
                    embedding,
                    dims_val,
                    k_val,
                    md_val,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::omega(
                    graph.inner(),
                    embedding,
                    dims_val,
                    k_val,
                    md_val,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::omega(
                    graph.inner(),
                    embedding,
                    dims_val,
                    k_val,
                    md_val,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::omega(
                    graph.inner(),
                    embedding,
                    dims_val,
                    k_val,
                    md_val,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Constructs a minimal-RAM SGD solver storing 1-hop springs and streaming non-edge negatives.
    #[wasm_bindgen(js_name = streamingNegatives)]
    pub fn streaming_negatives(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let mut neg_cfg = StreamingNegativeConfig::default();
        let (sched, seed_u64, target) =
            parse_sgd_schedule_options(options, "SgdSolver.streamingNegatives")?;
        let max_mu = sched.max_mu;
        if check_options_arg(options, "SgdSolver.streamingNegatives")? {
            neg_cfg.negatives_per_node =
                opt_usize_prop(options, "negativesPerNode", neg_cfg.negatives_per_node)?;
            neg_cfg.target_distance =
                opt_f64_prop(options, "targetDistance", neg_cfg.target_distance)?;
            neg_cfg.weight = opt_f64_prop(options, "weight", neg_cfg.weight)?;
            neg_cfg.hinge_only = opt_bool_prop(options, "hingeOnly", neg_cfg.hinge_only)?;
        }
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::streaming_negatives(
                    graph.inner(),
                    neg_cfg,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::streaming_negatives(
                    graph.inner(),
                    neg_cfg,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::streaming_negatives(
                    graph.inner(),
                    neg_cfg,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::streaming_negatives(
                    graph.inner(),
                    neg_cfg,
                    sched,
                    seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }
}
