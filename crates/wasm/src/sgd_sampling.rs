//! Sampling-based `SgdSolver` constructors (`randomPair`, `omega`, `streamingNegatives`).

use egraph::layout::{SgdScheduleConfig, SgdSolver, StreamingNegativeConfig};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::js_options::{
    is_plain_js_object, opt_bool_prop, opt_f64_prop, opt_usize_prop, reject_extra_args_with_options,
};
use crate::sgd::{JsSgdSolver, parse_sgd_schedule_and_seed, parse_sgd_schedule_kind};

const DEFAULT_RANDOM_PAIRS_PER_NODE: usize = 30;
const DEFAULT_OMEGA_MIN_DIST: f64 = 1e-3;

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a random-pair sparse SGD solver combining 1-hop edges with sampled random pairs.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = randomPair)]
    pub fn random_pair(
        graph: &JsCsrGraph,
        pairs_per_node_or_options: &JsValue,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let (pairs_per_node, sched, seed_u64) = if is_plain_js_object(pairs_per_node_or_options) {
            let has_extra = (!t_max_or_options.is_undefined() && !t_max_or_options.is_null())
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            let ppn = opt_usize_prop(
                pairs_per_node_or_options,
                "pairsPerNode",
                DEFAULT_RANDOM_PAIRS_PER_NODE,
            )?;
            let (s, sd) =
                parse_sgd_schedule_and_seed(pairs_per_node_or_options, None, None, None, None)?;
            (ppn, s, sd)
        } else {
            let ppn = if pairs_per_node_or_options.is_undefined()
                || pairs_per_node_or_options.is_null()
            {
                DEFAULT_RANDOM_PAIRS_PER_NODE
            } else {
                let v = pairs_per_node_or_options.as_f64().ok_or_else(|| {
                    JsError::new("pairsPerNodeOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "pairsPerNode")?
            };
            let (s, sd) =
                parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
            (ppn, s, sd)
        };
        let inner = SgdSolver::<f32>::random_pair(graph.inner(), pairs_per_node, sched, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs an Omega SGD solver from a flat row-major `(numNodes, dims)` spectral embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn omega(
        graph: &JsCsrGraph,
        embedding: &[f64],
        dims: f64,
        k_or_options: &JsValue,
        min_dist: Option<f64>,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let dims_val = parse_non_negative_usize(dims, "dims")?;
        let (k_val, md_val, sched, seed_u64) = if is_plain_js_object(k_or_options) {
            let has_extra = min_dist.is_some()
                || (!t_max_or_options.is_undefined() && !t_max_or_options.is_null())
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            let k_v = opt_usize_prop(k_or_options, "k", DEFAULT_RANDOM_PAIRS_PER_NODE)?;
            let md_v = opt_f64_prop(k_or_options, "minDist", DEFAULT_OMEGA_MIN_DIST)?;
            let (s, sd) = parse_sgd_schedule_and_seed(k_or_options, None, None, None, None)?;
            (k_v, md_v, s, sd)
        } else {
            let k_v = if k_or_options.is_undefined() || k_or_options.is_null() {
                DEFAULT_RANDOM_PAIRS_PER_NODE
            } else {
                let v = k_or_options.as_f64().ok_or_else(|| {
                    JsError::new(
                        "kOrOptions must be a non-negative integer, options object, null, or undefined",
                    )
                })?;
                parse_non_negative_usize(v, "k")?
            };
            let md_v = min_dist.unwrap_or(DEFAULT_OMEGA_MIN_DIST);
            let (s, sd) =
                parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
            (k_v, md_v, s, sd)
        };
        let inner = SgdSolver::<f32>::omega(
            graph.inner(),
            embedding,
            dims_val,
            k_val,
            md_val,
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a minimal-RAM SGD solver storing 1-hop springs and streaming non-edge negatives.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = streamingNegatives)]
    pub fn streaming_negatives(
        graph: &JsCsrGraph,
        negatives_or_options: &JsValue,
        target_distance: Option<f64>,
        weight: Option<f64>,
        hinge_only: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let mut neg_cfg = StreamingNegativeConfig::default();
        let mut sched = SgdScheduleConfig::default();
        let seed_u64 = if is_plain_js_object(negatives_or_options) {
            let has_extra = target_distance.is_some()
                || weight.is_some()
                || hinge_only.is_some()
                || t_max.is_some()
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            neg_cfg.negatives_per_node = opt_usize_prop(
                negatives_or_options,
                "negativesPerNode",
                neg_cfg.negatives_per_node,
            )?;
            neg_cfg.target_distance = opt_f64_prop(
                negatives_or_options,
                "targetDistance",
                neg_cfg.target_distance,
            )?;
            neg_cfg.weight = opt_f64_prop(negatives_or_options, "weight", neg_cfg.weight)?;
            neg_cfg.hinge_only =
                opt_bool_prop(negatives_or_options, "hingeOnly", neg_cfg.hinge_only)?;
            let (s, sd) =
                parse_sgd_schedule_and_seed(negatives_or_options, None, None, None, None)?;
            sched = s;
            sd
        } else {
            if !negatives_or_options.is_undefined() && !negatives_or_options.is_null() {
                let v = negatives_or_options.as_f64().ok_or_else(|| {
                    JsError::new("negativesOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                neg_cfg.negatives_per_node = parse_non_negative_usize(v, "negativesPerNode")?;
            }
            neg_cfg.target_distance = target_distance.unwrap_or(neg_cfg.target_distance);
            neg_cfg.weight = weight.unwrap_or(neg_cfg.weight);
            neg_cfg.hinge_only = hinge_only.unwrap_or(neg_cfg.hinge_only);
            sched.t_max = parse_optional_usize(t_max, sched.t_max, "tMax")?;
            sched.eps = eps.unwrap_or(sched.eps);
            if let Some(k_str) = kind {
                sched.kind = parse_sgd_schedule_kind(&k_str)?;
            }
            sched.step_scale = step_scale.unwrap_or(sched.step_scale);
            parse_seed_u64(seed)?
        };
        let inner = SgdSolver::<f32>::streaming_negatives(graph.inner(), neg_cfg, sched, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }
}
