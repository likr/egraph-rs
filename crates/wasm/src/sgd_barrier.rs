//! 3-tier Sparse Barrier SGD constructors.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d};
use egraph::layout::{BarrierSymmetry, SgdSolver, SparseBarrierConfig};
use wasm_bindgen::{JsCast, prelude::*};

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{core_err_to_js, invalid_input_err, parse_non_negative_usize};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop, opt_string_prop, opt_target_prop,
    opt_usize_prop, parse_dog_config_from_options,
};
use crate::sgd::{JsSgdSolver, SgdInner, parse_sgd_schedule_kind};
use crate::sgd_stats::{JsDogStats, JsSparseBarrierStats};

fn parse_barrier_symmetry(raw: &str) -> Result<BarrierSymmetry, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "symmax" => Ok(BarrierSymmetry::SymMax),
        "symavg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(invalid_input_err(format!(
            "unknown BarrierSymmetry {raw:?}; expected 'symMax' or 'symAvg'"
        ))),
    }
}

fn opt_u32_array_prop(obj: &JsValue, key: &str) -> Result<Option<Vec<u32>>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(Some(arr.to_vec()));
    }
    if !val.is_string() && val.is_object() {
        let iter = js_sys::try_iter(&val)
            .map_err(|err| invalid_input_err(format!("failed to iterate options.{key}: {err:?}")))?
            .ok_or_else(|| {
                invalid_input_err(format!(
                    "options.{key} must be a Uint32Array or number array"
                ))
            })?;
        let mut out = Vec::new();
        for item_res in iter {
            let item = item_res.map_err(|err| {
                invalid_input_err(format!("failed to read options.{key} entry: {err:?}"))
            })?;
            let num = item.as_f64().ok_or_else(|| {
                invalid_input_err(format!(
                    "options.{key} entries must be non-negative integers"
                ))
            })?;
            let idx = parse_non_negative_usize(num, key)?;
            let idx_u32 = u32::try_from(idx).map_err(|_| {
                invalid_input_err(format!("options.{key} entry {idx} exceeds u32::MAX"))
            })?;
            out.push(idx_u32);
        }
        return Ok(Some(out));
    }
    Err(invalid_input_err(format!(
        "options.{key} must be a Uint32Array or number array"
    )))
}

fn parse_sparse_barrier_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(SparseBarrierConfig, JsTargetKind, Option<Vec<u32>>), JsValue> {
    let mut cfg = SparseBarrierConfig::default();
    let mut target = JsTargetKind::Layout2d;
    let mut pivots = None;
    if check_options_arg(options, ctx)? {
        cfg.max_2hop_per_node = opt_usize_prop(options, "max2hopPerNode", cfg.max_2hop_per_node)?;
        cfg.hinge_repulsion = opt_bool_prop(options, "hingeRepulsion", cfg.hinge_repulsion)?;
        cfg.delta_rel = opt_f64_prop(options, "deltaRel", cfg.delta_rel)?;
        if let Some(sym_str) = opt_string_prop(options, "barrierSym")? {
            cfg.barrier_sym = parse_barrier_symmetry(&sym_str)?;
        }
        cfg.omega_1 = opt_f64_prop(options, "omega1", cfg.omega_1)?;
        cfg.omega_2 = opt_f64_prop(options, "omega2", cfg.omega_2)?;
        cfg.gamma_non2 = opt_f64_prop(options, "gammaNon2", cfg.gamma_non2)?;
        cfg.ht_repulsion = opt_bool_prop(options, "htRepulsion", cfg.ht_repulsion)?;
        cfg.min_ratio = opt_f64_prop(options, "minRatio", cfg.min_ratio)?;
        cfg.eps_dist = opt_f64_prop(options, "epsDist", cfg.eps_dist)?;
        cfg.neg_per_node = opt_maybe_usize_prop(options, "negPerNode")?;
        cfg.num_pivots = opt_usize_prop(options, "numPivots", cfg.num_pivots)?;
        cfg.core_pivots = opt_bool_prop(options, "corePivots", cfg.core_pivots)?;
        cfg.degree_weight_beta = opt_f64_prop(options, "degreeWeightBeta", cfg.degree_weight_beta)?;
        cfg.schedule.iterations = opt_usize_prop(options, "iterations", cfg.schedule.iterations)?;
        cfg.schedule.eps = opt_f64_prop(options, "eps", cfg.schedule.eps)?;
        if let Some(k_str) = opt_string_prop(options, "kind")? {
            cfg.schedule.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.schedule.step_scale = opt_f64_prop(options, "stepScale", cfg.schedule.step_scale)?;
        cfg.schedule.max_mu = opt_maybe_f64_prop(options, "maxMu")?;
        cfg.seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
        pivots = opt_u32_array_prop(options, "pivots")?;
    }
    Ok((cfg, target, pivots))
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a 3-tier Sparse Barrier SGD solver from precomputed `DogVaradhanLandmarks`.
    #[wasm_bindgen(js_name = sparseBarrierFromLandmarks)]
    pub fn sparse_barrier_from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let (cfg, target, pivots) =
            parse_sparse_barrier_options(options, "SgdSolver.sparseBarrierFromLandmarks")?;
        let (inner, stats) = match target {
            JsTargetKind::Layout2d => {
                let (s, st) = SgdSolver::<Layout2d<f32>>::sparse_barrier_from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    pivots.as_deref(),
                    &cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Dim2(s), st)
            }
            JsTargetKind::Layout3d => {
                let (s, st) = SgdSolver::<Layout3d<f32>>::sparse_barrier_from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    pivots.as_deref(),
                    &cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Dim3(s), st)
            }
            JsTargetKind::LayoutNd => {
                let (s, st) = SgdSolver::<LayoutNd<f32>>::sparse_barrier_from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    pivots.as_deref(),
                    &cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::DimN(s), st)
            }
            JsTargetKind::Manifold2d => {
                let (s, st) = SgdSolver::<ManifoldLayout2d<f32>>::sparse_barrier_from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    pivots.as_deref(),
                    &cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Manifold(s), st)
            }
        };
        Ok(Self::from_barrier_parts(
            inner,
            Some(stats),
            None,
            cfg.schedule.max_mu,
        ))
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a 3-tier Sparse Barrier SGD solver.
    #[wasm_bindgen(js_name = sparseBarrierDog)]
    pub fn sparse_barrier_dog(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let (barrier_cfg, target, _) =
            parse_sparse_barrier_options(options, "SgdSolver.sparseBarrierDog")?;
        let dog_cfg = parse_dog_config_from_options(options, "SgdSolver.sparseBarrierDog")?;
        let (inner, stats, dog_stats) = match target {
            JsTargetKind::Layout2d => {
                let (s, st, ds) = SgdSolver::<Layout2d<f32>>::sparse_barrier_dog(
                    graph.inner(),
                    &dog_cfg,
                    &barrier_cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Dim2(s), st, ds)
            }
            JsTargetKind::Layout3d => {
                let (s, st, ds) = SgdSolver::<Layout3d<f32>>::sparse_barrier_dog(
                    graph.inner(),
                    &dog_cfg,
                    &barrier_cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Dim3(s), st, ds)
            }
            JsTargetKind::LayoutNd => {
                let (s, st, ds) = SgdSolver::<LayoutNd<f32>>::sparse_barrier_dog(
                    graph.inner(),
                    &dog_cfg,
                    &barrier_cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::DimN(s), st, ds)
            }
            JsTargetKind::Manifold2d => {
                let (s, st, ds) = SgdSolver::<ManifoldLayout2d<f32>>::sparse_barrier_dog(
                    graph.inner(),
                    &dog_cfg,
                    &barrier_cfg,
                )
                .map_err(core_err_to_js)?;
                (SgdInner::Manifold(s), st, ds)
            }
        };
        Ok(Self::from_barrier_parts(
            inner,
            Some(stats),
            Some(dog_stats),
            barrier_cfg.schedule.max_mu,
        ))
    }

    /// Returns `SparseBarrierStats` when constructed via `sparseBarrierFromLandmarks` or
    /// `sparseBarrierDog`, or `undefined` otherwise.
    pub fn stats(&self) -> Option<JsSparseBarrierStats> {
        self.barrier_stats_inner()
            .map(JsSparseBarrierStats::from_inner)
    }

    /// Returns regularized DoG Varadhan landmark diagnostics `DogStats` when constructed via
    /// `sparseBarrierDog`, or `undefined` otherwise.
    #[wasm_bindgen(js_name = dogStats)]
    pub fn dog_stats(&self) -> Option<JsDogStats> {
        self.dog_stats_inner().map(JsDogStats::from_inner)
    }
}
