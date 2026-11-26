//! Pure-Rust pivoted `RHGH` landmark WebAssembly bindings (`AmgPcgSolver` + JL diagonal probes).

use egraph::neighborhood::{RhghLandmarkConfig, RhghLandmarkStats, compute_pivoted_rhgh_landmarks};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_usize_prop,
};

/// Output of `computePivotedRhghLandmarks`.
#[wasm_bindgen(js_name = RhghLandmarksResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsRhghLandmarksResult {
    landmarks: JsDogVaradhanLandmarks,
    stats: RhghLandmarkStats,
}

#[wasm_bindgen(js_class = RhghLandmarksResult)]
impl JsRhghLandmarksResult {
    /// Returns the `DogVaradhanLandmarks` handle usable with `SgdSolver` and `GuMapSolver`.
    pub fn landmarks(&self) -> JsDogVaradhanLandmarks {
        self.landmarks.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn gamma(&self) -> f64 {
        self.stats.gamma
    }

    #[wasm_bindgen(getter, js_name = pExp)]
    pub fn p_exp(&self) -> f64 {
        self.stats.p_exp
    }

    #[wasm_bindgen(getter, js_name = degCv)]
    pub fn deg_cv(&self) -> f64 {
        self.stats.deg_cv
    }

    #[wasm_bindgen(getter, js_name = leafFraction)]
    pub fn leaf_fraction(&self) -> f64 {
        self.stats.leaf_fraction
    }

    #[wasm_bindgen(getter, js_name = meanDegree)]
    pub fn mean_degree(&self) -> f64 {
        self.stats.mean_degree
    }

    #[wasm_bindgen(getter)]
    pub fn exp2(&self) -> f64 {
        self.stats.exp2
    }

    #[wasm_bindgen(getter, js_name = sSw)]
    pub fn s_sw(&self) -> f64 {
        self.stats.s_sw
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.stats.n_landmarks
    }

    #[wasm_bindgen(getter, js_name = numProbes)]
    pub fn num_probes(&self) -> usize {
        self.stats.num_probes
    }

    #[wasm_bindgen(getter, js_name = maxPcgIters)]
    pub fn max_pcg_iters(&self) -> usize {
        self.stats.max_pcg_iters
    }

    #[wasm_bindgen(getter, js_name = totalPcgIters)]
    pub fn total_pcg_iters(&self) -> usize {
        self.stats.total_pcg_iters
    }

    #[wasm_bindgen(getter, js_name = allConverged)]
    pub fn all_converged(&self) -> bool {
        self.stats.all_converged
    }

    #[wasm_bindgen(getter, js_name = edgeMedScale)]
    pub fn edge_med_scale(&self) -> f64 {
        self.stats.edge_med_scale
    }
}

/// Computes pure-Rust pivoted `RHGH` landmark tables via `AmgPcgSolver` + JL diagonal probes.
#[wasm_bindgen(js_name = computePivotedRhghLandmarks)]
pub fn js_compute_pivoted_rhgh_landmarks(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsRhghLandmarksResult, JsError> {
    let mut cfg = RhghLandmarkConfig::default();
    if check_options_arg(options, "computePivotedRhghLandmarks")? {
        cfg.gamma = opt_maybe_f64_prop(options, "gamma")?;
        cfg.p_exp = opt_maybe_f64_prop(options, "pExp")?;
        cfg.n_landmarks = opt_usize_prop(options, "numLandmarks", cfg.n_landmarks)?;
        cfg.min_degree = opt_usize_prop(options, "minDegree", cfg.min_degree)?;
        cfg.num_probes = opt_usize_prop(options, "numProbes", cfg.num_probes)?;
        cfg.rel_tol = opt_f64_prop(options, "relTol", cfg.rel_tol)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.normalize_1hop = opt_bool_prop(options, "normalize1hop", cfg.normalize_1hop)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let (inner, stats) =
        compute_pivoted_rhgh_landmarks(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsRhghLandmarksResult {
        landmarks: JsDogVaradhanLandmarks::from_rhgh_parts(inner, &stats),
        stats,
    })
}
