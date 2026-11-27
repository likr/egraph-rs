//! WebAssembly bindings for Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`)
//! across `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_rta_landmark_f64_sequence, parse_seed_u64,
    parse_u32_sequence, set_f64_prop, set_js_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use egraph::metrics::{
    RtaEvaluation, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_detail_manifold, landmark_stratified_rta_detail_nd,
    landmark_stratified_rta_from_landmarks_f64,
    landmark_stratified_rta_from_landmarks_f64_manifold,
    landmark_stratified_rta_from_landmarks_f64_nd, landmark_stratified_rta_manifold,
    landmark_stratified_rta_nd,
};
use wasm_bindgen::prelude::*;

const DEFAULT_RTA_NUM_LANDMARKS: usize = 64;
const DEFAULT_RTA_NUM_TRIPLETS: usize = 1_000_000;
const DEFAULT_RTA_MIN_DEGREE: usize = 2;

/// Detailed result returned by `landmarkStratifiedRtaDetail` and `landmarkStratifiedRtaFromLandmarks`.
#[wasm_bindgen(js_name = RtaEvaluation)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsRtaEvaluation {
    inner: RtaEvaluation,
}

#[wasm_bindgen(js_class = RtaEvaluation)]
impl JsRtaEvaluation {
    #[wasm_bindgen(getter)]
    pub fn rta(&self) -> f64 {
        self.inner.rta
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks
    }

    #[wasm_bindgen(getter, js_name = numTriplets)]
    pub fn num_triplets(&self) -> usize {
        self.inner.num_triplets
    }

    #[wasm_bindgen(getter, js_name = numStrata)]
    pub fn num_strata(&self) -> usize {
        self.inner.num_strata
    }

    #[wasm_bindgen(getter, js_name = isExact)]
    pub fn is_exact(&self) -> bool {
        self.inner.is_exact
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        for (k, v) in [
            ("rta", self.inner.rta),
            ("numLandmarks", self.inner.num_landmarks as f64),
            ("numTriplets", self.inner.num_triplets as f64),
            ("numStrata", self.inner.num_strata as f64),
        ] {
            set_f64_prop(&obj, k, v)?;
        }
        set_js_prop(&obj, "isExact", &JsValue::from_bool(self.inner.is_exact))?;
        Ok(obj)
    }
}

fn parse_rta_params(
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<(usize, usize, usize, u64), JsError> {
    let nl = match num_landmarks {
        None => DEFAULT_RTA_NUM_LANDMARKS,
        Some(v) => parse_non_negative_usize(v, "numLandmarks")?,
    };
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let md = match min_degree {
        None => DEFAULT_RTA_MIN_DEGREE,
        Some(v) => parse_non_negative_usize(v, "minDegree")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    Ok((nl, nt, md, seed_u64))
}

#[wasm_bindgen(js_name = _landmarkStratifiedRta2d)]
pub fn js_rta_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    let soa = layout.inner().to_layout2d();
    landmark_stratified_rta(graph.inner(), &soa, nl, nt, md, seed_u64).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaNd)]
pub fn js_rta_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    landmark_stratified_rta_nd(graph.inner(), layout.inner(), nl, nt, md, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaManifold)]
pub fn js_rta_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    landmark_stratified_rta_manifold(graph.inner(), layout.inner(), nl, nt, md, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaDetail2d)]
pub fn js_rta_detail_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    let soa = layout.inner().to_layout2d();
    let inner = landmark_stratified_rta_detail(graph.inner(), &soa, nl, nt, md, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaDetailNd)]
pub fn js_rta_detail_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    let inner =
        landmark_stratified_rta_detail_nd(graph.inner(), layout.inner(), nl, nt, md, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaDetailManifold)]
pub fn js_rta_detail_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let (nl, nt, md, seed_u64) = parse_rta_params(num_landmarks, num_triplets, min_degree, seed)?;
    let inner = landmark_stratified_rta_detail_manifold(
        graph.inner(),
        layout.inner(),
        nl,
        nt,
        md,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaFromLandmarks2d)]
pub fn js_rta_from_landmarks_2d(
    layout: &JsLayout2d,
    landmark_ids: &JsValue,
    landmark_dists: &JsValue,
    num_triplets: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let ids = parse_u32_sequence(landmark_ids, "landmarkIds")?;
    let dists = parse_rta_landmark_f64_sequence(landmark_dists, "landmarkDists")?;
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner = landmark_stratified_rta_from_landmarks_f64(&soa, &ids, &dists, nt, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaFromLandmarksNd)]
pub fn js_rta_from_landmarks_nd(
    layout: &JsLayoutNd,
    landmark_ids: &JsValue,
    landmark_dists: &JsValue,
    num_triplets: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let ids = parse_u32_sequence(landmark_ids, "landmarkIds")?;
    let dists = parse_rta_landmark_f64_sequence(landmark_dists, "landmarkDists")?;
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let inner =
        landmark_stratified_rta_from_landmarks_f64_nd(layout.inner(), &ids, &dists, nt, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

#[wasm_bindgen(js_name = _landmarkStratifiedRtaFromLandmarksManifold)]
pub fn js_rta_from_landmarks_manifold(
    layout: &JsManifoldLayout2d,
    landmark_ids: &JsValue,
    landmark_dists: &JsValue,
    num_triplets: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let ids = parse_u32_sequence(landmark_ids, "landmarkIds")?;
    let dists = parse_rta_landmark_f64_sequence(landmark_dists, "landmarkDists")?;
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = landmark_stratified_rta_from_landmarks_f64_manifold(
        layout.inner(),
        &ids,
        &dists,
        nt,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}
