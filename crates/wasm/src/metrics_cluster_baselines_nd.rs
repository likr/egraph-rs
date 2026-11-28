//! WebAssembly bindings for N-D (`LayoutNd`) and Manifold (`ManifoldLayout2d`) cluster separation
//! and clustering validity metrics.

use egraph::metrics::{
    calinski_harabasz_score, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
    distribution_consistency, neighborhood_hit, pairwise_distance_overlap_error, sepme_gab,
    sepme_gong, sepme_knng, sepme_rng, silhouette_score,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, parse_u32_sequence};
use crate::layout_buffers::JsLayoutNd;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_cluster_baselines::{
    JsPartitionAgreementResult, parse_cq_kmeans_options, parse_dc_grid_size, parse_gong_gamma,
    parse_nh_k, parse_pdoe_bins, parse_sepme_k,
};

#[wasm_bindgen(js_name = _cqKmeansNd)]
pub fn js_cq_kmeans_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let (n_init, max_iters, seed) = parse_cq_kmeans_options(options, "cqKmeans")?;
    let inner = cq_kmeans(layout.inner(), &parsed_labels, n_init, max_iters, seed)
        .map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqKmeansManifold)]
pub fn js_cq_kmeans_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let (n_init, max_iters, seed) = parse_cq_kmeans_options(options, "cqKmeans")?;
    let inner = cq_kmeans(layout.inner(), &parsed_labels, n_init, max_iters, seed)
        .map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqHacNd)]
pub fn js_cq_hac_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let inner = cq_hac(layout.inner(), &parsed_labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqHacManifold)]
pub fn js_cq_hac_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let inner = cq_hac(layout.inner(), &parsed_labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _distanceConsistencyNd)]
pub fn js_distance_consistency_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    distance_consistency(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distanceConsistencyManifold)]
pub fn js_distance_consistency_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    distance_consistency(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distributionConsistencyNd)]
pub fn js_distribution_consistency_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_dc_grid_size(options, "distributionConsistency")?;
    distribution_consistency(layout.inner(), &parsed_labels, gs).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distributionConsistencyManifold)]
pub fn js_distribution_consistency_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_dc_grid_size(options, "distributionConsistency")?;
    distribution_consistency(layout.inner(), &parsed_labels, gs).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeKnngNd)]
pub fn js_sepme_knng_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_sepme_k(options, "sepmeKnng")?;
    sepme_knng(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeKnngManifold)]
pub fn js_sepme_knng_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_sepme_k(options, "sepmeKnng")?;
    sepme_knng(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeRngNd)]
pub fn js_sepme_rng_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_rng(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeRngManifold)]
pub fn js_sepme_rng_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_rng(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGabNd)]
pub fn js_sepme_gab_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gab(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGabManifold)]
pub fn js_sepme_gab_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gab(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGongNd)]
pub fn js_sepme_gong_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let g = parse_gong_gamma(options, "sepmeGong")?;
    sepme_gong(layout.inner(), &parsed_labels, g).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGongManifold)]
pub fn js_sepme_gong_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let g = parse_gong_gamma(options, "sepmeGong")?;
    sepme_gong(layout.inner(), &parsed_labels, g).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodHitNd)]
pub fn js_neighborhood_hit_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_nh_k(options, "neighborhoodHit")?;
    neighborhood_hit(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodHitManifold)]
pub fn js_neighborhood_hit_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_nh_k(options, "neighborhoodHit")?;
    neighborhood_hit(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _silhouetteScoreNd)]
pub fn js_silhouette_score_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    silhouette_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _silhouetteScoreManifold)]
pub fn js_silhouette_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    silhouette_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _calinskiHarabaszScoreNd)]
pub fn js_calinski_harabasz_score_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    calinski_harabasz_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _calinskiHarabaszScoreManifold)]
pub fn js_calinski_harabasz_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    calinski_harabasz_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _daviesBouldinScoreNd)]
pub fn js_davies_bouldin_score_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    davies_bouldin_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _daviesBouldinScoreManifold)]
pub fn js_davies_bouldin_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    davies_bouldin_score(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _pairwiseDistanceOverlapErrorNd)]
pub fn js_pdoe_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_pdoe_bins(options, "pairwiseDistanceOverlapError")?;
    pairwise_distance_overlap_error(layout.inner(), &parsed_labels, bins).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _pairwiseDistanceOverlapErrorManifold)]
pub fn js_pdoe_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_pdoe_bins(options, "pairwiseDistanceOverlapError")?;
    pairwise_distance_overlap_error(layout.inner(), &parsed_labels, bins).map_err(core_err_to_js)
}
