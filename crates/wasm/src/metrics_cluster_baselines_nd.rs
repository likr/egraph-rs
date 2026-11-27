//! WebAssembly bindings for N-D (`LayoutNd`) and Manifold (`ManifoldLayout2d`) cluster separation
//! and clustering validity metrics.

use egraph::metrics::{
    ClusterBaselineConfig, calinski_harabasz_score_manifold, calinski_harabasz_score_nd,
    cq_hac_manifold, cq_hac_nd, cq_kmeans_manifold, cq_kmeans_nd, davies_bouldin_score_manifold,
    davies_bouldin_score_nd, distance_consistency_manifold, distance_consistency_nd,
    distribution_consistency_manifold, distribution_consistency_nd, neighborhood_hit_manifold,
    neighborhood_hit_nd, pairwise_distance_overlap_error_manifold,
    pairwise_distance_overlap_error_nd, sepme_gab_manifold, sepme_gab_nd, sepme_gong_manifold,
    sepme_gong_nd, sepme_knng_manifold, sepme_knng_nd, sepme_rng_manifold, sepme_rng_nd,
    silhouette_score_manifold, silhouette_score_nd,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, parse_optional_usize, parse_seed_u64, parse_u32_sequence};
use crate::layout_buffers::JsLayoutNd;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_cluster_baselines::JsPartitionAgreementResult;

#[wasm_bindgen(js_name = _cqKmeansNd)]
pub fn js_cq_kmeans_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    n_init: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
) -> Result<JsPartitionAgreementResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let n_init_val = parse_optional_usize(n_init, defaults.kmeans_n_init, "nInit")?;
    let max_iters_val =
        parse_optional_usize(max_iterations, defaults.kmeans_max_iters, "maxIterations")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = cq_kmeans_nd(
        layout.inner(),
        &parsed_labels,
        n_init_val,
        max_iters_val,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqKmeansManifold)]
pub fn js_cq_kmeans_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    n_init: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
) -> Result<JsPartitionAgreementResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let n_init_val = parse_optional_usize(n_init, defaults.kmeans_n_init, "nInit")?;
    let max_iters_val =
        parse_optional_usize(max_iterations, defaults.kmeans_max_iters, "maxIterations")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = cq_kmeans_manifold(
        layout.inner(),
        &parsed_labels,
        n_init_val,
        max_iters_val,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqHacNd)]
pub fn js_cq_hac_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let inner = cq_hac_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqHacManifold)]
pub fn js_cq_hac_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let inner = cq_hac_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _distanceConsistencyNd)]
pub fn js_distance_consistency_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    distance_consistency_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distanceConsistencyManifold)]
pub fn js_distance_consistency_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    distance_consistency_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distributionConsistencyNd)]
pub fn js_distribution_consistency_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    grid_size: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_optional_usize(grid_size, defaults.dc_grid_size, "gridSize")?;
    distribution_consistency_nd(layout.inner(), &parsed_labels, gs).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distributionConsistencyManifold)]
pub fn js_distribution_consistency_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    grid_size: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_optional_usize(grid_size, defaults.dc_grid_size, "gridSize")?;
    distribution_consistency_manifold(layout.inner(), &parsed_labels, gs).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeKnngNd)]
pub fn js_sepme_knng_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.sepme_knng_k, "k")?;
    sepme_knng_nd(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeKnngManifold)]
pub fn js_sepme_knng_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.sepme_knng_k, "k")?;
    sepme_knng_manifold(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeRngNd)]
pub fn js_sepme_rng_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_rng_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeRngManifold)]
pub fn js_sepme_rng_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_rng_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGabNd)]
pub fn js_sepme_gab_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gab_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGabManifold)]
pub fn js_sepme_gab_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gab_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGongNd)]
pub fn js_sepme_gong_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    gamma: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gong_nd(
        layout.inner(),
        &parsed_labels,
        gamma.unwrap_or(defaults.gong_gamma),
    )
    .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGongManifold)]
pub fn js_sepme_gong_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    gamma: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    sepme_gong_manifold(
        layout.inner(),
        &parsed_labels,
        gamma.unwrap_or(defaults.gong_gamma),
    )
    .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodHitNd)]
pub fn js_neighborhood_hit_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.neighborhood_hit_k, "k")?;
    neighborhood_hit_nd(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodHitManifold)]
pub fn js_neighborhood_hit_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.neighborhood_hit_k, "k")?;
    neighborhood_hit_manifold(layout.inner(), &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _silhouetteScoreNd)]
pub fn js_silhouette_score_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    silhouette_score_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _silhouetteScoreManifold)]
pub fn js_silhouette_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    silhouette_score_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _calinskiHarabaszScoreNd)]
pub fn js_calinski_harabasz_score_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    calinski_harabasz_score_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _calinskiHarabaszScoreManifold)]
pub fn js_calinski_harabasz_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    calinski_harabasz_score_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _daviesBouldinScoreNd)]
pub fn js_davies_bouldin_score_nd(layout: &JsLayoutNd, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    davies_bouldin_score_nd(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _daviesBouldinScoreManifold)]
pub fn js_davies_bouldin_score_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    davies_bouldin_score_manifold(layout.inner(), &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _pairwiseDistanceOverlapErrorNd)]
pub fn js_pdoe_nd(
    layout: &JsLayoutNd,
    labels: &JsValue,
    num_bins: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_optional_usize(num_bins, defaults.num_bins, "numBins")?;
    pairwise_distance_overlap_error_nd(layout.inner(), &parsed_labels, bins).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _pairwiseDistanceOverlapErrorManifold)]
pub fn js_pdoe_manifold(
    layout: &JsManifoldLayout2d,
    labels: &JsValue,
    num_bins: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_optional_usize(num_bins, defaults.num_bins, "numBins")?;
    pairwise_distance_overlap_error_manifold(layout.inner(), &parsed_labels, bins)
        .map_err(core_err_to_js)
}
