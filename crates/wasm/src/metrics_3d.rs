//! WebAssembly bindings for 3D (`Layout3d`) geometry, stress, neighborhood,
//! proximity, and cluster separation metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_u32_sequence};
use crate::layout3d::JsLayout3d;
use crate::metrics_cluster_baselines::{
    JsPartitionAgreementResult, parse_cq_kmeans_options, parse_dc_grid_size, parse_gong_gamma,
    parse_nh_k, parse_pdoe_bins, parse_sepme_k,
};
use crate::metrics_crossings::JsLayoutDegeneracy;
use crate::metrics_geometry::{parse_pivots_and_seed, parse_samples_and_seed};
use crate::metrics_summaries::JsStressEvaluation;
use egraph::metrics::{
    calinski_harabasz_score, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
    distribution_consistency, gabriel_graph_edges, gabriel_graph_preservation,
    gabriel_graph_property, ideal_edge_lengths, ideal_edge_lengths_with_unit, layout_degeneracy,
    neighborhood_hit, neighborhood_preservation_1hop, neighborhood_preservation_2hop,
    node_resolution, pairwise_distance_overlap_error, relative_neighborhood_edges,
    relative_neighborhood_preservation, sampled_neighborhood_preservation_1hop,
    sampled_neighborhood_preservation_2hop, sampled_scale_normalized_stress,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances, sepme_gab, sepme_gong, sepme_knng, sepme_rng,
    silhouette_score,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = _scaleNormalizedStress3d)]
pub fn js_sns_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    scale_normalized_stress(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressDetail3d)]
pub fn js_sns_detail_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
) -> Result<JsStressEvaluation, JsValue> {
    let soa = layout.inner().to_layout3d();
    let eval = scale_normalized_stress_detail(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsStressEvaluation::from_inner(eval))
}

#[wasm_bindgen(js_name = _sampledScaleNormalizedStress3d)]
pub fn js_sampled_sns_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (pivots, seed_u64) = parse_pivots_and_seed(options, "sampledScaleNormalizedStress")?;
    let soa = layout.inner().to_layout3d();
    sampled_scale_normalized_stress(graph.inner(), &soa, pivots, seed_u64).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistances3d)]
pub fn js_sns_from_distances_3d(
    layout: &JsLayout3d,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    scale_normalized_stress_from_distances(&soa, distances, weights.as_deref())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation1Hop3d)]
pub fn js_np1_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    neighborhood_preservation_1hop(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation2Hop3d)]
pub fn js_np2_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    neighborhood_preservation_2hop(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp13d)]
pub fn js_sampled_np1_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp1")?;
    let soa = layout.inner().to_layout3d();
    sampled_neighborhood_preservation_1hop(graph.inner(), &soa, samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp23d)]
pub fn js_sampled_np2_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp2")?;
    let soa = layout.inner().to_layout3d();
    sampled_neighborhood_preservation_2hop(graph.inner(), &soa, samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _gabrielGraphProperty3d)]
pub fn js_gabriel_property_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    gabriel_graph_property(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _nodeResolution3d)]
pub fn js_node_resolution_3d(layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    node_resolution(&soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _drngPreservation3d)]
pub fn js_drng_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    relative_neighborhood_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _dggPreservation3d)]
pub fn js_dgg_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    gabriel_graph_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengths3d)]
pub fn js_ideal_edge_lengths_3d(graph: &JsCsrGraph, layout: &JsLayout3d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    ideal_edge_lengths(graph.inner(), &soa).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnit3d)]
pub fn js_ideal_edge_lengths_with_unit_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
    unit_edge_length: f64,
) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout3d();
    ideal_edge_lengths_with_unit(graph.inner(), &soa, unit_edge_length).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _layoutDegeneracy3d)]
pub fn js_layout_degeneracy_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
) -> Result<JsLayoutDegeneracy, JsValue> {
    let soa = layout.inner().to_layout3d();
    let inner = layout_degeneracy(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy::from_inner(inner))
}

#[wasm_bindgen(js_name = _gabrielGraphEdges3d)]
pub fn js_gabriel_graph_edges_3d(layout: &JsLayout3d) -> Result<Vec<u32>, JsValue> {
    let soa = layout.inner().to_layout3d();
    let edges = gabriel_graph_edges(&soa).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _relativeNeighborhoodEdges3d)]
pub fn js_relative_neighborhood_edges_3d(layout: &JsLayout3d) -> Result<Vec<u32>, JsValue> {
    let soa = layout.inner().to_layout3d();
    let edges = relative_neighborhood_edges(&soa).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _cqKmeans3d)]
pub fn js_cq_kmeans_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let (n_init, max_iters, seed) = parse_cq_kmeans_options(options, "cqKmeans")?;
    let soa = layout.inner().to_layout3d();
    let inner = cq_kmeans(&soa, &parsed_labels, n_init, max_iters, seed).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _cqHac3d)]
pub fn js_cq_hac_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    let inner = cq_hac(&soa, &parsed_labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult::from_inner(inner))
}

#[wasm_bindgen(js_name = _distanceConsistency3d)]
pub fn js_distance_consistency_3d(layout: &JsLayout3d, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    distance_consistency(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _distributionConsistency3d)]
pub fn js_distribution_consistency_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_dc_grid_size(options, "distributionConsistency")?;
    let soa = layout.inner().to_layout3d();
    distribution_consistency(&soa, &parsed_labels, gs).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeKnng3d)]
pub fn js_sepme_knng_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_sepme_k(options, "sepmeKnng")?;
    let soa = layout.inner().to_layout3d();
    sepme_knng(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeRng3d)]
pub fn js_sepme_rng_3d(layout: &JsLayout3d, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    sepme_rng(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGab3d)]
pub fn js_sepme_gab_3d(layout: &JsLayout3d, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    sepme_gab(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sepmeGong3d)]
pub fn js_sepme_gong_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let g = parse_gong_gamma(options, "sepmeGong")?;
    let soa = layout.inner().to_layout3d();
    sepme_gong(&soa, &parsed_labels, g).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodHit3d)]
pub fn js_neighborhood_hit_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_nh_k(options, "neighborhoodHit")?;
    let soa = layout.inner().to_layout3d();
    neighborhood_hit(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _silhouetteScore3d)]
pub fn js_silhouette_score_3d(layout: &JsLayout3d, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    silhouette_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _calinskiHarabaszScore3d)]
pub fn js_calinski_harabasz_score_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    calinski_harabasz_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _daviesBouldinScore3d)]
pub fn js_davies_bouldin_score_3d(layout: &JsLayout3d, labels: &JsValue) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout3d();
    davies_bouldin_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _pairwiseDistanceOverlapError3d)]
pub fn js_pdoe_3d(
    layout: &JsLayout3d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_pdoe_bins(options, "pairwiseDistanceOverlapError")?;
    let soa = layout.inner().to_layout3d();
    pairwise_distance_overlap_error(&soa, &parsed_labels, bins).map_err(core_err_to_js)
}
