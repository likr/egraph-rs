//! WebAssembly bindings for N-D (`LayoutNd`) and Manifold (`ManifoldLayout2d`) geometry,
//! stress, neighborhood, and proximity metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::layout_buffers::JsLayoutNd;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_crossings::JsLayoutDegeneracy;
use crate::metrics_geometry::{parse_pivots_and_seed, parse_samples_and_seed};
use crate::metrics_summaries::JsStressEvaluation;
use egraph::metrics::{
    gabriel_graph_edges, gabriel_graph_preservation, gabriel_graph_property, ideal_edge_lengths,
    ideal_edge_lengths_with_unit, layout_degeneracy, neighborhood_preservation_1hop,
    neighborhood_preservation_2hop, node_resolution, relative_neighborhood_edges,
    relative_neighborhood_preservation, sampled_neighborhood_preservation_1hop,
    sampled_neighborhood_preservation_2hop, sampled_scale_normalized_stress,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = _scaleNormalizedStressNd)]
pub fn js_sns_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    scale_normalized_stress(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressManifold)]
pub fn js_sns_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    scale_normalized_stress(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressDetailNd)]
pub fn js_sns_detail_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
) -> Result<JsStressEvaluation, JsValue> {
    let eval =
        scale_normalized_stress_detail(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsStressEvaluation::from_inner(eval))
}

#[wasm_bindgen(js_name = _scaleNormalizedStressDetailManifold)]
pub fn js_sns_detail_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<JsStressEvaluation, JsValue> {
    let eval =
        scale_normalized_stress_detail(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsStressEvaluation::from_inner(eval))
}

#[wasm_bindgen(js_name = _sampledScaleNormalizedStressNd)]
pub fn js_sampled_sns_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (pivots, seed_u64) = parse_pivots_and_seed(options, "sampledScaleNormalizedStress")?;
    sampled_scale_normalized_stress(graph.inner(), layout.inner(), pivots, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledScaleNormalizedStressManifold)]
pub fn js_sampled_sns_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (pivots, seed_u64) = parse_pivots_and_seed(options, "sampledScaleNormalizedStress")?;
    sampled_scale_normalized_stress(graph.inner(), layout.inner(), pivots, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistancesNd)]
pub fn js_sns_from_distances_nd(
    layout: &JsLayoutNd,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsValue> {
    scale_normalized_stress_from_distances(layout.inner(), distances, weights.as_deref())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistancesManifold)]
pub fn js_sns_from_distances_manifold(
    layout: &JsManifoldLayout2d,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsValue> {
    scale_normalized_stress_from_distances(layout.inner(), distances, weights.as_deref())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation1HopNd)]
pub fn js_np1_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    neighborhood_preservation_1hop(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation1HopManifold)]
pub fn js_np1_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    neighborhood_preservation_1hop(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation2HopNd)]
pub fn js_np2_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    neighborhood_preservation_2hop(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation2HopManifold)]
pub fn js_np2_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    neighborhood_preservation_2hop(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp1Nd)]
pub fn js_sampled_np1_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp1")?;
    sampled_neighborhood_preservation_1hop(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp1Manifold)]
pub fn js_sampled_np1_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp1")?;
    sampled_neighborhood_preservation_1hop(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp2Nd)]
pub fn js_sampled_np2_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp2")?;
    sampled_neighborhood_preservation_2hop(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp2Manifold)]
pub fn js_sampled_np2_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_samples_and_seed(options, "sampledNp2")?;
    sampled_neighborhood_preservation_2hop(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _gabrielGraphPropertyNd)]
pub fn js_gabriel_property_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    gabriel_graph_property(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _gabrielGraphPropertyManifold)]
pub fn js_gabriel_property_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<f64, JsValue> {
    gabriel_graph_property(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _nodeResolutionNd)]
pub fn js_node_resolution_nd(layout: &JsLayoutNd) -> Result<f64, JsValue> {
    node_resolution(layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _nodeResolutionManifold)]
pub fn js_node_resolution_manifold(layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    node_resolution(layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _drngPreservationNd)]
pub fn js_drng_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    relative_neighborhood_preservation(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _drngPreservationManifold)]
pub fn js_drng_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    relative_neighborhood_preservation(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _dggPreservationNd)]
pub fn js_dgg_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    gabriel_graph_preservation(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _dggPreservationManifold)]
pub fn js_dgg_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsValue> {
    gabriel_graph_preservation(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsNd)]
pub fn js_ideal_edge_lengths_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsValue> {
    ideal_edge_lengths(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsManifold)]
pub fn js_ideal_edge_lengths_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<f64, JsValue> {
    ideal_edge_lengths(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnitNd)]
pub fn js_ideal_edge_lengths_with_unit_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    unit_edge_length: f64,
) -> Result<f64, JsValue> {
    ideal_edge_lengths_with_unit(graph.inner(), layout.inner(), unit_edge_length)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnitManifold)]
pub fn js_ideal_edge_lengths_with_unit_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    unit_edge_length: f64,
) -> Result<f64, JsValue> {
    ideal_edge_lengths_with_unit(graph.inner(), layout.inner(), unit_edge_length)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _layoutDegeneracyNd)]
pub fn js_layout_degeneracy_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
) -> Result<JsLayoutDegeneracy, JsValue> {
    let inner = layout_degeneracy(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy::from_inner(inner))
}

#[wasm_bindgen(js_name = _layoutDegeneracyManifold)]
pub fn js_layout_degeneracy_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<JsLayoutDegeneracy, JsValue> {
    let inner = layout_degeneracy(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy::from_inner(inner))
}

#[wasm_bindgen(js_name = _gabrielGraphEdgesNd)]
pub fn js_gabriel_graph_edges_nd(layout: &JsLayoutNd) -> Result<Vec<u32>, JsValue> {
    let edges = gabriel_graph_edges(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _gabrielGraphEdgesManifold)]
pub fn js_gabriel_graph_edges_manifold(layout: &JsManifoldLayout2d) -> Result<Vec<u32>, JsValue> {
    let edges = gabriel_graph_edges(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _relativeNeighborhoodEdgesNd)]
pub fn js_relative_neighborhood_edges_nd(layout: &JsLayoutNd) -> Result<Vec<u32>, JsValue> {
    let edges = relative_neighborhood_edges(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _relativeNeighborhoodEdgesManifold)]
pub fn js_relative_neighborhood_edges_manifold(
    layout: &JsManifoldLayout2d,
) -> Result<Vec<u32>, JsValue> {
    let edges = relative_neighborhood_edges(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}
