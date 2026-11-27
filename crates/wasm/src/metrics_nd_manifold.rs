//! WebAssembly bindings for N-D (`LayoutNd`) and Manifold (`ManifoldLayout2d`) geometry,
//! stress, neighborhood, and proximity metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::layout_buffers::JsLayoutNd;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_crossings::JsLayoutDegeneracy;
use crate::metrics_summaries::JsStressEvaluation;
use egraph::metrics::{
    gabriel_graph_edges_manifold, gabriel_graph_edges_nd, gabriel_graph_preservation_manifold,
    gabriel_graph_preservation_nd, gabriel_graph_property_manifold, gabriel_graph_property_nd,
    ideal_edge_lengths_manifold, ideal_edge_lengths_nd, ideal_edge_lengths_with_unit_manifold,
    ideal_edge_lengths_with_unit_nd, layout_degeneracy_manifold, layout_degeneracy_nd,
    neighborhood_preservation_1hop_manifold, neighborhood_preservation_1hop_nd,
    neighborhood_preservation_2hop_manifold, neighborhood_preservation_2hop_nd,
    node_resolution_manifold, node_resolution_nd, relative_neighborhood_edges_manifold,
    relative_neighborhood_edges_nd, relative_neighborhood_preservation_manifold,
    relative_neighborhood_preservation_nd, sampled_neighborhood_preservation_1hop_manifold,
    sampled_neighborhood_preservation_1hop_nd, sampled_neighborhood_preservation_2hop_manifold,
    sampled_neighborhood_preservation_2hop_nd, sampled_scale_normalized_stress_manifold,
    sampled_scale_normalized_stress_nd, scale_normalized_stress_detail_manifold,
    scale_normalized_stress_detail_nd, scale_normalized_stress_from_distances_manifold,
    scale_normalized_stress_from_distances_nd, scale_normalized_stress_manifold,
    scale_normalized_stress_nd,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;

#[wasm_bindgen(js_name = _scaleNormalizedStressNd)]
pub fn js_sns_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    scale_normalized_stress_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressManifold)]
pub fn js_sns_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    scale_normalized_stress_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressDetailNd)]
pub fn js_sns_detail_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
) -> Result<JsStressEvaluation, JsError> {
    let eval =
        scale_normalized_stress_detail_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsStressEvaluation::from_inner(eval))
}

#[wasm_bindgen(js_name = _scaleNormalizedStressDetailManifold)]
pub fn js_sns_detail_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<JsStressEvaluation, JsError> {
    let eval = scale_normalized_stress_detail_manifold(graph.inner(), layout.inner())
        .map_err(core_err_to_js)?;
    Ok(JsStressEvaluation::from_inner(eval))
}

#[wasm_bindgen(js_name = _sampledScaleNormalizedStressNd)]
pub fn js_sampled_sns_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    num_pivots: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let pivots = match num_pivots {
        None => DEFAULT_NUM_PIVOTS,
        Some(v) => parse_non_negative_usize(v, "numPivots")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_scale_normalized_stress_nd(graph.inner(), layout.inner(), pivots, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledScaleNormalizedStressManifold)]
pub fn js_sampled_sns_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    num_pivots: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let pivots = match num_pivots {
        None => DEFAULT_NUM_PIVOTS,
        Some(v) => parse_non_negative_usize(v, "numPivots")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_scale_normalized_stress_manifold(graph.inner(), layout.inner(), pivots, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistancesNd)]
pub fn js_sns_from_distances_nd(
    layout: &JsLayoutNd,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsError> {
    scale_normalized_stress_from_distances_nd(layout.inner(), distances, weights.as_deref())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistancesManifold)]
pub fn js_sns_from_distances_manifold(
    layout: &JsManifoldLayout2d,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsError> {
    scale_normalized_stress_from_distances_manifold(layout.inner(), distances, weights.as_deref())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation1HopNd)]
pub fn js_np1_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    neighborhood_preservation_1hop_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation1HopManifold)]
pub fn js_np1_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    neighborhood_preservation_1hop_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation2HopNd)]
pub fn js_np2_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    neighborhood_preservation_2hop_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _neighborhoodPreservation2HopManifold)]
pub fn js_np2_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    neighborhood_preservation_2hop_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp1Nd)]
pub fn js_sampled_np1_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_neighborhood_preservation_1hop_nd(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp1Manifold)]
pub fn js_sampled_np1_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_neighborhood_preservation_1hop_manifold(
        graph.inner(),
        layout.inner(),
        samples,
        seed_u64,
    )
    .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp2Nd)]
pub fn js_sampled_np2_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_neighborhood_preservation_2hop_nd(graph.inner(), layout.inner(), samples, seed_u64)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _sampledNp2Manifold)]
pub fn js_sampled_np2_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    sampled_neighborhood_preservation_2hop_manifold(
        graph.inner(),
        layout.inner(),
        samples,
        seed_u64,
    )
    .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _gabrielGraphPropertyNd)]
pub fn js_gabriel_property_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    gabriel_graph_property_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _gabrielGraphPropertyManifold)]
pub fn js_gabriel_property_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<f64, JsError> {
    gabriel_graph_property_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _nodeResolutionNd)]
pub fn js_node_resolution_nd(layout: &JsLayoutNd) -> Result<f64, JsError> {
    node_resolution_nd(layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _nodeResolutionManifold)]
pub fn js_node_resolution_manifold(layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    node_resolution_manifold(layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _drngPreservationNd)]
pub fn js_drng_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    relative_neighborhood_preservation_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _drngPreservationManifold)]
pub fn js_drng_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    relative_neighborhood_preservation_manifold(graph.inner(), layout.inner())
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _dggPreservationNd)]
pub fn js_dgg_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    gabriel_graph_preservation_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _dggPreservationManifold)]
pub fn js_dgg_manifold(graph: &JsCsrGraph, layout: &JsManifoldLayout2d) -> Result<f64, JsError> {
    gabriel_graph_preservation_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsNd)]
pub fn js_ideal_edge_lengths_nd(graph: &JsCsrGraph, layout: &JsLayoutNd) -> Result<f64, JsError> {
    ideal_edge_lengths_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsManifold)]
pub fn js_ideal_edge_lengths_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<f64, JsError> {
    ideal_edge_lengths_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnitNd)]
pub fn js_ideal_edge_lengths_with_unit_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    unit_edge_length: f64,
) -> Result<f64, JsError> {
    ideal_edge_lengths_with_unit_nd(graph.inner(), layout.inner(), unit_edge_length)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnitManifold)]
pub fn js_ideal_edge_lengths_with_unit_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    unit_edge_length: f64,
) -> Result<f64, JsError> {
    ideal_edge_lengths_with_unit_manifold(graph.inner(), layout.inner(), unit_edge_length)
        .map_err(core_err_to_js)
}

#[wasm_bindgen(js_name = _layoutDegeneracyNd)]
pub fn js_layout_degeneracy_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
) -> Result<JsLayoutDegeneracy, JsError> {
    let inner = layout_degeneracy_nd(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy::from_inner(inner))
}

#[wasm_bindgen(js_name = _layoutDegeneracyManifold)]
pub fn js_layout_degeneracy_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<JsLayoutDegeneracy, JsError> {
    let inner =
        layout_degeneracy_manifold(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy::from_inner(inner))
}

#[wasm_bindgen(js_name = _gabrielGraphEdgesNd)]
pub fn js_gabriel_graph_edges_nd(layout: &JsLayoutNd) -> Result<Vec<u32>, JsError> {
    let edges = gabriel_graph_edges_nd(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _gabrielGraphEdgesManifold)]
pub fn js_gabriel_graph_edges_manifold(layout: &JsManifoldLayout2d) -> Result<Vec<u32>, JsError> {
    let edges = gabriel_graph_edges_manifold(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

#[wasm_bindgen(js_name = _relativeNeighborhoodEdgesNd)]
pub fn js_relative_neighborhood_edges_nd(layout: &JsLayoutNd) -> Result<Vec<u32>, JsError> {
    let edges = relative_neighborhood_edges_nd(layout.inner()).map_err(core_err_to_js)?;
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
) -> Result<Vec<u32>, JsError> {
    let edges = relative_neighborhood_edges_manifold(layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}
