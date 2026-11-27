//! WebAssembly bindings for ideal edge lengths, proximity edge extraction,
//! PRISM distortion, and Adjusted Patch-NIG metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_seed_u64, parse_u32_sequence,
};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_summaries::{JsAdjustedPatchNigDetail, JsPrismDistortion};
use egraph::core::CoreError;
use egraph::metrics::{
    adjusted_patch_nig, adjusted_patch_nig_detail, adjusted_patch_nig_multiscale,
    gabriel_graph_edges, ideal_edge_lengths, ideal_edge_lengths_with_unit, prism_distortion,
    prism_distortion_with_edges, relative_neighborhood_edges,
};
use wasm_bindgen::prelude::*;

const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;

/// Evaluates ideal edge-length preservation error sum for `graph` under `layout`.
#[wasm_bindgen(js_name = _idealEdgeLengths2d)]
pub fn js_ideal_edge_lengths(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Evaluates ideal edge-length preservation error against a uniform `unitEdgeLength > 0`.
#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnit2d)]
pub fn js_ideal_edge_lengths_with_unit(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    unit_edge_length: f64,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths_with_unit(graph.inner(), &soa, unit_edge_length).map_err(core_err_to_js)
}

/// Computes PRISM layout-preservation distortion (`sigmaDist`, `sigmaDisp`) comparing `originalLayout`
/// and `adjustedLayout`.
#[wasm_bindgen(js_name = prismDistortion)]
pub fn js_prism_distortion(
    original_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
) -> Result<JsPrismDistortion, JsError> {
    let orig_soa = original_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let res = prism_distortion(&orig_soa, &adj_soa).map_err(core_err_to_js)?;
    Ok(JsPrismDistortion::new(res.sigma_dist, res.sigma_disp))
}

/// Computes PRISM layout-preservation distortion using flat `[u0, v0, ...]` proximity `edges`.
#[wasm_bindgen(js_name = prismDistortionWithEdges)]
pub fn js_prism_distortion_with_edges(
    original_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
    edges: &[u32],
) -> Result<JsPrismDistortion, JsError> {
    if !edges.len().is_multiple_of(2) {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            edges.len() + 1,
            edges.len(),
            "prismDistortionWithEdges::edges",
        )));
    }
    let mut pairs = Vec::with_capacity(edges.len() / 2);
    for &[u, v] in edges.as_chunks::<2>().0 {
        pairs.push((u, v));
    }
    let orig_soa = original_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let res = prism_distortion_with_edges(&orig_soa, &adj_soa, &pairs).map_err(core_err_to_js)?;
    Ok(JsPrismDistortion::new(res.sigma_dist, res.sigma_disp))
}

/// Extracts exact undirected Gabriel Graph (`GG`) edges as flat `[u0, v0, ...]` (`Uint32Array`).
#[wasm_bindgen(js_name = _gabrielGraphEdges2d)]
pub fn js_gabriel_graph_edges(layout: &JsLayout2d) -> Result<Vec<u32>, JsError> {
    let soa = layout.inner().to_layout2d();
    let edges = gabriel_graph_edges(&soa).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

/// Extracts exact undirected Relative Neighborhood Graph (`RNG`) edges as flat `[u0, v0, ...]` (`Uint32Array`).
#[wasm_bindgen(js_name = _relativeNeighborhoodEdges2d)]
pub fn js_relative_neighborhood_edges(layout: &JsLayout2d) -> Result<Vec<u32>, JsError> {
    let soa = layout.inner().to_layout2d();
    let edges = relative_neighborhood_edges(&soa).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

/// Computes Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`).
#[wasm_bindgen(js_name = adjustedPatchNig)]
pub fn js_adjusted_patch_nig(
    layout: &JsLayout2d,
    labels: &JsValue,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let radius = match patch_radius {
        None => DEFAULT_PATCH_NIG_RADIUS,
        Some(v) => parse_non_negative_usize(v, "patchRadius")?,
    };
    let c_sz = match canvas_size {
        None => DEFAULT_PATCH_NIG_CANVAS_SIZE,
        Some(v) => parse_non_negative_usize(v, "canvasSize")?,
    };
    let pad = pad_frac.unwrap_or(DEFAULT_PATCH_NIG_PAD_FRAC);
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    adjusted_patch_nig(&soa, &parsed_labels, radius, c_sz, pad, seed_u64).map_err(core_err_to_js)
}

/// Computes single-scale Adjusted Patch-NIG with patch occupancy diagnostics (`AdjustedPatchNigDetail`).
#[wasm_bindgen(js_name = adjustedPatchNigDetail)]
pub fn js_adjusted_patch_nig_detail(
    layout: &JsLayout2d,
    labels: &JsValue,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    seed: Option<f64>,
) -> Result<JsAdjustedPatchNigDetail, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let radius = match patch_radius {
        None => DEFAULT_PATCH_NIG_RADIUS,
        Some(v) => parse_non_negative_usize(v, "patchRadius")?,
    };
    let c_sz = match canvas_size {
        None => DEFAULT_PATCH_NIG_CANVAS_SIZE,
        Some(v) => parse_non_negative_usize(v, "canvasSize")?,
    };
    let pad = pad_frac.unwrap_or(DEFAULT_PATCH_NIG_PAD_FRAC);
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner = adjusted_patch_nig_detail(&soa, &parsed_labels, radius, c_sz, pad, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsAdjustedPatchNigDetail::from_inner(inner))
}

/// Computes multi-scale Adjusted Patch-NIG across `radii` (default `[2, 8, 32]`).
#[wasm_bindgen(js_name = adjustedPatchNigMultiscale)]
pub fn js_adjusted_patch_nig_multiscale(
    layout: &JsLayout2d,
    labels: &JsValue,
    radii: Option<Vec<u32>>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let radii_usize: Vec<usize> = match radii {
        Some(rs) => rs.into_iter().map(|r| r as usize).collect(),
        None => vec![2, 8, 32],
    };
    let c_sz = match canvas_size {
        None => DEFAULT_PATCH_NIG_CANVAS_SIZE,
        Some(v) => parse_non_negative_usize(v, "canvasSize")?,
    };
    let pad = pad_frac.unwrap_or(DEFAULT_PATCH_NIG_PAD_FRAC);
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    adjusted_patch_nig_multiscale(&soa, &parsed_labels, &radii_usize, c_sz, pad, seed_u64)
        .map_err(core_err_to_js)
}
