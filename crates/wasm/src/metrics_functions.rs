//! WebAssembly bindings for ideal edge lengths, proximity edge extraction,
//! PRISM distortion, and Adjusted Patch-NIG metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, invalid_input_err, parse_non_negative_usize, parse_u32_sequence,
};
use crate::js_options::{check_options_arg, get_prop, opt_f64_prop, opt_seed_prop, opt_usize_prop};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_summaries::{JsAdjustedPatchNigDetail, JsPrismDistortion};
use egraph::core::CoreError;
use egraph::metrics::{
    adjusted_patch_nig, adjusted_patch_nig_detail, adjusted_patch_nig_multiscale,
    gabriel_graph_edges, ideal_edge_lengths, ideal_edge_lengths_with_unit, prism_distortion,
    prism_distortion_with_edges, relative_neighborhood_edges,
};
use wasm_bindgen::{JsCast, prelude::*};

const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;

fn parse_radii_prop(options: &JsValue) -> Result<Option<Vec<usize>>, JsValue> {
    let val = get_prop(options, "radii")?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(Some(arr.to_vec().into_iter().map(|v| v as usize).collect()));
    }
    if !val.is_string() && val.is_object() {
        let iter = js_sys::try_iter(&val)
            .map_err(|err| invalid_input_err(format!("failed to iterate options.radii: {err:?}")))?
            .ok_or_else(|| invalid_input_err("options.radii must be an array of integers"))?;
        let mut out = Vec::new();
        for item_res in iter {
            let item = item_res
                .map_err(|err| invalid_input_err(format!("failed to read radii entry: {err:?}")))?;
            let num = item
                .as_f64()
                .ok_or_else(|| invalid_input_err("options.radii entries must be integers"))?;
            out.push(parse_non_negative_usize(num, "radii")?);
        }
        return Ok(Some(out));
    }
    Err(invalid_input_err(
        "options.radii must be a Uint32Array or integer array",
    ))
}

struct ParsedPatchNigOptions {
    radius: usize,
    canvas_size: usize,
    pad_frac: f64,
    seed: u64,
    radii: Option<Vec<usize>>,
}

fn parse_patch_nig_options(options: &JsValue, ctx: &str) -> Result<ParsedPatchNigOptions, JsValue> {
    let mut radius = DEFAULT_PATCH_NIG_RADIUS;
    let mut canvas_size = DEFAULT_PATCH_NIG_CANVAS_SIZE;
    let mut pad_frac = DEFAULT_PATCH_NIG_PAD_FRAC;
    let mut seed = DEFAULT_SEED;
    let mut radii = None;
    if check_options_arg(options, ctx)? {
        radius = opt_usize_prop(options, "patchRadius", radius)?;
        canvas_size = opt_usize_prop(options, "canvasSize", canvas_size)?;
        pad_frac = opt_f64_prop(options, "padFrac", pad_frac)?;
        seed = opt_seed_prop(options)?;
        radii = parse_radii_prop(options)?;
    }
    Ok(ParsedPatchNigOptions {
        radius,
        canvas_size,
        pad_frac,
        seed,
        radii,
    })
}

/// Evaluates ideal edge-length preservation error sum for `graph` under `layout`.
#[wasm_bindgen(js_name = _idealEdgeLengths2d)]
pub fn js_ideal_edge_lengths(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Evaluates ideal edge-length preservation error against a uniform `unitEdgeLength > 0`.
#[wasm_bindgen(js_name = _idealEdgeLengthsWithUnit2d)]
pub fn js_ideal_edge_lengths_with_unit(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    unit_edge_length: f64,
) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths_with_unit(graph.inner(), &soa, unit_edge_length).map_err(core_err_to_js)
}

/// Computes PRISM layout-preservation distortion (`sigmaDist`, `sigmaDisp`) comparing `originalLayout`
/// and `adjustedLayout`.
#[wasm_bindgen(js_name = prismDistortion)]
pub fn js_prism_distortion(
    original_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
) -> Result<JsPrismDistortion, JsValue> {
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
) -> Result<JsPrismDistortion, JsValue> {
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
pub fn js_gabriel_graph_edges(layout: &JsLayout2d) -> Result<Vec<u32>, JsValue> {
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
pub fn js_relative_neighborhood_edges(layout: &JsLayout2d) -> Result<Vec<u32>, JsValue> {
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
///
/// `options?: { patchRadius?, canvasSize?, padFrac?, seed? }`.
#[wasm_bindgen(js_name = adjustedPatchNig)]
pub fn js_adjusted_patch_nig(
    layout: &JsLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let opts = parse_patch_nig_options(options, "adjustedPatchNig")?;
    let soa = layout.inner().to_layout2d();
    adjusted_patch_nig(
        &soa,
        &parsed_labels,
        opts.radius,
        opts.canvas_size,
        opts.pad_frac,
        opts.seed,
    )
    .map_err(core_err_to_js)
}

/// Computes single-scale Adjusted Patch-NIG with patch occupancy diagnostics (`AdjustedPatchNigDetail`).
///
/// `options?: { patchRadius?, canvasSize?, padFrac?, seed? }`.
#[wasm_bindgen(js_name = adjustedPatchNigDetail)]
pub fn js_adjusted_patch_nig_detail(
    layout: &JsLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<JsAdjustedPatchNigDetail, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let opts = parse_patch_nig_options(options, "adjustedPatchNigDetail")?;
    let soa = layout.inner().to_layout2d();
    let inner = adjusted_patch_nig_detail(
        &soa,
        &parsed_labels,
        opts.radius,
        opts.canvas_size,
        opts.pad_frac,
        opts.seed,
    )
    .map_err(core_err_to_js)?;
    Ok(JsAdjustedPatchNigDetail::from_inner(inner))
}

/// Computes multi-scale Adjusted Patch-NIG across `options.radii` (default `[2, 8, 32]`).
///
/// `options?: { radii?, canvasSize?, padFrac?, seed? }`.
#[wasm_bindgen(js_name = adjustedPatchNigMultiscale)]
pub fn js_adjusted_patch_nig_multiscale(
    layout: &JsLayout2d,
    labels: &JsValue,
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let opts = parse_patch_nig_options(options, "adjustedPatchNigMultiscale")?;
    let radii_usize = opts.radii.unwrap_or_else(|| vec![2, 8, 32]);
    let soa = layout.inner().to_layout2d();
    adjusted_patch_nig_multiscale(
        &soa,
        &parsed_labels,
        &radii_usize,
        opts.canvas_size,
        opts.pad_frac,
        opts.seed,
    )
    .map_err(core_err_to_js)
}
