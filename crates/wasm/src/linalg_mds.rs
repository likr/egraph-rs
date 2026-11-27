//! WebAssembly bindings for N-D Classical MDS and distance/pivot MDS constructors.

use egraph::core::{Layout2d, LayoutNd};
use egraph::linalg::{
    PivotMdsConfig, classical_mds, classical_mds_from_distances, pivot_mds,
    pivot_mds_from_distances, pivot_mds_with_pivots,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};

const DEFAULT_NUM_PIVOTS: usize = 50;
const DEFAULT_MIN_DEGREE: usize = 2;

fn parse_optional_target_std(value: &JsValue) -> Result<Option<f64>, JsError> {
    if value.is_undefined() {
        return Ok(Some(1.0));
    }
    if value.is_null() {
        return Ok(None);
    }
    if let Some(num) = value.as_f64() {
        return Ok(Some(num));
    }
    Err(JsError::new(
        "targetStd must be a positive finite number, null, or undefined",
    ))
}

/// Computes a `dims`-dimensional Classical MDS `LayoutNd` (`f32`, default `dims = 2`).
#[wasm_bindgen(js_name = classicalMds)]
pub fn js_classical_mds(graph: &JsCsrGraph, dims: Option<f64>) -> Result<JsLayoutNd, JsError> {
    let d = parse_optional_usize(dims, 2, "dims")?;
    let layout: LayoutNd<f32> = classical_mds(graph.inner(), d).map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Computes a `dims`-dimensional Classical MDS `LayoutNd` (`f32`) from a flat `(numNodes, numNodes)`
/// row-major distance matrix.
#[wasm_bindgen(js_name = classicalMdsFromDistances)]
pub fn js_classical_mds_from_distances(
    distances_nn: &[f64],
    num_nodes: f64,
    dims: Option<f64>,
) -> Result<JsLayoutNd, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_optional_usize(dims, 2, "dims")?;
    let layout: LayoutNd<f32> =
        classical_mds_from_distances(distances_nn, n, d).map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Computes a 2D `Layout2d` (`f32`) via Core-First MaxMin PivotMDS.
#[wasm_bindgen(js_name = pivotMds)]
pub fn js_pivot_mds(
    graph: &JsCsrGraph,
    num_pivots: Option<f64>,
    seed: Option<f64>,
    min_degree: Option<f64>,
    target_std: &JsValue,
) -> Result<JsLayout2d, JsError> {
    let pivots = parse_optional_usize(num_pivots, DEFAULT_NUM_PIVOTS, "numPivots")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let min_deg = parse_optional_usize(min_degree, DEFAULT_MIN_DEGREE, "minDegree")?;
    let t_std = parse_optional_target_std(target_std)?;
    let cfg = PivotMdsConfig {
        num_pivots: pivots,
        min_degree: min_deg,
        seed: seed_u64,
        target_std: t_std,
    };
    let layout: Layout2d<f32> = pivot_mds(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a 2D `Layout2d` (`f32`) via PivotMDS from a flat `(numNodes, numPivots)` distance matrix.
#[wasm_bindgen(js_name = pivotMdsFromDistances)]
pub fn js_pivot_mds_from_distances(
    distances_nk: &[f64],
    num_nodes: f64,
    num_pivots: f64,
    target_std: &JsValue,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let k = parse_non_negative_usize(num_pivots, "numPivots")?;
    let t_std = parse_optional_target_std(target_std)?;
    let seed_u64 = parse_seed_u64(seed)?;
    let layout: Layout2d<f32> =
        pivot_mds_from_distances(distances_nk, n, k, t_std, seed_u64).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a 2D `Layout2d` (`f32`) via PivotMDS from an explicit slice of `pivots`.
#[wasm_bindgen(js_name = pivotMdsWithPivots)]
pub fn js_pivot_mds_with_pivots(
    graph: &JsCsrGraph,
    pivots: &[u32],
    target_std: &JsValue,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let t_std = parse_optional_target_std(target_std)?;
    let seed_u64 = parse_seed_u64(seed)?;
    let layout: Layout2d<f32> =
        pivot_mds_with_pivots(graph.inner(), pivots, t_std, seed_u64).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}
