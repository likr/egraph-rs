//! WebAssembly bindings for N-D Classical MDS and distance/pivot MDS constructors.

use egraph::core::{Layout2d, LayoutGenerator, LayoutNd};
use egraph::linalg::{ClassicalMds, PivotMds, PivotMdsConfig};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, invalid_input_err, parse_non_negative_usize, parse_optional_usize,
};
use crate::js_options::{check_options_arg, get_prop, opt_seed_prop, opt_usize_prop};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};

const DEFAULT_NUM_PIVOTS: usize = 50;
const DEFAULT_MIN_DEGREE: usize = 2;

pub(crate) fn parse_optional_target_std(value: &JsValue) -> Result<Option<f64>, JsValue> {
    if value.is_undefined() {
        return Ok(Some(1.0));
    }
    if value.is_null() {
        return Ok(None);
    }
    if let Some(num) = value.as_f64() {
        return Ok(Some(num));
    }
    Err(invalid_input_err(
        "targetStd must be a positive finite number, null, or undefined",
    ))
}

/// Computes a `dims`-dimensional Classical MDS `LayoutNd` (`f32`, default `dims = 2`).
#[wasm_bindgen(js_name = classicalMds)]
pub fn js_classical_mds(graph: &JsCsrGraph, dims: Option<f64>) -> Result<JsLayoutNd, JsValue> {
    let d = parse_optional_usize(dims, 2, "dims")?;
    let layout = ClassicalMds::<LayoutNd<f32>, f32>::new(graph.inner())
        .with_dims(d)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Computes a `dims`-dimensional Classical MDS `LayoutNd` (`f32`) from a flat `(numNodes, numNodes)`
/// row-major distance matrix.
#[wasm_bindgen(js_name = classicalMdsFromDistances)]
pub fn js_classical_mds_from_distances(
    distances_nn: &[f64],
    num_nodes: f64,
    dims: Option<f64>,
) -> Result<JsLayoutNd, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_optional_usize(dims, 2, "dims")?;
    let layout = ClassicalMds::<LayoutNd<f32>, f64>::from_distances(distances_nn, n)
        .with_dims(d)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Computes a 2D `Layout2d` (`f32`) via Core-First MaxMin PivotMDS.
#[wasm_bindgen(js_name = pivotMds)]
pub fn js_pivot_mds(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayout2d, JsValue> {
    let mut cfg = PivotMdsConfig::default();
    if check_options_arg(options, "pivotMds")? {
        cfg.num_pivots = opt_usize_prop(options, "numPivots", DEFAULT_NUM_PIVOTS)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.min_degree = opt_usize_prop(options, "minDegree", DEFAULT_MIN_DEGREE)?;
        let ts_val = get_prop(options, "targetStd")?;
        cfg.target_std = parse_optional_target_std(&ts_val)?;
    }
    let layout = PivotMds::<Layout2d<f32>, f32>::from_config(graph.inner(), cfg)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a 2D `Layout2d` (`f32`) via PivotMDS from a flat `(numNodes, numPivots)` distance matrix.
#[wasm_bindgen(js_name = pivotMdsFromDistances)]
pub fn js_pivot_mds_from_distances(
    distances_nk: &[f64],
    num_nodes: f64,
    num_pivots: f64,
    options: &JsValue,
) -> Result<JsLayout2d, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let k = parse_non_negative_usize(num_pivots, "numPivots")?;
    let (t_std, seed_u64) = if check_options_arg(options, "pivotMdsFromDistances")? {
        let ts_val = get_prop(options, "targetStd")?;
        (parse_optional_target_std(&ts_val)?, opt_seed_prop(options)?)
    } else {
        (Some(1.0), DEFAULT_SEED)
    };
    let layout = PivotMds::<Layout2d<f32>, f64>::from_distances(distances_nk, n, k)
        .with_target_std(t_std)
        .with_seed(seed_u64)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a 2D `Layout2d` (`f32`) via PivotMDS from an explicit slice of `pivots`.
#[wasm_bindgen(js_name = pivotMdsWithPivots)]
pub fn js_pivot_mds_with_pivots(
    graph: &JsCsrGraph,
    pivots: &[u32],
    options: &JsValue,
) -> Result<JsLayout2d, JsValue> {
    let (t_std, seed_u64) = if check_options_arg(options, "pivotMdsWithPivots")? {
        let ts_val = get_prop(options, "targetStd")?;
        (parse_optional_target_std(&ts_val)?, opt_seed_prop(options)?)
    } else {
        (Some(1.0), DEFAULT_SEED)
    };
    let layout = PivotMds::<Layout2d<f32>, f32>::new(graph.inner())
        .with_pivots(pivots)
        .with_target_std(t_std)
        .with_seed(seed_u64)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}
