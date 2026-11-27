//! Polymorphic `evaluateMetrics` entry points across `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::metrics::{MetricInputs, evaluate_metrics_with_inputs};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_eval::parse_eval_args;
use crate::metrics_report::JsMetricReport;

#[allow(clippy::too_many_arguments)]
fn evaluate_metrics_2d_inner(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: &JsValue,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
    cluster_baseline_options: &JsValue,
    original_layout: Option<&JsLayout2d>,
) -> Result<JsMetricReport, JsError> {
    let parsed = parse_eval_args(
        targets,
        num_pivots_or_options,
        num_samples,
        seed,
        ideal_edge_length,
        labels,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
        node_weights,
        cluster_weights,
        cluster_baseline_options,
    )?;
    let orig_soa = original_layout.map(|l| l.inner().to_layout2d());
    let inputs = MetricInputs {
        labels: parsed.labels.as_deref(),
        node_weights: parsed.node_weights.as_deref(),
        cluster_weights: parsed.cluster_weights.as_deref(),
        original_layout: orig_soa.as_ref(),
        sprawlter_half_widths: parsed.sprawlter_half_widths.as_deref(),
        sprawlter_half_heights: parsed.sprawlter_half_heights.as_deref(),
    };
    let soa = layout.inner().to_layout2d();
    let report = evaluate_metrics_with_inputs(
        graph.inner(),
        &soa,
        &parsed.targets,
        &parsed.config,
        &inputs,
    )
    .map_err(core_err_to_js)?;
    Ok(JsMetricReport::from_inner(report))
}

/// Selectively evaluates quality metrics on `Layout2d` without an `originalLayout`.
#[wasm_bindgen(js_name = _evaluateMetrics2d)]
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: &JsValue,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
    cluster_baseline_options: &JsValue,
) -> Result<JsMetricReport, JsError> {
    evaluate_metrics_2d_inner(
        graph,
        layout,
        targets,
        num_pivots_or_options,
        num_samples,
        seed,
        ideal_edge_length,
        labels,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
        node_weights,
        cluster_weights,
        cluster_baseline_options,
        None,
    )
}

/// Selectively evaluates quality metrics on `Layout2d` with a reference `originalLayout`.
#[wasm_bindgen(js_name = _evaluateMetrics2dWithOriginal)]
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_2d_with_original(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    original_layout: &JsLayout2d,
) -> Result<JsMetricReport, JsError> {
    evaluate_metrics_2d_inner(
        graph,
        layout,
        targets,
        num_pivots_or_options,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        Some(original_layout),
    )
}

/// Selectively evaluates quality metrics on `LayoutNd`, skipping unrequested work.
#[wasm_bindgen(js_name = _evaluateMetricsNd)]
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: &JsValue,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
    cluster_baseline_options: &JsValue,
) -> Result<JsMetricReport, JsError> {
    let parsed = parse_eval_args(
        targets,
        num_pivots_or_options,
        num_samples,
        seed,
        ideal_edge_length,
        labels,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
        node_weights,
        cluster_weights,
        cluster_baseline_options,
    )?;
    let inputs = MetricInputs {
        labels: parsed.labels.as_deref(),
        node_weights: parsed.node_weights.as_deref(),
        cluster_weights: parsed.cluster_weights.as_deref(),
        original_layout: None,
        sprawlter_half_widths: parsed.sprawlter_half_widths.as_deref(),
        sprawlter_half_heights: parsed.sprawlter_half_heights.as_deref(),
    };
    let report = evaluate_metrics_with_inputs(
        graph.inner(),
        layout.inner(),
        &parsed.targets,
        &parsed.config,
        &inputs,
    )
    .map_err(core_err_to_js)?;
    Ok(JsMetricReport::from_inner(report))
}

/// Selectively evaluates quality metrics on `ManifoldLayout2d`, skipping unrequested work.
#[wasm_bindgen(js_name = _evaluateMetricsManifold)]
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: &JsValue,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
    cluster_baseline_options: &JsValue,
) -> Result<JsMetricReport, JsError> {
    let parsed = parse_eval_args(
        targets,
        num_pivots_or_options,
        num_samples,
        seed,
        ideal_edge_length,
        labels,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
        node_weights,
        cluster_weights,
        cluster_baseline_options,
    )?;
    let inputs = MetricInputs {
        labels: parsed.labels.as_deref(),
        node_weights: parsed.node_weights.as_deref(),
        cluster_weights: parsed.cluster_weights.as_deref(),
        original_layout: None,
        sprawlter_half_widths: parsed.sprawlter_half_widths.as_deref(),
        sprawlter_half_heights: parsed.sprawlter_half_heights.as_deref(),
    };
    let report = evaluate_metrics_with_inputs(
        graph.inner(),
        layout.inner(),
        &parsed.targets,
        &parsed.config,
        &inputs,
    )
    .map_err(core_err_to_js)?;
    Ok(JsMetricReport::from_inner(report))
}

/// Rust-side 22-argument compatibility helper used by Rust WASM tests.
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_with_options(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: &JsValue,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
) -> Result<JsMetricReport, JsError> {
    let np_js = num_pivots
        .map(JsValue::from_f64)
        .unwrap_or(JsValue::UNDEFINED);
    js_evaluate_metrics_2d(
        graph,
        layout,
        targets,
        &np_js,
        num_samples,
        seed,
        ideal_edge_length,
        labels,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
        node_weights,
        cluster_weights,
        &JsValue::UNDEFINED,
    )
}

/// Rust-side 14-argument compatibility helper used by Rust WASM tests.
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    labels: Option<Vec<u32>>,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
) -> Result<JsMetricReport, JsError> {
    let np_js = num_pivots
        .map(JsValue::from_f64)
        .unwrap_or(JsValue::UNDEFINED);
    let lbl_js = match labels {
        Some(v) => js_sys::Uint32Array::from(v.as_slice()).into(),
        None => JsValue::UNDEFINED,
    };
    js_evaluate_metrics_2d(
        graph,
        layout,
        targets,
        &np_js,
        num_samples,
        seed,
        ideal_edge_length,
        &lbl_js,
        num_landmarks,
        num_triplets,
        min_degree,
        patch_radius,
        canvas_size,
        pad_frac,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
}
