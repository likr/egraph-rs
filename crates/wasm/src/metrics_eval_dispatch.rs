//! Polymorphic `evaluateMetrics` entry points across `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::NodeExtents2d;
use egraph::metrics::{MetricInputs, evaluate_metrics_with_inputs};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_eval::parse_eval_options;
use crate::metrics_report::JsMetricReport;

fn evaluate_metrics_2d_inner(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    options: &JsValue,
    original_layout: Option<&JsLayout2d>,
) -> Result<JsMetricReport, JsValue> {
    let parsed = parse_eval_options(targets, options)?;
    let orig_soa = original_layout.map(|l| l.inner().to_layout2d());
    let extents = match (
        parsed.sprawlter_half_widths.as_deref(),
        parsed.sprawlter_half_heights.as_deref(),
    ) {
        (Some(hw), Some(hh)) => Some(NodeExtents2d::new(hw, hh).map_err(core_err_to_js)?),
        (Some(hw), None) => Some(NodeExtents2d::new(hw, hw).map_err(core_err_to_js)?),
        (None, Some(hh)) => Some(NodeExtents2d::new(hh, hh).map_err(core_err_to_js)?),
        (None, None) => None,
    };
    let inputs = MetricInputs::new()
        .with_labels(parsed.labels.as_deref())
        .with_node_weights(parsed.node_weights.as_deref())
        .with_cluster_weights(parsed.cluster_weights.as_deref())
        .with_original_layout(orig_soa.as_ref())
        .with_sprawlter_extents(extents);
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
pub fn js_evaluate_metrics_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    options: &JsValue,
) -> Result<JsMetricReport, JsValue> {
    evaluate_metrics_2d_inner(graph, layout, targets, options, None)
}

/// Selectively evaluates quality metrics on `Layout2d` with a reference `originalLayout`.
#[wasm_bindgen(js_name = _evaluateMetrics2dWithOriginal)]
pub fn js_evaluate_metrics_2d_with_original(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    options: &JsValue,
    original_layout: &JsLayout2d,
) -> Result<JsMetricReport, JsValue> {
    evaluate_metrics_2d_inner(graph, layout, targets, options, Some(original_layout))
}

/// Selectively evaluates quality metrics on `Layout3d`, skipping unrequested work.
#[wasm_bindgen(js_name = _evaluateMetrics3d)]
pub fn js_evaluate_metrics_3d(
    graph: &JsCsrGraph,
    layout: &JsLayout3d,
    targets: &JsValue,
    options: &JsValue,
) -> Result<JsMetricReport, JsValue> {
    let parsed = parse_eval_options(targets, options)?;
    let inputs = MetricInputs::new()
        .with_labels(parsed.labels.as_deref())
        .with_node_weights(parsed.node_weights.as_deref())
        .with_cluster_weights(parsed.cluster_weights.as_deref());
    let soa = layout.inner().to_layout3d();
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

/// Selectively evaluates quality metrics on `LayoutNd`, skipping unrequested work.
#[wasm_bindgen(js_name = _evaluateMetricsNd)]
pub fn js_evaluate_metrics_nd(
    graph: &JsCsrGraph,
    layout: &JsLayoutNd,
    targets: &JsValue,
    options: &JsValue,
) -> Result<JsMetricReport, JsValue> {
    let parsed = parse_eval_options(targets, options)?;
    let inputs = MetricInputs::new()
        .with_labels(parsed.labels.as_deref())
        .with_node_weights(parsed.node_weights.as_deref())
        .with_cluster_weights(parsed.cluster_weights.as_deref());
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
pub fn js_evaluate_metrics_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
    targets: &JsValue,
    options: &JsValue,
) -> Result<JsMetricReport, JsValue> {
    let parsed = parse_eval_options(targets, options)?;
    let inputs = MetricInputs::new()
        .with_labels(parsed.labels.as_deref())
        .with_node_weights(parsed.node_weights.as_deref())
        .with_cluster_weights(parsed.cluster_weights.as_deref());
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
