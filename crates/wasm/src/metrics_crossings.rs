//! WebAssembly bindings for Euclidean edge crossings, angular resolution, aspect ratio,
//! node resolution, and layout degeneracy metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{DEFAULT_SEED, core_err_to_js, set_f64_prop, set_js_prop};
use crate::js_options::{check_options_arg, opt_seed_prop, opt_usize_prop};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_summaries::JsMinimumAngleSummary;
use egraph::metrics::{
    CrossingPair, CrossingSummary, LayoutDegeneracy, angular_resolution, aspect_ratio,
    crossing_angle, crossing_edges, crossing_number, crossing_summary, crosslessness, edge_angle,
    layout_degeneracy, minimum_angle, minimum_angle_summary, node_resolution,
    sampled_crossing_number, sampled_crosslessness,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_SAMPLES: usize = 4096;

fn parse_sampled_crossing_options(options: &JsValue, ctx: &str) -> Result<(usize, u64), JsValue> {
    let mut num_samples = DEFAULT_NUM_SAMPLES;
    let mut seed = DEFAULT_SEED;
    if check_options_arg(options, ctx)? {
        num_samples = opt_usize_prop(options, "numSamples", num_samples)?;
        seed = opt_seed_prop(options)?;
    }
    Ok((num_samples, seed))
}

/// Coordinate-collapse and zero-length edge degeneracy diagnostics returned by `layoutDegeneracy`.
#[wasm_bindgen(js_name = LayoutDegeneracy)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLayoutDegeneracy {
    inner: LayoutDegeneracy,
}

impl JsLayoutDegeneracy {
    pub(crate) fn from_inner(inner: LayoutDegeneracy) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = LayoutDegeneracy)]
impl JsLayoutDegeneracy {
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    #[wasm_bindgen(getter, js_name = duplicateNodesF32)]
    pub fn duplicate_nodes_f32(&self) -> usize {
        self.inner.duplicate_nodes_f32
    }

    #[wasm_bindgen(getter, js_name = duplicateRatioF32)]
    pub fn duplicate_ratio_f32(&self) -> f64 {
        self.inner.duplicate_ratio_f32
    }

    #[wasm_bindgen(getter, js_name = duplicateNodesF64)]
    pub fn duplicate_nodes_f64(&self) -> usize {
        self.inner.duplicate_nodes_f64
    }

    #[wasm_bindgen(getter, js_name = duplicateRatioF64)]
    pub fn duplicate_ratio_f64(&self) -> f64 {
        self.inner.duplicate_ratio_f64
    }

    #[wasm_bindgen(getter, js_name = excessDuplicateRatio)]
    pub fn excess_duplicate_ratio(&self) -> f64 {
        self.inner.excess_duplicate_ratio
    }

    #[wasm_bindgen(getter, js_name = nearCollisionRatio)]
    pub fn near_collision_ratio(&self) -> f64 {
        self.inner.near_collision_ratio
    }

    #[wasm_bindgen(getter, js_name = zeroLengthEdges)]
    pub fn zero_length_edges(&self) -> usize {
        self.inner.zero_length_edges
    }

    #[wasm_bindgen(getter, js_name = zeroLengthEdgeRatio)]
    pub fn zero_length_edge_ratio(&self) -> f64 {
        self.inner.zero_length_edge_ratio
    }

    #[wasm_bindgen(getter, js_name = bboxWidth)]
    pub fn bbox_width(&self) -> f64 {
        self.inner.bbox_width
    }

    #[wasm_bindgen(getter, js_name = bboxHeight)]
    pub fn bbox_height(&self) -> f64 {
        self.inner.bbox_height
    }

    #[wasm_bindgen(getter, js_name = isBboxDegenerate)]
    pub fn is_bbox_degenerate(&self) -> bool {
        self.inner.is_bbox_degenerate
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        for (k, v) in [
            ("numNodes", self.inner.num_nodes as f64),
            ("numEdges", self.inner.num_edges as f64),
            ("duplicateNodesF32", self.inner.duplicate_nodes_f32 as f64),
            ("duplicateRatioF32", self.inner.duplicate_ratio_f32),
            ("duplicateNodesF64", self.inner.duplicate_nodes_f64 as f64),
            ("duplicateRatioF64", self.inner.duplicate_ratio_f64),
            ("excessDuplicateRatio", self.inner.excess_duplicate_ratio),
            ("nearCollisionRatio", self.inner.near_collision_ratio),
            ("zeroLengthEdges", self.inner.zero_length_edges as f64),
            ("zeroLengthEdgeRatio", self.inner.zero_length_edge_ratio),
            ("bboxWidth", self.inner.bbox_width),
            ("bboxHeight", self.inner.bbox_height),
        ] {
            set_f64_prop(&obj, k, v)?;
        }
        set_js_prop(
            &obj,
            "isBboxDegenerate",
            &JsValue::from_bool(self.inner.is_bbox_degenerate),
        )?;
        Ok(obj)
    }
}

/// Counts the exact number of non-adjacent edge crossings via $O((|E|+K)\log|E|)$ sweep-line.
#[wasm_bindgen(js_name = sweepCrossingNumber)]
pub fn js_sweep_crossing_number(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<usize, JsValue> {
    let soa = layout.inner().to_layout2d();
    crossing_number(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes the exact sweep-line crossing-angle penalty $\sum \cos^2(\theta)$.
#[wasm_bindgen(js_name = sweepCrossingAngle)]
pub fn js_sweep_crossing_angle(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    crossing_angle(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Estimates edge crossings in $O(\text{numSamples})$ time (`options?: { numSamples?, seed? }`).
#[wasm_bindgen(js_name = sampledCrossingNumber)]
pub fn js_sampled_crossing_number(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_sampled_crossing_options(options, "sampledCrossingNumber")?;
    let soa = layout.inner().to_layout2d();
    sampled_crossing_number(graph.inner(), &soa, samples, seed_u64).map_err(core_err_to_js)
}

/// Computes the one-sided hinge node-resolution violation sum of `layout`.
#[wasm_bindgen(js_name = _nodeResolution2d)]
pub fn js_node_resolution(layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    node_resolution(&soa).map_err(core_err_to_js)
}

/// Computes the PCA aspect ratio $\sigma_2 / \sigma_1 \in [0.0, 1.0]$ of `layout`.
#[wasm_bindgen(js_name = aspectRatio)]
pub fn js_aspect_ratio(layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    aspect_ratio(&soa).map_err(core_err_to_js)
}

/// Computes the exponential angular-resolution penalty sum over incident edge pairs.
#[wasm_bindgen(js_name = angularResolution)]
pub fn js_angular_resolution(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    angular_resolution(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes the undirected angle in `[0.0, pi]` radians between 2D vectors `(x1, y1)` and `(x2, y2)`,
/// returning `undefined` if either vector is zero-length or non-finite.
#[wasm_bindgen(js_name = edgeAngle)]
pub fn js_edge_angle(x1: f64, y1: f64, x2: f64, y2: f64) -> Option<f64> {
    edge_angle(x1, y1, x2, y2)
}

/// Computes exact DRGraph Eq. (17) crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$.
#[wasm_bindgen(js_name = crosslessness)]
pub fn js_crosslessness(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    crosslessness(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Estimates DRGraph Eq. (17) crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$
/// (`options?: { numSamples?, seed? }`).
#[wasm_bindgen(js_name = sampledCrosslessness)]
pub fn js_sampled_crosslessness(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let (samples, seed_u64) = parse_sampled_crossing_options(options, "sampledCrosslessness")?;
    let soa = layout.inner().to_layout2d();
    sampled_crosslessness(graph.inner(), &soa, samples, seed_u64).map_err(core_err_to_js)
}

/// Computes DRGraph Eq. (19) strict minimum-angle resolution (`A2`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = minimumAngle)]
pub fn js_minimum_angle(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsValue> {
    let soa = layout.inner().to_layout2d();
    minimum_angle(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`).
#[wasm_bindgen(js_name = minimumAngleSummary)]
pub fn js_minimum_angle_summary(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<JsMinimumAngleSummary, JsValue> {
    let soa = layout.inner().to_layout2d();
    let inner = minimum_angle_summary(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsMinimumAngleSummary::from_inner(inner))
}

/// Evaluates duplicate-node (`f32`/`f64`), zero-length edge, and bounding-box degeneracies.
#[wasm_bindgen(js_name = _layoutDegeneracy2d)]
pub fn js_layout_degeneracy(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<JsLayoutDegeneracy, JsValue> {
    let soa = layout.inner().to_layout2d();
    let inner = layout_degeneracy(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy { inner })
}

/// Non-adjacent crossing edge pair details returned by `crossingEdges`, `crossingEdgesSpherical`,
/// and `crossingEdgesHyperbolic`.
#[wasm_bindgen(js_name = CrossingPair)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsCrossingPair {
    inner: CrossingPair,
}

impl JsCrossingPair {
    pub(crate) fn from_inner(inner: CrossingPair) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = CrossingPair)]
impl JsCrossingPair {
    #[wasm_bindgen(getter, js_name = edgeA)]
    pub fn edge_a(&self) -> usize {
        self.inner.edge_a
    }

    #[wasm_bindgen(getter, js_name = edgeB)]
    pub fn edge_b(&self) -> usize {
        self.inner.edge_b
    }

    #[wasm_bindgen(getter)]
    pub fn u1(&self) -> u32 {
        self.inner.u1
    }

    #[wasm_bindgen(getter)]
    pub fn v1(&self) -> u32 {
        self.inner.v1
    }

    #[wasm_bindgen(getter)]
    pub fn u2(&self) -> u32 {
        self.inner.u2
    }

    #[wasm_bindgen(getter)]
    pub fn v2(&self) -> u32 {
        self.inner.v2
    }

    #[wasm_bindgen(getter, js_name = angleRad)]
    pub fn angle_rad(&self) -> Option<f64> {
        self.inner.angle_rad
    }

    #[wasm_bindgen(getter, js_name = cosSq)]
    pub fn cos_sq(&self) -> f64 {
        self.inner.cos_sq
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "edgeA", self.inner.edge_a as f64)?;
        set_f64_prop(&obj, "edgeB", self.inner.edge_b as f64)?;
        set_f64_prop(&obj, "u1", self.inner.u1 as f64)?;
        set_f64_prop(&obj, "v1", self.inner.v1 as f64)?;
        set_f64_prop(&obj, "u2", self.inner.u2 as f64)?;
        set_f64_prop(&obj, "v2", self.inner.v2 as f64)?;
        if let Some(ang) = self.inner.angle_rad {
            set_f64_prop(&obj, "angleRad", ang)?;
        }
        set_f64_prop(&obj, "cosSq", self.inner.cos_sq)?;
        Ok(obj)
    }
}

/// Returns all non-adjacent crossing edge pairs in `graph` under `layout`.
#[wasm_bindgen(js_name = crossingEdges)]
pub fn js_crossing_edges(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<Vec<JsCrossingPair>, JsValue> {
    let soa = layout.inner().to_layout2d();
    let pairs = crossing_edges(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(pairs.into_iter().map(JsCrossingPair::from_inner).collect())
}

/// Exact edge-crossing count and angle statistics returned by `crossingSummary`.
#[wasm_bindgen(js_name = CrossingSummary)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsCrossingSummary {
    inner: CrossingSummary,
}

#[wasm_bindgen(js_class = CrossingSummary)]
impl JsCrossingSummary {
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.inner.count
    }

    #[wasm_bindgen(getter, js_name = angleCosSqSum)]
    pub fn angle_cos_sq_sum(&self) -> f64 {
        self.inner.angle_cos_sq_sum
    }

    #[wasm_bindgen(getter, js_name = minAngleRad)]
    pub fn min_angle_rad(&self) -> Option<f64> {
        self.inner.min_angle_rad
    }
}

/// Computes both exact crossing count and crossing-angle statistics in a single sweep pass.
#[wasm_bindgen(js_name = crossingSummary)]
pub fn js_crossing_summary(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<JsCrossingSummary, JsValue> {
    let soa = layout.inner().to_layout2d();
    let inner = crossing_summary(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsCrossingSummary { inner })
}
