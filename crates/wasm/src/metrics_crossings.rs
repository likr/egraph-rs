//! WebAssembly bindings for edge crossings, angular resolution, aspect ratio, node resolution, and layout degeneracy metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_report::JsMinimumAngleSummary;
use egraph::metrics::{
    CrossingPair, CrossingSummary, LayoutDegeneracy, TorusCrossing, angular_resolution,
    aspect_ratio, crossing_angle, crossing_edges, crossing_edges_torus, crossing_number,
    crossing_summary, crosslessness, edge_angle, layout_degeneracy, minimum_angle,
    minimum_angle_summary, node_resolution, sampled_crossing_number, sampled_crosslessness,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_SAMPLES: usize = 4096;

/// Coordinate-collapse and zero-length edge degeneracy diagnostics returned by `layoutDegeneracy`.
#[wasm_bindgen(js_name = LayoutDegeneracy)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsLayoutDegeneracy {
    inner: LayoutDegeneracy,
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
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let num_entries: [(&str, f64); 10] = [
            ("numNodes", self.inner.num_nodes as f64),
            ("numEdges", self.inner.num_edges as f64),
            ("duplicateNodesF32", self.inner.duplicate_nodes_f32 as f64),
            ("duplicateRatioF32", self.inner.duplicate_ratio_f32),
            ("duplicateNodesF64", self.inner.duplicate_nodes_f64 as f64),
            ("duplicateRatioF64", self.inner.duplicate_ratio_f64),
            ("zeroLengthEdges", self.inner.zero_length_edges as f64),
            ("zeroLengthEdgeRatio", self.inner.zero_length_edge_ratio),
            ("bboxWidth", self.inner.bbox_width),
            ("bboxHeight", self.inner.bbox_height),
        ];
        for (k, v) in num_entries {
            js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v))
                .map_err(|e| JsError::new(&format!("failed to set {k}: {e:?}")))?;
        }
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("isBboxDegenerate"),
            &JsValue::from_bool(self.inner.is_bbox_degenerate),
        )
        .map_err(|e| JsError::new(&format!("failed to set isBboxDegenerate: {e:?}")))?;
        Ok(obj)
    }
}

/// Counts the exact number of non-adjacent edge crossings via $O((|E|+K)\log|E|)$ sweep-line.
#[wasm_bindgen(js_name = sweepCrossingNumber)]
pub fn js_sweep_crossing_number(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<usize, JsError> {
    let soa = layout.inner().to_layout2d();
    crossing_number(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes the exact sweep-line crossing-angle penalty $\sum \cos^2(\theta)$.
#[wasm_bindgen(js_name = sweepCrossingAngle)]
pub fn js_sweep_crossing_angle(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    crossing_angle(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Estimates edge crossings in $O(\text{numSamples})$ time (`numSamples` default `4096`, `seed` default `42`).
#[wasm_bindgen(js_name = sampledCrossingNumber)]
pub fn js_sampled_crossing_number(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    sampled_crossing_number(graph.inner(), &soa, samples, seed_u64).map_err(core_err_to_js)
}

/// Counts non-adjacent edge sub-segment crossings on the flat unit torus `[0, 1)^2`.
#[wasm_bindgen(js_name = torusCrossingNumber)]
pub fn js_torus_crossing_number(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<usize, JsError> {
    let soa = layout.inner().to_layout2d();
    crossing_edges_torus(graph.inner(), &soa)
        .map(|crossings| crossings.len())
        .map_err(core_err_to_js)
}

/// Computes the one-sided hinge node-resolution violation sum of `layout`.
#[wasm_bindgen(js_name = nodeResolutionV6)]
pub fn js_node_resolution_v6(layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    node_resolution(&soa).map_err(core_err_to_js)
}

/// Computes the PCA aspect ratio $\sigma_2 / \sigma_1 \in [0.0, 1.0]$ of `layout`.
#[wasm_bindgen(js_name = aspectRatioV6)]
pub fn js_aspect_ratio_v6(layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    aspect_ratio(&soa).map_err(core_err_to_js)
}

/// Computes the exponential angular-resolution penalty sum over incident edge pairs.
#[wasm_bindgen(js_name = angularResolutionV6)]
pub fn js_angular_resolution_v6(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
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
pub fn js_crosslessness(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    crosslessness(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Estimates DRGraph Eq. (17) crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$
/// (`numSamples` default `4096`, `seed` default `42`).
#[wasm_bindgen(js_name = sampledCrosslessness)]
pub fn js_sampled_crosslessness(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    sampled_crosslessness(graph.inner(), &soa, samples, seed_u64).map_err(core_err_to_js)
}

/// Computes DRGraph Eq. (19) strict minimum-angle resolution (`A2`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = minimumAngle)]
pub fn js_minimum_angle(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    minimum_angle(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`).
#[wasm_bindgen(js_name = minimumAngleSummary)]
pub fn js_minimum_angle_summary(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<JsMinimumAngleSummary, JsError> {
    let soa = layout.inner().to_layout2d();
    let inner = minimum_angle_summary(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsMinimumAngleSummary::from_inner(inner))
}

/// Evaluates duplicate-node (`f32`/`f64`), zero-length edge, and bounding-box degeneracies.
#[wasm_bindgen(js_name = layoutDegeneracy)]
pub fn js_layout_degeneracy(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<JsLayoutDegeneracy, JsError> {
    let soa = layout.inner().to_layout2d();
    let inner = layout_degeneracy(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsLayoutDegeneracy { inner })
}

/// Non-adjacent crossing edge pair details returned by `crossingEdges`.
#[wasm_bindgen(js_name = CrossingPair)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsCrossingPair {
    inner: CrossingPair,
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
}

/// Returns all non-adjacent crossing edge pairs in `graph` under `layout`.
#[wasm_bindgen(js_name = crossingEdges)]
pub fn js_crossing_edges(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<Vec<JsCrossingPair>, JsError> {
    let soa = layout.inner().to_layout2d();
    let pairs = crossing_edges(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(pairs
        .into_iter()
        .map(|inner| JsCrossingPair { inner })
        .collect())
}

/// Periodic `[0, 1)^2` torus crossing edge sub-segment pair returned by `crossingEdgesTorus`.
#[wasm_bindgen(js_name = TorusCrossing)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsTorusCrossing {
    inner: TorusCrossing,
}

#[wasm_bindgen(js_class = TorusCrossing)]
impl JsTorusCrossing {
    #[wasm_bindgen(getter, js_name = edgeA)]
    pub fn edge_a(&self) -> usize {
        self.inner.edge_a
    }

    #[wasm_bindgen(getter, js_name = edgeB)]
    pub fn edge_b(&self) -> usize {
        self.inner.edge_b
    }

    /// Returns `[x1, y1, x2, y2]` coordinates of the first crossing sub-segment.
    #[wasm_bindgen(js_name = segA)]
    pub fn seg_a(&self) -> Vec<f64> {
        self.inner.seg_a.to_vec()
    }

    /// Returns `[x1, y1, x2, y2]` coordinates of the second crossing sub-segment.
    #[wasm_bindgen(js_name = segB)]
    pub fn seg_b(&self) -> Vec<f64> {
        self.inner.seg_b.to_vec()
    }
}

/// Finds all crossing edge sub-segment pairs on the unit torus `[0, 1)^2`.
#[wasm_bindgen(js_name = crossingEdgesTorus)]
pub fn js_crossing_edges_torus(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<Vec<JsTorusCrossing>, JsError> {
    let soa = layout.inner().to_layout2d();
    let crossings = crossing_edges_torus(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(crossings
        .into_iter()
        .map(|inner| JsTorusCrossing { inner })
        .collect())
}

/// Exact edge-crossing count and angle statistics returned by `crossingSummary`.
#[wasm_bindgen(js_name = CrossingSummary)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
) -> Result<JsCrossingSummary, JsError> {
    let soa = layout.inner().to_layout2d();
    let inner = crossing_summary(graph.inner(), &soa).map_err(core_err_to_js)?;
    Ok(JsCrossingSummary { inner })
}
