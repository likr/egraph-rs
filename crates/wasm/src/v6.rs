//! v6 zero-copy `CsrGraph`, WebAssembly linear-memory `Layout2d`, and M1 `egraph-metrics` /
//! `egraph-spatial` bindings.

use egraph_core::{CoreError, CsrGraph};
use egraph_ffi::{FfiError, InterleavedLayout2d, build_csr_f32, build_edges_f32};
use egraph_metrics::{
    MetricConfig, MetricReport, MetricTarget, angular_resolution, aspect_ratio, crossing_angle,
    crossing_edges_torus, crossing_number, edge_angle, evaluate_metrics,
    gabriel_graph_preservation, gabriel_graph_property, ideal_edge_lengths,
    neighborhood_preservation_1hop, neighborhood_preservation_2hop, node_resolution,
    prism_distortion, relative_neighborhood_preservation, sampled_crossing_number,
    sampled_scale_normalized_stress, scale_normalized_stress,
};
#[cfg(feature = "delaunay")]
use egraph_spatial::Delaunay2d;
use wasm_bindgen::{JsCast, prelude::*};

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 10_000;
const DEFAULT_SEED: u64 = 42;
// IEEE-754 double precision represents integers exactly up to 2^53 - 1 (`Number.MAX_SAFE_INTEGER`),
// allowing JS callers to pass numeric seeds like `42` without requiring `BigInt`.
const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_991.0;

fn ffi_err_to_js(err: FfiError) -> JsError {
    JsError::new(&err.to_string())
}

fn core_err_to_js(err: CoreError) -> JsError {
    JsError::new(&err.to_string())
}

fn parse_non_negative_usize(value: f64, name: &str) -> Result<usize, JsError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > (u32::MAX as f64) {
        return Err(JsError::new(&format!(
            "{name} must be a non-negative integer <= u32::MAX, got {value}"
        )));
    }
    Ok(value as usize)
}

fn parse_seed_u64(seed: Option<f64>) -> Result<u64, JsError> {
    match seed {
        None => Ok(DEFAULT_SEED),
        Some(value) => {
            if !value.is_finite()
                || value < 0.0
                || value.fract() != 0.0
                || value > MAX_SAFE_INTEGER_F64
            {
                return Err(JsError::new(&format!(
                    "seed must be a non-negative safe integer <= 9007199254740991, got {value}"
                )));
            }
            Ok(value as u64)
        }
    }
}

fn parse_metric_target(raw: &str) -> Result<MetricTarget, JsError> {
    let normalized: String = raw
        .chars()
        .filter(|ch| !matches!(ch, '_' | '-' | ' '))
        .map(|ch| ch.to_ascii_lowercase())
        .collect();
    match normalized.as_str() {
        "scalenormalizedstress" | "sns" => Ok(MetricTarget::ScaleNormalizedStress),
        "sampledscalenormalizedstress" | "sampledsns" => {
            Ok(MetricTarget::SampledScaleNormalizedStress)
        }
        "crossingnumber" | "sweepcrossingnumber" | "crossings" => Ok(MetricTarget::CrossingNumber),
        "crossingangle" | "sweepcrossingangle" => Ok(MetricTarget::CrossingAngle),
        "sampledcrossingnumber" | "sampledcrossings" => Ok(MetricTarget::SampledCrossingNumber),
        "neighborhoodpreservation1hop" | "neighborhoodpreservation" | "np1" => {
            Ok(MetricTarget::NeighborhoodPreservation1Hop)
        }
        "neighborhoodpreservation2hop" | "neighborhoodpreservation2hopv6" | "np2" => {
            Ok(MetricTarget::NeighborhoodPreservation2Hop)
        }
        "gabrielgraphproperty" | "gabrielgraphpropertyv6" | "gabrielproperty" => {
            Ok(MetricTarget::GabrielGraphProperty)
        }
        "noderesolution" | "noderesolutionv6" => Ok(MetricTarget::NodeResolution),
        "relativeneighborhoodpreservation" | "drngpreservation" | "drng" => {
            Ok(MetricTarget::RelativeNeighborhoodPreservation)
        }
        "gabrielgraphpreservation" | "dggpreservation" | "dgg" => {
            Ok(MetricTarget::GabrielGraphPreservation)
        }
        "aspectratio" | "aspectratiov6" => Ok(MetricTarget::AspectRatio),
        "angularresolution" | "angularresolutionv6" => Ok(MetricTarget::AngularResolution),
        "idealedgelengths" | "idealedgelengthsv6" => Ok(MetricTarget::IdealEdgeLengths),
        _ => Err(JsError::new(&format!("unknown metric target: {raw:?}"))),
    }
}

fn parse_metric_targets(targets: &JsValue) -> Result<Vec<MetricTarget>, JsError> {
    if targets.is_null() || targets.is_undefined() || targets.is_string() || !targets.is_object() {
        return Err(JsError::new(
            "targets must be an array or iterable of metric target strings",
        ));
    }
    let iter = js_sys::try_iter(targets)
        .map_err(|err| JsError::new(&format!("failed to iterate targets: {err:?}")))?
        .ok_or_else(|| JsError::new("targets must be an array or iterable of strings"))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res
            .map_err(|err| JsError::new(&format!("failed to read target item: {err:?}")))?;
        let name = item
            .as_string()
            .ok_or_else(|| JsError::new("each entry in targets must be a string"))?;
        out.push(parse_metric_target(&name)?);
    }
    Ok(out)
}

fn parse_csr_options(
    directed_or_options: &JsValue,
    sanitize: Option<bool>,
) -> Result<(bool, bool), JsError> {
    if directed_or_options.is_undefined() || directed_or_options.is_null() {
        return Ok((false, sanitize.unwrap_or(true)));
    }
    if let Some(directed) = directed_or_options.as_bool() {
        return Ok((directed, sanitize.unwrap_or(true)));
    }
    if directed_or_options.is_object()
        && !js_sys::Array::is_array(directed_or_options)
        && !js_sys::ArrayBuffer::is_view(directed_or_options)
        && !directed_or_options.is_function()
    {
        if sanitize.is_some() {
            return Err(JsError::new(
                "unexpected sanitize argument when passing an options object",
            ));
        }
        let directed_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("directed")).map_err(
                |err| JsError::new(&format!("failed to read options.directed: {err:?}")),
            )?;
        let directed = if directed_val.is_undefined() {
            false
        } else {
            directed_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.directed must be a boolean"))?
        };

        let sanitize_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("sanitize")).map_err(
                |err| JsError::new(&format!("failed to read options.sanitize: {err:?}")),
            )?;
        let sanitize_flag = if sanitize_val.is_undefined() {
            true
        } else {
            sanitize_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.sanitize must be a boolean"))?
        };

        return Ok((directed, sanitize_flag));
    }
    Err(JsError::new(
        "expected boolean `directed` or options object `{ directed?: boolean, sanitize?: boolean }`",
    ))
}

/// Returns the active WebAssembly linear `WebAssembly.Memory` instance.
///
/// The JavaScript wrapper uses `wasmMemory().buffer` to detect `memory.grow` buffer detachments
/// in $O(1)$ time before returning a cached `Float32Array` view from `Layout2d.coordsView()`.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> js_sys::WebAssembly::Memory {
    wasm_bindgen::memory().unchecked_into()
}

/// Compact `u32`-indexed Compressed Sparse Row (CSR) graph storing `f32` edge weights.
#[wasm_bindgen(js_name = CsrGraph)]
#[derive(Debug, Clone)]
pub struct JsCsrGraph {
    inner: CsrGraph<f32>,
}

impl JsCsrGraph {
    /// Returns a reference to the underlying `egraph_core::CsrGraph<f32>`.
    pub fn inner(&self) -> &CsrGraph<f32> {
        &self.inner
    }
}

#[wasm_bindgen(js_class = CsrGraph)]
impl JsCsrGraph {
    /// Constructs a `CsrGraph` from CSR `indptr`, `indices`, and optional `weights`.
    ///
    /// The optional trailing parameters accept either `(directed, sanitize)` booleans or an
    /// options object `{ directed?: boolean, sanitize?: boolean }`. Defaults are
    /// `directed = false` and `sanitize = true`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        indptr: &[u32],
        indices: &[u32],
        weights: Option<Vec<f32>>,
        directed_or_options: &JsValue,
        sanitize: Option<bool>,
    ) -> Result<JsCsrGraph, JsError> {
        let (directed, sanitize_flag) = parse_csr_options(directed_or_options, sanitize)?;
        let inner = build_csr_f32(indptr, indices, weights.as_deref(), directed, sanitize_flag)
            .map_err(ffi_err_to_js)?;
        Ok(JsCsrGraph { inner })
    }

    /// Constructs a `CsrGraph` from a flat interleaved edge list `[u0, v0, u1, v1, ...]`
    /// and optional `weights`.
    #[wasm_bindgen(js_name = fromEdgeList)]
    pub fn from_edge_list(
        num_nodes: f64,
        edges: &[u32],
        weights: Option<Vec<f32>>,
        directed_or_options: &JsValue,
        sanitize: Option<bool>,
    ) -> Result<JsCsrGraph, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (directed, sanitize_flag) = parse_csr_options(directed_or_options, sanitize)?;
        let inner = build_edges_f32(n, edges, weights.as_deref(), directed, sanitize_flag)
            .map_err(ffi_err_to_js)?;
        Ok(JsCsrGraph { inner })
    }

    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of logical edges in the graph.
    #[wasm_bindgen(js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns `true` if the graph is directed.
    #[wasm_bindgen(js_name = isDirected)]
    pub fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes() + 1`).
    pub fn indptr(&self) -> Vec<u32> {
        self.inner.indptr().to_vec()
    }

    /// Returns a copy of the flat CSR `indices` neighbor array.
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of the flat CSR `weights` array, or `undefined` if unweighted.
    pub fn weights(&self) -> Option<Vec<f32>> {
        self.inner.weights().map(|w| w.to_vec())
    }
}

/// Contiguous interleaved `[x0, y0, x1, y1, ...]` 2D node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = Layout2d)]
#[derive(Debug, Clone)]
pub struct JsLayout2d {
    inner: InterleavedLayout2d<f32>,
}

impl JsLayout2d {
    /// Returns a shared reference to the underlying `InterleavedLayout2d<f32>`.
    pub fn inner(&self) -> &InterleavedLayout2d<f32> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying `InterleavedLayout2d<f32>`.
    pub fn inner_mut(&mut self) -> &mut InterleavedLayout2d<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = Layout2d)]
impl JsLayout2d {
    /// Creates a zero-initialized `Layout2d` for `n` vertices.
    pub fn zeros(n: f64) -> Result<JsLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        let inner = InterleavedLayout2d::<f32>::try_zeros(count).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
    }

    /// Creates a `Layout2d` from an even-length interleaved `[x0, y0, ...]` array.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(coords: &[f32]) -> Result<JsLayout2d, JsError> {
        let inner = InterleavedLayout2d::<f32>::from_interleaved(coords).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
    }

    /// Returns the number of vertices in the layout.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the byte offset of the interleaved `f32` buffer in WebAssembly linear memory.
    #[wasm_bindgen(js_name = coordsPtr)]
    pub fn coords_ptr(&self) -> usize {
        self.inner.coords_ptr() as usize
    }

    /// Returns the total scalar length of the interleaved buffer (`2 * len()`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.inner.coords_len()
    }

    /// Returns the `x` coordinate of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.x(idx).map_err(ffi_err_to_js)
    }

    /// Returns the `y` coordinate of vertex `index`.
    pub fn y(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.y(idx).map_err(ffi_err_to_js)
    }

    /// Sets the `x` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, index: f64, value: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_x(idx, value).map_err(ffi_err_to_js)
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, index: f64, value: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_y(idx, value).map_err(ffi_err_to_js)
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}

/// PRISM layout-preservation distortion metrics comparing an original and adjusted layout.
#[wasm_bindgen(js_name = PrismDistortion)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsPrismDistortion {
    sigma_dist: f64,
    sigma_disp: f64,
}

#[wasm_bindgen(js_class = PrismDistortion)]
impl JsPrismDistortion {
    /// Returns the proximity edge-length ratio coefficient of variation (`sigma_dist`).
    #[wasm_bindgen(getter, js_name = sigmaDist)]
    pub fn sigma_dist(&self) -> f64 {
        self.sigma_dist
    }

    /// Returns the scale-, translation-, and rotation-invariant Procrustes disparity (`sigma_disp`)
    /// lying in `[0.0, 1.0]`.
    #[wasm_bindgen(getter, js_name = sigmaDisp)]
    pub fn sigma_disp(&self) -> f64 {
        self.sigma_disp
    }
}

/// Selective quality metric report returned by `evaluateMetrics`.
///
/// Unrequested metrics return `undefined` without requiring `serde` in minimal builds.
#[wasm_bindgen(js_name = MetricReport)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsMetricReport {
    inner: MetricReport,
}

#[wasm_bindgen(js_class = MetricReport)]
impl JsMetricReport {
    /// Returns closed-form Scale-Normalized Stress (`SNS`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = scaleNormalizedStress)]
    pub fn scale_normalized_stress(&self) -> Option<f64> {
        self.inner.scale_normalized_stress
    }

    /// Returns pivot-sampled Scale-Normalized Stress (`SNS`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledScaleNormalizedStress)]
    pub fn sampled_scale_normalized_stress(&self) -> Option<f64> {
        self.inner.sampled_scale_normalized_stress
    }

    /// Returns exact sweep-line edge crossing count, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = crossingNumber)]
    pub fn crossing_number(&self) -> Option<usize> {
        self.inner.crossing_number
    }

    /// Returns exact sweep-line edge crossing-angle penalty sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = crossingAngle)]
    pub fn crossing_angle(&self) -> Option<f64> {
        self.inner.crossing_angle
    }

    /// Returns deterministic sampled crossing estimate, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledCrossingNumber)]
    pub fn sampled_crossing_number(&self) -> Option<f64> {
        self.inner.sampled_crossing_number
    }

    /// Returns 1-hop neighborhood preservation (`NP1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodPreservation1Hop)]
    pub fn neighborhood_preservation_1hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_1hop
    }

    /// Returns 2-hop neighborhood preservation (`NP2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodPreservation2Hop)]
    pub fn neighborhood_preservation_2hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_2hop
    }

    /// Returns Gabriel graph property violation sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = gabrielGraphProperty)]
    pub fn gabriel_graph_property(&self) -> Option<f64> {
        self.inner.gabriel_graph_property
    }

    /// Returns one-sided hinge node-resolution violation sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = nodeResolution)]
    pub fn node_resolution(&self) -> Option<f64> {
        self.inner.node_resolution
    }

    /// Returns shape-based Relative Neighborhood Graph preservation (`dRNG`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = relativeNeighborhoodPreservation)]
    pub fn relative_neighborhood_preservation(&self) -> Option<f64> {
        self.inner.relative_neighborhood_preservation
    }

    /// Returns shape-based Gabriel Graph preservation (`dGG`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = gabrielGraphPreservation)]
    pub fn gabriel_graph_preservation(&self) -> Option<f64> {
        self.inner.gabriel_graph_preservation
    }

    /// Returns PCA aspect ratio, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = aspectRatio)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        self.inner.aspect_ratio
    }

    /// Returns exponential angular-resolution penalty sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = angularResolution)]
    pub fn angular_resolution(&self) -> Option<f64> {
        self.inner.angular_resolution
    }

    /// Returns ideal edge-length preservation error sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = idealEdgeLengths)]
    pub fn ideal_edge_lengths(&self) -> Option<f64> {
        self.inner.ideal_edge_lengths
    }

    /// Converts requested metrics into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let entries: [(&str, Option<f64>); 14] = [
            ("scaleNormalizedStress", self.inner.scale_normalized_stress),
            (
                "sampledScaleNormalizedStress",
                self.inner.sampled_scale_normalized_stress,
            ),
            (
                "crossingNumber",
                self.inner.crossing_number.map(|v| v as f64),
            ),
            ("crossingAngle", self.inner.crossing_angle),
            ("sampledCrossingNumber", self.inner.sampled_crossing_number),
            (
                "neighborhoodPreservation1Hop",
                self.inner.neighborhood_preservation_1hop,
            ),
            (
                "neighborhoodPreservation2Hop",
                self.inner.neighborhood_preservation_2hop,
            ),
            ("gabrielGraphProperty", self.inner.gabriel_graph_property),
            ("nodeResolution", self.inner.node_resolution),
            (
                "relativeNeighborhoodPreservation",
                self.inner.relative_neighborhood_preservation,
            ),
            (
                "gabrielGraphPreservation",
                self.inner.gabriel_graph_preservation,
            ),
            ("aspectRatio", self.inner.aspect_ratio),
            ("angularResolution", self.inner.angular_resolution),
            ("idealEdgeLengths", self.inner.ideal_edge_lengths),
        ];
        for (key, maybe_val) in entries {
            if let Some(val) = maybe_val {
                js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_f64(val))
                    .map_err(|err| {
                        JsError::new(&format!("failed to set MetricReport.{key}: {err:?}"))
                    })?;
            }
        }
        Ok(obj)
    }
}

/// Deterministic 2D Delaunay triangulation result returned by `delaunayTriangulation`.
#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_name = DelaunayResult)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsDelaunayResult {
    num_nodes: usize,
    edges: Vec<u32>,
    triangles: Vec<u32>,
}

#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_class = DelaunayResult)]
impl JsDelaunayResult {
    /// Returns the number of vertices in the triangulated layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns a flat `[u0, v0, u1, v1, ...]` `Uint32Array` of canonical undirected edges (`u < v`).
    pub fn edges(&self) -> Vec<u32> {
        self.edges.clone()
    }

    /// Returns a flat `[a0, b0, c0, a1, b1, c1, ...]` `Uint32Array` of canonical Delaunay triangles.
    pub fn triangles(&self) -> Vec<u32> {
        self.triangles.clone()
    }
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` for `graph` and `layout`.
#[wasm_bindgen(js_name = scaleNormalizedStress)]
pub fn js_scale_normalized_stress(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    scale_normalized_stress(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`) from `numPivots` (default `64`)
/// deterministic BFS sources with optional safe-integer `seed` (default `42`).
#[wasm_bindgen(js_name = sampledScaleNormalizedStress)]
pub fn js_sampled_scale_normalized_stress(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_pivots: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let pivots = match num_pivots {
        None => DEFAULT_NUM_PIVOTS,
        Some(v) => parse_non_negative_usize(v, "numPivots")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    sampled_scale_normalized_stress(graph.inner(), &soa, pivots, seed_u64).map_err(core_err_to_js)
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

/// Estimates edge crossings in $O(\text{numSamples})$ time (`numSamples` default `10000`, `seed` default `42`).
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

/// Computes `StaticKdTree`-accelerated 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = neighborhoodPreservation1Hop)]
pub fn js_neighborhood_preservation_1hop(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    neighborhood_preservation_1hop(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes `StaticKdTree`-accelerated 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = neighborhoodPreservation2HopV6)]
pub fn js_neighborhood_preservation_2hop_v6(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    neighborhood_preservation_2hop(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes Gabriel graph condition violations across `graph` edges under `layout`.
#[wasm_bindgen(js_name = gabrielGraphPropertyV6)]
pub fn js_gabriel_graph_property_v6(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_property(graph.inner(), &soa).map_err(core_err_to_js)
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

/// Evaluates ideal edge-length preservation error sum for `graph` under `layout`.
#[wasm_bindgen(js_name = idealEdgeLengthsV6)]
pub fn js_ideal_edge_lengths_v6(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths(graph.inner(), &soa).map_err(core_err_to_js)
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
    Ok(JsPrismDistortion {
        sigma_dist: res.sigma_dist,
        sigma_disp: res.sigma_disp,
    })
}

/// Computes shape-based Relative Neighborhood Graph preservation (`dRNG`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = drngPreservation)]
pub fn js_drng_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    relative_neighborhood_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes shape-based Gabriel Graph preservation (`dGG`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = dggPreservation)]
pub fn js_dgg_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Selectively evaluates only the quality metrics listed in `targets`, skipping unrequested work.
#[wasm_bindgen(js_name = evaluateMetrics)]
pub fn js_evaluate_metrics(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
) -> Result<JsMetricReport, JsError> {
    let parsed_targets = parse_metric_targets(targets)?;
    let sns_num_pivots = match num_pivots {
        None => DEFAULT_NUM_PIVOTS,
        Some(v) => parse_non_negative_usize(v, "numPivots")?,
    };
    let crossing_num_samples = match num_samples {
        None => DEFAULT_NUM_SAMPLES,
        Some(v) => parse_non_negative_usize(v, "numSamples")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let cfg = MetricConfig {
        sns_num_pivots,
        sns_seed: seed_u64,
        crossing_num_samples,
        crossing_seed: seed_u64,
        ideal_edge_length: 1.0,
    };
    let soa = layout.inner().to_layout2d();
    let report =
        evaluate_metrics(graph.inner(), &soa, &parsed_targets, &cfg).map_err(core_err_to_js)?;
    Ok(JsMetricReport { inner: report })
}

/// Computes the deterministic 2D Delaunay triangulation of `layout` (`delaunay` feature).
#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_name = delaunayTriangulation)]
pub fn js_delaunay_triangulation(layout: &JsLayout2d) -> Result<JsDelaunayResult, JsError> {
    let soa = layout.inner().to_layout2d();
    let dt = Delaunay2d::triangulate(&soa).map_err(core_err_to_js)?;
    let mut flat_edges = Vec::with_capacity(dt.edges().len() * 2);
    for &(u, v) in dt.edges() {
        flat_edges.push(u);
        flat_edges.push(v);
    }
    let mut flat_triangles = Vec::with_capacity(dt.triangles().len() * 3);
    for &[a, b, c] in dt.triangles() {
        flat_triangles.push(a);
        flat_triangles.push(b);
        flat_triangles.push(c);
    }
    Ok(JsDelaunayResult {
        num_nodes: dt.num_nodes(),
        edges: flat_edges,
        triangles: flat_triangles,
    })
}
