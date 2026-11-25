//! v6 zero-copy `CsrGraph`, WebAssembly linear-memory `Layout2d`, and M1 `egraph-metrics` /
//! `egraph-spatial` bindings.

pub use crate::v6_m2::*;
pub use crate::v6_m3::*;
pub use crate::v6_m4::*;
pub use crate::v6_m5::*;
pub use crate::v6_w1::*;
pub use crate::v6_w2::*;

use egraph::core::{CoreError, CsrGraph, DeterministicRng, Layout2d, LayoutNd};
use egraph::metrics::{
    ClusterAreaConfig, ClusterAreaDetail, ClusterBaselineConfig, CrossingPair, CrossingSummary,
    LayoutDegeneracy, MetricConfig, MetricReport, MetricTarget, MinimumAngleSummary, RtaEvaluation,
    TorusCrossing, adjusted_patch_nig, adjusted_patch_nig_multiscale, all_pairs_bfs,
    angular_resolution, aspect_ratio, bfs_single_source,
    cluster_area_faithfulness_detail_with_weights, cluster_area_faithfulness_with_weights,
    crossing_angle, crossing_edges, crossing_edges_torus, crossing_number, crossing_summary,
    crosslessness, edge_angle, evaluate_metrics_with_community_weights, gabriel_graph_edges,
    gabriel_graph_preservation, gabriel_graph_property, ideal_edge_lengths,
    ideal_edge_lengths_with_unit, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_from_landmarks, layout_degeneracy, minimum_angle,
    minimum_angle_summary, neighborhood_preservation_1hop, neighborhood_preservation_2hop,
    node_resolution, prism_distortion, prism_distortion_with_edges, relative_neighborhood_edges,
    relative_neighborhood_preservation, sampled_crossing_number, sampled_crosslessness,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    sampled_scale_normalized_stress, scale_normalized_stress,
    scale_normalized_stress_from_distances,
};
#[cfg(feature = "delaunay")]
use egraph::spatial::Delaunay2d;
use egraph::spatial::RectangleOverlapSweep;
use egraph_dataset::parse_csr_csv;
use egraph_ffi::{FfiError, InterleavedLayout2d, build_csr_f32, build_edges_f32};
use wasm_bindgen::{JsCast, prelude::*};

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_UNIT_EDGE_LENGTH: f64 = 1.0;
const DEFAULT_RTA_NUM_LANDMARKS: usize = 64;
const DEFAULT_RTA_NUM_TRIPLETS: usize = 1_000_000;
const DEFAULT_RTA_MIN_DEGREE: usize = 2;
const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;
// IEEE-754 double precision represents integers exactly up to 2^53 - 1 (`Number.MAX_SAFE_INTEGER`),
// allowing JS callers to pass numeric seeds like `42` without requiring `BigInt`.
const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_991.0;

pub(crate) fn ffi_err_to_js(err: FfiError) -> JsError {
    JsError::new(&err.to_string())
}

pub(crate) fn core_err_to_js(err: CoreError) -> JsError {
    JsError::new(&err.to_string())
}

pub(crate) fn parse_non_negative_usize(value: f64, name: &str) -> Result<usize, JsError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > (u32::MAX as f64) {
        return Err(JsError::new(&format!(
            "{name} must be a non-negative integer <= u32::MAX, got {value}"
        )));
    }
    Ok(value as usize)
}

pub(crate) fn parse_non_negative_u32(value: f64, name: &str) -> Result<u32, JsError> {
    let v = parse_non_negative_usize(value, name)?;
    u32::try_from(v).map_err(|_| JsError::new(&format!("{name} must be <= u32::MAX, got {value}")))
}

pub(crate) fn parse_u32_sequence(val: &JsValue, name: &str) -> Result<Vec<u32>, JsError> {
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(arr.to_vec());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Uint32Array or integer array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| JsError::new(&format!("{name} must be a Uint32Array or integer array")))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be integers")))?;
        out.push(parse_non_negative_u32(num, name)?);
    }
    Ok(out)
}

pub(crate) fn parse_optional_f64_sequence(
    val: &JsValue,
    name: &str,
) -> Result<Option<Vec<f64>>, JsError> {
    if val.is_null() || val.is_undefined() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(Some(arr.to_vec()));
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(Some(arr.to_vec().into_iter().map(f64::from).collect()));
    }
    if val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Float64Array, Float32Array, or numeric array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            JsError::new(&format!(
                "{name} must be a Float64Array, Float32Array, or numeric array"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(Some(out))
}

pub(crate) fn parse_seed_u64(seed: Option<f64>) -> Result<u64, JsError> {
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
        "crosslessness" => Ok(MetricTarget::Crosslessness),
        "sampledcrosslessness" => Ok(MetricTarget::SampledCrosslessness),
        "neighborhoodpreservation1hop" | "neighborhoodpreservation" | "np1" => {
            Ok(MetricTarget::NeighborhoodPreservation1Hop)
        }
        "neighborhoodpreservation2hop" | "neighborhoodpreservation2hopv6" | "np2" => {
            Ok(MetricTarget::NeighborhoodPreservation2Hop)
        }
        "sampledneighborhoodpreservation1hop"
        | "sampledneighborhoodpreservation"
        | "samplednp1" => Ok(MetricTarget::SampledNp1),
        "sampledneighborhoodpreservation2hop"
        | "sampledneighborhoodpreservation2hopv6"
        | "samplednp2" => Ok(MetricTarget::SampledNp2),
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
        "minimumangle" | "minangle" | "minimumanglea2" | "minanglea2" => {
            Ok(MetricTarget::MinimumAngle)
        }
        "minimumanglea1" | "minanglea1" => Ok(MetricTarget::MinimumAngleA1),
        "layoutdegeneracy" => Ok(MetricTarget::LayoutDegeneracy),
        "idealedgelengths" | "idealedgelengthsv6" => Ok(MetricTarget::IdealEdgeLengths),
        "adjustedpatchnig" | "patchnig" => Ok(MetricTarget::AdjustedPatchNig),
        "clusterareafaithfulness" | "clusterarea" | "gcaf" => {
            Ok(MetricTarget::ClusterAreaFaithfulness)
        }
        "landmarkstratifiedrta" | "lstratifiedrta" | "rta" => {
            Ok(MetricTarget::LandmarkStratifiedRta)
        }
        "dsc" | "distanceconsistency" => Ok(MetricTarget::Dsc),
        "dc" | "distributionconsistency" => Ok(MetricTarget::Dc),
        "cdm" | "classdensitymeasure" => Ok(MetricTarget::Cdm),
        "hdm" | "2dhdm" | "histogramdensitymeasure" => Ok(MetricTarget::Hdm),
        "sepmeknng" | "sepmeknng2" => Ok(MetricTarget::SepmeKnng),
        "sepmerng" => Ok(MetricTarget::SepmeRng),
        "sepmegab" => Ok(MetricTarget::SepmeGab),
        "sepmegong" => Ok(MetricTarget::SepmeGong),
        "neighborhoodhit" | "mnh" => Ok(MetricTarget::NeighborhoodHit),
        "silhouette" | "silhouettescore" => Ok(MetricTarget::Silhouette),
        "calinskiharabasz" | "calinskiharabaszscore" | "ch" => Ok(MetricTarget::CalinskiHarabasz),
        "daviesbouldin" | "daviesbouldinscore" | "db" => Ok(MetricTarget::DaviesBouldin),
        "cqkmeansari" => Ok(MetricTarget::CqKmeansAri),
        "cqkmeansami" => Ok(MetricTarget::CqKmeansAmi),
        "cqhachom" | "cqhachomogeneity" => Ok(MetricTarget::CqHacHom),
        "cqhacami" => Ok(MetricTarget::CqHacAmi),
        "pairwisedistanceoverlaperror" | "pdoe" => Ok(MetricTarget::PairwiseDistanceOverlapError),
        "sprawlternn" => Ok(MetricTarget::SprawlterNn),
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
    /// Wraps an existing `egraph_core::CsrGraph<f32>` in a `JsCsrGraph`.
    pub(crate) fn from_inner(inner: CsrGraph<f32>) -> Self {
        Self { inner }
    }

    /// Returns a reference to the underlying `egraph_core::CsrGraph<f32>`.
    pub fn inner(&self) -> &CsrGraph<f32> {
        &self.inner
    }
}

/// Largest connected component result returned by `CsrGraph.largestConnectedComponent()`.
#[wasm_bindgen(js_name = LargestConnectedComponentResult)]
#[derive(Debug, Clone)]
pub struct JsLargestConnectedComponent {
    graph: CsrGraph<f32>,
    original_node_ids: Vec<u32>,
}

#[wasm_bindgen(js_class = LargestConnectedComponentResult)]
impl JsLargestConnectedComponent {
    /// Returns a cloned handle to the induced `CsrGraph` of the largest connected component.
    pub fn graph(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.graph.clone())
    }

    /// Returns the sorted original vertex IDs (`Uint32Array`) corresponding to compact vertices `0..k`.
    #[wasm_bindgen(js_name = originalNodeIds)]
    pub fn original_node_ids(&self) -> Vec<u32> {
        self.original_node_ids.clone()
    }

    /// Alias for `originalNodeIds()`.
    #[wasm_bindgen(js_name = nodeIds)]
    pub fn node_ids(&self) -> Vec<u32> {
        self.original_node_ids.clone()
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

    /// Returns the number of stored CSR half-edge entries (`indices().length`).
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns `true` if the graph is directed.
    #[wasm_bindgen(js_name = isDirected)]
    pub fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Returns `true` if explicit edge weights are stored.
    #[wasm_bindgen(js_name = isWeighted)]
    pub fn is_weighted(&self) -> bool {
        self.inner.is_weighted()
    }

    /// Returns the degree (outgoing degree for directed graphs) of vertex `u`.
    pub fn degree(&self, u: f64) -> Result<usize, JsError> {
        let idx = parse_non_negative_usize(u, "u")?;
        self.inner.try_degree(idx).map_err(core_err_to_js)
    }

    /// Returns a copy of the neighbor indices adjacent to vertex `u`.
    pub fn neighbors(&self, u: f64) -> Result<Vec<u32>, JsError> {
        let idx = parse_non_negative_usize(u, "u")?;
        self.inner
            .try_neighbors(idx)
            .map(|slice| slice.to_vec())
            .map_err(core_err_to_js)
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

    /// Extracts the induced subgraph on `nodes` with compact `0..nodes.length` vertex IDs.
    #[wasm_bindgen(js_name = inducedSubgraph)]
    pub fn induced_subgraph(&self, nodes: &[u32]) -> Result<JsCsrGraph, JsError> {
        let sub = self.inner.induced_subgraph(nodes).map_err(core_err_to_js)?;
        Ok(JsCsrGraph { inner: sub })
    }

    /// Extracts the largest (weakly) connected component as `{ graph(), originalNodeIds() }`.
    #[wasm_bindgen(js_name = largestConnectedComponent)]
    pub fn largest_connected_component(&self) -> JsLargestConnectedComponent {
        let (graph, original_node_ids) = self.inner.largest_connected_component();
        JsLargestConnectedComponent {
            graph,
            original_node_ids,
        }
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
    /// Creates a `JsLayout2d` by copying coordinates from a core SoA `Layout2d<f32>`.
    pub(crate) fn from_layout2d(layout: &Layout2d<f32>) -> Self {
        Self {
            inner: InterleavedLayout2d::from_layout2d(layout),
        }
    }

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

    /// Creates a deterministically initialized `Layout2d` for `n` vertices using `seed` (default `42`).
    pub fn init(n: f64, seed: Option<f64>) -> Result<JsLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        let seed_u64 = parse_seed_u64(seed)?;
        let mut rng = DeterministicRng::from_seed_u64(seed_u64);
        let mut coords = Vec::with_capacity(count.saturating_mul(2));
        for _ in 0..count {
            coords.push((rng.next_f64() - 0.5) as f32);
            coords.push((rng.next_f64() - 0.5) as f32);
        }
        let inner = InterleavedLayout2d::<f32>::from_interleaved(&coords).map_err(ffi_err_to_js)?;
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

    /// Returns the number of vertices in the layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
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

    /// Returns a freshly allocated `Float32Array` copy of `[x0, y0, ...]` coordinates.
    #[wasm_bindgen(js_name = copyCoords)]
    pub fn copy_coords(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
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

    /// Centers coordinates at zero mean and scales isotropically to `targetStd` (default `1.0`).
    #[wasm_bindgen(js_name = normalizeStd)]
    pub fn normalize_std(&mut self, target_std: Option<f64>) -> Result<(), JsError> {
        let t_std = target_std.unwrap_or(1.0);
        if !t_std.is_finite() || t_std <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "targetStd must be finite and strictly positive",
            )));
        }
        let t_f32 = t_std as f32;
        if !t_f32.is_finite() || t_f32 <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "targetStd overflows f32",
            )));
        }
        let mut soa = self.inner.to_layout2d();
        soa.center_and_normalize_to_std(t_f32)
            .map_err(core_err_to_js)?;
        self.inner.copy_from_layout2d(&soa).map_err(ffi_err_to_js)?;
        Ok(())
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}

/// Contiguous row-major `(numNodes * dim)` $d$-dimensional node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = LayoutNd)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLayoutNd {
    inner: LayoutNd<f32>,
}

impl JsLayoutNd {
    /// Creates a `JsLayoutNd` from a core `LayoutNd<f32>`.
    pub(crate) fn from_inner(inner: LayoutNd<f32>) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = LayoutNd)]
impl JsLayoutNd {
    /// Creates a zero-initialized `dim`-dimensional layout for `n` vertices.
    pub fn zeros(n: f64, dim: f64) -> Result<JsLayoutNd, JsError> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let dims = parse_non_negative_usize(dim, "dim")?;
        let inner = LayoutNd::<f32>::zeros(num_nodes, dims).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `LayoutNd` from flat row-major coordinates of length `n * dim`.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(n: f64, dim: f64, coords: &[f32]) -> Result<JsLayoutNd, JsError> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let dims = parse_non_negative_usize(dim, "dim")?;
        let inner =
            LayoutNd::<f32>::from_flat(num_nodes, dims, coords.to_vec()).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the number of vertices in the layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the dimensionality of the embedding space.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.inner.dims()
    }

    /// Returns the number of vertices in the layout.
    pub fn len(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns `true` if the layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the byte offset of the row-major `f32` buffer in WebAssembly linear memory.
    #[wasm_bindgen(js_name = coordsPtr)]
    pub fn coords_ptr(&self) -> usize {
        self.inner.coords().as_ptr() as usize
    }

    /// Returns the total scalar length of the row-major buffer (`numNodes * dim`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.inner.coords().len()
    }

    /// Returns a freshly allocated `Float32Array` copy of the row-major coordinate buffer.
    #[wasm_bindgen(js_name = copyCoords)]
    pub fn copy_coords(&self) -> Vec<f32> {
        self.inner.coords().to_vec()
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

    /// Returns exact DRGraph Eq. (17) crosslessness, or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn crosslessness(&self) -> Option<f64> {
        self.inner.crosslessness
    }

    /// Returns sampled DRGraph Eq. (17) crosslessness, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledCrosslessness)]
    pub fn sampled_crosslessness(&self) -> Option<f64> {
        self.inner.sampled_crosslessness
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

    /// Returns deterministic vertex-sampled 1-hop neighborhood preservation (`NP1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledNp1)]
    pub fn sampled_np1(&self) -> Option<f64> {
        self.inner.sampled_np1
    }

    /// Returns deterministic vertex-sampled 2-hop neighborhood preservation (`NP2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledNp2)]
    pub fn sampled_np2(&self) -> Option<f64> {
        self.inner.sampled_np2
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

    /// Returns Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
    /// (`S_RNG`, distinct from deferred Hong et al. 2022 `dRNG` and Fang et al. 2025 `dcRNG`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = relativeNeighborhoodPreservation)]
    pub fn relative_neighborhood_preservation(&self) -> Option<f64> {
        self.inner.relative_neighborhood_preservation
    }

    /// Returns Eades et al. (2017) unweighted shape-based Gabriel Graph preservation
    /// (`S_GG`, distinct from deferred Hong et al. 2022 `dGG` and Fang et al. 2025 `dcGG`), or `undefined` if not requested.
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

    /// Returns DRGraph Eq. (19) strict minimum-angle resolution (`A2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = minimumAngle)]
    pub fn minimum_angle(&self) -> Option<f64> {
        self.inner.minimum_angle
    }

    /// Returns DRGraph Eq. (19) minimum-angle resolution (`A1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = minimumAngleA1)]
    pub fn minimum_angle_a1(&self) -> Option<f64> {
        self.inner.minimum_angle_a1
    }

    /// Returns scalar layout-degeneracy summary, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = layoutDegeneracy)]
    pub fn layout_degeneracy(&self) -> Option<f64> {
        self.inner.layout_degeneracy
    }

    /// Returns ideal edge-length preservation error sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = idealEdgeLengths)]
    pub fn ideal_edge_lengths(&self) -> Option<f64> {
        self.inner.ideal_edge_lengths
    }

    /// Returns Adjusted Patch-NIG score, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = adjustedPatchNig)]
    pub fn adjusted_patch_nig(&self) -> Option<f64> {
        self.inner.adjusted_patch_nig
    }

    /// Returns Grid-based Cluster Area Faithfulness (`G-CAF`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = clusterAreaFaithfulness)]
    pub fn cluster_area_faithfulness(&self) -> Option<f64> {
        self.inner.cluster_area_faithfulness
    }

    /// Returns Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = landmarkStratifiedRta)]
    pub fn landmark_stratified_rta(&self) -> Option<f64> {
        self.inner.landmark_stratified_rta
    }

    /// Returns Distance Consistency (`DSC`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn dsc(&self) -> Option<f64> {
        self.inner.dsc
    }

    /// Returns Distribution Consistency (`DC`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn dc(&self) -> Option<f64> {
        self.inner.dc
    }

    /// Returns Class Density Measure (`CDM`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn cdm(&self) -> Option<f64> {
        self.inner.cdm
    }

    /// Returns 2D Histogram Density Measure (`2D-HDM`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn hdm(&self) -> Option<f64> {
        self.inner.hdm
    }

    /// Returns SepMe $k$-NN graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeKnng)]
    pub fn sepme_knng(&self) -> Option<f64> {
        self.inner.sepme_knng
    }

    /// Returns SepMe Relative Neighborhood Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeRng)]
    pub fn sepme_rng(&self) -> Option<f64> {
        self.inner.sepme_rng
    }

    /// Returns SepMe Gabriel Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeGab)]
    pub fn sepme_gab(&self) -> Option<f64> {
        self.inner.sepme_gab
    }

    /// Returns SepMe $\gamma$-Observable Neighbor Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeGong)]
    pub fn sepme_gong(&self) -> Option<f64> {
        self.inner.sepme_gong
    }

    /// Returns Neighborhood Hit ($M_{\mathrm{NH}}$), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodHit)]
    pub fn neighborhood_hit(&self) -> Option<f64> {
        self.inner.neighborhood_hit
    }

    /// Returns Silhouette coefficient, or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn silhouette(&self) -> Option<f64> {
        self.inner.silhouette
    }

    /// Returns Calinski-Harabasz Index, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = calinskiHarabasz)]
    pub fn calinski_harabasz(&self) -> Option<f64> {
        self.inner.calinski_harabasz
    }

    /// Returns Davies-Bouldin Index, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = daviesBouldin)]
    pub fn davies_bouldin(&self) -> Option<f64> {
        self.inner.davies_bouldin
    }

    /// Returns $k$-means Cluster Quality (`CQ-kmeans` ARI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqKmeansAri)]
    pub fn cq_kmeans_ari(&self) -> Option<f64> {
        self.inner.cq_kmeans_ari
    }

    /// Returns $k$-means Cluster Quality (`CQ-kmeans` AMI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqKmeansAmi)]
    pub fn cq_kmeans_ami(&self) -> Option<f64> {
        self.inner.cq_kmeans_ami
    }

    /// Returns Average-linkage HAC Cluster Quality (`CQ-HAC` Homogeneity), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqHacHom)]
    pub fn cq_hac_hom(&self) -> Option<f64> {
        self.inner.cq_hac_hom
    }

    /// Returns Average-linkage HAC Cluster Quality (`CQ-HAC` AMI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqHacAmi)]
    pub fn cq_hac_ami(&self) -> Option<f64> {
        self.inner.cq_hac_ami
    }

    /// Returns Pairwise Distance-Overlap Error (`PDOE`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = pairwiseDistanceOverlapError)]
    pub fn pairwise_distance_overlap_error(&self) -> Option<f64> {
        self.inner.pairwise_distance_overlap_error
    }

    /// Returns Sprawlter-NN composite overlap clutter measure, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sprawlterNn)]
    pub fn sprawlter_nn(&self) -> Option<f64> {
        self.inner.sprawlter_nn
    }

    /// Converts requested metrics into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let entries: [(&str, Option<f64>); 42] = [
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
            ("crosslessness", self.inner.crosslessness),
            ("sampledCrosslessness", self.inner.sampled_crosslessness),
            (
                "neighborhoodPreservation1Hop",
                self.inner.neighborhood_preservation_1hop,
            ),
            (
                "neighborhoodPreservation2Hop",
                self.inner.neighborhood_preservation_2hop,
            ),
            ("sampledNp1", self.inner.sampled_np1),
            ("sampledNp2", self.inner.sampled_np2),
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
            ("minimumAngle", self.inner.minimum_angle),
            ("minimumAngleA1", self.inner.minimum_angle_a1),
            ("layoutDegeneracy", self.inner.layout_degeneracy),
            ("idealEdgeLengths", self.inner.ideal_edge_lengths),
            ("adjustedPatchNig", self.inner.adjusted_patch_nig),
            (
                "clusterAreaFaithfulness",
                self.inner.cluster_area_faithfulness,
            ),
            ("landmarkStratifiedRta", self.inner.landmark_stratified_rta),
            ("dsc", self.inner.dsc),
            ("dc", self.inner.dc),
            ("cdm", self.inner.cdm),
            ("hdm", self.inner.hdm),
            ("sepmeKnng", self.inner.sepme_knng),
            ("sepmeRng", self.inner.sepme_rng),
            ("sepmeGab", self.inner.sepme_gab),
            ("sepmeGong", self.inner.sepme_gong),
            ("neighborhoodHit", self.inner.neighborhood_hit),
            ("silhouette", self.inner.silhouette),
            ("calinskiHarabasz", self.inner.calinski_harabasz),
            ("daviesBouldin", self.inner.davies_bouldin),
            ("cqKmeansAri", self.inner.cq_kmeans_ari),
            ("cqKmeansAmi", self.inner.cq_kmeans_ami),
            ("cqHacHom", self.inner.cq_hac_hom),
            ("cqHacAmi", self.inner.cq_hac_ami),
            (
                "pairwiseDistanceOverlapError",
                self.inner.pairwise_distance_overlap_error,
            ),
            ("sprawlterNn", self.inner.sprawlter_nn),
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

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`).
#[wasm_bindgen(js_name = MinimumAngleSummary)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsMinimumAngleSummary {
    inner: MinimumAngleSummary,
}

#[wasm_bindgen(js_class = MinimumAngleSummary)]
impl JsMinimumAngleSummary {
    #[wasm_bindgen(getter)]
    pub fn a1(&self) -> f64 {
        self.inner.a1
    }

    #[wasm_bindgen(getter)]
    pub fn a2(&self) -> f64 {
        self.inner.a2
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("a1"),
            &JsValue::from_f64(self.inner.a1),
        )
        .map_err(|e| JsError::new(&format!("failed to set a1: {e:?}")))?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("a2"),
            &JsValue::from_f64(self.inner.a2),
        )
        .map_err(|e| JsError::new(&format!("failed to set a2: {e:?}")))?;
        Ok(obj)
    }
}

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

/// Detailed result returned by `landmarkStratifiedRtaDetail` and `landmarkStratifiedRtaFromLandmarks`.
#[wasm_bindgen(js_name = RtaEvaluation)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsRtaEvaluation {
    inner: RtaEvaluation,
}

#[wasm_bindgen(js_class = RtaEvaluation)]
impl JsRtaEvaluation {
    #[wasm_bindgen(getter)]
    pub fn rta(&self) -> f64 {
        self.inner.rta
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks
    }

    #[wasm_bindgen(getter, js_name = numTriplets)]
    pub fn num_triplets(&self) -> usize {
        self.inner.num_triplets
    }

    #[wasm_bindgen(getter, js_name = numStrata)]
    pub fn num_strata(&self) -> usize {
        self.inner.num_strata
    }

    #[wasm_bindgen(getter, js_name = isExact)]
    pub fn is_exact(&self) -> bool {
        self.inner.is_exact
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        for (k, v) in [
            ("rta", self.inner.rta),
            ("numLandmarks", self.inner.num_landmarks as f64),
            ("numTriplets", self.inner.num_triplets as f64),
            ("numStrata", self.inner.num_strata as f64),
        ] {
            js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v))
                .map_err(|e| JsError::new(&format!("failed to set {k}: {e:?}")))?;
        }
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("isExact"),
            &JsValue::from_bool(self.inner.is_exact),
        )
        .map_err(|e| JsError::new(&format!("failed to set isExact: {e:?}")))?;
        Ok(obj)
    }
}

/// Detailed Grid-based Cluster Area Faithfulness (`G-CAF`) report returned by
/// `clusterAreaFaithfulnessDetail`.
#[wasm_bindgen(js_name = ClusterAreaReport)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsClusterAreaReport {
    inner: ClusterAreaDetail,
}

#[wasm_bindgen(js_class = ClusterAreaReport)]
impl JsClusterAreaReport {
    #[wasm_bindgen(getter)]
    pub fn caf(&self) -> f64 {
        self.inner.report.caf
    }

    #[wasm_bindgen(getter, js_name = purityAdj)]
    pub fn purity_adj(&self) -> f64 {
        self.inner.report.purity_adj
    }

    #[wasm_bindgen(getter, js_name = propVis)]
    pub fn prop_vis(&self) -> f64 {
        self.inner.report.prop_vis
    }

    #[wasm_bindgen(getter, js_name = resDensity)]
    pub fn res_density(&self) -> f64 {
        self.inner.report.res_density
    }

    #[wasm_bindgen(getter, js_name = resDeconv)]
    pub fn res_deconv(&self) -> f64 {
        self.inner.report.res_deconv
    }

    #[wasm_bindgen(getter, js_name = resCombined)]
    pub fn res_combined(&self) -> f64 {
        self.inner.report.res_combined
    }

    #[wasm_bindgen(getter, js_name = spatialCoherence)]
    pub fn spatial_coherence(&self) -> f64 {
        self.inner.report.spatial_coherence
    }

    #[wasm_bindgen(getter, js_name = inFrac)]
    pub fn in_frac(&self) -> f64 {
        self.inner.report.in_frac
    }

    #[wasm_bindgen(getter, js_name = effectiveBoundaryR95)]
    pub fn effective_boundary_r95(&self) -> f64 {
        self.inner.report.effective_boundary_r95
    }

    #[wasm_bindgen(getter, js_name = gEff)]
    pub fn g_eff(&self) -> usize {
        self.inner.report.g_eff
    }

    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.report.num_clusters
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.report.num_nodes
    }

    #[wasm_bindgen(js_name = clusterIds)]
    pub fn cluster_ids(&self) -> Vec<u32> {
        self.inner.cluster_ids.clone()
    }

    pub fn counts(&self) -> Vec<u32> {
        self.inner.counts.iter().map(|&c| c as u32).collect()
    }

    pub fn p(&self) -> Vec<f64> {
        self.inner.p.clone()
    }

    pub fn q(&self) -> Vec<f64> {
        self.inner.q.clone()
    }

    pub fn purity(&self) -> Vec<f64> {
        self.inner.purity.clone()
    }

    pub fn resolution(&self) -> Vec<f64> {
        self.inner.resolution.clone()
    }

    #[wasm_bindgen(js_name = resDensityPerCluster)]
    pub fn res_density_per_cluster(&self) -> Vec<f64> {
        self.inner.res_density.clone()
    }

    #[wasm_bindgen(js_name = resDeconvPerCluster)]
    pub fn res_deconv_per_cluster(&self) -> Vec<f64> {
        self.inner.res_deconv.clone()
    }

    #[wasm_bindgen(js_name = cafPerCluster)]
    pub fn caf_per_cluster(&self) -> Vec<f64> {
        self.inner.caf_per_cluster.clone()
    }

    pub fn coherence(&self) -> Vec<f64> {
        self.inner.coherence.clone()
    }

    /// Converts summary and per-cluster diagnostics into a plain JavaScript object without `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let set = |k: &str, v: &JsValue| -> Result<(), JsError> {
            js_sys::Reflect::set(&obj, &JsValue::from_str(k), v)
                .map(|_| ())
                .map_err(|e| JsError::new(&format!("failed to set {k}: {e:?}")))
        };
        let r = &self.inner.report;
        for (k, v) in [
            ("caf", r.caf),
            ("purityAdj", r.purity_adj),
            ("propVis", r.prop_vis),
            ("resDensity", r.res_density),
            ("resDeconv", r.res_deconv),
            ("resCombined", r.res_combined),
            ("spatialCoherence", r.spatial_coherence),
            ("inFrac", r.in_frac),
            ("effectiveBoundaryR95", r.effective_boundary_r95),
            ("gEff", r.g_eff as f64),
            ("numClusters", r.num_clusters as f64),
            ("numNodes", r.num_nodes as f64),
        ] {
            set(k, &JsValue::from_f64(v))?;
        }
        let counts_u32 = self.counts();
        for (k, arr) in [
            ("clusterIds", self.inner.cluster_ids.as_slice()),
            ("counts", counts_u32.as_slice()),
        ] {
            set(k, &js_sys::Uint32Array::from(arr).into())?;
        }
        for (k, arr) in [
            ("p", self.inner.p.as_slice()),
            ("q", self.inner.q.as_slice()),
            ("purity", self.inner.purity.as_slice()),
            ("resolution", self.inner.resolution.as_slice()),
            ("resDensityPerCluster", self.inner.res_density.as_slice()),
            ("resDeconvPerCluster", self.inner.res_deconv.as_slice()),
            ("cafPerCluster", self.inner.caf_per_cluster.as_slice()),
            ("coherence", self.inner.coherence.as_slice()),
        ] {
            set(k, &js_sys::Float64Array::from(arr).into())?;
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

/// Estimates 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` via deterministic vertex sampling (`numSamples` default `4096`, `seed` default `42`).
#[wasm_bindgen(js_name = sampledNp1)]
pub fn js_sampled_np1(
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
    sampled_neighborhood_preservation_1hop(graph.inner(), &soa, samples, seed_u64)
        .map_err(core_err_to_js)
}

/// Estimates 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` via deterministic vertex sampling (`numSamples` default `4096`, `seed` default `42`).
#[wasm_bindgen(js_name = sampledNp2)]
pub fn js_sampled_np2(
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
    sampled_neighborhood_preservation_2hop(graph.inner(), &soa, samples, seed_u64)
        .map_err(core_err_to_js)
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

/// Computes Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
/// (`S_RNG`) in `[0.0, 1.0]`.
///
/// Note: Implements classical unweighted `S_RNG` (Eades et al., JGAA 2017), distinct from
/// deferred Hong et al. (2022) degree-sensitive `dRNG` and Fang et al. (2025) degree-constrained
/// `dcRNG` (both deferred pending primary PDFs).
#[wasm_bindgen(js_name = drngPreservation)]
pub fn js_drng_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    relative_neighborhood_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes Eades et al. (2017) unweighted shape-based Gabriel Graph preservation
/// (`S_GG`) in `[0.0, 1.0]`.
///
/// Note: Implements classical unweighted `S_GG` (Eades et al., JGAA 2017), distinct from
/// deferred Hong et al. (2022) degree-sensitive `dGG` and Fang et al. (2025) degree-constrained
/// `dcGG` (both deferred pending primary PDFs).
#[wasm_bindgen(js_name = dggPreservation)]
pub fn js_dgg_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_preservation(graph.inner(), &soa).map_err(core_err_to_js)
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
    Ok(JsMinimumAngleSummary { inner })
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

/// Computes Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`).
#[wasm_bindgen(js_name = adjustedPatchNig)]
pub fn js_adjusted_patch_nig(
    layout: &JsLayout2d,
    community_labels: &[u32],
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
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
    adjusted_patch_nig(&soa, community_labels, radius, c_sz, pad, seed_u64).map_err(core_err_to_js)
}

/// Computes multi-scale Adjusted Patch-NIG across `radii` (default `[2, 8, 32]`).
#[wasm_bindgen(js_name = adjustedPatchNigMultiscale)]
pub fn js_adjusted_patch_nig_multiscale(
    layout: &JsLayout2d,
    community_labels: &[u32],
    radii: Option<Vec<u32>>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<f64>, JsError> {
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
    adjusted_patch_nig_multiscale(&soa, community_labels, &radii_usize, c_sz, pad, seed_u64)
        .map_err(core_err_to_js)
}

fn parse_cluster_area_config(
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
) -> Result<ClusterAreaConfig, JsError> {
    let def = ClusterAreaConfig::default();
    Ok(ClusterAreaConfig {
        g_max: match g_max {
            None => def.g_max,
            Some(v) => parse_non_negative_usize(v, "gMax")?,
        },
        sigma_cells: sigma_cells.unwrap_or(def.sigma_cells),
        kappa: kappa.unwrap_or(def.kappa),
        tau_frac: tau_frac.unwrap_or(def.tau_frac),
        boundary_r95: boundary_r95.unwrap_or(def.boundary_r95),
        stevens_gamma: stevens_gamma.unwrap_or(def.stevens_gamma),
    })
}

/// Computes Grid-based Cluster Area Faithfulness (`G-CAF`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = clusterAreaFaithfulness)]
#[allow(clippy::too_many_arguments)]
pub fn js_cluster_area_faithfulness(
    layout: &JsLayout2d,
    community_labels: &JsValue,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
) -> Result<f64, JsError> {
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let cfg = parse_cluster_area_config(
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
    )?;
    let soa = layout.inner().to_layout2d();
    cluster_area_faithfulness_with_weights(&soa, &labels, &cfg, nw.as_deref(), cw.as_deref())
        .map(|r| r.caf)
        .map_err(core_err_to_js)
}

/// Computes Grid-based Cluster Area Faithfulness (`G-CAF`) with per-cluster diagnostics ([`ClusterAreaReport`]).
#[wasm_bindgen(js_name = clusterAreaFaithfulnessDetail)]
#[allow(clippy::too_many_arguments)]
pub fn js_cluster_area_faithfulness_detail(
    layout: &JsLayout2d,
    community_labels: &JsValue,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
) -> Result<JsClusterAreaReport, JsError> {
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let cfg = parse_cluster_area_config(
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
    )?;
    let soa = layout.inner().to_layout2d();
    let inner = cluster_area_faithfulness_detail_with_weights(
        &soa,
        &labels,
        &cfg,
        nw.as_deref(),
        cw.as_deref(),
    )
    .map_err(core_err_to_js)?;
    Ok(JsClusterAreaReport { inner })
}

/// Computes Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = landmarkStratifiedRta)]
pub fn js_landmark_stratified_rta(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<f64, JsError> {
    let nl = match num_landmarks {
        None => DEFAULT_RTA_NUM_LANDMARKS,
        Some(v) => parse_non_negative_usize(v, "numLandmarks")?,
    };
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let md = match min_degree {
        None => DEFAULT_RTA_MIN_DEGREE,
        Some(v) => parse_non_negative_usize(v, "minDegree")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    landmark_stratified_rta(graph.inner(), &soa, nl, nt, md, seed_u64).map_err(core_err_to_js)
}

/// Computes Landmark-Stratified Random Triplet Accuracy with detailed metadata ([`RtaEvaluation`]).
#[wasm_bindgen(js_name = landmarkStratifiedRtaDetail)]
pub fn js_landmark_stratified_rta_detail(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let nl = match num_landmarks {
        None => DEFAULT_RTA_NUM_LANDMARKS,
        Some(v) => parse_non_negative_usize(v, "numLandmarks")?,
    };
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let md = match min_degree {
        None => DEFAULT_RTA_MIN_DEGREE,
        Some(v) => parse_non_negative_usize(v, "minDegree")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner = landmark_stratified_rta_detail(graph.inner(), &soa, nl, nt, md, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

/// Computes Landmark-Stratified RTA from a flat row-major `(numLandmarks * N)` `Uint32Array` hop table.
#[wasm_bindgen(js_name = landmarkStratifiedRtaFromLandmarks)]
pub fn js_landmark_stratified_rta_from_landmarks(
    layout: &JsLayout2d,
    landmark_ids: &[u32],
    landmark_dists: &[u32],
    num_triplets: Option<f64>,
    seed: Option<f64>,
) -> Result<JsRtaEvaluation, JsError> {
    let nt = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner =
        landmark_stratified_rta_from_landmarks(&soa, landmark_ids, landmark_dists, nt, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsRtaEvaluation { inner })
}

fn parse_cluster_baseline_config(
    options: &JsValue,
    default_seed: u64,
) -> Result<ClusterBaselineConfig, JsError> {
    let mut cfg = ClusterBaselineConfig {
        seed: default_seed,
        ..ClusterBaselineConfig::default()
    };
    cfg.sepme_knng_k = crate::v6_m3::opt_usize_prop(options, "sepmeKnngK", cfg.sepme_knng_k)?;
    cfg.neighborhood_hit_k =
        crate::v6_m3::opt_usize_prop(options, "neighborhoodHitK", cfg.neighborhood_hit_k)?;
    cfg.dc_grid_size = crate::v6_m3::opt_usize_prop(options, "dcGridSize", cfg.dc_grid_size)?;
    cfg.cdm_grid_size = crate::v6_m3::opt_usize_prop(options, "cdmGridSize", cfg.cdm_grid_size)?;
    if let Some(bw) = crate::v6_m3::opt_maybe_f64_prop(options, "cdmBandwidth")? {
        cfg.cdm_bandwidth = Some(bw);
    }
    cfg.num_bins = crate::v6_m3::opt_usize_prop(options, "numBins", cfg.num_bins)?;
    cfg.gong_gamma = crate::v6_m3::opt_f64_prop(options, "gongGamma", cfg.gong_gamma)?;
    cfg.kmeans_n_init = crate::v6_m3::opt_usize_prop(options, "kmeansNInit", cfg.kmeans_n_init)?;
    cfg.kmeans_max_iters =
        crate::v6_m3::opt_usize_prop(options, "kmeansMaxIters", cfg.kmeans_max_iters)?;
    let seed_val = crate::v6_m3::get_prop(options, "seed")?;
    if !seed_val.is_undefined() && !seed_val.is_null() {
        let num = seed_val
            .as_f64()
            .ok_or_else(|| JsError::new("options.seed must be a non-negative safe integer"))?;
        cfg.seed = parse_seed_u64(Some(num))?;
    }
    cfg.sprawlter_node_radius =
        crate::v6_m3::opt_f64_prop(options, "sprawlterNodeRadius", cfg.sprawlter_node_radius)?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_metrics_impl(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    community_labels: Option<&[u32]>,
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
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
    cluster_baselines_opt: Option<ClusterBaselineConfig>,
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
    let ideal_len = ideal_edge_length.unwrap_or(DEFAULT_UNIT_EDGE_LENGTH);
    let rta_num_landmarks = match num_landmarks {
        None => DEFAULT_RTA_NUM_LANDMARKS,
        Some(v) => parse_non_negative_usize(v, "numLandmarks")?,
    };
    let rta_num_triplets = match num_triplets {
        None => DEFAULT_RTA_NUM_TRIPLETS,
        Some(v) => parse_non_negative_usize(v, "numTriplets")?,
    };
    let rta_min_degree = match min_degree {
        None => DEFAULT_RTA_MIN_DEGREE,
        Some(v) => parse_non_negative_usize(v, "minDegree")?,
    };
    let patch_nig_radius = match patch_radius {
        None => DEFAULT_PATCH_NIG_RADIUS,
        Some(v) => parse_non_negative_usize(v, "patchRadius")?,
    };
    let patch_nig_canvas_size = match canvas_size {
        None => DEFAULT_PATCH_NIG_CANVAS_SIZE,
        Some(v) => parse_non_negative_usize(v, "canvasSize")?,
    };
    let patch_nig_pad_frac = pad_frac.unwrap_or(DEFAULT_PATCH_NIG_PAD_FRAC);
    let cluster_area = parse_cluster_area_config(
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
    )?;
    let cluster_baselines = cluster_baselines_opt.unwrap_or(ClusterBaselineConfig {
        seed: seed_u64,
        ..ClusterBaselineConfig::default()
    });

    let cfg = MetricConfig {
        sns_num_pivots,
        sns_seed: seed_u64,
        crossing_num_samples,
        crossing_seed: seed_u64,
        ideal_edge_length: ideal_len,
        rta_num_landmarks,
        rta_num_triplets,
        rta_min_degree,
        rta_seed: seed_u64,
        patch_nig_radius,
        patch_nig_canvas_size,
        patch_nig_pad_frac,
        patch_nig_seed: seed_u64,
        cluster_area,
        cluster_baselines,
    };
    let soa = layout.inner().to_layout2d();
    let report = evaluate_metrics_with_community_weights(
        graph.inner(),
        &soa,
        &parsed_targets,
        &cfg,
        community_labels,
        node_weights,
        cluster_weights,
    )
    .map_err(core_err_to_js)?;
    Ok(JsMetricReport { inner: report })
}

/// Selectively evaluates only the quality metrics listed in `targets`, skipping unrequested work.
#[wasm_bindgen(js_name = evaluateMetrics)]
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_entry(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots_or_options: &JsValue,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    community_labels: &JsValue,
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
    if crate::v6_m3::is_plain_js_object(num_pivots_or_options) {
        let has_extra = num_samples.is_some()
            || seed.is_some()
            || ideal_edge_length.is_some()
            || (!community_labels.is_undefined() && !community_labels.is_null())
            || num_landmarks.is_some()
            || num_triplets.is_some()
            || min_degree.is_some()
            || patch_radius.is_some()
            || canvas_size.is_some()
            || pad_frac.is_some()
            || g_max.is_some()
            || sigma_cells.is_some()
            || kappa.is_some()
            || tau_frac.is_some()
            || boundary_r95.is_some()
            || stevens_gamma.is_some()
            || (!node_weights.is_undefined() && !node_weights.is_null())
            || (!cluster_weights.is_undefined() && !cluster_weights.is_null())
            || (!cluster_baseline_options.is_undefined() && !cluster_baseline_options.is_null());
        crate::v6_m3::reject_extra_args_with_options(has_extra)?;

        let opts = num_pivots_or_options;
        let opt_num_pivots = crate::v6_m3::opt_maybe_f64_prop(opts, "numPivots")?;
        let opt_num_samples = crate::v6_m3::opt_maybe_f64_prop(opts, "numSamples")?;
        let opt_seed = crate::v6_m3::opt_maybe_f64_prop(opts, "seed")?;
        let opt_seed_u64 = parse_seed_u64(opt_seed)?;
        let opt_ideal_len = crate::v6_m3::opt_maybe_f64_prop(opts, "idealEdgeLength")?;
        let labels_val = crate::v6_m3::get_prop(opts, "communityLabels")?;
        let opt_labels = if labels_val.is_null() || labels_val.is_undefined() {
            None
        } else {
            Some(parse_u32_sequence(&labels_val, "communityLabels")?)
        };
        let opt_num_landmarks = crate::v6_m3::opt_maybe_f64_prop(opts, "numLandmarks")?;
        let opt_num_triplets = crate::v6_m3::opt_maybe_f64_prop(opts, "numTriplets")?;
        let opt_min_degree = crate::v6_m3::opt_maybe_f64_prop(opts, "minDegree")?;
        let opt_patch_radius = crate::v6_m3::opt_maybe_f64_prop(opts, "patchRadius")?;
        let opt_canvas_size = crate::v6_m3::opt_maybe_f64_prop(opts, "canvasSize")?;
        let opt_pad_frac = crate::v6_m3::opt_maybe_f64_prop(opts, "padFrac")?;
        let opt_g_max = crate::v6_m3::opt_maybe_f64_prop(opts, "gMax")?;
        let opt_sigma_cells = crate::v6_m3::opt_maybe_f64_prop(opts, "sigmaCells")?;
        let opt_kappa = crate::v6_m3::opt_maybe_f64_prop(opts, "kappa")?;
        let opt_tau_frac = crate::v6_m3::opt_maybe_f64_prop(opts, "tauFrac")?;
        let opt_boundary_r95 = crate::v6_m3::opt_maybe_f64_prop(opts, "boundaryR95")?;
        let opt_stevens_gamma = crate::v6_m3::opt_maybe_f64_prop(opts, "stevensGamma")?;
        let opt_nw = parse_optional_f64_sequence(
            &crate::v6_m3::get_prop(opts, "nodeWeights")?,
            "nodeWeights",
        )?;
        let opt_cw = parse_optional_f64_sequence(
            &crate::v6_m3::get_prop(opts, "clusterWeights")?,
            "clusterWeights",
        )?;
        let cb_nested = crate::v6_m3::get_prop(opts, "clusterBaselines")?;
        let cb_cfg =
            if crate::v6_m4::check_options_arg(&cb_nested, "evaluateMetrics::clusterBaselines")? {
                parse_cluster_baseline_config(&cb_nested, opt_seed_u64)?
            } else {
                parse_cluster_baseline_config(opts, opt_seed_u64)?
            };
        return evaluate_metrics_impl(
            graph,
            layout,
            targets,
            opt_num_pivots,
            opt_num_samples,
            opt_seed,
            opt_ideal_len,
            opt_labels.as_deref(),
            opt_num_landmarks,
            opt_num_triplets,
            opt_min_degree,
            opt_patch_radius,
            opt_canvas_size,
            opt_pad_frac,
            opt_g_max,
            opt_sigma_cells,
            opt_kappa,
            opt_tau_frac,
            opt_boundary_r95,
            opt_stevens_gamma,
            opt_nw.as_deref(),
            opt_cw.as_deref(),
            Some(cb_cfg),
        );
    }

    let num_pivots = if num_pivots_or_options.is_undefined() || num_pivots_or_options.is_null() {
        None
    } else if let Some(v) = num_pivots_or_options.as_f64() {
        Some(v)
    } else {
        return Err(JsError::new(
            "numPivotsOrOptions must be a number, options object, null, or undefined",
        ));
    };
    let parsed_labels = if community_labels.is_null() || community_labels.is_undefined() {
        None
    } else {
        Some(parse_u32_sequence(community_labels, "communityLabels")?)
    };
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let cb_cfg = if crate::v6_m4::check_options_arg(
        cluster_baseline_options,
        "evaluateMetrics::clusterBaselineOptions",
    )? {
        Some(parse_cluster_baseline_config(
            cluster_baseline_options,
            seed_u64,
        )?)
    } else {
        None
    };
    evaluate_metrics_impl(
        graph,
        layout,
        targets,
        num_pivots,
        num_samples,
        seed,
        ideal_edge_length,
        parsed_labels.as_deref(),
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
        nw.as_deref(),
        cw.as_deref(),
        cb_cfg,
    )
}

/// Rust-side 22-argument compatibility wrapper forwarding to `evaluate_metrics_impl`.
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics_with_options(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    community_labels: &JsValue,
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
    let parsed_labels = if community_labels.is_null() || community_labels.is_undefined() {
        None
    } else {
        Some(parse_u32_sequence(community_labels, "communityLabels")?)
    };
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    evaluate_metrics_impl(
        graph,
        layout,
        targets,
        num_pivots,
        num_samples,
        seed,
        ideal_edge_length,
        parsed_labels.as_deref(),
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
        nw.as_deref(),
        cw.as_deref(),
        None,
    )
}

/// Rust-side 14-argument compatibility wrapper forwarding to `evaluate_metrics_impl`.
#[allow(clippy::too_many_arguments)]
pub fn js_evaluate_metrics(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    targets: &JsValue,
    num_pivots: Option<f64>,
    num_samples: Option<f64>,
    seed: Option<f64>,
    ideal_edge_length: Option<f64>,
    community_labels: Option<Vec<u32>>,
    num_landmarks: Option<f64>,
    num_triplets: Option<f64>,
    min_degree: Option<f64>,
    patch_radius: Option<f64>,
    canvas_size: Option<f64>,
    pad_frac: Option<f64>,
) -> Result<JsMetricReport, JsError> {
    evaluate_metrics_impl(
        graph,
        layout,
        targets,
        num_pivots,
        num_samples,
        seed,
        ideal_edge_length,
        community_labels.as_deref(),
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
        None,
        None,
        None,
    )
}

/// Evaluates ideal edge-length preservation error against a uniform `unitEdgeLength > 0`.
#[wasm_bindgen(js_name = idealEdgeLengthsWithUnit)]
pub fn js_ideal_edge_lengths_with_unit(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
    unit_edge_length: f64,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    ideal_edge_lengths_with_unit(graph.inner(), &soa, unit_edge_length).map_err(core_err_to_js)
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

/// Extracts exact undirected Gabriel Graph (`GG`) edges as flat `[u0, v0, ...]` (`Uint32Array`).
#[wasm_bindgen(js_name = gabrielGraphEdges)]
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
#[wasm_bindgen(js_name = relativeNeighborhoodEdges)]
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

/// Computes an all-pairs shortest-path flat row-major `N * N` distance matrix via BFS.
#[wasm_bindgen(js_name = allPairsBfs)]
pub fn js_all_pairs_bfs(
    graph: &JsCsrGraph,
    unit_edge_length: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    all_pairs_bfs(
        graph.inner(),
        unit_edge_length.unwrap_or(DEFAULT_UNIT_EDGE_LENGTH),
    )
    .map_err(core_err_to_js)
}

/// Runs single-source unweighted BFS from `source`, scaling each hop by `unitEdgeLength` (default `1.0`).
#[wasm_bindgen(js_name = bfsSingleSource)]
pub fn js_bfs_single_source(
    graph: &JsCsrGraph,
    source: f64,
    unit_edge_length: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let src = parse_non_negative_usize(source, "source")?;
    let mut scratch = egraph::metrics::BfsScratch::new(graph.inner().num_nodes());
    let dists = bfs_single_source(
        graph.inner(),
        src,
        unit_edge_length.unwrap_or(DEFAULT_UNIT_EDGE_LENGTH),
        &mut scratch,
    )
    .map_err(core_err_to_js)?;
    Ok(dists.to_vec())
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) from a flat row-major `N * N` `distances`
/// matrix and optional `weights` matrix (`w_ij = 1 / d_ij^2` when omitted).
#[wasm_bindgen(js_name = scaleNormalizedStressFromDistances)]
pub fn js_scale_normalized_stress_from_distances(
    layout: &JsLayout2d,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    scale_normalized_stress_from_distances(&soa, distances, weights.as_deref())
        .map_err(core_err_to_js)
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
    Ok(JsPrismDistortion {
        sigma_dist: res.sigma_dist,
        sigma_disp: res.sigma_disp,
    })
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

#[cfg(feature = "delaunay")]
fn csr_f64_to_f32_unweighted_or_checked(g64: CsrGraph<f64>) -> Result<CsrGraph<f32>, JsError> {
    let weights_f32: Option<Vec<f32>> = g64
        .weights()
        .map(|ws| {
            ws.iter()
                .map(|&w| {
                    let wf = w as f32;
                    if !wf.is_finite() {
                        return Err(core_err_to_js(CoreError::non_finite("CsrGraph weight")));
                    }
                    Ok(wf)
                })
                .collect::<Result<Vec<f32>, JsError>>()
        })
        .transpose()?;
    CsrGraph::<f32>::from_csr(
        g64.indptr(),
        g64.indices(),
        weights_f32.as_deref(),
        g64.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}

/// Converts the 2D Delaunay triangulation of `layout` directly into an unweighted `CsrGraph`.
#[wasm_bindgen(js_name = delaunayToCsr)]
pub fn js_delaunay_to_csr(layout: &JsLayout2d) -> Result<JsCsrGraph, JsError> {
    #[cfg(feature = "delaunay")]
    {
        let soa = layout.inner().to_layout2d();
        let dt = Delaunay2d::triangulate(&soa).map_err(core_err_to_js)?;
        let csr64 = dt.to_csr().map_err(core_err_to_js)?;
        let csr32 = csr_f64_to_f32_unweighted_or_checked(csr64)?;
        Ok(JsCsrGraph::from_inner(csr32))
    }
    #[cfg(not(feature = "delaunay"))]
    {
        let _ = layout;
        Err(JsError::new(
            "delaunayToCsr requires the 'delaunay' feature to be enabled",
        ))
    }
}

/// Finds all overlapping rectangle pairs `[i0, j0, i1, j1, ...]` (`i < j`) for `layout`
/// with per-node `halfWidths` and `halfHeights` in $O(N \log N + K)$ time.
#[wasm_bindgen(js_name = rectangleOverlaps)]
pub fn js_rectangle_overlaps(
    layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<Vec<u32>, JsError> {
    let soa = layout.inner().to_layout2d();
    let pairs = RectangleOverlapSweep::find_layout_overlaps(&soa, half_widths, half_heights)
        .map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (i, j) in pairs {
        let iu = u32::try_from(i)
            .map_err(|_| JsError::new("rectangleOverlaps index i exceeds u32::MAX"))?;
        let ju = u32::try_from(j)
            .map_err(|_| JsError::new("rectangleOverlaps index j exceeds u32::MAX"))?;
        flat.push(iu);
        flat.push(ju);
    }
    Ok(flat)
}

/// Parses an edge-list CSV string (`num_nodes,num_edges` header + `u,v` rows) into a `CsrGraph`.
#[wasm_bindgen(js_name = parseCsrCsv)]
pub fn js_parse_csr_csv(
    csv: &str,
    directed: Option<bool>,
    sanitize: Option<bool>,
) -> Result<JsCsrGraph, JsError> {
    let g32 = parse_csr_csv::<f32>(csv, directed.unwrap_or(false), sanitize.unwrap_or(false))
        .map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(g32))
}
