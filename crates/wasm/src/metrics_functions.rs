//! WebAssembly bindings for 2D stress, geodesic distortion, neighborhood preservation,
//! proximity preservation, and BFS shortest-path metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::layout_buffers::JsLayout2d;
use crate::manifold::JsManifoldLayout2d;
use egraph::metrics::{
    BfsScratch, all_pairs_bfs, bfs_single_source, gabriel_graph_preservation,
    gabriel_graph_property, geodesic_distortion, neighborhood_preservation_1hop,
    neighborhood_preservation_2hop, relative_neighborhood_preservation,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    sampled_scale_normalized_stress, scale_normalized_stress,
    scale_normalized_stress_from_distances,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;
const DEFAULT_UNIT_EDGE_LENGTH: f64 = 1.0;

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` for `graph` and `layout`.
#[wasm_bindgen(js_name = _scaleNormalizedStress2d)]
pub fn js_scale_normalized_stress(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    scale_normalized_stress(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`) from `numPivots` (default `64`)
/// deterministic BFS sources with optional safe-integer `seed` (default `42`).
#[wasm_bindgen(js_name = _sampledScaleNormalizedStress2d)]
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

/// Computes closed-form Scale-Normalized Stress (`SNS`) from a flat row-major `N * N` `distances`
/// matrix and optional `weights` matrix (`w_ij = 1 / d_ij^2` when omitted).
#[wasm_bindgen(js_name = _scaleNormalizedStressFromDistances2d)]
pub fn js_scale_normalized_stress_from_distances(
    layout: &JsLayout2d,
    distances: &[f64],
    weights: Option<Vec<f64>>,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    scale_normalized_stress_from_distances(&soa, distances, weights.as_deref())
        .map_err(core_err_to_js)
}

/// Computes Miller et al. (2024) Eq. (2) Geodesic Distortion without post-hoc rescaling
/// on a `ManifoldLayout2d`.
#[wasm_bindgen(js_name = geodesicDistortion)]
pub fn js_geodesic_distortion(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<f64, JsError> {
    geodesic_distortion(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

/// Computes `StaticKdTree`-accelerated 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = _neighborhoodPreservation1Hop2d)]
pub fn js_neighborhood_preservation_1hop(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    neighborhood_preservation_1hop(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes `StaticKdTree`-accelerated 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = _neighborhoodPreservation2Hop2d)]
pub fn js_neighborhood_preservation_2hop(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    neighborhood_preservation_2hop(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Estimates 1-hop neighborhood preservation (`NP1`) via vertex sampling (`numSamples` default `4096`).
#[wasm_bindgen(js_name = _sampledNp12d)]
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

/// Estimates 2-hop neighborhood preservation (`NP2`) via vertex sampling (`numSamples` default `4096`).
#[wasm_bindgen(js_name = _sampledNp22d)]
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
#[wasm_bindgen(js_name = _gabrielGraphProperty2d)]
pub fn js_gabriel_graph_property(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_property(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation (`S_RNG`).
#[wasm_bindgen(js_name = _drngPreservation2d)]
pub fn js_drng_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    relative_neighborhood_preservation(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Computes Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`).
#[wasm_bindgen(js_name = _dggPreservation2d)]
pub fn js_dgg_preservation(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_preservation(graph.inner(), &soa).map_err(core_err_to_js)
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
    let mut scratch = BfsScratch::new(graph.inner().num_nodes());
    let dists = bfs_single_source(
        graph.inner(),
        src,
        unit_edge_length.unwrap_or(DEFAULT_UNIT_EDGE_LENGTH),
        &mut scratch,
    )
    .map_err(core_err_to_js)?;
    Ok(dists.to_vec())
}
