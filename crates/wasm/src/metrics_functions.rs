//! WebAssembly bindings for stress, neighborhood preservation, proximity graphs, PRISM distortion, Patch-NIG, and BFS distance metrics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_report::JsPrismDistortion;
use egraph::core::CoreError;
use egraph::metrics::{
    BfsScratch, MetricTarget, adjusted_patch_nig, adjusted_patch_nig_multiscale, all_pairs_bfs,
    bfs_single_source, gabriel_graph_edges, gabriel_graph_preservation, gabriel_graph_property,
    ideal_edge_lengths, ideal_edge_lengths_with_unit, neighborhood_preservation_1hop,
    neighborhood_preservation_2hop, prism_distortion, prism_distortion_with_edges,
    relative_neighborhood_edges, relative_neighborhood_preservation,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    sampled_scale_normalized_stress, scale_normalized_stress,
    scale_normalized_stress_from_distances,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;
const DEFAULT_UNIT_EDGE_LENGTH: f64 = 1.0;
const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;

pub(crate) fn parse_metric_target(raw: &str) -> Result<MetricTarget, JsError> {
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
        "neighborhoodpreservation1hop" | "np1" => Ok(MetricTarget::NeighborhoodPreservation1Hop),
        "neighborhoodpreservation2hop" | "np2" => Ok(MetricTarget::NeighborhoodPreservation2Hop),
        "sampledneighborhoodpreservation1hop"
        | "sampledneighborhoodpreservation"
        | "samplednp1" => Ok(MetricTarget::SampledNp1),
        "sampledneighborhoodpreservation2hop" | "samplednp2" => Ok(MetricTarget::SampledNp2),
        "gabrielgraphproperty" | "gabrielproperty" => Ok(MetricTarget::GabrielGraphProperty),
        "noderesolution" => Ok(MetricTarget::NodeResolution),
        "relativeneighborhoodpreservation" | "drngpreservation" | "drng" => {
            Ok(MetricTarget::RelativeNeighborhoodPreservation)
        }
        "gabrielgraphpreservation" | "dggpreservation" | "dgg" => {
            Ok(MetricTarget::GabrielGraphPreservation)
        }
        "aspectratio" => Ok(MetricTarget::AspectRatio),
        "angularresolution" => Ok(MetricTarget::AngularResolution),
        "minimumangle" | "minangle" | "minimumanglea2" | "minanglea2" => {
            Ok(MetricTarget::MinimumAngle)
        }
        "minimumanglea1" | "minanglea1" => Ok(MetricTarget::MinimumAngleA1),
        "layoutdegeneracy" => Ok(MetricTarget::LayoutDegeneracy),
        "idealedgelengths" => Ok(MetricTarget::IdealEdgeLengths),
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

pub(crate) fn parse_metric_targets(targets: &JsValue) -> Result<Vec<MetricTarget>, JsError> {
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
#[wasm_bindgen(js_name = neighborhoodPreservation2Hop)]
pub fn js_neighborhood_preservation_2hop(
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
#[wasm_bindgen(js_name = gabrielGraphProperty)]
pub fn js_gabriel_graph_property(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    gabriel_graph_property(graph.inner(), &soa).map_err(core_err_to_js)
}

/// Evaluates ideal edge-length preservation error sum for `graph` under `layout`.
#[wasm_bindgen(js_name = idealEdgeLengths)]
pub fn js_ideal_edge_lengths(graph: &JsCsrGraph, layout: &JsLayout2d) -> Result<f64, JsError> {
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
    Ok(JsPrismDistortion::new(res.sigma_dist, res.sigma_disp))
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
    Ok(JsPrismDistortion::new(res.sigma_dist, res.sigma_disp))
}
