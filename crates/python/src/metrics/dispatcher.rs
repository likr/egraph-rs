//! Selective multi-metric evaluation dispatcher (`evaluate_metrics`).

use std::collections::BTreeMap;

use egraph::metrics as egraph_metrics;
use egraph_metrics::{ClusterAreaConfig, ClusterBaselineConfig, MetricConfig, MetricTarget};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, extract_optional_weights_1d, extract_u32_1d,
    is_non_sequence_container,
};
use crate::layout_buffers::{PyLayout2d, dispatch_layout2d};

pub(super) fn parse_metric_target(raw: &str) -> Result<MetricTarget, String> {
    let normalized: String = raw
        .trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match normalized.as_str() {
        "scalenormalizedstress" | "sns" => Ok(MetricTarget::ScaleNormalizedStress),
        "sampledscalenormalizedstress" | "sampledsns" => {
            Ok(MetricTarget::SampledScaleNormalizedStress)
        }
        "crossingnumber" | "sweepcrossingnumber" => Ok(MetricTarget::CrossingNumber),
        "crossingangle" | "sweepcrossingangle" => Ok(MetricTarget::CrossingAngle),
        "sampledcrossingnumber" | "sampledcrossing" => Ok(MetricTarget::SampledCrossingNumber),
        "crosslessness" => Ok(MetricTarget::Crosslessness),
        "sampledcrosslessness" => Ok(MetricTarget::SampledCrosslessness),
        "neighborhoodpreservation1hop" | "neighborhoodpreservation" | "np1" => {
            Ok(MetricTarget::NeighborhoodPreservation1Hop)
        }
        "neighborhoodpreservation2hop" | "neighborhoodpreservation2hopv6" | "np2" => {
            Ok(MetricTarget::NeighborhoodPreservation2Hop)
        }
        "samplednp1" | "sampledneighborhoodpreservation1hop" => Ok(MetricTarget::SampledNp1),
        "samplednp2" | "sampledneighborhoodpreservation2hop" => Ok(MetricTarget::SampledNp2),
        "gabrielgraphproperty" | "gabrielgraphpropertyv6" | "gabrielproperty" => {
            Ok(MetricTarget::GabrielGraphProperty)
        }
        "noderesolution" | "noderesolutionv6" => Ok(MetricTarget::NodeResolution),
        "relativeneighborhoodpreservation" | "drng" | "drngpreservation" => {
            Ok(MetricTarget::RelativeNeighborhoodPreservation)
        }
        "gabrielgraphpreservation" | "dgg" | "dggpreservation" => {
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
        _ => Err(format!("unknown metric target {raw:?}")),
    }
}

fn extract_metric_targets(targets: &Bound<'_, PyAny>) -> PyResult<Vec<MetricTarget>> {
    if is_non_sequence_container(targets) {
        return Err(PyTypeError::new_err(
            "targets must be a sequence or iterable of metric target strings",
        ));
    }
    let iter = targets.try_iter().map_err(|_| {
        PyTypeError::new_err("targets must be a sequence or iterable of metric target strings")
    })?;
    let mut parsed = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let raw = item
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("metric target entries must be strings"))?;
        parsed.push(parse_metric_target(&raw).map_err(PyValueError::new_err)?);
    }
    Ok(parsed)
}

/// Selectively evaluate requested quality metrics on `(graph, layout)` without running unrequested passes.
///
/// Supported target names and aliases include `"scale_normalized_stress"` (`"sns"`),
/// `"sampled_scale_normalized_stress"` (`"sampled_sns"`), `"crossing_number"`,
/// `"crossing_angle"`, `"sampled_crossing_number"`, `"crosslessness"`,
/// `"sampled_crosslessness"`, `"neighborhood_preservation_1hop"` (`"np1"`),
/// `"neighborhood_preservation_2hop"` (`"np2"`), `"sampled_np1"`, `"sampled_np2"`,
/// `"gabriel_graph_property"`, `"node_resolution"`, `"relative_neighborhood_preservation"`
/// (`"drng"`, Eades et al. 2017 unweighted `S_RNG`), `"gabriel_graph_preservation"`
/// (`"dgg"`, Eades et al. 2017 unweighted `S_GG`), `"aspect_ratio"`,
/// `"angular_resolution"`, `"minimum_angle"` (`"min_angle"`), `"minimum_angle_a1"`,
/// `"layout_degeneracy"`, `"ideal_edge_lengths"`, `"adjusted_patch_nig"` (requires
/// `community_labels`), `"cluster_area_faithfulness"` (`"gcaf"`, requires
/// `community_labels`), `"landmark_stratified_rta"` (`"rta"`), `"dsc"`, `"dc"`, `"cdm"`,
/// `"hdm"`, `"sepme_knng"` (`"sepme_knng2"`), `"sepme_rng"`, `"sepme_gab"`, `"sepme_gong"`,
/// `"neighborhood_hit"`, `"silhouette"`, `"calinski_harabasz"`, `"davies_bouldin"`,
/// `"cq_kmeans_ari"`, `"cq_kmeans_ami"`, `"cq_hac_hom"`, `"cq_hac_ami"`,
/// `"pairwise_distance_overlap_error"` (`"pdoe"`), and `"sprawlter_nn"`.
///
/// Releases the Python GIL during computation and returns a dict keyed by canonical
/// `MetricReport` snake_case field names.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, targets, *, num_pivots = StrictUsize(64), num_samples = StrictUsize(4096), seed = StrictU64(42), ideal_edge_length = StrictF64(1.0), community_labels = None, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), patch_radius = StrictUsize(8), canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None, sepme_knng_k = StrictUsize(2), neighborhood_hit_k = StrictUsize(7), dc_grid_size = StrictUsize(20), cdm_grid_size = StrictUsize(64), cdm_bandwidth = None, num_bins = StrictUsize(100), gong_gamma = StrictF64(0.35), kmeans_n_init = StrictUsize(5), kmeans_max_iters = StrictUsize(100), sprawlter_node_radius = StrictF64(1.0)),
    text_signature = "(graph, layout, targets, *, num_pivots=64, num_samples=4096, seed=42, ideal_edge_length=1.0, community_labels=None, num_landmarks=64, num_triplets=1000000, min_degree=2, patch_radius=8, canvas_size=2048, pad_frac=0.04, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None, sepme_knng_k=2, neighborhood_hit_k=7, dc_grid_size=20, cdm_grid_size=64, cdm_bandwidth=None, num_bins=100, gong_gamma=0.35, kmeans_n_init=5, kmeans_max_iters=100, sprawlter_node_radius=1.0)"
)]
#[allow(clippy::too_many_arguments)]
fn evaluate_metrics(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    targets: &Bound<'_, PyAny>,
    num_pivots: StrictUsize,
    num_samples: StrictUsize,
    seed: StrictU64,
    ideal_edge_length: StrictF64,
    community_labels: Option<&Bound<'_, PyAny>>,
    num_landmarks: StrictUsize,
    num_triplets: StrictUsize,
    min_degree: StrictUsize,
    patch_radius: StrictUsize,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    g_max: StrictUsize,
    sigma_cells: StrictF64,
    kappa: StrictF64,
    tau_frac: StrictF64,
    boundary_r95: StrictF64,
    stevens_gamma: StrictF64,
    node_weights: Option<&Bound<'_, PyAny>>,
    cluster_weights: Option<&Bound<'_, PyAny>>,
    sepme_knng_k: StrictUsize,
    neighborhood_hit_k: StrictUsize,
    dc_grid_size: StrictUsize,
    cdm_grid_size: StrictUsize,
    cdm_bandwidth: Option<StrictF64>,
    num_bins: StrictUsize,
    gong_gamma: StrictF64,
    kmeans_n_init: StrictUsize,
    kmeans_max_iters: StrictUsize,
    sprawlter_node_radius: StrictF64,
) -> PyResult<BTreeMap<String, f64>> {
    let target_list = extract_metric_targets(targets)?;
    let comm_labels_vec = match community_labels {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "community_labels")?),
        _ => None,
    };
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let config = MetricConfig {
        sns_num_pivots: num_pivots.0,
        sns_seed: seed.0,
        crossing_num_samples: num_samples.0,
        crossing_seed: seed.0,
        ideal_edge_length: ideal_edge_length.0,
        rta_num_landmarks: num_landmarks.0,
        rta_num_triplets: num_triplets.0,
        rta_min_degree: min_degree.0,
        rta_seed: seed.0,
        patch_nig_radius: patch_radius.0,
        patch_nig_canvas_size: canvas_size.0,
        patch_nig_pad_frac: pad_frac.0,
        patch_nig_seed: seed.0,
        cluster_area: ClusterAreaConfig {
            g_max: g_max.0,
            sigma_cells: sigma_cells.0,
            kappa: kappa.0,
            tau_frac: tau_frac.0,
            boundary_r95: boundary_r95.0,
            stevens_gamma: stevens_gamma.0,
        },
        cluster_baselines: ClusterBaselineConfig {
            sepme_knng_k: sepme_knng_k.0,
            neighborhood_hit_k: neighborhood_hit_k.0,
            dc_grid_size: dc_grid_size.0,
            cdm_grid_size: cdm_grid_size.0,
            cdm_bandwidth: cdm_bandwidth.map(|v| v.0),
            num_bins: num_bins.0,
            gong_gamma: gong_gamma.0,
            kmeans_n_init: kmeans_n_init.0,
            kmeans_max_iters: kmeans_max_iters.0,
            seed: seed.0,
            sprawlter_node_radius: sprawlter_node_radius.0,
        },
    };
    let g = graph.inner();
    let report = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::evaluate_metrics_with_community_weights(
            g,
            &soa,
            &target_list,
            &config,
            comm_labels_vec.as_deref(),
            nw.as_deref(),
            cw.as_deref(),
        )
    })?;

    let mut out = BTreeMap::new();
    if let Some(v) = report.scale_normalized_stress {
        out.insert("scale_normalized_stress".to_owned(), v);
    }
    if let Some(v) = report.sampled_scale_normalized_stress {
        out.insert("sampled_scale_normalized_stress".to_owned(), v);
    }
    if let Some(v) = report.crossing_number {
        out.insert("crossing_number".to_owned(), v as f64);
    }
    if let Some(v) = report.crossing_angle {
        out.insert("crossing_angle".to_owned(), v);
    }
    if let Some(v) = report.sampled_crossing_number {
        out.insert("sampled_crossing_number".to_owned(), v);
    }
    if let Some(v) = report.crosslessness {
        out.insert("crosslessness".to_owned(), v);
    }
    if let Some(v) = report.sampled_crosslessness {
        out.insert("sampled_crosslessness".to_owned(), v);
    }
    if let Some(v) = report.neighborhood_preservation_1hop {
        out.insert("neighborhood_preservation_1hop".to_owned(), v);
    }
    if let Some(v) = report.neighborhood_preservation_2hop {
        out.insert("neighborhood_preservation_2hop".to_owned(), v);
    }
    if let Some(v) = report.sampled_np1 {
        out.insert("sampled_np1".to_owned(), v);
    }
    if let Some(v) = report.sampled_np2 {
        out.insert("sampled_np2".to_owned(), v);
    }
    if let Some(v) = report.gabriel_graph_property {
        out.insert("gabriel_graph_property".to_owned(), v);
    }
    if let Some(v) = report.node_resolution {
        out.insert("node_resolution".to_owned(), v);
    }
    if let Some(v) = report.relative_neighborhood_preservation {
        out.insert("relative_neighborhood_preservation".to_owned(), v);
    }
    if let Some(v) = report.gabriel_graph_preservation {
        out.insert("gabriel_graph_preservation".to_owned(), v);
    }
    if let Some(v) = report.aspect_ratio {
        out.insert("aspect_ratio".to_owned(), v);
    }
    if let Some(v) = report.angular_resolution {
        out.insert("angular_resolution".to_owned(), v);
    }
    if let Some(v) = report.minimum_angle {
        out.insert("minimum_angle".to_owned(), v);
    }
    if let Some(v) = report.minimum_angle_a1 {
        out.insert("minimum_angle_a1".to_owned(), v);
    }
    if let Some(v) = report.layout_degeneracy {
        out.insert("layout_degeneracy".to_owned(), v);
    }
    if let Some(v) = report.ideal_edge_lengths {
        out.insert("ideal_edge_lengths".to_owned(), v);
    }
    if let Some(v) = report.adjusted_patch_nig {
        out.insert("adjusted_patch_nig".to_owned(), v);
    }
    if let Some(v) = report.cluster_area_faithfulness {
        out.insert("cluster_area_faithfulness".to_owned(), v);
    }
    if let Some(v) = report.landmark_stratified_rta {
        out.insert("landmark_stratified_rta".to_owned(), v);
    }
    if let Some(v) = report.dsc {
        out.insert("dsc".to_owned(), v);
    }
    if let Some(v) = report.dc {
        out.insert("dc".to_owned(), v);
    }
    if let Some(v) = report.cdm {
        out.insert("cdm".to_owned(), v);
    }
    if let Some(v) = report.hdm {
        out.insert("hdm".to_owned(), v);
    }
    if let Some(v) = report.sepme_knng {
        out.insert("sepme_knng".to_owned(), v);
    }
    if let Some(v) = report.sepme_rng {
        out.insert("sepme_rng".to_owned(), v);
    }
    if let Some(v) = report.sepme_gab {
        out.insert("sepme_gab".to_owned(), v);
    }
    if let Some(v) = report.sepme_gong {
        out.insert("sepme_gong".to_owned(), v);
    }
    if let Some(v) = report.neighborhood_hit {
        out.insert("neighborhood_hit".to_owned(), v);
    }
    if let Some(v) = report.silhouette {
        out.insert("silhouette".to_owned(), v);
    }
    if let Some(v) = report.calinski_harabasz {
        out.insert("calinski_harabasz".to_owned(), v);
    }
    if let Some(v) = report.davies_bouldin {
        out.insert("davies_bouldin".to_owned(), v);
    }
    if let Some(v) = report.cq_kmeans_ari {
        out.insert("cq_kmeans_ari".to_owned(), v);
    }
    if let Some(v) = report.cq_kmeans_ami {
        out.insert("cq_kmeans_ami".to_owned(), v);
    }
    if let Some(v) = report.cq_hac_hom {
        out.insert("cq_hac_hom".to_owned(), v);
    }
    if let Some(v) = report.cq_hac_ami {
        out.insert("cq_hac_ami".to_owned(), v);
    }
    if let Some(v) = report.pairwise_distance_overlap_error {
        out.insert("pairwise_distance_overlap_error".to_owned(), v);
    }
    if let Some(v) = report.sprawlter_nn {
        out.insert("sprawlter_nn".to_owned(), v);
    }
    Ok(out)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_metrics, m)?)?;
    Ok(())
}
