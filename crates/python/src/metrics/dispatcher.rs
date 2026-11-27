//! Selective multi-metric evaluation dispatcher (`evaluate_metrics`).

use egraph::metrics as egraph_metrics;
use egraph_metrics::{
    ClusterAreaConfig, ClusterBaselineConfig, MetricConfig, MetricInputs, MetricTarget,
    evaluate_metrics_with_inputs,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    extract_u32_1d, is_non_sequence_container,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, layout_f32_to_f64};
use crate::solver_common::vec_f64_to_f32_checked;

use super::nd_manifold_helpers::{MetricLayoutInput, OwnedMetricLayout};
use super::report::PyMetricReport;

pub(super) fn parse_metric_target(raw: &str) -> Result<MetricTarget, String> {
    let normalized: String = raw
        .trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && !c.is_whitespace())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match normalized.as_str() {
        "scalenormalizedstress" | "sns" => Ok(MetricTarget::ScaleNormalizedStress),
        "sampledscalenormalizedstress" | "sampledsns" => {
            Ok(MetricTarget::SampledScaleNormalizedStress)
        }
        "geodesicdistortion" => Ok(MetricTarget::GeodesicDistortion),
        "crossingnumber" | "sweepcrossingnumber" => Ok(MetricTarget::CrossingNumber),
        "crossingedges" => Ok(MetricTarget::CrossingEdges),
        "crossingangle" | "sweepcrossingangle" => Ok(MetricTarget::CrossingAngle),
        "sampledcrossingnumber" | "sampledcrossing" => Ok(MetricTarget::SampledCrossingNumber),
        "crosslessness" => Ok(MetricTarget::Crosslessness),
        "sampledcrosslessness" => Ok(MetricTarget::SampledCrosslessness),
        "toruscrossingnumber" | "toruscrossing" => Ok(MetricTarget::TorusCrossingNumber),
        "crossingedgestorus" | "toruscrossingedges" => Ok(MetricTarget::CrossingEdgesTorus),
        "neighborhoodpreservation1hop" | "np1" => Ok(MetricTarget::NeighborhoodPreservation1Hop),
        "neighborhoodpreservation2hop" | "np2" => Ok(MetricTarget::NeighborhoodPreservation2Hop),
        "samplednp1" | "sampledneighborhoodpreservation1hop" => Ok(MetricTarget::SampledNp1),
        "samplednp2" | "sampledneighborhoodpreservation2hop" => Ok(MetricTarget::SampledNp2),
        "gabrielgraphproperty" | "gabrielproperty" => Ok(MetricTarget::GabrielGraphProperty),
        "noderesolution" => Ok(MetricTarget::NodeResolution),
        "relativeneighborhoodpreservation" | "drng" | "drngpreservation" => {
            Ok(MetricTarget::RelativeNeighborhoodPreservation)
        }
        "gabrielgraphpreservation" | "dgg" | "dggpreservation" => {
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
        "prismdistortion" | "prism" => Ok(MetricTarget::PrismDistortion),
        "adjustedpatchnig" | "patchnig" => Ok(MetricTarget::AdjustedPatchNig),
        "clusterareafaithfulness" | "clusterarea" | "gcaf" => {
            Ok(MetricTarget::ClusterAreaFaithfulness)
        }
        "landmarkstratifiedrta" | "lstratifiedrta" | "rta" => {
            Ok(MetricTarget::LandmarkStratifiedRta)
        }
        "dsc" | "distanceconsistency" => Ok(MetricTarget::DistanceConsistency),
        "dc" | "distributionconsistency" => Ok(MetricTarget::DistributionConsistency),
        "cdm" | "classdensitymeasure" => Ok(MetricTarget::ClassDensityMeasure),
        "hdm" | "2dhdm" | "histogramdensitymeasure" => Ok(MetricTarget::HistogramDensityMeasure),
        "sepmeknng" | "sepmeknng2" => Ok(MetricTarget::SepmeKnng),
        "sepmerng" => Ok(MetricTarget::SepmeRng),
        "sepmegab" => Ok(MetricTarget::SepmeGab),
        "sepmegong" => Ok(MetricTarget::SepmeGong),
        "neighborhoodhit" | "mnh" => Ok(MetricTarget::NeighborhoodHit),
        "silhouette" | "silhouettescore" => Ok(MetricTarget::SilhouetteScore),
        "calinskiharabasz" | "calinskiharabaszscore" | "ch" => {
            Ok(MetricTarget::CalinskiHarabaszScore)
        }
        "daviesbouldin" | "daviesbouldinscore" | "db" => Ok(MetricTarget::DaviesBouldinScore),
        "cqkmeansari" => Ok(MetricTarget::CqKmeansAri),
        "cqkmeansami" => Ok(MetricTarget::CqKmeansAmi),
        "cqhachom" | "cqhachomogeneity" => Ok(MetricTarget::CqHacHom),
        "cqhacami" => Ok(MetricTarget::CqHacAmi),
        "pairwisedistanceoverlaperror" | "pdoe" => Ok(MetricTarget::PairwiseDistanceOverlapError),
        "sprawlternn" => Ok(MetricTarget::SprawlterNn),
        "normetrics" | "nor" => Ok(MetricTarget::NorMetrics),
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

enum ExtractedOriginal {
    F64(egraph::core::Layout2d<f64>),
    F32(egraph::core::Layout2d<f32>),
}

/// Selectively evaluate requested quality metrics on `(graph, layout)` without running unrequested passes.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during
/// computation and returns a typed `MetricReport` whose unrequested attributes evaluate to `None`.
/// Note: `drng_preservation` (`S_RNG`) and `dgg_preservation` (`S_GG`) measure graph-edge
/// proximity preservation, distinct from `sepme_rng` and `sepme_gab` cluster-separation scores.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, targets, *, num_pivots = StrictUsize(64), num_samples = StrictUsize(4096), seed = StrictU64(42), ideal_edge_length = StrictF64(1.0), labels = None, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), patch_radius = StrictUsize(8), canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None, original_layout = None, sprawlter_half_widths = None, sprawlter_half_heights = None, sepme_knng_k = StrictUsize(2), neighborhood_hit_k = StrictUsize(7), dc_grid_size = StrictUsize(20), cdm_grid_size = StrictUsize(64), cdm_bandwidth = None, num_bins = StrictUsize(100), gong_gamma = StrictF64(0.35), kmeans_n_init = StrictUsize(5), kmeans_max_iterations = StrictUsize(100), sprawlter_node_radius = StrictF64(1.0)),
    text_signature = "(graph, layout, targets, *, num_pivots=64, num_samples=4096, seed=42, ideal_edge_length=1.0, labels=None, num_landmarks=64, num_triplets=1000000, min_degree=2, patch_radius=8, canvas_size=2048, pad_frac=0.04, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None, original_layout=None, sprawlter_half_widths=None, sprawlter_half_heights=None, sepme_knng_k=2, neighborhood_hit_k=7, dc_grid_size=20, cdm_grid_size=64, cdm_bandwidth=None, num_bins=100, gong_gamma=0.35, kmeans_n_init=5, kmeans_max_iterations=100, sprawlter_node_radius=1.0)"
)]
#[allow(clippy::too_many_arguments)]
fn evaluate_metrics(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    targets: &Bound<'_, PyAny>,
    num_pivots: StrictUsize,
    num_samples: StrictUsize,
    seed: StrictU64,
    ideal_edge_length: StrictF64,
    labels: Option<&Bound<'_, PyAny>>,
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
    original_layout: Option<&PyLayout2d>,
    sprawlter_half_widths: Option<&Bound<'_, PyAny>>,
    sprawlter_half_heights: Option<&Bound<'_, PyAny>>,
    sepme_knng_k: StrictUsize,
    neighborhood_hit_k: StrictUsize,
    dc_grid_size: StrictUsize,
    cdm_grid_size: StrictUsize,
    cdm_bandwidth: Option<StrictF64>,
    num_bins: StrictUsize,
    gong_gamma: StrictF64,
    kmeans_n_init: StrictUsize,
    kmeans_max_iterations: StrictUsize,
    sprawlter_node_radius: StrictF64,
) -> PyResult<PyMetricReport> {
    let target_list = extract_metric_targets(targets)?;
    let comm_labels_vec = match labels {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "labels")?),
        _ => None,
    };
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let hw_f64 = match sprawlter_half_widths {
        Some(obj) if !obj.is_none() => Some(extract_f64_1d(obj, "sprawlter_half_widths")?),
        _ => None,
    };
    let hh_f64 = match sprawlter_half_heights {
        Some(obj) if !obj.is_none() => Some(extract_f64_1d(obj, "sprawlter_half_heights")?),
        _ => None,
    };
    let orig_extracted = original_layout.map(|ol| match ol.inner() {
        Layout2dInner::F64(buf) => ExtractedOriginal::F64(buf.to_layout2d()),
        Layout2dInner::F32(buf) => ExtractedOriginal::F32(buf.to_layout2d()),
    });
    let owned_layout = layout.extract_owned()?;

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
            kmeans_max_iters: kmeans_max_iterations.0,
            seed: seed.0,
            sprawlter_node_radius: sprawlter_node_radius.0,
        },
    };

    let hw_f32 = match (&owned_layout, hw_f64.as_deref()) {
        (
            OwnedMetricLayout::F32_2d(_)
            | OwnedMetricLayout::F32Nd(_)
            | OwnedMetricLayout::F32Manifold(_),
            Some(hw),
        ) if !matches!(orig_extracted, Some(ExtractedOriginal::F64(_))) => {
            Some(vec_f64_to_f32_checked(hw, "sprawlter_half_widths")?)
        }
        _ => None,
    };
    let hh_f32 = match (&owned_layout, hh_f64.as_deref()) {
        (
            OwnedMetricLayout::F32_2d(_)
            | OwnedMetricLayout::F32Nd(_)
            | OwnedMetricLayout::F32Manifold(_),
            Some(hh),
        ) if !matches!(orig_extracted, Some(ExtractedOriginal::F64(_))) => {
            Some(vec_f64_to_f32_checked(hh, "sprawlter_half_heights")?)
        }
        _ => None,
    };

    let g = graph.inner();
    let report = py
        .detach(|| match (&owned_layout, &orig_extracted) {
            (OwnedMetricLayout::F64_2d(soa), opt_orig) => {
                let promoted;
                let orig_ref = match opt_orig {
                    Some(ExtractedOriginal::F64(o)) => Some(o),
                    Some(ExtractedOriginal::F32(o)) => {
                        promoted = layout_f32_to_f64(o);
                        Some(&promoted)
                    }
                    None => None,
                };
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: orig_ref,
                    sprawlter_half_widths: hw_f64.as_deref(),
                    sprawlter_half_heights: hh_f64.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, soa, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F32_2d(soa), Some(ExtractedOriginal::F64(o))) => {
                let soa64 = layout_f32_to_f64(soa);
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: Some(o),
                    sprawlter_half_widths: hw_f64.as_deref(),
                    sprawlter_half_heights: hh_f64.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, &soa64, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F32_2d(soa), opt_orig) => {
                let orig_ref = match opt_orig {
                    Some(ExtractedOriginal::F32(o)) => Some(o),
                    _ => None,
                };
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: orig_ref,
                    sprawlter_half_widths: hw_f32.as_deref(),
                    sprawlter_half_heights: hh_f32.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, soa, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F64Nd(nd), _) => {
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: None,
                    sprawlter_half_widths: hw_f64.as_deref(),
                    sprawlter_half_heights: hh_f64.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, nd, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F32Nd(nd), _) => {
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: None,
                    sprawlter_half_widths: hw_f32.as_deref(),
                    sprawlter_half_heights: hh_f32.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, nd, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F64Manifold(m), _) => {
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: None,
                    sprawlter_half_widths: hw_f64.as_deref(),
                    sprawlter_half_heights: hh_f64.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, m, &target_list, &config, &inputs)
            }
            (OwnedMetricLayout::F32Manifold(m), _) => {
                let inputs = MetricInputs {
                    labels: comm_labels_vec.as_deref(),
                    node_weights: nw.as_deref(),
                    cluster_weights: cw.as_deref(),
                    original_layout: None,
                    sprawlter_half_widths: hw_f32.as_deref(),
                    sprawlter_half_heights: hh_f32.as_deref(),
                };
                evaluate_metrics_with_inputs(&*g, m, &target_list, &config, &inputs)
            }
        })
        .map_err(core_error_to_py)?;

    Ok(PyMetricReport { inner: report })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetricReport>()?;
    m.add_function(wrap_pyfunction!(evaluate_metrics, m)?)?;
    Ok(())
}
