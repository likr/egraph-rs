//! Target parsing and configuration extraction for `evaluateMetrics`.

use egraph::metrics::{ClusterBaselineConfig, MetricConfig, MetricTarget};
use wasm_bindgen::prelude::*;

use crate::js_convert::{
    parse_non_negative_usize, parse_optional_f32_sequence, parse_optional_f64_sequence,
    parse_seed_u64, parse_u32_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, is_plain_js_object, normalize_enum_token, opt_f64_prop,
    opt_maybe_f64_prop, opt_usize_prop, reject_extra_args_with_options,
};
use crate::metrics_cluster_rta::parse_cluster_area_config;

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;
const DEFAULT_UNIT_EDGE_LENGTH: f64 = 1.0;
const DEFAULT_RTA_NUM_LANDMARKS: usize = 64;
const DEFAULT_RTA_NUM_TRIPLETS: usize = 1_000_000;
const DEFAULT_RTA_MIN_DEGREE: usize = 2;
const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;

pub(crate) fn parse_metric_target(raw: &str) -> Result<MetricTarget, JsError> {
    match normalize_enum_token(raw).as_str() {
        "scalenormalizedstress" | "sns" => Ok(MetricTarget::ScaleNormalizedStress),
        "sampledscalenormalizedstress" | "sampledsns" => {
            Ok(MetricTarget::SampledScaleNormalizedStress)
        }
        "geodesicdistortion" => Ok(MetricTarget::GeodesicDistortion),
        "crossingnumber" | "sweepcrossingnumber" | "crossings" => Ok(MetricTarget::CrossingNumber),
        "crossingedges" => Ok(MetricTarget::CrossingEdges),
        "crossingangle" | "sweepcrossingangle" => Ok(MetricTarget::CrossingAngle),
        "sampledcrossingnumber" | "sampledcrossings" => Ok(MetricTarget::SampledCrossingNumber),
        "crosslessness" => Ok(MetricTarget::Crosslessness),
        "sampledcrosslessness" => Ok(MetricTarget::SampledCrosslessness),
        "toruscrossingnumber" | "toruscrossings" => Ok(MetricTarget::TorusCrossingNumber),
        "crossingedgestorus" => Ok(MetricTarget::CrossingEdgesTorus),
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
        "prismdistortion" => Ok(MetricTarget::PrismDistortion),
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

pub(crate) fn parse_cluster_baseline_config(
    options: &JsValue,
    default_seed: u64,
) -> Result<ClusterBaselineConfig, JsError> {
    let mut cfg = ClusterBaselineConfig {
        seed: default_seed,
        ..ClusterBaselineConfig::default()
    };
    cfg.sepme_knng_k = opt_usize_prop(options, "sepmeKnngK", cfg.sepme_knng_k)?;
    cfg.neighborhood_hit_k = opt_usize_prop(options, "neighborhoodHitK", cfg.neighborhood_hit_k)?;
    cfg.dc_grid_size = opt_usize_prop(options, "dcGridSize", cfg.dc_grid_size)?;
    cfg.cdm_grid_size = opt_usize_prop(options, "cdmGridSize", cfg.cdm_grid_size)?;
    if let Some(bw) = opt_maybe_f64_prop(options, "cdmBandwidth")? {
        cfg.cdm_bandwidth = Some(bw);
    }
    cfg.num_bins = opt_usize_prop(options, "numBins", cfg.num_bins)?;
    cfg.gong_gamma = opt_f64_prop(options, "gongGamma", cfg.gong_gamma)?;
    cfg.kmeans_n_init = opt_usize_prop(options, "kmeansNInit", cfg.kmeans_n_init)?;
    cfg.kmeans_max_iters = opt_usize_prop(options, "kmeansMaxIterations", cfg.kmeans_max_iters)?;
    let seed_val = get_prop(options, "seed")?;
    if !seed_val.is_undefined() && !seed_val.is_null() {
        let num = seed_val
            .as_f64()
            .ok_or_else(|| JsError::new("options.seed must be a non-negative safe integer"))?;
        cfg.seed = parse_seed_u64(Some(num))?;
    }
    cfg.sprawlter_node_radius =
        opt_f64_prop(options, "sprawlterNodeRadius", cfg.sprawlter_node_radius)?;
    Ok(cfg)
}

pub(crate) struct ParsedMetricEvalInputs {
    pub(crate) targets: Vec<MetricTarget>,
    pub(crate) config: MetricConfig,
    pub(crate) labels: Option<Vec<u32>>,
    pub(crate) node_weights: Option<Vec<f64>>,
    pub(crate) cluster_weights: Option<Vec<f64>>,
    pub(crate) sprawlter_half_widths: Option<Vec<f32>>,
    pub(crate) sprawlter_half_heights: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn build_parsed_inputs(
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
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: Option<Vec<f64>>,
    cluster_weights: Option<Vec<f64>>,
    cluster_baselines_opt: Option<ClusterBaselineConfig>,
    sprawlter_half_widths: Option<Vec<f32>>,
    sprawlter_half_heights: Option<Vec<f32>>,
) -> Result<ParsedMetricEvalInputs, JsError> {
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
    let config = MetricConfig {
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
    Ok(ParsedMetricEvalInputs {
        targets: parsed_targets,
        config,
        labels,
        node_weights,
        cluster_weights,
        sprawlter_half_widths,
        sprawlter_half_heights,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_eval_args(
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
) -> Result<ParsedMetricEvalInputs, JsError> {
    if is_plain_js_object(num_pivots_or_options) {
        let has_extra = num_samples.is_some()
            || seed.is_some()
            || ideal_edge_length.is_some()
            || (!labels.is_undefined() && !labels.is_null())
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
        reject_extra_args_with_options(has_extra)?;

        let opts = num_pivots_or_options;
        let opt_num_pivots = opt_maybe_f64_prop(opts, "numPivots")?;
        let opt_num_samples = opt_maybe_f64_prop(opts, "numSamples")?;
        let opt_seed = opt_maybe_f64_prop(opts, "seed")?;
        let opt_seed_u64 = parse_seed_u64(opt_seed)?;
        let opt_ideal_len = opt_maybe_f64_prop(opts, "idealEdgeLength")?;
        let labels_val = get_prop(opts, "labels")?;
        let opt_labels = if labels_val.is_null() || labels_val.is_undefined() {
            None
        } else {
            Some(parse_u32_sequence(&labels_val, "labels")?)
        };
        let opt_num_landmarks = opt_maybe_f64_prop(opts, "numLandmarks")?;
        let opt_num_triplets = opt_maybe_f64_prop(opts, "numTriplets")?;
        let opt_min_degree = opt_maybe_f64_prop(opts, "minDegree")?;
        let opt_patch_radius = opt_maybe_f64_prop(opts, "patchRadius")?;
        let opt_canvas_size = opt_maybe_f64_prop(opts, "canvasSize")?;
        let opt_pad_frac = opt_maybe_f64_prop(opts, "padFrac")?;
        let opt_g_max = opt_maybe_f64_prop(opts, "gMax")?;
        let opt_sigma_cells = opt_maybe_f64_prop(opts, "sigmaCells")?;
        let opt_kappa = opt_maybe_f64_prop(opts, "kappa")?;
        let opt_tau_frac = opt_maybe_f64_prop(opts, "tauFrac")?;
        let opt_boundary_r95 = opt_maybe_f64_prop(opts, "boundaryR95")?;
        let opt_stevens_gamma = opt_maybe_f64_prop(opts, "stevensGamma")?;
        let opt_nw = parse_optional_f64_sequence(&get_prop(opts, "nodeWeights")?, "nodeWeights")?;
        let opt_cw =
            parse_optional_f64_sequence(&get_prop(opts, "clusterWeights")?, "clusterWeights")?;
        let opt_hw = parse_optional_f32_sequence(
            &get_prop(opts, "sprawlterHalfWidths")?,
            "sprawlterHalfWidths",
        )?;
        let opt_hh = parse_optional_f32_sequence(
            &get_prop(opts, "sprawlterHalfHeights")?,
            "sprawlterHalfHeights",
        )?;
        let cb_nested = get_prop(opts, "clusterBaselines")?;
        let cb_cfg = if check_options_arg(&cb_nested, "evaluateMetrics::clusterBaselines")? {
            parse_cluster_baseline_config(&cb_nested, opt_seed_u64)?
        } else {
            parse_cluster_baseline_config(opts, opt_seed_u64)?
        };
        return build_parsed_inputs(
            targets,
            opt_num_pivots,
            opt_num_samples,
            opt_seed,
            opt_ideal_len,
            opt_labels,
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
            opt_nw,
            opt_cw,
            Some(cb_cfg),
            opt_hw,
            opt_hh,
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
    let parsed_labels = if labels.is_null() || labels.is_undefined() {
        None
    } else {
        Some(parse_u32_sequence(labels, "labels")?)
    };
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let cb_cfg = if check_options_arg(
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
    build_parsed_inputs(
        targets,
        num_pivots,
        num_samples,
        seed,
        ideal_edge_length,
        parsed_labels,
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
        nw,
        cw,
        cb_cfg,
        None,
        None,
    )
}
