//! Target parsing and `{ ...options }` extraction for `evaluateMetrics`.

use egraph::metrics::{ClusterAreaConfig, ClusterBaselineConfig, MetricConfig, MetricTarget};
use wasm_bindgen::prelude::*;

use crate::js_convert::{
    DEFAULT_SEED, invalid_input_err, parse_optional_f32_sequence, parse_optional_f64_sequence,
    parse_u32_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_usize_prop,
};

pub(crate) fn parse_metric_target(raw: &str) -> Result<MetricTarget, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
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
        _ => Err(invalid_input_err(format!("unknown metric target: {raw:?}"))),
    }
}

pub(crate) fn parse_metric_targets(targets: &JsValue) -> Result<Vec<MetricTarget>, JsValue> {
    if targets.is_null() || targets.is_undefined() || targets.is_string() || !targets.is_object() {
        return Err(invalid_input_err(
            "targets must be an array or iterable of metric target strings",
        ));
    }
    let iter = js_sys::try_iter(targets)
        .map_err(|err| invalid_input_err(format!("failed to iterate targets: {err:?}")))?
        .ok_or_else(|| invalid_input_err("targets must be an array or iterable of strings"))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res
            .map_err(|err| invalid_input_err(format!("failed to read target item: {err:?}")))?;
        let name = item
            .as_string()
            .ok_or_else(|| invalid_input_err("each entry in targets must be a string"))?;
        out.push(parse_metric_target(&name)?);
    }
    Ok(out)
}

pub(crate) fn parse_cluster_baseline_config(
    options: &JsValue,
    default_seed: u64,
) -> Result<ClusterBaselineConfig, JsValue> {
    let mut cfg = ClusterBaselineConfig::default().with_seed(default_seed);
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
        cfg.seed = opt_seed_prop(options)?;
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

pub(crate) fn parse_eval_options(
    targets: &JsValue,
    options: &JsValue,
) -> Result<ParsedMetricEvalInputs, JsValue> {
    let parsed_targets = parse_metric_targets(targets)?;
    let mut cfg = MetricConfig::default();
    let mut labels = None;
    let mut node_weights = None;
    let mut cluster_weights = None;
    let mut sprawlter_half_widths = None;
    let mut sprawlter_half_heights = None;

    if check_options_arg(options, "evaluateMetrics")? {
        let seed_u64 = opt_seed_prop(options)?;
        cfg.sns_num_pivots = opt_usize_prop(options, "numPivots", cfg.sns_num_pivots)?;
        cfg.sns_seed = seed_u64;
        let samples = opt_usize_prop(options, "numSamples", cfg.crossing_num_samples)?;
        cfg.crossing_num_samples = samples;
        cfg.crossing_seed = seed_u64;
        cfg.np_num_samples = opt_usize_prop(options, "npNumSamples", samples)?;
        cfg.np_seed = seed_u64;
        cfg.ideal_edge_length = opt_f64_prop(options, "idealEdgeLength", cfg.ideal_edge_length)?;
        cfg.rta_num_landmarks = opt_usize_prop(options, "numLandmarks", cfg.rta_num_landmarks)?;
        cfg.rta_num_triplets = opt_usize_prop(options, "numTriplets", cfg.rta_num_triplets)?;
        cfg.rta_min_degree = opt_usize_prop(options, "minDegree", cfg.rta_min_degree)?;
        cfg.rta_seed = seed_u64;
        cfg.patch_nig_radius = opt_usize_prop(options, "patchRadius", cfg.patch_nig_radius)?;
        cfg.patch_nig_canvas_size =
            opt_usize_prop(options, "canvasSize", cfg.patch_nig_canvas_size)?;
        cfg.patch_nig_pad_frac = opt_f64_prop(options, "padFrac", cfg.patch_nig_pad_frac)?;
        cfg.patch_nig_seed = seed_u64;

        let mut ca = ClusterAreaConfig::default();
        ca.g_max = opt_usize_prop(options, "gMax", ca.g_max)?;
        ca.sigma_cells = opt_f64_prop(options, "sigmaCells", ca.sigma_cells)?;
        ca.kappa = opt_f64_prop(options, "kappa", ca.kappa)?;
        ca.tau_frac = opt_f64_prop(options, "tauFrac", ca.tau_frac)?;
        ca.boundary_r95 = opt_f64_prop(options, "boundaryR95", ca.boundary_r95)?;
        ca.stevens_gamma = opt_f64_prop(options, "stevensGamma", ca.stevens_gamma)?;
        cfg.cluster_area = ca;

        let cb_nested = get_prop(options, "clusterBaselines")?;
        let cb_cfg = if check_options_arg(&cb_nested, "evaluateMetrics::clusterBaselines")? {
            parse_cluster_baseline_config(&cb_nested, seed_u64)?
        } else {
            parse_cluster_baseline_config(options, seed_u64)?
        };
        cfg = cfg.with_baseline_config(cb_cfg);

        let labels_val = get_prop(options, "labels")?;
        if !labels_val.is_null() && !labels_val.is_undefined() {
            labels = Some(parse_u32_sequence(&labels_val, "labels")?);
        }
        node_weights =
            parse_optional_f64_sequence(&get_prop(options, "nodeWeights")?, "nodeWeights")?;
        cluster_weights =
            parse_optional_f64_sequence(&get_prop(options, "clusterWeights")?, "clusterWeights")?;
        sprawlter_half_widths = parse_optional_f32_sequence(
            &get_prop(options, "sprawlterHalfWidths")?,
            "sprawlterHalfWidths",
        )?;
        sprawlter_half_heights = parse_optional_f32_sequence(
            &get_prop(options, "sprawlterHalfHeights")?,
            "sprawlterHalfHeights",
        )?;
    } else {
        let cb_cfg = ClusterBaselineConfig::default().with_seed(DEFAULT_SEED);
        cfg = cfg.with_baseline_config(cb_cfg);
    }

    Ok(ParsedMetricEvalInputs {
        targets: parsed_targets,
        config: cfg,
        labels,
        node_weights,
        cluster_weights,
        sprawlter_half_widths,
        sprawlter_half_heights,
    })
}
