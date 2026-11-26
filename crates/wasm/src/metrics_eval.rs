//! Selective quality metric dispatcher (`evaluateMetrics`) for WebAssembly.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_f64_sequence, parse_seed_u64,
    parse_u32_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, is_plain_js_object, opt_f64_prop, opt_maybe_f64_prop,
    opt_usize_prop, reject_extra_args_with_options,
};
use crate::layout_buffers::JsLayout2d;
use crate::metrics_cluster_rta::parse_cluster_area_config;
use crate::metrics_functions::parse_metric_targets;
use crate::metrics_report::JsMetricReport;
use egraph::metrics::{
    ClusterBaselineConfig, MetricConfig, evaluate_metrics_with_community_weights,
};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 64;
const DEFAULT_NUM_SAMPLES: usize = 4096;
const DEFAULT_UNIT_EDGE_LENGTH: f64 = 1.0;
const DEFAULT_RTA_NUM_LANDMARKS: usize = 64;
const DEFAULT_RTA_NUM_TRIPLETS: usize = 1_000_000;
const DEFAULT_RTA_MIN_DEGREE: usize = 2;
const DEFAULT_PATCH_NIG_RADIUS: usize = 8;
const DEFAULT_PATCH_NIG_CANVAS_SIZE: usize = 2048;
const DEFAULT_PATCH_NIG_PAD_FRAC: f64 = 0.04;

fn parse_cluster_baseline_config(
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
    cfg.kmeans_max_iters = opt_usize_prop(options, "kmeansMaxIters", cfg.kmeans_max_iters)?;
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
    Ok(JsMetricReport::from_inner(report))
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
    if is_plain_js_object(num_pivots_or_options) {
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
        reject_extra_args_with_options(has_extra)?;

        let opts = num_pivots_or_options;
        let opt_num_pivots = opt_maybe_f64_prop(opts, "numPivots")?;
        let opt_num_samples = opt_maybe_f64_prop(opts, "numSamples")?;
        let opt_seed = opt_maybe_f64_prop(opts, "seed")?;
        let opt_seed_u64 = parse_seed_u64(opt_seed)?;
        let opt_ideal_len = opt_maybe_f64_prop(opts, "idealEdgeLength")?;
        let labels_val = get_prop(opts, "communityLabels")?;
        let opt_labels = if labels_val.is_null() || labels_val.is_undefined() {
            None
        } else {
            Some(parse_u32_sequence(&labels_val, "communityLabels")?)
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
        let cb_nested = get_prop(opts, "clusterBaselines")?;
        let cb_cfg = if check_options_arg(&cb_nested, "evaluateMetrics::clusterBaselines")? {
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
