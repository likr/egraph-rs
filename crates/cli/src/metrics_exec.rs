use std::fs;
use std::path::Path;
use std::time::Instant;

use egraph::prelude::{
    ClusterAreaConfig, ClusterBaselineConfig, CoreError, MetricConfig, MetricTarget,
    cluster_area_faithfulness, cluster_area_faithfulness_detail, evaluate_metrics_with_communities,
};

use crate::args::MetricsArgs;
use crate::io::{CliError, CliResult, load_layout_from_path, resolve_graph};
use crate::output::MetricsOutput;

const ALL_STANDARD_METRIC_TARGETS: [MetricTarget; 22] = [
    MetricTarget::ScaleNormalizedStress,
    MetricTarget::SampledScaleNormalizedStress,
    MetricTarget::CrossingNumber,
    MetricTarget::CrossingAngle,
    MetricTarget::SampledCrossingNumber,
    MetricTarget::Crosslessness,
    MetricTarget::SampledCrosslessness,
    MetricTarget::NeighborhoodPreservation1Hop,
    MetricTarget::NeighborhoodPreservation2Hop,
    MetricTarget::SampledNp1,
    MetricTarget::SampledNp2,
    MetricTarget::GabrielGraphProperty,
    MetricTarget::NodeResolution,
    MetricTarget::RelativeNeighborhoodPreservation,
    MetricTarget::GabrielGraphPreservation,
    MetricTarget::AspectRatio,
    MetricTarget::AngularResolution,
    MetricTarget::MinimumAngle,
    MetricTarget::MinimumAngleA1,
    MetricTarget::LayoutDegeneracy,
    MetricTarget::IdealEdgeLengths,
    MetricTarget::LandmarkStratifiedRta,
];

fn push_unique_target(targets: &mut Vec<MetricTarget>, target: MetricTarget) {
    if !targets.contains(&target) {
        targets.push(target);
    }
}

/// Parses comma-separated metric target names into [`MetricTarget`] values.
pub fn parse_metric_targets(spec: &str) -> CliResult<Vec<MetricTarget>> {
    let mut targets = Vec::new();
    for raw in spec.split(',') {
        let token = raw.trim().to_ascii_lowercase().replace('_', "-");
        if token.is_empty() {
            continue;
        }
        match token.as_str() {
            "all" => {
                // Expand `all` in place instead of returning early so combinations
                // like `all,gcaf` and trailing invalid tokens are both processed.
                for &target in &ALL_STANDARD_METRIC_TARGETS {
                    push_unique_target(&mut targets, target);
                }
            }
            "sns" | "scale-normalized-stress" => {
                push_unique_target(&mut targets, MetricTarget::ScaleNormalizedStress);
            }
            "sampled-sns" | "sampled-scale-normalized-stress" => {
                push_unique_target(&mut targets, MetricTarget::SampledScaleNormalizedStress);
            }
            "crossings" | "crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::CrossingNumber);
            }
            "crossing-angle" => {
                push_unique_target(&mut targets, MetricTarget::CrossingAngle);
            }
            "sampled-crossings" | "sampled-crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::SampledCrossingNumber);
            }
            "crosslessness" => {
                push_unique_target(&mut targets, MetricTarget::Crosslessness);
            }
            "sampled-crosslessness" => {
                push_unique_target(&mut targets, MetricTarget::SampledCrosslessness);
            }
            "np1" | "neighborhood-preservation-1hop" => {
                push_unique_target(&mut targets, MetricTarget::NeighborhoodPreservation1Hop);
            }
            "np2" | "neighborhood-preservation-2hop" => {
                push_unique_target(&mut targets, MetricTarget::NeighborhoodPreservation2Hop);
            }
            "sampled-np1" | "sampled-neighborhood-preservation-1hop" => {
                push_unique_target(&mut targets, MetricTarget::SampledNp1);
            }
            "sampled-np2" | "sampled-neighborhood-preservation-2hop" => {
                push_unique_target(&mut targets, MetricTarget::SampledNp2);
            }
            "gabriel" | "gabriel-graph-property" => {
                push_unique_target(&mut targets, MetricTarget::GabrielGraphProperty);
            }
            "node-resolution" => {
                push_unique_target(&mut targets, MetricTarget::NodeResolution);
            }
            // Eades et al. (2017) unweighted shape-based S_RNG / S_GG (distinct from
            // deferred Hong et al. 2022 dRNG/dGG and Fang et al. 2025 dcRNG/dcGG);
            // keep the legacy `drng`/`dgg` CLI target tokens intact for compatibility.
            "drng" | "relative-neighborhood-preservation" => {
                push_unique_target(&mut targets, MetricTarget::RelativeNeighborhoodPreservation);
            }
            "dgg" | "gabriel-graph-preservation" => {
                push_unique_target(&mut targets, MetricTarget::GabrielGraphPreservation);
            }
            "aspect-ratio" => {
                push_unique_target(&mut targets, MetricTarget::AspectRatio);
            }
            "angular-resolution" => {
                push_unique_target(&mut targets, MetricTarget::AngularResolution);
            }
            "minimum-angle" | "min-angle" | "minimum-angle-a2" | "min-angle-a2" => {
                push_unique_target(&mut targets, MetricTarget::MinimumAngle);
            }
            "minimum-angle-a1" | "min-angle-a1" => {
                push_unique_target(&mut targets, MetricTarget::MinimumAngleA1);
            }
            "layout-degeneracy" => {
                push_unique_target(&mut targets, MetricTarget::LayoutDegeneracy);
            }
            "ideal-edge-lengths" => {
                push_unique_target(&mut targets, MetricTarget::IdealEdgeLengths);
            }
            "adjusted-patch-nig" | "patch-nig" => {
                push_unique_target(&mut targets, MetricTarget::AdjustedPatchNig);
            }
            "cluster-area-faithfulness" | "cluster-area" | "gcaf" | "g-caf" => {
                push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
            }
            "rta" | "l-stratified-rta" | "landmark-stratified-rta" => {
                push_unique_target(&mut targets, MetricTarget::LandmarkStratifiedRta);
            }
            "dsc" | "distance-consistency" => {
                push_unique_target(&mut targets, MetricTarget::DistanceConsistency);
            }
            "dc" | "distribution-consistency" => {
                push_unique_target(&mut targets, MetricTarget::DistributionConsistency);
            }
            "cdm" | "class-density-measure" => {
                push_unique_target(&mut targets, MetricTarget::ClassDensityMeasure);
            }
            "hdm" | "2d-hdm" | "histogram-density-measure" => {
                push_unique_target(&mut targets, MetricTarget::HistogramDensityMeasure);
            }
            "sepme-knng" | "sepme-knng2" => {
                push_unique_target(&mut targets, MetricTarget::SepmeKnng);
            }
            "sepme-rng" => {
                push_unique_target(&mut targets, MetricTarget::SepmeRng);
            }
            "sepme-gab" => {
                push_unique_target(&mut targets, MetricTarget::SepmeGab);
            }
            "sepme-gong" => {
                push_unique_target(&mut targets, MetricTarget::SepmeGong);
            }
            "neighborhood-hit" | "m-nh" | "nh" => {
                push_unique_target(&mut targets, MetricTarget::NeighborhoodHit);
            }
            "silhouette" | "silhouette-score" => {
                push_unique_target(&mut targets, MetricTarget::SilhouetteScore);
            }
            "calinski-harabasz" | "calinski-harabasz-score" | "ch" => {
                push_unique_target(&mut targets, MetricTarget::CalinskiHarabaszScore);
            }
            "davies-bouldin" | "davies-bouldin-score" | "db" => {
                push_unique_target(&mut targets, MetricTarget::DaviesBouldinScore);
            }
            "cq-kmeans-ari" => {
                push_unique_target(&mut targets, MetricTarget::CqKmeansAri);
            }
            "cq-kmeans-ami" => {
                push_unique_target(&mut targets, MetricTarget::CqKmeansAmi);
            }
            "cq-hac-hom" => {
                push_unique_target(&mut targets, MetricTarget::CqHacHom);
            }
            "cq-hac-ami" => {
                push_unique_target(&mut targets, MetricTarget::CqHacAmi);
            }
            "pairwise-distance-overlap-error" | "pdoe" => {
                push_unique_target(&mut targets, MetricTarget::PairwiseDistanceOverlapError);
            }
            "sprawlter-nn" | "sprawlter" => {
                push_unique_target(&mut targets, MetricTarget::SprawlterNn);
            }
            other => {
                return Err(CliError::InvalidInput(format!(
                    "unknown metric target '{other}'"
                )));
            }
        }
    }
    Ok(targets)
}

/// Loads community labels from a JSON file (`{"labels": [...]}` / `{"community_labels": [...]}` / `[...]`)
/// or a plain-text/CSV integer list, validating that all entries are `u32` integers and that
/// the label count matches `expected_nodes`.
pub fn load_community_labels_from_path(path: &Path, expected_nodes: usize) -> CliResult<Vec<u32>> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        || trimmed.starts_with('{')
        || trimmed.starts_with('[');

    let labels = if is_json {
        let val: serde_json::Value = serde_json::from_str(trimmed)?;
        let items = if let Some(arr) = val.as_array() {
            arr
        } else if let Some(field) = val.get("labels") {
            field.as_array().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "'labels' field in '{}' must be a JSON array of integers",
                    path.display()
                ))
            })?
        } else if let Some(field) = val.get("community_labels") {
            field.as_array().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "'community_labels' field in '{}' must be a JSON array of integers",
                    path.display()
                ))
            })?
        } else {
            return Err(CliError::InvalidInput(format!(
                "unrecognized community labels JSON schema in '{}'",
                path.display()
            )));
        };

        let mut parsed = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let raw = item.as_u64().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "community label at index {idx} in '{}' must be a non-negative integer",
                    path.display()
                ))
            })?;
            let lbl = u32::try_from(raw).map_err(|_| {
                CliError::InvalidInput(format!(
                    "community label {raw} at index {idx} in '{}' exceeds u32::MAX",
                    path.display()
                ))
            })?;
            parsed.push(lbl);
        }
        parsed
    } else {
        let mut parsed = Vec::new();
        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            for token in line
                .split(|c: char| c.is_ascii_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
            {
                let lbl = token.parse::<u32>().map_err(|_| {
                    CliError::InvalidInput(format!(
                        "invalid community label '{token}' on line {} of '{}'",
                        line_idx + 1,
                        path.display()
                    ))
                })?;
                parsed.push(lbl);
            }
        }
        parsed
    };

    if labels.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            labels.len(),
            "load_community_labels_from_path",
        )));
    }
    Ok(labels)
}

/// Executes the `metrics` subcommand and returns the structured JSON output.
pub fn execute_metrics(args: &MetricsArgs) -> CliResult<MetricsOutput> {
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;
    let layout = load_layout_from_path(&args.layout, graph.num_nodes())?;
    let mut targets = parse_metric_targets(&args.targets)?;
    let community_labels = args
        .community_labels
        .as_deref()
        .map(|path| load_community_labels_from_path(path, graph.num_nodes()))
        .transpose()?;
    let has_all = args
        .targets
        .split(',')
        .map(|t| t.trim().to_ascii_lowercase().replace('_', "-"))
        .any(|t| t == "all");
    if has_all && community_labels.is_some() {
        push_unique_target(&mut targets, MetricTarget::AdjustedPatchNig);
        push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
    }
    if args.gcaf_detail {
        push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
    }
    let cfg = MetricConfig {
        sns_num_pivots: args.num_pivots,
        sns_seed: args.seed,
        crossing_num_samples: args.num_samples,
        crossing_seed: args.seed,
        ideal_edge_length: args.ideal_edge_length,
        cluster_area: ClusterAreaConfig {
            g_max: args.gcaf_g_max,
            sigma_cells: args.gcaf_sigma_cells,
            kappa: args.gcaf_kappa,
            tau_frac: args.gcaf_tau_frac,
            boundary_r95: args.gcaf_boundary_r95,
            stevens_gamma: args.gcaf_stevens_gamma,
        },
        cluster_baselines: ClusterBaselineConfig {
            seed: args.seed,
            ..ClusterBaselineConfig::default()
        },
        ..MetricConfig::default()
    };
    cfg.validate()?;

    let t0 = Instant::now();
    let want_gcaf = targets.contains(&MetricTarget::ClusterAreaFaithfulness);
    let non_gcaf_targets: Vec<MetricTarget> = targets
        .iter()
        .copied()
        .filter(|t| *t != MetricTarget::ClusterAreaFaithfulness)
        .collect();
    let report = evaluate_metrics_with_communities(
        &graph,
        &layout,
        &non_gcaf_targets,
        &cfg,
        community_labels.as_deref(),
    )?;
    let (gcaf_report, gcaf_detail) = if want_gcaf {
        let labels = community_labels.as_deref().ok_or_else(|| {
            CoreError::invalid_input("ClusterAreaFaithfulness target requires community_labels")
        })?;
        if args.gcaf_detail {
            let det = cluster_area_faithfulness_detail(&layout, labels, &cfg.cluster_area)?;
            (Some(det.report), Some(det))
        } else {
            let rep = cluster_area_faithfulness(&layout, labels, &cfg.cluster_area)?;
            (Some(rep), None)
        }
    } else {
        (None, None)
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(MetricsOutput::from_report(
        source,
        graph.num_nodes(),
        graph.num_edges(),
        elapsed_ms,
        report,
        gcaf_report,
        gcaf_detail,
    ))
}
