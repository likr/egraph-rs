use std::fs;
use std::path::Path;

use egraph::prelude::{CoreError, MetricTarget};

use crate::io::{CliError, CliResult};

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

pub(crate) fn push_unique_target(targets: &mut Vec<MetricTarget>, target: MetricTarget) {
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
            "geodesic-distortion" => {
                push_unique_target(&mut targets, MetricTarget::GeodesicDistortion);
            }
            "crossings" | "crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::CrossingNumber);
            }
            "crossing-edges" => {
                push_unique_target(&mut targets, MetricTarget::CrossingEdges);
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
            "torus-crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::TorusCrossingNumber);
            }
            "crossing-edges-torus" => {
                push_unique_target(&mut targets, MetricTarget::CrossingEdgesTorus);
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
            "prism-distortion" => {
                push_unique_target(&mut targets, MetricTarget::PrismDistortion);
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
            "nor-metrics" | "nor" => {
                push_unique_target(&mut targets, MetricTarget::NorMetrics);
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

/// Loads cluster/community labels from a JSON file (`{"labels": [...]}` / `{"community_labels": [...]}` / `[...]`)
/// or a plain-text/CSV integer list, validating that all entries are `u32` integers and that
/// the label count matches `expected_nodes`.
pub fn load_labels_from_path(path: &Path, expected_nodes: usize) -> CliResult<Vec<u32>> {
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
                "unrecognized labels JSON schema in '{}'",
                path.display()
            )));
        };

        let mut parsed = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let raw = item.as_u64().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "label at index {idx} in '{}' must be a non-negative integer",
                    path.display()
                ))
            })?;
            let lbl = u32::try_from(raw).map_err(|_| {
                CliError::InvalidInput(format!(
                    "label {raw} at index {idx} in '{}' exceeds u32::MAX",
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
                        "invalid label '{token}' on line {} of '{}'",
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
            "load_labels_from_path",
        )));
    }
    Ok(labels)
}

/// Loads a per-node positive finite `f64` series from a JSON array or plain-text/CSV list.
pub(crate) fn load_positive_f64_series_from_path(
    path: &Path,
    expected_nodes: usize,
    field_name: &str,
) -> CliResult<Vec<f64>> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        || trimmed.starts_with('[')
        || trimmed.starts_with('{');

    let values = if is_json {
        let val: serde_json::Value = serde_json::from_str(trimmed)?;
        let items = val
            .as_array()
            .or_else(|| val.get(field_name).and_then(|v| v.as_array()))
            .or_else(|| val.get("values").and_then(|v| v.as_array()))
            .ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "expected JSON array for {field_name} in '{}'",
                    path.display()
                ))
            })?;
        let mut parsed = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let v = item.as_f64().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "{field_name} entry at index {idx} in '{}' must be a finite number",
                    path.display()
                ))
            })?;
            if !v.is_finite() || v <= 0.0 {
                return Err(CliError::InvalidInput(format!(
                    "{field_name} entry at index {idx} in '{}' must be finite and > 0.0, got {v}",
                    path.display()
                )));
            }
            parsed.push(v);
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
                let v = token.parse::<f64>().map_err(|_| {
                    CliError::InvalidInput(format!(
                        "invalid {field_name} value '{token}' on line {} of '{}'",
                        line_idx + 1,
                        path.display()
                    ))
                })?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(CliError::InvalidInput(format!(
                        "{field_name} value on line {} of '{}' must be finite and > 0.0, got {v}",
                        line_idx + 1,
                        path.display()
                    )));
                }
                parsed.push(v);
            }
        }
        parsed
    };

    if values.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            values.len(),
            "load_positive_f64_series_from_path",
        )));
    }
    Ok(values)
}
