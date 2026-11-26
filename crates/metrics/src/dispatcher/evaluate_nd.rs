use egraph_core::{CoreError, CoreResult, CsrGraph, LayoutNd, Scalar};
use egraph_metrics_geometry::{
    NdSpatialIndex, build_nd_spatial_index,
    geometry::{ideal_edge_lengths_scaled_nd, layout_degeneracy_nd_with_index},
    neighborhood::{
        neighborhood_preservation_1hop_nd_with_index, neighborhood_preservation_2hop_nd_with_index,
        sampled_neighborhood_preservation_1hop_nd_with_index,
        sampled_neighborhood_preservation_2hop_nd_with_index,
    },
    proximity::{
        gabriel_graph_preservation_nd_with_index, gabriel_graph_property_nd_with_index,
        node_resolution_nd_with_index, relative_neighborhood_preservation_nd_with_index,
    },
    rta::landmark_stratified_rta_nd,
    stress::{sampled_scale_normalized_stress_nd, scale_normalized_stress_nd},
    validate_graph_layout_nd,
};

#[cfg(feature = "eval-baselines")]
use super::baselines_nd::evaluate_eval_baseline_targets_nd;
use super::config::{MetricConfig, MetricInputs};
use super::report::{MetricReport, MetricTarget, RequestedFlags};

/// Evaluates strictly the N-D-supported metrics listed in `targets` on a [`LayoutNd`],
/// skipping all unrequested work.
///
/// Shares a single `NdSpatialIndex` across requested spatial/neighborhood/degeneracy targets.
/// Returns [`CoreError::UnsupportedOperation`] if any explicitly requested target is 2D-only.
pub fn evaluate_metrics_nd<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    evaluate_metrics_nd_with_inputs(graph, layout, targets, config, &MetricInputs::default())
}

/// Evaluates strictly the N-D-supported metrics listed in `targets` on a [`LayoutNd`] with
/// optional `labels` (required when `eval-baselines` cluster/separation targets are requested).
pub fn evaluate_metrics_nd_with_communities<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    labels: Option<&[u32]>,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs {
        labels,
        ..MetricInputs::default()
    };
    evaluate_metrics_nd_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the N-D-supported metrics listed in `targets` on a [`LayoutNd`] with
/// optional `labels` and optional weight slices forwarded via [`MetricInputs`].
pub fn evaluate_metrics_nd_with_community_weights<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    labels: Option<&[u32]>,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs {
        labels,
        node_weights,
        cluster_weights,
        ..MetricInputs::default()
    };
    evaluate_metrics_nd_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the N-D-supported metrics listed in `targets` on a [`LayoutNd`] using
/// optional auxiliary [`MetricInputs`].
///
/// Returns [`CoreError::UnsupportedOperation`] if any requested target is inherently 2D-only.
pub fn evaluate_metrics_nd_with_inputs<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
) -> CoreResult<MetricReport> {
    validate_graph_layout_nd(graph, layout)?;
    let flags = RequestedFlags::from_targets(targets);
    if let Some(target_name) = first_unsupported_nd_target(&flags) {
        return Err(CoreError::unsupported(format!(
            "MetricTarget::{target_name} is only supported on 2D layouts"
        )));
    }

    let mut report = MetricReport::default();

    if flags.sns {
        report.scale_normalized_stress = Some(scale_normalized_stress_nd(graph, layout)?);
    }
    if flags.sampled_sns {
        report.sampled_scale_normalized_stress = Some(sampled_scale_normalized_stress_nd(
            graph,
            layout,
            config.sns_num_pivots,
            config.sns_seed,
        )?);
    }

    if (flags.sampled_np1 || flags.sampled_np2) && config.crossing_num_samples == 0 {
        return Err(CoreError::invalid_input(
            "crossing_num_samples must be >= 1",
        ));
    }

    let nd_index: Option<NdSpatialIndex> = if needs_nd_spatial_index(&flags) {
        Some(build_nd_spatial_index(layout)?)
    } else {
        None
    };

    if let Some(index) = nd_index.as_ref() {
        evaluate_nd_spatial_targets(graph, layout, index, &flags, config, &mut report)?;
    }

    if flags.ideal_edge_lengths {
        if !config.ideal_edge_length.is_finite() {
            return Err(CoreError::non_finite("MetricConfig::ideal_edge_length"));
        }
        if config.ideal_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(
                "ideal_edge_length must be strictly positive",
            ));
        }
        report.ideal_edge_lengths = Some(ideal_edge_lengths_scaled_nd(
            graph,
            layout,
            config.ideal_edge_length,
        )?);
    }

    if flags.landmark_stratified_rta {
        report.landmark_stratified_rta = Some(landmark_stratified_rta_nd(
            graph,
            layout,
            config.rta_num_landmarks,
            config.rta_num_triplets,
            config.rta_min_degree,
            config.rta_seed,
        )?);
    }

    #[cfg(feature = "eval-baselines")]
    evaluate_eval_baseline_targets_nd(layout, &flags, config, inputs, &mut report)?;
    #[cfg(not(feature = "eval-baselines"))]
    let _ = inputs;

    Ok(report)
}

fn first_unsupported_nd_target(flags: &RequestedFlags) -> Option<&'static str> {
    if flags.crossing_number {
        return Some("CrossingNumber");
    }
    if flags.crossing_edges {
        return Some("CrossingEdges");
    }
    if flags.crossing_angle {
        return Some("CrossingAngle");
    }
    if flags.sampled_crossing_number {
        return Some("SampledCrossingNumber");
    }
    if flags.crosslessness {
        return Some("Crosslessness");
    }
    if flags.sampled_crosslessness {
        return Some("SampledCrosslessness");
    }
    if flags.torus_crossing_number {
        return Some("TorusCrossingNumber");
    }
    if flags.crossing_edges_torus {
        return Some("CrossingEdgesTorus");
    }
    if flags.aspect_ratio {
        return Some("AspectRatio");
    }
    if flags.angular_resolution {
        return Some("AngularResolution");
    }
    if flags.minimum_angle {
        return Some("MinimumAngle");
    }
    if flags.minimum_angle_a1 {
        return Some("MinimumAngleA1");
    }
    if flags.prism_distortion {
        return Some("PrismDistortion");
    }
    if flags.adjusted_patch_nig {
        return Some("AdjustedPatchNig");
    }
    if flags.cluster_area_faithfulness {
        return Some("ClusterAreaFaithfulness");
    }
    #[cfg(feature = "eval-baselines")]
    {
        if flags.class_density_measure {
            return Some("ClassDensityMeasure");
        }
        if flags.histogram_density_measure {
            return Some("HistogramDensityMeasure");
        }
        if flags.sprawlter_nn {
            return Some("SprawlterNn");
        }
        if flags.nor_metrics {
            return Some("NorMetrics");
        }
    }
    None
}

#[inline]
fn needs_nd_spatial_index(flags: &RequestedFlags) -> bool {
    flags.np1
        || flags.np2
        || flags.sampled_np1
        || flags.sampled_np2
        || flags.gabriel_property
        || flags.node_resolution
        || flags.s_rng
        || flags.s_gg
        || flags.layout_degeneracy
}

fn evaluate_nd_spatial_targets<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
    flags: &RequestedFlags,
    config: &MetricConfig,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.np1 {
        report.neighborhood_preservation_1hop = Some(neighborhood_preservation_1hop_nd_with_index(
            graph, layout, index,
        )?);
    }
    if flags.np2 {
        report.neighborhood_preservation_2hop = Some(neighborhood_preservation_2hop_nd_with_index(
            graph, layout, index,
        )?);
    }
    if flags.sampled_np1 {
        report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop_nd_with_index(
            graph,
            layout,
            index,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.sampled_np2 {
        report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop_nd_with_index(
            graph,
            layout,
            index,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.gabriel_property {
        report.gabriel_graph_property =
            Some(gabriel_graph_property_nd_with_index(graph, layout, index)?);
    }
    if flags.node_resolution {
        report.node_resolution = Some(node_resolution_nd_with_index(layout, index)?);
    }
    if flags.s_rng {
        report.relative_neighborhood_preservation = Some(
            relative_neighborhood_preservation_nd_with_index(graph, layout, index)?,
        );
    }
    if flags.s_gg {
        report.gabriel_graph_preservation = Some(gabriel_graph_preservation_nd_with_index(
            graph, layout, index,
        )?);
    }
    if flags.layout_degeneracy {
        let deg = layout_degeneracy_nd_with_index(graph, layout, index)?;
        let bbox_penalty = if deg.is_bbox_degenerate && layout.num_nodes() > 1 {
            1.0
        } else {
            0.0
        };
        report.layout_degeneracy = Some(
            deg.duplicate_ratio_f32
                .max(deg.zero_length_edge_ratio)
                .max(bbox_penalty),
        );
    }
    Ok(())
}
