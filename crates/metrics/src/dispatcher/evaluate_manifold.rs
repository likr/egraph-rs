use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldKind, ManifoldLayout2d, Scalar};
use egraph_metrics_geometry::{
    crossings::crossing_edges_torus,
    geometry::{ideal_edge_lengths_scaled_manifold, layout_degeneracy_manifold},
    neighborhood::{
        neighborhood_preservation_1hop_manifold, neighborhood_preservation_2hop_manifold,
        sampled_neighborhood_preservation_1hop_manifold,
        sampled_neighborhood_preservation_2hop_manifold,
    },
    proximity::{
        gabriel_graph_preservation_manifold, gabriel_graph_property_manifold,
        node_resolution_manifold, relative_neighborhood_preservation_manifold,
    },
    rta::landmark_stratified_rta_manifold,
    stress::{
        geodesic_distortion, sampled_scale_normalized_stress_manifold,
        scale_normalized_stress_manifold,
    },
    validate_graph_manifold_layout,
};

#[cfg(feature = "eval-baselines")]
use super::baselines_manifold::evaluate_eval_baseline_targets_manifold;
use super::config::{MetricConfig, MetricInputs};
use super::report::{MetricReport, MetricTarget, RequestedFlags};

pub(super) fn evaluate_metrics_manifold_with_inputs<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
) -> CoreResult<MetricReport> {
    validate_graph_manifold_layout(graph, layout, "evaluate_metrics")?;
    let flags = RequestedFlags::from_targets(targets);
    if let Some(target_name) = first_unsupported_manifold_target(&flags, layout.kind()) {
        return Err(CoreError::unsupported(format!(
            "MetricTarget::{target_name} is not supported on {:?} manifold layouts",
            layout.kind()
        )));
    }

    let mut report = MetricReport::default();

    if flags.sns {
        report.scale_normalized_stress = Some(scale_normalized_stress_manifold(graph, layout)?);
    }
    if flags.sampled_sns {
        report.sampled_scale_normalized_stress = Some(sampled_scale_normalized_stress_manifold(
            graph,
            layout,
            config.sns_num_pivots,
            config.sns_seed,
        )?);
    }
    if flags.geodesic_distortion {
        report.geodesic_distortion = Some(geodesic_distortion(graph, layout)?);
    }

    if flags.torus_crossing_number || flags.crossing_edges_torus {
        let pairs = crossing_edges_torus(graph, layout)?;
        if flags.torus_crossing_number {
            report.torus_crossing_number = Some(pairs.len());
        }
        if flags.crossing_edges_torus {
            report.crossing_edges_torus = Some(pairs);
        }
    }

    if (flags.sampled_np1 || flags.sampled_np2) && config.crossing_num_samples == 0 {
        return Err(CoreError::invalid_input(
            "crossing_num_samples must be >= 1",
        ));
    }

    if flags.np1 {
        report.neighborhood_preservation_1hop =
            Some(neighborhood_preservation_1hop_manifold(graph, layout)?);
    }
    if flags.np2 {
        report.neighborhood_preservation_2hop =
            Some(neighborhood_preservation_2hop_manifold(graph, layout)?);
    }
    if flags.sampled_np1 {
        report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop_manifold(
            graph,
            layout,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.sampled_np2 {
        report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop_manifold(
            graph,
            layout,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.gabriel_property {
        report.gabriel_graph_property = Some(gabriel_graph_property_manifold(graph, layout)?);
    }
    if flags.node_resolution {
        report.node_resolution = Some(node_resolution_manifold(layout)?);
    }
    if flags.s_rng {
        report.relative_neighborhood_preservation =
            Some(relative_neighborhood_preservation_manifold(graph, layout)?);
    }
    if flags.s_gg {
        report.gabriel_graph_preservation =
            Some(gabriel_graph_preservation_manifold(graph, layout)?);
    }
    if flags.layout_degeneracy {
        let deg = layout_degeneracy_manifold(graph, layout)?;
        let bbox_penalty = if deg.is_bbox_degenerate && layout.len() > 1 {
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
    if flags.ideal_edge_lengths {
        if !config.ideal_edge_length.is_finite() {
            return Err(CoreError::non_finite("MetricConfig::ideal_edge_length"));
        }
        if config.ideal_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(
                "ideal_edge_length must be strictly positive",
            ));
        }
        report.ideal_edge_lengths = Some(ideal_edge_lengths_scaled_manifold(
            graph,
            layout,
            config.ideal_edge_length,
        )?);
    }
    if flags.landmark_stratified_rta {
        report.landmark_stratified_rta = Some(landmark_stratified_rta_manifold(
            graph,
            layout,
            config.rta_num_landmarks,
            config.rta_num_triplets,
            config.rta_min_degree,
            config.rta_seed,
        )?);
    }

    #[cfg(feature = "eval-baselines")]
    evaluate_eval_baseline_targets_manifold(layout, &flags, config, inputs, &mut report)?;
    #[cfg(not(feature = "eval-baselines"))]
    let _ = inputs;

    Ok(report)
}

fn first_unsupported_manifold_target(
    flags: &RequestedFlags,
    kind: ManifoldKind,
) -> Option<&'static str> {
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
    if kind != ManifoldKind::Torus2d {
        if flags.torus_crossing_number {
            return Some("TorusCrossingNumber");
        }
        if flags.crossing_edges_torus {
            return Some("CrossingEdgesTorus");
        }
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
