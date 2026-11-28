use std::any::Any;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, MetricSpace, Scalar};
use egraph_metrics_cluster::{adjusted_patch_nig, cluster_area_faithfulness_with_weights};
use egraph_metrics_geometry::{
    angular_resolution, aspect_ratio, crossing_angle_from_pairs, crossing_edges,
    crossing_edges_torus, crossing_number, crossing_number_from_pairs, crossing_summary,
    crosslessness_from_count, geodesic_distortion, ideal_edge_lengths_scaled,
    ideal_edge_lengths_with_unit, landmark_stratified_rta, minimum_angle_summary, prism_distortion,
    sampled_crossing_number, sampled_scale_normalized_stress, scale_normalized_stress,
};
use egraph_spatial_trees::NearestNeighborIndex;

#[cfg(feature = "eval-baselines")]
use super::baselines::evaluate_eval_baseline_targets;
use super::config::MetricConfig;
use super::inputs::MetricInputs;
use super::report::{MetricReport, MetricTarget, RequestedFlags};
use super::spatial::{
    SpaceClass, evaluate_spatial_and_degeneracy_targets, geodesic_distortion_unweighted,
    try_build_spatial_index,
};

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Works uniformly across [`Layout2d`], [`egraph_core::Layout3d`], [`egraph_core::LayoutNd`],
/// [`egraph_core::ManifoldLayout2d`], and arbitrary external [`MetricSpace`] implementations.
/// For Euclidean layouts, shares a single [`NearestNeighborIndex`] across requested
/// spatial, neighborhood, proximity, degeneracy, and separation targets; for 2D Euclidean layouts,
/// shares a single sweep-line pass across crossing targets, a single sampling pass across
/// sampled crossing targets, and a single angular scan across minimum-angle targets.
/// Returns [`CoreError::UnsupportedOperation`] when a target is inherently restricted to a
/// narrower geometry (such as 2D-only crossing/area metrics, `Torus2d` crossings, or
/// `ManifoldLayout2d` geodesic distortion).
pub fn evaluate_metrics<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs::<L>::new();
    evaluate_metrics_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the metrics listed in `targets` with optional `labels`
/// (required when [`MetricTarget::AdjustedPatchNig`], [`MetricTarget::ClusterAreaFaithfulness`],
/// or labeled `eval-baselines` targets are requested).
pub fn evaluate_metrics_with_communities<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
    labels: Option<&[u32]>,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs::<L>::new().with_labels(labels);
    evaluate_metrics_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the metrics listed in `targets` with optional `labels` and
/// optional positive finite G-CAF target weights (`node_weights` or `cluster_weights`,
/// mutually exclusive, evaluated lazily when [`MetricTarget::ClusterAreaFaithfulness`] is requested).
pub fn evaluate_metrics_with_community_weights<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
    labels: Option<&[u32]>,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs::<L>::new()
        .with_labels(labels)
        .with_node_weights(node_weights)
        .with_cluster_weights(cluster_weights);
    evaluate_metrics_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the metrics listed in `targets` using optional auxiliary [`MetricInputs`]
/// (`labels`, `node_weights`, `cluster_weights`, `original_layout`, `sprawlter_extents`),
/// validating inputs lazily per requested target.
pub fn evaluate_metrics_with_inputs<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
    inputs: &MetricInputs<'_, L>,
) -> CoreResult<MetricReport> {
    if graph.num_nodes() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            graph.num_nodes(),
            layout.len(),
            "evaluate_metrics",
        ));
    }
    layout.validate_finite()?;

    let flags = RequestedFlags::from_targets(targets);
    let space = SpaceClass::classify(layout);
    validate_space_support(&flags, &space)?;

    if (flags.sampled_np1 || flags.sampled_np2) && config.np_num_samples == 0 {
        return Err(CoreError::invalid_input("np_num_samples must be >= 1"));
    }

    let mut report = MetricReport::new();

    if flags.sns {
        report.scale_normalized_stress = Some(scale_normalized_stress(graph, layout)?);
    }
    if flags.sampled_sns {
        report.sampled_scale_normalized_stress = Some(sampled_scale_normalized_stress(
            graph,
            layout,
            config.sns_num_pivots,
            config.sns_seed,
        )?);
    }
    if flags.geodesic_distortion {
        report.geodesic_distortion = Some(eval_geodesic_distortion(graph, layout)?);
    }

    if let SpaceClass::Euclidean2d(l2) = &space {
        evaluate_2d_crossings_and_geometry(graph, *l2, &flags, config, inputs, &mut report)?;
    }

    if let SpaceClass::Manifold(ml) = &space
        && (flags.torus_crossing_number || flags.crossing_edges_torus)
    {
        let pairs = crossing_edges_torus(graph, *ml)?;
        if flags.torus_crossing_number {
            report.torus_crossing_number = Some(pairs.len());
        }
        if flags.crossing_edges_torus {
            report.crossing_edges_torus = Some(pairs);
        }
    }

    let spatial_index: Option<Box<dyn NearestNeighborIndex>> = if flags.needs_spatial_index() {
        try_build_spatial_index(&space)?
    } else {
        None
    };

    evaluate_spatial_and_degeneracy_targets(
        graph,
        layout,
        &space,
        spatial_index.as_deref(),
        &flags,
        config,
        &mut report,
    )?;

    if flags.angular_resolution {
        report.angular_resolution = Some(eval_angular_resolution(graph, &space)?);
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
        report.ideal_edge_lengths = Some(eval_ideal_edge_lengths(
            graph,
            layout,
            config.ideal_edge_length,
        )?);
    }
    if flags.landmark_stratified_rta {
        report.landmark_stratified_rta = Some(landmark_stratified_rta(
            graph,
            layout,
            config.rta_num_landmarks,
            config.rta_num_triplets,
            config.rta_min_degree,
            config.rta_seed,
        )?);
    }

    #[cfg(feature = "eval-baselines")]
    evaluate_eval_baseline_targets(
        graph,
        layout,
        &space,
        spatial_index.as_deref(),
        &flags,
        config,
        inputs,
        &mut report,
    )?;
    #[cfg(not(feature = "eval-baselines"))]
    let _ = inputs;

    Ok(report)
}

fn validate_space_support<S: Scalar>(
    flags: &RequestedFlags,
    space: &SpaceClass<'_, S>,
) -> CoreResult<()> {
    if flags.geodesic_distortion && !matches!(space, SpaceClass::Manifold(_)) {
        return Err(CoreError::unsupported(
            "MetricTarget::GeodesicDistortion is only supported on ManifoldLayout2d",
        ));
    }

    if flags.torus_crossing_number || flags.crossing_edges_torus {
        let is_torus =
            matches!(space, SpaceClass::Manifold(ml) if ml.kind() == ManifoldKind::Torus2d);
        if !is_torus {
            let name = if flags.torus_crossing_number {
                "TorusCrossingNumber"
            } else {
                "CrossingEdgesTorus"
            };
            return Err(CoreError::unsupported(format!(
                "MetricTarget::{name} is only supported on Torus2d ManifoldLayout2d"
            )));
        }
    }

    if flags.angular_resolution && matches!(space, SpaceClass::ExternalMetric) {
        return Err(CoreError::unsupported(
            "MetricTarget::AngularResolution requires a RiemannianLayout",
        ));
    }

    if !matches!(space, SpaceClass::Euclidean2d(_))
        && let Some(target_name) = first_2d_only_target(flags)
    {
        return Err(CoreError::unsupported(format!(
            "MetricTarget::{target_name} is only supported on 2D Layout2d"
        )));
    }

    Ok(())
}

fn first_2d_only_target(flags: &RequestedFlags) -> Option<&'static str> {
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
    if flags.aspect_ratio {
        return Some("AspectRatio");
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

fn evaluate_2d_crossings_and_geometry<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<L::Scalar>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    inputs: &MetricInputs<'_, L>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.crossing_edges {
        let pairs = crossing_edges(graph, layout)?;
        if flags.crossing_number {
            report.crossing_number = Some(crossing_number_from_pairs(&pairs));
        }
        if flags.crossing_angle {
            report.crossing_angle = Some(crossing_angle_from_pairs(&pairs));
        }
        if flags.crosslessness {
            report.crosslessness = Some(crosslessness_from_count(graph, pairs.len() as f64)?);
        }
        report.crossing_edges = Some(pairs);
    } else if flags.crossing_angle {
        let summary = crossing_summary(graph, layout)?;
        report.crossing_angle = Some(summary.angle_cos_sq_sum);
        if flags.crossing_number {
            report.crossing_number = Some(summary.count);
        }
        if flags.crosslessness {
            report.crosslessness = Some(crosslessness_from_count(graph, summary.count as f64)?);
        }
    } else if flags.crossing_number || flags.crosslessness {
        let count = crossing_number(graph, layout)?;
        if flags.crossing_number {
            report.crossing_number = Some(count);
        }
        if flags.crosslessness {
            report.crosslessness = Some(crosslessness_from_count(graph, count as f64)?);
        }
    }

    if flags.sampled_crossing_number || flags.sampled_crosslessness {
        let est = sampled_crossing_number(
            graph,
            layout,
            config.crossing_num_samples,
            config.crossing_seed,
        )?;
        if flags.sampled_crossing_number {
            report.sampled_crossing_number = Some(est);
        }
        if flags.sampled_crosslessness {
            report.sampled_crosslessness = Some(crosslessness_from_count(graph, est)?);
        }
    }

    if flags.aspect_ratio {
        report.aspect_ratio = Some(aspect_ratio(layout)?);
    }
    if flags.minimum_angle || flags.minimum_angle_a1 {
        let summary = minimum_angle_summary(graph, layout)?;
        if flags.minimum_angle {
            report.minimum_angle = Some(summary.a2);
        }
        if flags.minimum_angle_a1 {
            report.minimum_angle_a1 = Some(summary.a1);
        }
    }
    if flags.prism_distortion {
        let orig = inputs
            .original_layout
            .and_then(|o| (o as &dyn Any).downcast_ref::<Layout2d<L::Scalar>>())
            .ok_or_else(|| {
                CoreError::invalid_input("PrismDistortion target requires original_layout")
            })?;
        let pd = prism_distortion(orig, layout)?;
        report.prism_sigma_dist = Some(pd.sigma_dist);
        report.prism_sigma_disp = Some(pd.sigma_disp);
    }
    if flags.adjusted_patch_nig {
        let labels = inputs
            .labels
            .ok_or_else(|| CoreError::invalid_input("AdjustedPatchNig target requires labels"))?;
        report.adjusted_patch_nig = Some(adjusted_patch_nig(
            layout,
            labels,
            config.patch_nig_radius,
            config.patch_nig_canvas_size,
            config.patch_nig_pad_frac,
            config.patch_nig_seed,
        )?);
    }
    if flags.cluster_area_faithfulness {
        let labels = inputs.labels.ok_or_else(|| {
            CoreError::invalid_input("ClusterAreaFaithfulness target requires labels")
        })?;
        report.cluster_area_faithfulness = Some(
            cluster_area_faithfulness_with_weights(
                layout,
                labels,
                &config.cluster_area,
                inputs.node_weights,
                inputs.cluster_weights,
            )?
            .caf,
        );
    }

    Ok(())
}

fn eval_angular_resolution<W, S: Scalar>(
    graph: &CsrGraph<W>,
    space: &SpaceClass<'_, S>,
) -> CoreResult<f64> {
    match space {
        SpaceClass::Euclidean2d(l) => angular_resolution(graph, *l),
        SpaceClass::Euclidean3d(l) => angular_resolution(graph, *l),
        SpaceClass::EuclideanNd(l) => angular_resolution(graph, *l),
        SpaceClass::Manifold(l) => angular_resolution(graph, *l),
        SpaceClass::ExternalMetric => Err(CoreError::unsupported(
            "MetricTarget::AngularResolution requires a RiemannianLayout",
        )),
    }
}

fn eval_geodesic_distortion<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    let any_g: &dyn Any = graph;
    if let Some(g64) = any_g.downcast_ref::<CsrGraph<f64>>() {
        return geodesic_distortion(g64, layout);
    }
    if let Some(g32) = any_g.downcast_ref::<CsrGraph<f32>>() {
        return geodesic_distortion(g32, layout);
    }
    geodesic_distortion_unweighted(graph, layout)
}

fn eval_ideal_edge_lengths<W: 'static, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    let any_g: &dyn Any = graph;
    if let Some(g64) = any_g.downcast_ref::<CsrGraph<f64>>() {
        return ideal_edge_lengths_scaled(g64, layout, unit_edge_length);
    }
    if let Some(g32) = any_g.downcast_ref::<CsrGraph<f32>>() {
        return ideal_edge_lengths_scaled(g32, layout, unit_edge_length);
    }
    if graph.is_weighted() {
        return Err(CoreError::unsupported(
            "IdealEdgeLengths on weighted graphs requires f32 or f64 edge weights",
        ));
    }
    ideal_edge_lengths_with_unit(graph, layout, unit_edge_length)
}
