use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, ManifoldLayout2d, Scalar};
use egraph_metrics_cluster::{
    cluster_area::cluster_area_faithfulness_with_weights, community::adjusted_patch_nig,
};
use egraph_metrics_geometry::{
    build_kdtree_f64,
    crossings::{
        crossing_angle_from_pairs, crossing_edges, crossing_edges_torus, crossing_number,
        crossing_number_from_pairs, crossing_summary, crosslessness_from_count,
        sampled_crossing_number,
    },
    geometry::{
        angular_resolution, aspect_ratio, ideal_edge_lengths_scaled, layout_degeneracy,
        minimum_angle_summary, prism_distortion,
    },
    neighborhood::{
        neighborhood_preservation_1hop_with_tree, neighborhood_preservation_2hop_with_tree,
        sampled_neighborhood_preservation_1hop_with_tree,
        sampled_neighborhood_preservation_2hop_with_tree,
    },
    proximity::{
        gabriel_graph_preservation_with_tree, gabriel_graph_property_with_tree,
        node_resolution_with_tree, relative_neighborhood_preservation_with_tree,
    },
    rta::landmark_stratified_rta,
    stress::{sampled_scale_normalized_stress, scale_normalized_stress},
    validate_graph_layout,
};
use egraph_spatial_trees::StaticKdTree;

#[cfg(feature = "eval-baselines")]
use egraph_metrics_cluster::cluster_baselines::{
    neighborhood_hit_with_tree, sepme_gab_with_tree, sepme_gong_with_tree, sepme_knng_with_tree,
    sepme_rng_with_tree,
};

#[cfg(feature = "eval-baselines")]
use super::baselines::evaluate_eval_baseline_targets;
use super::config::{MetricConfig, MetricInputs};
use super::evaluate_manifold::evaluate_metrics_manifold_with_inputs;
use super::evaluate_nd::evaluate_metrics_nd_with_inputs;
use super::report::{MetricReport, MetricTarget, RequestedFlags};

/// Layout types supported by [`evaluate_metrics`] and related dispatcher entry points.
pub trait MetricLayout<S: Scalar> {
    /// Dispatches target-selective evaluation against `graph` using auxiliary `inputs`.
    #[doc(hidden)]
    fn evaluate_with_inputs_dispatch<W: Scalar>(
        &self,
        graph: &CsrGraph<W>,
        targets: &[MetricTarget],
        config: &MetricConfig,
        inputs: &MetricInputs<'_, S>,
    ) -> CoreResult<MetricReport>;
}

impl<S: Scalar> MetricLayout<S> for Layout2d<S> {
    #[inline]
    fn evaluate_with_inputs_dispatch<W: Scalar>(
        &self,
        graph: &CsrGraph<W>,
        targets: &[MetricTarget],
        config: &MetricConfig,
        inputs: &MetricInputs<'_, S>,
    ) -> CoreResult<MetricReport> {
        evaluate_metrics_2d_with_inputs(graph, self, targets, config, inputs)
    }
}

impl<S: Scalar> MetricLayout<S> for LayoutNd<S> {
    #[inline]
    fn evaluate_with_inputs_dispatch<W: Scalar>(
        &self,
        graph: &CsrGraph<W>,
        targets: &[MetricTarget],
        config: &MetricConfig,
        inputs: &MetricInputs<'_, S>,
    ) -> CoreResult<MetricReport> {
        evaluate_metrics_nd_with_inputs(graph, self, targets, config, inputs)
    }
}

impl<S: Scalar> MetricLayout<S> for ManifoldLayout2d<S> {
    #[inline]
    fn evaluate_with_inputs_dispatch<W: Scalar>(
        &self,
        graph: &CsrGraph<W>,
        targets: &[MetricTarget],
        config: &MetricConfig,
        inputs: &MetricInputs<'_, S>,
    ) -> CoreResult<MetricReport> {
        evaluate_metrics_manifold_with_inputs(graph, self, targets, config, inputs)
    }
}

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Supports both 2D [`Layout2d`] and N-D [`LayoutNd`] coordinates via [`MetricLayout`].
/// On [`Layout2d`], shares a single [`StaticKdTree<2, f64>`] across requested spatial/neighborhood
/// targets, a single sweep-line pass across `CrossingEdges`/`CrossingNumber`/`CrossingAngle`/`Crosslessness`,
/// a single torus crossing pass across `TorusCrossingNumber`/`CrossingEdgesTorus`,
/// a single sampling pass across `SampledCrossingNumber`/`SampledCrosslessness`, and
/// a single angular scan across `MinimumAngle`/`MinimumAngleA1`.
/// On [`LayoutNd`], shares one `NdSpatialIndex` (and one `DynamicKdTree<f64>` for `eval-baselines`
/// separation targets) and returns [`CoreError::UnsupportedOperation`] for 2D-only targets.
/// For [`MetricTarget::AdjustedPatchNig`], [`MetricTarget::ClusterAreaFaithfulness`],
/// [`MetricTarget::PrismDistortion`], or `eval-baselines` targets, use
/// [`evaluate_metrics_with_communities`] or [`evaluate_metrics_with_inputs`].
pub fn evaluate_metrics<W: Scalar, S: Scalar, L: MetricLayout<S>>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_inputs(graph, layout, targets, config, &MetricInputs::default())
}

/// Evaluates strictly the metrics listed in `targets` with optional `labels`
/// (required when [`MetricTarget::AdjustedPatchNig`], [`MetricTarget::ClusterAreaFaithfulness`],
/// or labeled `eval-baselines` targets are requested).
pub fn evaluate_metrics_with_communities<W: Scalar, S: Scalar, L: MetricLayout<S>>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
    labels: Option<&[u32]>,
) -> CoreResult<MetricReport> {
    let inputs = MetricInputs {
        labels,
        ..MetricInputs::default()
    };
    evaluate_metrics_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the metrics listed in `targets` with optional `labels` and
/// optional positive finite G-CAF target weights (`node_weights` or `cluster_weights`,
/// mutually exclusive, evaluated lazily when [`MetricTarget::ClusterAreaFaithfulness`] is requested).
pub fn evaluate_metrics_with_community_weights<W: Scalar, S: Scalar, L: MetricLayout<S>>(
    graph: &CsrGraph<W>,
    layout: &L,
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
    evaluate_metrics_with_inputs(graph, layout, targets, config, &inputs)
}

/// Evaluates strictly the metrics listed in `targets` using optional auxiliary [`MetricInputs`]
/// (`labels`, `node_weights`, `cluster_weights`, `original_layout`, `sprawlter_half_widths`,
/// `sprawlter_half_heights`), validating inputs lazily per requested target.
pub fn evaluate_metrics_with_inputs<W: Scalar, S: Scalar, L: MetricLayout<S>>(
    graph: &CsrGraph<W>,
    layout: &L,
    targets: &[MetricTarget],
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
) -> CoreResult<MetricReport> {
    layout.evaluate_with_inputs_dispatch(graph, targets, config, inputs)
}

fn evaluate_metrics_2d_with_inputs<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
) -> CoreResult<MetricReport> {
    validate_graph_layout(graph, layout, "evaluate_metrics")?;
    let flags = RequestedFlags::from_targets(targets);
    if flags.geodesic_distortion {
        return Err(CoreError::unsupported(
            "MetricTarget::GeodesicDistortion is only supported on ManifoldLayout2d",
        ));
    }
    let mut report = MetricReport::default();

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

    if flags.torus_crossing_number || flags.crossing_edges_torus {
        let pairs = crossing_edges_torus(graph, layout)?;
        if flags.torus_crossing_number {
            report.torus_crossing_number = Some(pairs.len());
        }
        if flags.crossing_edges_torus {
            report.crossing_edges_torus = Some(pairs);
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
    if (flags.sampled_np1 || flags.sampled_np2) && config.crossing_num_samples == 0 {
        return Err(CoreError::invalid_input(
            "crossing_num_samples must be >= 1",
        ));
    }

    let kdtree: Option<StaticKdTree<2, f64>> = if flags.needs_kdtree() {
        Some(build_kdtree_f64(layout)?)
    } else {
        None
    };

    if let Some(tree) = kdtree.as_ref() {
        evaluate_kdtree_targets(
            graph,
            layout,
            tree,
            &flags,
            config,
            inputs.labels,
            &mut report,
        )?;
    }

    if flags.aspect_ratio {
        report.aspect_ratio = Some(aspect_ratio(layout)?);
    }
    if flags.angular_resolution {
        report.angular_resolution = Some(angular_resolution(graph, layout)?);
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
    if flags.layout_degeneracy {
        let deg = layout_degeneracy(graph, layout)?;
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
        report.ideal_edge_lengths = Some(ideal_edge_lengths_scaled(
            graph,
            layout,
            config.ideal_edge_length,
        )?);
    }
    if flags.prism_distortion {
        let orig = inputs.original_layout.ok_or_else(|| {
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
    evaluate_eval_baseline_targets(graph, layout, &flags, config, inputs, &mut report)?;

    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_kdtree_targets<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    labels: Option<&[u32]>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.np1 {
        report.neighborhood_preservation_1hop = Some(neighborhood_preservation_1hop_with_tree(
            graph, layout, tree,
        )?);
    }
    if flags.np2 {
        report.neighborhood_preservation_2hop = Some(neighborhood_preservation_2hop_with_tree(
            graph, layout, tree,
        )?);
    }
    if flags.sampled_np1 {
        report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop_with_tree(
            graph,
            layout,
            tree,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.sampled_np2 {
        report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop_with_tree(
            graph,
            layout,
            tree,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
    }
    if flags.gabriel_property {
        report.gabriel_graph_property =
            Some(gabriel_graph_property_with_tree(graph, layout, tree)?);
    }
    if flags.node_resolution {
        report.node_resolution = Some(node_resolution_with_tree(layout, tree)?);
    }
    if flags.s_rng {
        report.relative_neighborhood_preservation = Some(
            relative_neighborhood_preservation_with_tree(graph, layout, tree)?,
        );
    }
    if flags.s_gg {
        report.gabriel_graph_preservation =
            Some(gabriel_graph_preservation_with_tree(graph, layout, tree)?);
    }
    #[cfg(not(feature = "eval-baselines"))]
    let _ = labels;
    #[cfg(feature = "eval-baselines")]
    {
        if flags.sepme_knng {
            let lbls = labels
                .ok_or_else(|| CoreError::invalid_input("SepmeKnng target requires labels"))?;
            report.sepme_knng = Some(sepme_knng_with_tree(
                layout,
                lbls,
                config.cluster_baselines.sepme_knng_k,
                tree,
            )?);
        }
        if flags.sepme_rng {
            let lbls = labels
                .ok_or_else(|| CoreError::invalid_input("SepmeRng target requires labels"))?;
            report.sepme_rng = Some(sepme_rng_with_tree(layout, lbls, tree)?);
        }
        if flags.sepme_gab {
            let lbls = labels
                .ok_or_else(|| CoreError::invalid_input("SepmeGab target requires labels"))?;
            report.sepme_gab = Some(sepme_gab_with_tree(layout, lbls, tree)?);
        }
        if flags.sepme_gong {
            let lbls = labels
                .ok_or_else(|| CoreError::invalid_input("SepmeGong target requires labels"))?;
            report.sepme_gong = Some(sepme_gong_with_tree(
                layout,
                lbls,
                config.cluster_baselines.gong_gamma,
                tree,
            )?);
        }
        if flags.neighborhood_hit {
            let lbls = labels.ok_or_else(|| {
                CoreError::invalid_input("NeighborhoodHit target requires labels")
            })?;
            report.neighborhood_hit = Some(neighborhood_hit_with_tree(
                layout,
                lbls,
                config.cluster_baselines.neighborhood_hit_k,
                tree,
            )?);
        }
    }
    Ok(())
}
