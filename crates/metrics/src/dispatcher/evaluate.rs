use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_metrics_cluster::{
    cluster_area::cluster_area_faithfulness_with_weights, community::adjusted_patch_nig,
};
use egraph_metrics_geometry::{
    build_kdtree_f64,
    crossings::{
        crossing_number, crossing_summary, crosslessness_from_count, sampled_crossing_number,
    },
    geometry::{
        angular_resolution, aspect_ratio, ideal_edge_lengths, ideal_edge_lengths_with_unit,
        layout_degeneracy, minimum_angle_summary,
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
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, neighborhood_hit_with_tree,
    pairwise_distance_overlap_error, sepme_gab_with_tree, sepme_gong_with_tree,
    sepme_knng_with_tree, sepme_rng_with_tree, silhouette_score, sprawlter_nn_radii,
};

use super::config::MetricConfig;
use super::report::{MetricReport, MetricTarget, RequestedFlags};

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Shares a single [`StaticKdTree<2, f64>`] across requested spatial/neighborhood targets,
/// a single sweep-line pass across `CrossingNumber`/`CrossingAngle`/`Crosslessness`,
/// a single sampling pass across `SampledCrossingNumber`/`SampledCrosslessness`, and
/// a single angular scan across `MinimumAngle`/`MinimumAngleA1`.
/// For [`MetricTarget::AdjustedPatchNig`] and [`MetricTarget::ClusterAreaFaithfulness`],
/// use [`evaluate_metrics_with_communities`].
pub fn evaluate_metrics<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_communities(graph, layout, targets, config, None)
}

/// Evaluates strictly the metrics listed in `targets` with optional `community_labels`
/// (required when [`MetricTarget::AdjustedPatchNig`] or [`MetricTarget::ClusterAreaFaithfulness`]
/// is requested).
pub fn evaluate_metrics_with_communities<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_community_weights(
        graph,
        layout,
        targets,
        config,
        community_labels,
        None,
        None,
    )
}

/// Evaluates strictly the metrics listed in `targets` with optional `community_labels` and
/// optional positive finite G-CAF target weights (`node_weights` or `cluster_weights`,
/// mutually exclusive, evaluated lazily when [`MetricTarget::ClusterAreaFaithfulness`] is requested).
pub fn evaluate_metrics_with_community_weights<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<MetricReport> {
    validate_graph_layout(graph, layout, "evaluate_metrics")?;
    let flags = RequestedFlags::from_targets(targets);
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

    if flags.crossing_angle {
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
            community_labels,
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
        let val = if graph.is_weighted() {
            ideal_edge_lengths(graph, layout)?
        } else {
            ideal_edge_lengths_with_unit(graph, layout, config.ideal_edge_length)?
        };
        report.ideal_edge_lengths = Some(val);
    }
    if flags.adjusted_patch_nig {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("AdjustedPatchNig target requires community_labels")
        })?;
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
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("ClusterAreaFaithfulness target requires community_labels")
        })?;
        report.cluster_area_faithfulness = Some(
            cluster_area_faithfulness_with_weights(
                layout,
                labels,
                &config.cluster_area,
                node_weights,
                cluster_weights,
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
    evaluate_eval_baseline_targets(layout, &flags, config, community_labels, &mut report)?;

    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_kdtree_targets<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
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
    let _ = community_labels;
    #[cfg(feature = "eval-baselines")]
    {
        if flags.sepme_knng {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("SepmeKnng target requires community_labels")
            })?;
            report.sepme_knng = Some(sepme_knng_with_tree(
                layout,
                labels,
                config.cluster_baselines.sepme_knng_k,
                tree,
            )?);
        }
        if flags.sepme_rng {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("SepmeRng target requires community_labels")
            })?;
            report.sepme_rng = Some(sepme_rng_with_tree(layout, labels, tree)?);
        }
        if flags.sepme_gab {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("SepmeGab target requires community_labels")
            })?;
            report.sepme_gab = Some(sepme_gab_with_tree(layout, labels, tree)?);
        }
        if flags.sepme_gong {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("SepmeGong target requires community_labels")
            })?;
            report.sepme_gong = Some(sepme_gong_with_tree(
                layout,
                labels,
                config.cluster_baselines.gong_gamma,
                tree,
            )?);
        }
        if flags.neighborhood_hit {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("NeighborhoodHit target requires community_labels")
            })?;
            report.neighborhood_hit = Some(neighborhood_hit_with_tree(
                layout,
                labels,
                config.cluster_baselines.neighborhood_hit_k,
                tree,
            )?);
        }
    }
    Ok(())
}

#[cfg(feature = "eval-baselines")]
fn evaluate_eval_baseline_targets<S: Scalar>(
    layout: &Layout2d<S>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.dsc {
        let labels = community_labels
            .ok_or_else(|| CoreError::invalid_input("Dsc target requires community_labels"))?;
        report.dsc = Some(distance_consistency(layout, labels)?);
    }
    if flags.dc {
        let labels = community_labels
            .ok_or_else(|| CoreError::invalid_input("Dc target requires community_labels"))?;
        report.dc = Some(distribution_consistency(
            layout,
            labels,
            config.cluster_baselines.dc_grid_size,
        )?);
    }
    if flags.cdm || flags.hdm {
        let labels = community_labels
            .ok_or_else(|| CoreError::invalid_input("Cdm/Hdm target requires community_labels"))?;
        let (cdm, hdm) = class_density_measure(
            layout,
            labels,
            config.cluster_baselines.cdm_grid_size,
            config.cluster_baselines.cdm_bandwidth,
        )?;
        if flags.cdm {
            report.cdm = Some(cdm);
        }
        if flags.hdm {
            report.hdm = Some(hdm);
        }
    }
    if flags.silhouette {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("Silhouette target requires community_labels")
        })?;
        report.silhouette = Some(silhouette_score(layout, labels)?);
    }
    if flags.calinski_harabasz {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("CalinskiHarabasz target requires community_labels")
        })?;
        report.calinski_harabasz = Some(calinski_harabasz_score(layout, labels)?);
    }
    if flags.davies_bouldin {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("DaviesBouldin target requires community_labels")
        })?;
        report.davies_bouldin = Some(davies_bouldin_score(layout, labels)?);
    }
    if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
        let labels = community_labels
            .ok_or_else(|| CoreError::invalid_input("CqKmeans target requires community_labels"))?;
        let pa = cq_kmeans(
            layout,
            labels,
            config.cluster_baselines.kmeans_n_init,
            config.cluster_baselines.kmeans_max_iters,
            config.cluster_baselines.seed,
        )?;
        if flags.cq_kmeans_ari {
            report.cq_kmeans_ari = Some(pa.ari);
        }
        if flags.cq_kmeans_ami {
            report.cq_kmeans_ami = Some(pa.ami);
        }
    }
    if flags.cq_hac_hom || flags.cq_hac_ami {
        let labels = community_labels
            .ok_or_else(|| CoreError::invalid_input("CqHac target requires community_labels"))?;
        let pa = cq_hac(layout, labels)?;
        if flags.cq_hac_hom {
            report.cq_hac_hom = Some(pa.homogeneity);
        }
        if flags.cq_hac_ami {
            report.cq_hac_ami = Some(pa.ami);
        }
    }
    if flags.pairwise_distance_overlap_error {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input(
                "PairwiseDistanceOverlapError target requires community_labels",
            )
        })?;
        report.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error(
            layout,
            labels,
            config.cluster_baselines.num_bins,
        )?);
    }
    if flags.sprawlter_nn {
        let r_f64 = config.cluster_baselines.sprawlter_node_radius;
        if !r_f64.is_finite() {
            return Err(CoreError::non_finite(
                "ClusterBaselineConfig::sprawlter_node_radius",
            ));
        }
        if r_f64 <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::sprawlter_node_radius must be strictly positive",
            ));
        }
        let r = S::from_f64(r_f64);
        let radii = vec![r; layout.len()];
        report.sprawlter_nn = Some(sprawlter_nn_radii(layout, &radii)?);
    }
    Ok(())
}
