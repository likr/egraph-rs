use super::super::*;
use crate::{
    ClusterAreaConfig, adjusted_patch_nig, angular_resolution, aspect_ratio,
    cluster_area_faithfulness, cluster_area_faithfulness_with_weights, crossing_angle,
    crossing_number, crosslessness, gabriel_graph_preservation, gabriel_graph_property,
    ideal_edge_lengths, landmark_stratified_rta, minimum_angle_summary,
    neighborhood_preservation_1hop, neighborhood_preservation_2hop, node_resolution,
    relative_neighborhood_preservation, sampled_crossing_number, sampled_crosslessness,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    sampled_scale_normalized_stress, scale_normalized_stress,
};
use egraph_core::{CoreError, CsrGraph, Layout2d};

fn build_test_cfg() -> MetricConfig {
    MetricConfig::new()
        .with_sns_num_pivots(2)
        .with_sns_seed(9)
        .with_crossing_num_samples(64)
        .with_crossing_seed(9)
        .with_np_num_samples(64)
        .with_np_seed(9)
        .with_ideal_edge_length(1.0)
        .with_rta_num_landmarks(4)
        .with_rta_num_triplets(10_000)
        .with_rta_min_degree(1)
        .with_rta_seed(9)
        .with_patch_nig_radius(16)
        .with_patch_nig_canvas_size(1024)
        .with_patch_nig_pad_frac(0.04)
        .with_patch_nig_seed(9)
        .with_cluster_area(ClusterAreaConfig::new())
}

#[test]
fn test_selective_dispatcher_computes_only_requested_targets() {
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let layout = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let cfg = build_test_cfg();
    assert_eq!(MetricConfig::default().sns_num_pivots, 64);
    assert_eq!(MetricConfig::default().np_num_samples, 4096);
    assert_eq!(MetricConfig::default().np_seed, 42);

    // Request strictly NP2 + SNS: all other fields and baseline_metrics must remain None.
    let subset = evaluate_metrics(
        &g,
        &layout,
        &[
            MetricTarget::NeighborhoodPreservation2Hop,
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::NeighborhoodPreservation2Hop,
        ],
        &cfg,
    )
    .unwrap();
    assert_eq!(
        subset.scale_normalized_stress,
        Some(scale_normalized_stress(&g, &layout).unwrap())
    );
    assert_eq!(
        subset.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop(&g, &layout).unwrap())
    );
    assert!(subset.baseline_metrics().is_none());
    assert!(subset.crossing_number.is_none());
    assert!(subset.crossing_angle.is_none());
    assert!(subset.sampled_crossing_number.is_none());
    assert!(subset.crosslessness.is_none());
    assert!(subset.sampled_crosslessness.is_none());
    assert!(subset.sampled_scale_normalized_stress.is_none());
    assert!(subset.neighborhood_preservation_1hop.is_none());
    assert!(subset.sampled_np1.is_none());
    assert!(subset.sampled_np2.is_none());
    assert!(subset.gabriel_graph_property.is_none());
    assert!(subset.node_resolution.is_none());
    assert!(subset.relative_neighborhood_preservation.is_none());
    assert!(subset.gabriel_graph_preservation.is_none());
    assert!(subset.aspect_ratio.is_none());
    assert!(subset.angular_resolution.is_none());
    assert!(subset.minimum_angle.is_none());
    assert!(subset.minimum_angle_a1.is_none());
    assert!(subset.layout_degeneracy.is_none());
    assert!(subset.ideal_edge_lengths.is_none());
    assert!(subset.adjusted_patch_nig.is_none());
    assert!(subset.cluster_area_faithfulness.is_none());
    assert!(subset.landmark_stratified_rta.is_none());

    // Empty targets list returns all None cleanly.
    assert_eq!(
        evaluate_metrics(&g, &layout, &[], &cfg).unwrap(),
        MetricReport::default()
    );

    // Request every 2D target and verify exact parity with standalone calls.
    let all_targets = [
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
        MetricTarget::AdjustedPatchNig,
        MetricTarget::ClusterAreaFaithfulness,
        MetricTarget::LandmarkStratifiedRta,
    ];
    let communities = [0_u32, 0, 1, 1];
    let full =
        evaluate_metrics_with_communities(&g, &layout, &all_targets, &cfg, Some(&communities))
            .unwrap();
    assert_eq!(
        full.scale_normalized_stress,
        Some(scale_normalized_stress(&g, &layout).unwrap())
    );
    assert_eq!(
        full.sampled_scale_normalized_stress,
        Some(sampled_scale_normalized_stress(&g, &layout, 2, 9).unwrap())
    );
    assert_eq!(
        full.crossing_number,
        Some(crossing_number(&g, &layout).unwrap())
    );
    assert_eq!(
        full.crossing_angle,
        Some(crossing_angle(&g, &layout).unwrap())
    );
    assert_eq!(
        full.sampled_crossing_number,
        Some(sampled_crossing_number(&g, &layout, 64, 9).unwrap())
    );
    assert_eq!(
        full.crosslessness,
        Some(crosslessness(&g, &layout).unwrap())
    );
    assert_eq!(
        full.sampled_crosslessness,
        Some(sampled_crosslessness(&g, &layout, 64, 9).unwrap())
    );
    assert_eq!(
        full.neighborhood_preservation_1hop,
        Some(neighborhood_preservation_1hop(&g, &layout).unwrap())
    );
    assert_eq!(
        full.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop(&g, &layout).unwrap())
    );
    assert_eq!(
        full.sampled_np1,
        Some(sampled_neighborhood_preservation_1hop(&g, &layout, 64, 9).unwrap())
    );
    assert_eq!(
        full.sampled_np2,
        Some(sampled_neighborhood_preservation_2hop(&g, &layout, 64, 9).unwrap())
    );
    assert_eq!(
        full.gabriel_graph_property,
        Some(gabriel_graph_property(&g, &layout).unwrap())
    );
    assert_eq!(
        full.node_resolution,
        Some(node_resolution(&layout).unwrap())
    );
    assert_eq!(
        full.relative_neighborhood_preservation,
        Some(relative_neighborhood_preservation(&g, &layout).unwrap())
    );
    assert_eq!(
        full.gabriel_graph_preservation,
        Some(gabriel_graph_preservation(&g, &layout).unwrap())
    );
    assert_eq!(full.aspect_ratio, Some(aspect_ratio(&layout).unwrap()));
    assert_eq!(
        full.angular_resolution,
        Some(angular_resolution(&g, &layout).unwrap())
    );
    let min_ang = minimum_angle_summary(&g, &layout).unwrap();
    assert_eq!(full.minimum_angle, Some(min_ang.a2));
    assert_eq!(full.minimum_angle_a1, Some(min_ang.a1));
    assert_eq!(full.layout_degeneracy, Some(0.0));
    assert_eq!(
        full.ideal_edge_lengths,
        Some(ideal_edge_lengths(&g, &layout).unwrap())
    );
    assert_eq!(
        full.adjusted_patch_nig,
        Some(adjusted_patch_nig(&layout, &communities, 16, 1024, 0.04, 9).unwrap())
    );
    assert_eq!(
        full.cluster_area_faithfulness,
        Some(
            cluster_area_faithfulness(&layout, &communities, &cfg.cluster_area)
                .unwrap()
                .caf
        )
    );
    assert_eq!(
        full.landmark_stratified_rta,
        Some(landmark_stratified_rta(&g, &layout, 4, 10_000, 1, 9).unwrap())
    );
    assert!(matches!(
        evaluate_metrics(&g, &layout, &[MetricTarget::AdjustedPatchNig], &cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        evaluate_metrics(&g, &layout, &[MetricTarget::ClusterAreaFaithfulness], &cfg),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_f32_unit_graph_and_weighted_gcaf_targets() {
    let cfg = build_test_cfg();
    let communities = [0_u32, 0, 1, 1];

    // Verify CsrGraph<()> (unweighted unit graph) and f32 layouts work uniformly.
    let g_unit =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, false)
            .unwrap();
    let l32 =
        Layout2d::from_xy(vec![0.0_f32, 1.0, 1.0, 0.0], vec![0.0_f32, 0.0, 1.0, 1.0]).unwrap();
    let rep_unit32 = evaluate_metrics_with_communities(
        &g_unit,
        &l32,
        &[
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::CrossingNumber,
            MetricTarget::AspectRatio,
            MetricTarget::IdealEdgeLengths,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
            MetricTarget::ClusterAreaFaithfulness,
        ],
        &cfg,
        Some(&communities),
    )
    .unwrap();
    assert_eq!(rep_unit32.crossing_number, Some(0));
    assert!((rep_unit32.aspect_ratio.unwrap() - 1.0).abs() < 1e-6);
    assert!(rep_unit32.ideal_edge_lengths.unwrap().abs() < 1e-6);
    assert!((rep_unit32.relative_neighborhood_preservation.unwrap() - 1.0).abs() < 1e-6);
    assert!((rep_unit32.gabriel_graph_preservation.unwrap() - 1.0).abs() < 1e-6);
    assert!(rep_unit32.cluster_area_faithfulness.unwrap().is_finite());

    // Weighted G-CAF via evaluate_metrics_with_community_weights + lazy weight validation.
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let layout = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let nw = [1.0, 2.0, 3.0, 4.0];
    let cw = [3.0, 7.0];
    let rep_nw = evaluate_metrics_with_community_weights(
        &g,
        &layout,
        &[MetricTarget::ClusterAreaFaithfulness],
        &cfg,
        Some(&communities),
        Some(&nw),
        None,
    )
    .unwrap();
    let rep_cw = evaluate_metrics_with_community_weights(
        &g,
        &layout,
        &[MetricTarget::ClusterAreaFaithfulness],
        &cfg,
        Some(&communities),
        None,
        Some(&cw),
    )
    .unwrap();
    assert_eq!(
        rep_nw.cluster_area_faithfulness,
        Some(
            cluster_area_faithfulness_with_weights(
                &layout,
                &communities,
                &cfg.cluster_area,
                Some(&nw),
                None
            )
            .unwrap()
            .caf
        )
    );
    assert!(
        (rep_nw.cluster_area_faithfulness.unwrap() - rep_cw.cluster_area_faithfulness.unwrap())
            .abs()
            < 1e-12
    );
    assert!(
        evaluate_metrics_with_community_weights(
            &g,
            &layout,
            &[MetricTarget::AspectRatio],
            &cfg,
            Some(&communities),
            Some(&[-1.0]),
            Some(&[0.0]),
        )
        .is_ok()
    );
    assert!(matches!(
        evaluate_metrics_with_community_weights(
            &g,
            &layout,
            &[MetricTarget::ClusterAreaFaithfulness],
            &cfg,
            Some(&communities),
            Some(&nw),
            Some(&cw),
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // Weighted graphs scale edge weights by MetricConfig::ideal_edge_length.
    let g_weighted =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[1.0, 2.0]), false, false).unwrap();
    let l_weighted = Layout2d::from_xy(vec![0.0, 2.5, 7.5], vec![0.0, 0.0, 0.0]).unwrap();
    let scaled_cfg = cfg.clone().with_ideal_edge_length(2.5);
    let rep_scaled = evaluate_metrics(
        &g_weighted,
        &l_weighted,
        &[MetricTarget::IdealEdgeLengths],
        &scaled_cfg,
    )
    .unwrap();
    assert!(rep_scaled.ideal_edge_lengths.unwrap().abs() < 1e-12);
}
