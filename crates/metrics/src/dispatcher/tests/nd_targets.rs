use super::super::*;
use crate::{
    ClusterAreaConfig, gabriel_graph_preservation_nd, gabriel_graph_property_nd,
    ideal_edge_lengths_scaled_nd, landmark_stratified_rta_nd, layout_degeneracy_nd,
    neighborhood_preservation_1hop_nd, neighborhood_preservation_2hop_nd, node_resolution_nd,
    relative_neighborhood_preservation_nd, sampled_neighborhood_preservation_1hop_nd,
    sampled_neighborhood_preservation_2hop_nd, sampled_scale_normalized_stress_nd,
    scale_normalized_stress_nd,
};
use egraph_core::{CoreError, CsrGraph, Layout2d, LayoutNd};

const ALL_ND_BASE_TARGETS: [MetricTarget; 13] = [
    MetricTarget::ScaleNormalizedStress,
    MetricTarget::SampledScaleNormalizedStress,
    MetricTarget::NeighborhoodPreservation1Hop,
    MetricTarget::NeighborhoodPreservation2Hop,
    MetricTarget::SampledNp1,
    MetricTarget::SampledNp2,
    MetricTarget::GabrielGraphProperty,
    MetricTarget::NodeResolution,
    MetricTarget::RelativeNeighborhoodPreservation,
    MetricTarget::GabrielGraphPreservation,
    MetricTarget::LayoutDegeneracy,
    MetricTarget::IdealEdgeLengths,
    MetricTarget::LandmarkStratifiedRta,
];

fn test_cfg() -> MetricConfig {
    MetricConfig {
        sns_num_pivots: 3,
        sns_seed: 11,
        crossing_num_samples: 64,
        crossing_seed: 11,
        ideal_edge_length: 1.5,
        rta_num_landmarks: 4,
        rta_num_triplets: 4096,
        rta_min_degree: 1,
        rta_seed: 11,
        patch_nig_radius: 8,
        patch_nig_canvas_size: 256,
        patch_nig_pad_frac: 0.04,
        patch_nig_seed: 11,
        cluster_area: ClusterAreaConfig::default(),
        #[cfg(feature = "eval-baselines")]
        cluster_baselines: ClusterBaselineConfig::default(),
    }
}

fn make_sample_graph_6() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        None,
        false,
        false,
    )
    .unwrap()
}

#[test]
fn test_nd_dispatcher_selective_empty_and_2d_parity() {
    let g = make_sample_graph_6();
    let xs = vec![0.0, 0.4, 0.2, 4.0, 4.4, 4.2];
    let ys = vec![0.0, 0.1, -0.3, 4.0, 4.1, 3.7];
    let layout2d = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let mut flat = Vec::with_capacity(12);
    for i in 0..6 {
        flat.push(xs[i]);
        flat.push(ys[i]);
    }
    let layout_nd = LayoutNd::from_flat(6, 2, flat).unwrap();
    let cfg = test_cfg();

    assert_eq!(
        evaluate_metrics(&g, &layout_nd, &[], &cfg).unwrap(),
        MetricReport::default()
    );
    assert_eq!(
        evaluate_metrics_nd(&g, &layout_nd, &[], &cfg).unwrap(),
        MetricReport::default()
    );

    // Partial selection keeps all unrequested fields as None.
    let subset = evaluate_metrics(
        &g,
        &layout_nd,
        &[
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::NeighborhoodPreservation2Hop,
        ],
        &cfg,
    )
    .unwrap();
    assert_eq!(
        subset.scale_normalized_stress,
        Some(scale_normalized_stress_nd(&g, &layout_nd).unwrap())
    );
    assert_eq!(
        subset.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop_nd(&g, &layout_nd).unwrap())
    );
    assert!(subset.sampled_scale_normalized_stress.is_none());
    assert!(subset.neighborhood_preservation_1hop.is_none());
    assert!(subset.node_resolution.is_none());
    assert!(subset.layout_degeneracy.is_none());
    assert!(subset.landmark_stratified_rta.is_none());

    // Exact parity between Layout2d and dims == 2 LayoutNd on all 13 base N-D targets.
    let rep_2d = evaluate_metrics(&g, &layout2d, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    let rep_nd_generic = evaluate_metrics(&g, &layout_nd, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    let rep_nd_explicit = evaluate_metrics_nd(&g, &layout_nd, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(rep_nd_generic, rep_nd_explicit);
    assert_eq!(rep_nd_generic, rep_2d);
}

#[test]
fn test_nd_dispatcher_dims_1_3_4_8_10_f64_and_f32_parity() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();

    for &dims in &[1usize, 3, 4, 8, 10] {
        let mut flat64 = Vec::with_capacity(6 * dims);
        for i in 0..6 {
            let cluster_shift = if i < 3 { 0.0 } else { 5.0 };
            for d in 0..dims {
                flat64.push(cluster_shift + ((i * 7 + d * 3) as f64) * 0.13);
            }
        }
        let l64 = LayoutNd::from_flat(6, dims, flat64.clone()).unwrap();
        let l32 = LayoutNd::from_flat(6, dims, flat64.iter().map(|&v| v as f32).collect()).unwrap();

        let rep64 = evaluate_metrics(&g, &l64, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
        let rep64_explicit = evaluate_metrics_nd(&g, &l64, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
        assert_eq!(rep64, rep64_explicit);

        assert_eq!(
            rep64.scale_normalized_stress,
            Some(scale_normalized_stress_nd(&g, &l64).unwrap())
        );
        assert_eq!(
            rep64.sampled_scale_normalized_stress,
            Some(
                sampled_scale_normalized_stress_nd(&g, &l64, cfg.sns_num_pivots, cfg.sns_seed)
                    .unwrap()
            )
        );
        assert_eq!(
            rep64.neighborhood_preservation_1hop,
            Some(neighborhood_preservation_1hop_nd(&g, &l64).unwrap())
        );
        assert_eq!(
            rep64.neighborhood_preservation_2hop,
            Some(neighborhood_preservation_2hop_nd(&g, &l64).unwrap())
        );
        assert_eq!(
            rep64.sampled_np1,
            Some(
                sampled_neighborhood_preservation_1hop_nd(
                    &g,
                    &l64,
                    cfg.crossing_num_samples,
                    cfg.crossing_seed
                )
                .unwrap()
            )
        );
        assert_eq!(
            rep64.sampled_np2,
            Some(
                sampled_neighborhood_preservation_2hop_nd(
                    &g,
                    &l64,
                    cfg.crossing_num_samples,
                    cfg.crossing_seed
                )
                .unwrap()
            )
        );
        assert_eq!(
            rep64.gabriel_graph_property,
            Some(gabriel_graph_property_nd(&g, &l64).unwrap())
        );
        assert_eq!(
            rep64.node_resolution,
            Some(node_resolution_nd(&l64).unwrap())
        );
        assert_eq!(
            rep64.relative_neighborhood_preservation,
            Some(relative_neighborhood_preservation_nd(&g, &l64).unwrap())
        );
        assert_eq!(
            rep64.gabriel_graph_preservation,
            Some(gabriel_graph_preservation_nd(&g, &l64).unwrap())
        );
        let deg64 = layout_degeneracy_nd(&g, &l64).unwrap();
        let bbox_pen = if deg64.is_bbox_degenerate { 1.0 } else { 0.0 };
        assert_eq!(
            rep64.layout_degeneracy,
            Some(
                deg64
                    .duplicate_ratio_f32
                    .max(deg64.zero_length_edge_ratio)
                    .max(bbox_pen)
            )
        );
        assert_eq!(
            rep64.ideal_edge_lengths,
            Some(ideal_edge_lengths_scaled_nd(&g, &l64, cfg.ideal_edge_length).unwrap())
        );
        assert_eq!(
            rep64.landmark_stratified_rta,
            Some(
                landmark_stratified_rta_nd(
                    &g,
                    &l64,
                    cfg.rta_num_landmarks,
                    cfg.rta_num_triplets,
                    cfg.rta_min_degree,
                    cfg.rta_seed,
                )
                .unwrap()
            )
        );

        let rep32 = evaluate_metrics(&g, &l32, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
        assert_eq!(
            rep32.scale_normalized_stress,
            Some(scale_normalized_stress_nd(&g, &l32).unwrap())
        );
        assert_eq!(
            rep32.neighborhood_preservation_2hop,
            Some(neighborhood_preservation_2hop_nd(&g, &l32).unwrap())
        );
        assert_eq!(
            rep32.node_resolution,
            Some(node_resolution_nd(&l32).unwrap())
        );
    }
}

#[test]
fn test_nd_dispatcher_rejects_2d_only_targets_and_boundary_errors() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();
    let l3 = LayoutNd::<f64>::init(6, 3, 7, 1.0).unwrap();

    for &target in &[
        MetricTarget::CrossingNumber,
        MetricTarget::CrossingEdges,
        MetricTarget::CrossingAngle,
        MetricTarget::SampledCrossingNumber,
        MetricTarget::Crosslessness,
        MetricTarget::SampledCrosslessness,
        MetricTarget::TorusCrossingNumber,
        MetricTarget::CrossingEdgesTorus,
        MetricTarget::AspectRatio,
        MetricTarget::AngularResolution,
        MetricTarget::MinimumAngle,
        MetricTarget::MinimumAngleA1,
        MetricTarget::PrismDistortion,
        MetricTarget::AdjustedPatchNig,
        MetricTarget::ClusterAreaFaithfulness,
    ] {
        assert!(
            matches!(
                evaluate_metrics(
                    &g,
                    &l3,
                    &[MetricTarget::ScaleNormalizedStress, target],
                    &cfg
                ),
                Err(CoreError::UnsupportedOperation(_))
            ),
            "expected UnsupportedOperation for {target:?} on LayoutNd"
        );
    }

    // Empty (n=0), singleton (n=1), and edgeless (n=4, 0 edges) graphs succeed cleanly.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let l0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    let r0 = evaluate_metrics(&g0, &l0, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(r0.scale_normalized_stress, Some(0.0));
    assert_eq!(r0.layout_degeneracy, Some(0.0));

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let l1 = LayoutNd::<f64>::from_flat(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    let r1 = evaluate_metrics(&g1, &l1, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(r1.scale_normalized_stress, Some(0.0));
    assert_eq!(r1.layout_degeneracy, Some(0.0));

    let g4_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 3, 42, 1.0).unwrap();
    let r4 = evaluate_metrics(&g4_edgeless, &l4, &ALL_ND_BASE_TARGETS, &cfg).unwrap();
    assert!(r4.scale_normalized_stress.unwrap().is_finite());

    // Node count mismatch -> DimensionMismatch.
    let l_mismatch = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert!(matches!(
        evaluate_metrics(
            &g,
            &l_mismatch,
            &[MetricTarget::ScaleNormalizedStress],
            &cfg
        ),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Non-finite coordinates -> NonFiniteValue.
    let mut nan_coords = vec![0.0_f64; 18];
    nan_coords[5] = f64::NAN;
    let l_nan = LayoutNd::from_flat(6, 3, nan_coords).unwrap();
    assert!(matches!(
        evaluate_metrics(&g, &l_nan, &[MetricTarget::ScaleNormalizedStress], &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut inf_coords = vec![0.0_f64; 18];
    inf_coords[2] = f64::INFINITY;
    let l_inf = LayoutNd::from_flat(6, 3, inf_coords).unwrap();
    assert!(matches!(
        evaluate_metrics(&g, &l_inf, &[MetricTarget::ScaleNormalizedStress], &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Config validation errors.
    let bad_samples = MetricConfig {
        crossing_num_samples: 0,
        ..cfg
    };
    assert!(matches!(
        evaluate_metrics(&g, &l3, &[MetricTarget::SampledNp1], &bad_samples),
        Err(CoreError::InvalidInput(_))
    ));
    let bad_iel = MetricConfig {
        ideal_edge_length: 0.0,
        ..cfg
    };
    assert!(matches!(
        evaluate_metrics(&g, &l3, &[MetricTarget::IdealEdgeLengths], &bad_iel),
        Err(CoreError::InvalidInput(_))
    ));
}
