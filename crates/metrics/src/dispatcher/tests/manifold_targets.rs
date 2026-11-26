use super::super::*;
use crate::{
    ClusterAreaConfig, crossing_edges_torus, gabriel_graph_preservation_manifold,
    gabriel_graph_property_manifold, geodesic_distortion, ideal_edge_lengths_scaled_manifold,
    landmark_stratified_rta_manifold, layout_degeneracy_manifold,
    neighborhood_preservation_1hop_manifold, neighborhood_preservation_2hop_manifold,
    node_resolution_manifold, relative_neighborhood_preservation_manifold,
    sampled_neighborhood_preservation_1hop_manifold,
    sampled_neighborhood_preservation_2hop_manifold, sampled_scale_normalized_stress_manifold,
    scale_normalized_stress_manifold,
};
use egraph_core::{
    CoreError, CsrGraph, Layout2d, LayoutNd, ManifoldKind, ManifoldLayout2d, Scalar,
};

const MANIFOLD_BASE_TARGETS: [MetricTarget; 14] = [
    MetricTarget::ScaleNormalizedStress,
    MetricTarget::SampledScaleNormalizedStress,
    MetricTarget::GeodesicDistortion,
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

fn make_manifold_layout<S: Scalar>(kind: ManifoldKind, scale: f64) -> ManifoldLayout2d<S> {
    let mut layout = ManifoldLayout2d::<S>::init(kind, 6, 42, 0.3).unwrap();
    layout.set_scale(scale).unwrap();
    layout
}

fn assert_manifold_parity<S: Scalar>(kind: ManifoldKind, scale: f64) {
    let g = make_sample_graph_6();
    let cfg = test_cfg();
    let layout = make_manifold_layout::<S>(kind, scale);

    let rep = evaluate_metrics(&g, &layout, &MANIFOLD_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(
        rep.scale_normalized_stress,
        Some(scale_normalized_stress_manifold(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.sampled_scale_normalized_stress,
        Some(
            sampled_scale_normalized_stress_manifold(&g, &layout, cfg.sns_num_pivots, cfg.sns_seed)
                .unwrap()
        )
    );
    assert_eq!(
        rep.geodesic_distortion,
        Some(geodesic_distortion(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.neighborhood_preservation_1hop,
        Some(neighborhood_preservation_1hop_manifold(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop_manifold(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.sampled_np1,
        Some(
            sampled_neighborhood_preservation_1hop_manifold(
                &g,
                &layout,
                cfg.crossing_num_samples,
                cfg.crossing_seed
            )
            .unwrap()
        )
    );
    assert_eq!(
        rep.sampled_np2,
        Some(
            sampled_neighborhood_preservation_2hop_manifold(
                &g,
                &layout,
                cfg.crossing_num_samples,
                cfg.crossing_seed
            )
            .unwrap()
        )
    );
    assert_eq!(
        rep.gabriel_graph_property,
        Some(gabriel_graph_property_manifold(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.node_resolution,
        Some(node_resolution_manifold(&layout).unwrap())
    );
    assert_eq!(
        rep.relative_neighborhood_preservation,
        Some(relative_neighborhood_preservation_manifold(&g, &layout).unwrap())
    );
    assert_eq!(
        rep.gabriel_graph_preservation,
        Some(gabriel_graph_preservation_manifold(&g, &layout).unwrap())
    );
    let deg = layout_degeneracy_manifold(&g, &layout).unwrap();
    let bbox_pen = if deg.is_bbox_degenerate { 1.0 } else { 0.0 };
    assert_eq!(
        rep.layout_degeneracy,
        Some(
            deg.duplicate_ratio_f32
                .max(deg.zero_length_edge_ratio)
                .max(bbox_pen)
        )
    );
    assert_eq!(
        rep.ideal_edge_lengths,
        Some(ideal_edge_lengths_scaled_manifold(&g, &layout, cfg.ideal_edge_length).unwrap())
    );
    assert_eq!(
        rep.landmark_stratified_rta,
        Some(
            landmark_stratified_rta_manifold(
                &g,
                &layout,
                cfg.rta_num_landmarks,
                cfg.rta_num_triplets,
                cfg.rta_min_degree,
                cfg.rta_seed,
            )
            .unwrap()
        )
    );
}

#[test]
fn test_manifold_dispatcher_all_kinds_f64_f32_and_selective() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();
    let poincare = make_manifold_layout::<f64>(ManifoldKind::HyperbolicPoincare, 1.5);

    assert_eq!(
        evaluate_metrics(&g, &poincare, &[], &cfg).unwrap(),
        MetricReport::default()
    );

    let subset = evaluate_metrics(
        &g,
        &poincare,
        &[
            MetricTarget::GeodesicDistortion,
            MetricTarget::ScaleNormalizedStress,
        ],
        &cfg,
    )
    .unwrap();
    assert_eq!(
        subset.geodesic_distortion,
        Some(geodesic_distortion(&g, &poincare).unwrap())
    );
    assert_eq!(
        subset.scale_normalized_stress,
        Some(scale_normalized_stress_manifold(&g, &poincare).unwrap())
    );
    assert!(subset.sampled_scale_normalized_stress.is_none());
    assert!(subset.neighborhood_preservation_1hop.is_none());
    assert!(subset.neighborhood_preservation_2hop.is_none());
    assert!(subset.torus_crossing_number.is_none());
    assert!(subset.crossing_edges_torus.is_none());
    assert!(subset.node_resolution.is_none());
    assert!(subset.layout_degeneracy.is_none());
    assert!(subset.landmark_stratified_rta.is_none());

    for &kind in &[
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        assert_manifold_parity::<f64>(kind, 2.0);
        assert_manifold_parity::<f32>(kind, 1.25);
    }
}

#[test]
fn test_manifold_dispatcher_torus_crossings_and_metric_sensitivity() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 2), (1, 3)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.4, 0.4],
        vec![0.1, 0.4, 0.4, 0.1],
    )
    .unwrap();
    torus.set_torus_extents(3.0, 1.5).unwrap();
    let cfg = test_cfg();

    let rep = evaluate_metrics(
        &g,
        &torus,
        &[
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
            MetricTarget::GeodesicDistortion,
        ],
        &cfg,
    )
    .unwrap();
    let expected_pairs = crossing_edges_torus(&g, &torus).unwrap();
    assert_eq!(expected_pairs.len(), 1);
    assert_eq!(rep.torus_crossing_number, Some(1));
    assert_eq!(rep.crossing_edges_torus, Some(expected_pairs.clone()));
    assert_eq!(
        rep.geodesic_distortion,
        Some(geodesic_distortion(&g, &torus).unwrap())
    );

    // Non-unity aspect changes physical crossing angle on Torus2d.
    let mut unit_torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.4, 0.4],
        vec![0.1, 0.4, 0.4, 0.1],
    )
    .unwrap();
    unit_torus.set_torus_extents(1.0, 1.0).unwrap();
    let rep_unit = evaluate_metrics(
        &g,
        &unit_torus,
        &[
            MetricTarget::CrossingEdgesTorus,
            MetricTarget::GeodesicDistortion,
        ],
        &cfg,
    )
    .unwrap();
    let pairs_rect = rep.crossing_edges_torus.unwrap();
    let pairs_unit = rep_unit.crossing_edges_torus.unwrap();
    assert!((pairs_unit[0].cos_sq - 0.0).abs() < 1e-12);
    assert!(pairs_rect[0].cos_sq > 0.2);
    assert_ne!(rep.geodesic_distortion, rep_unit.geodesic_distortion);

    // Non-torus manifolds reject TorusCrossingNumber and CrossingEdgesTorus.
    for &non_torus in &[
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
    ] {
        let ml = ManifoldLayout2d::<f64>::init(non_torus, 4, 42, 0.3).unwrap();
        for &target in &[
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
        ] {
            assert!(
                matches!(
                    evaluate_metrics(&g, &ml, &[target], &cfg),
                    Err(CoreError::UnsupportedOperation(_))
                ),
                "expected UnsupportedOperation for {target:?} on {non_torus:?}"
            );
        }
    }
}

#[test]
fn test_manifold_dispatcher_rejections_and_boundary_cases() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();
    let ml = make_manifold_layout::<f64>(ManifoldKind::HyperbolicLorentz, 1.0);

    for &target in &[
        MetricTarget::CrossingNumber,
        MetricTarget::CrossingEdges,
        MetricTarget::CrossingAngle,
        MetricTarget::SampledCrossingNumber,
        MetricTarget::Crosslessness,
        MetricTarget::SampledCrosslessness,
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
                    &ml,
                    &[MetricTarget::ScaleNormalizedStress, target],
                    &cfg
                ),
                Err(CoreError::UnsupportedOperation(_))
            ),
            "expected UnsupportedOperation for {target:?} on ManifoldLayout2d"
        );
    }

    // GeodesicDistortion is rejected on Layout2d and LayoutNd.
    let l2d = Layout2d::<f64>::init(6, 7, 1.0).unwrap();
    assert!(matches!(
        evaluate_metrics(&g, &l2d, &[MetricTarget::GeodesicDistortion], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
    let lnd = LayoutNd::<f64>::init(6, 3, 7, 1.0).unwrap();
    assert!(matches!(
        evaluate_metrics(&g, &lnd, &[MetricTarget::GeodesicDistortion], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // Empty (n=0), singleton (n=1), and edgeless (n=4) graphs succeed cleanly.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let m0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0);
    let r0 = evaluate_metrics(&g0, &m0, &MANIFOLD_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(r0.geodesic_distortion, Some(0.0));
    assert_eq!(r0.scale_normalized_stress, Some(0.0));
    assert_eq!(r0.layout_degeneracy, Some(0.0));

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let m1 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 1);
    let r1 = evaluate_metrics(&g1, &m1, &MANIFOLD_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(r1.geodesic_distortion, Some(0.0));
    assert_eq!(r1.scale_normalized_stress, Some(0.0));

    let g4 = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let m4 = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 4, 42, 0.5).unwrap();
    let r4 = evaluate_metrics(&g4, &m4, &MANIFOLD_BASE_TARGETS, &cfg).unwrap();
    assert_eq!(r4.geodesic_distortion, Some(0.0));
    assert!(r4.node_resolution.unwrap().is_finite());

    // Node count mismatch -> DimensionMismatch.
    let m_mismatch = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 5);
    assert!(matches!(
        evaluate_metrics(&g, &m_mismatch, &[MetricTarget::GeodesicDistortion], &cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Non-finite coordinates are rejected on ManifoldLayout2d construction.
    assert!(matches!(
        ManifoldLayout2d::<f64>::from_xy(
            ManifoldKind::HyperbolicLorentz,
            vec![0.0, f64::NAN, 0.2, 0.3, 0.4, 0.5],
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Config validation errors.
    let bad_samples = MetricConfig {
        crossing_num_samples: 0,
        ..cfg
    };
    assert!(matches!(
        evaluate_metrics(&g, &ml, &[MetricTarget::SampledNp1], &bad_samples),
        Err(CoreError::InvalidInput(_))
    ));
    let bad_iel = MetricConfig {
        ideal_edge_length: 0.0,
        ..cfg
    };
    assert!(matches!(
        evaluate_metrics(&g, &ml, &[MetricTarget::IdealEdgeLengths], &bad_iel),
        Err(CoreError::InvalidInput(_))
    ));
}
