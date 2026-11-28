use super::super::*;
use crate::{
    ClusterAreaConfig, angular_resolution, crossing_edges_torus, gabriel_graph_preservation,
    gabriel_graph_property, geodesic_distortion, ideal_edge_lengths_scaled,
    landmark_stratified_rta, layout_degeneracy, neighborhood_preservation_1hop,
    neighborhood_preservation_2hop, node_resolution, relative_neighborhood_preservation,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    sampled_scale_normalized_stress, scale_normalized_stress,
};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, MetricSpace, Scalar,
};

const GENERAL_METRIC_TARGETS: [MetricTarget; 13] = [
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
    MetricConfig::new()
        .with_sns_num_pivots(3)
        .with_sns_seed(11)
        .with_crossing_num_samples(64)
        .with_crossing_seed(11)
        .with_np_num_samples(64)
        .with_np_seed(11)
        .with_ideal_edge_length(1.5)
        .with_rta_num_landmarks(4)
        .with_rta_num_triplets(4096)
        .with_rta_min_degree(1)
        .with_rta_seed(11)
        .with_patch_nig_radius(8)
        .with_patch_nig_canvas_size(256)
        .with_patch_nig_pad_frac(0.04)
        .with_patch_nig_seed(11)
        .with_cluster_area(ClusterAreaConfig::new())
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

fn assert_general_targets_parity<W: Scalar, L: MetricSpace + 'static>(
    g: &CsrGraph<W>,
    layout: &L,
    cfg: &MetricConfig,
) {
    let rep = evaluate_metrics(g, layout, &GENERAL_METRIC_TARGETS, cfg).unwrap();
    assert!(rep.baseline_metrics().is_none());
    assert_eq!(
        rep.scale_normalized_stress,
        Some(scale_normalized_stress(g, layout).unwrap())
    );
    assert_eq!(
        rep.sampled_scale_normalized_stress,
        Some(sampled_scale_normalized_stress(g, layout, cfg.sns_num_pivots, cfg.sns_seed).unwrap())
    );
    assert_eq!(
        rep.neighborhood_preservation_1hop,
        Some(neighborhood_preservation_1hop(g, layout).unwrap())
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop(g, layout).unwrap())
    );
    assert_eq!(
        rep.sampled_np1,
        Some(
            sampled_neighborhood_preservation_1hop(g, layout, cfg.np_num_samples, cfg.np_seed)
                .unwrap()
        )
    );
    assert_eq!(
        rep.sampled_np2,
        Some(
            sampled_neighborhood_preservation_2hop(g, layout, cfg.np_num_samples, cfg.np_seed)
                .unwrap()
        )
    );
    assert_eq!(
        rep.gabriel_graph_property,
        Some(gabriel_graph_property(g, layout).unwrap())
    );
    assert_eq!(rep.node_resolution, Some(node_resolution(layout).unwrap()));
    assert_eq!(
        rep.relative_neighborhood_preservation,
        Some(relative_neighborhood_preservation(g, layout).unwrap())
    );
    assert_eq!(
        rep.gabriel_graph_preservation,
        Some(gabriel_graph_preservation(g, layout).unwrap())
    );
    let deg = layout_degeneracy(g, layout).unwrap();
    let bbox_pen = if deg.is_bbox_degenerate && layout.len() > 1 {
        1.0
    } else {
        0.0
    };
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
        Some(ideal_edge_lengths_scaled(g, layout, cfg.ideal_edge_length).unwrap())
    );
    assert_eq!(
        rep.landmark_stratified_rta,
        Some(
            landmark_stratified_rta(
                g,
                layout,
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
fn test_euclidean_2d_3d_nd_parity_and_selectivity() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();

    // 2D parity between Layout2d and dims == 2 LayoutNd.
    let xs = vec![0.0, 0.4, 0.2, 4.0, 4.4, 4.2];
    let ys = vec![0.0, 0.1, -0.3, 4.0, 4.1, 3.7];
    let layout2d = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let mut flat2 = Vec::with_capacity(12);
    for i in 0..6 {
        flat2.push(xs[i]);
        flat2.push(ys[i]);
    }
    let layout_nd2 = LayoutNd::from_flat(6, 2, &flat2).unwrap();
    assert_eq!(
        evaluate_metrics(&g, &layout_nd2, &[], &cfg).unwrap(),
        MetricReport::default()
    );
    assert_eq!(
        evaluate_metrics(&g, &layout2d, &GENERAL_METRIC_TARGETS, &cfg).unwrap(),
        evaluate_metrics(&g, &layout_nd2, &GENERAL_METRIC_TARGETS, &cfg).unwrap()
    );
    assert_eq!(
        evaluate_metrics(&g, &layout2d, &[MetricTarget::AngularResolution], &cfg)
            .unwrap()
            .angular_resolution,
        evaluate_metrics(&g, &layout_nd2, &[MetricTarget::AngularResolution], &cfg)
            .unwrap()
            .angular_resolution
    );

    // 3D parity between Layout3d and dims == 3 LayoutNd (both f64 and f32).
    let x3 = vec![0.0, 0.4, 0.2, 5.0, 5.4, 5.2];
    let y3 = vec![0.0, 0.1, -0.3, 5.0, 5.1, 4.7];
    let z3 = vec![0.0, 0.2, 0.4, 5.0, 5.2, 5.4];
    let l3_64 = Layout3d::from_xyz(x3.clone(), y3.clone(), z3.clone()).unwrap();
    let l3_32 = Layout3d::from_xyz(
        x3.iter().map(|&v| v as f32).collect(),
        y3.iter().map(|&v| v as f32).collect(),
        z3.iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut flat3 = Vec::with_capacity(18);
    for i in 0..6 {
        flat3.push(x3[i]);
        flat3.push(y3[i]);
        flat3.push(z3[i]);
    }
    let lnd3 = LayoutNd::from_flat(6, 3, &flat3).unwrap();
    assert_general_targets_parity(&g, &l3_64, &cfg);
    assert_general_targets_parity(&g, &l3_32, &cfg);
    assert_eq!(
        evaluate_metrics(&g, &l3_64, &GENERAL_METRIC_TARGETS, &cfg).unwrap(),
        evaluate_metrics(&g, &lnd3, &GENERAL_METRIC_TARGETS, &cfg).unwrap()
    );
    assert_eq!(
        evaluate_metrics(&g, &l3_64, &[MetricTarget::AngularResolution], &cfg)
            .unwrap()
            .angular_resolution,
        Some(angular_resolution(&g, &l3_64).unwrap())
    );

    // Multi-dimensional LayoutNd (1D, 4D, 8D, 10D) f64 and f32 parity.
    for &dims in &[1usize, 4, 8, 10] {
        let mut flat64 = Vec::with_capacity(6 * dims);
        for i in 0..6 {
            let cluster_shift = if i < 3 { 0.0 } else { 5.0 };
            for d in 0..dims {
                flat64.push(cluster_shift + ((i * 7 + d * 3) as f64) * 0.13);
            }
        }
        let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
        let l64 = LayoutNd::from_flat(6, dims, &flat64).unwrap();
        let l32 = LayoutNd::from_flat(6, dims, &flat32).unwrap();
        assert_general_targets_parity(&g, &l64, &cfg);
        assert_general_targets_parity(&g, &l32, &cfg);
    }

    // 2D-only targets and GeodesicDistortion rejected on Layout3d / LayoutNd.
    for target in [
        MetricTarget::GeodesicDistortion,
        MetricTarget::CrossingNumber,
        MetricTarget::CrossingEdges,
        MetricTarget::CrossingAngle,
        MetricTarget::SampledCrossingNumber,
        MetricTarget::Crosslessness,
        MetricTarget::SampledCrosslessness,
        MetricTarget::TorusCrossingNumber,
        MetricTarget::CrossingEdgesTorus,
        MetricTarget::AspectRatio,
        MetricTarget::MinimumAngle,
        MetricTarget::MinimumAngleA1,
        MetricTarget::PrismDistortion,
        MetricTarget::AdjustedPatchNig,
        MetricTarget::ClusterAreaFaithfulness,
    ] {
        assert!(matches!(
            evaluate_metrics(&g, &l3_64, &[target], &cfg),
            Err(CoreError::UnsupportedOperation(_))
        ));
        assert!(matches!(
            evaluate_metrics(&g, &lnd3, &[target], &cfg),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }
}

#[test]
fn test_manifold_layouts_all_kinds_and_torus_crossings() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();

    for &kind in &[
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut m64 = ManifoldLayout2d::<f64>::init(kind, 6, 42, 0.3).unwrap();
        m64.set_scale(2.0).unwrap();
        assert_general_targets_parity(&g, &m64, &cfg);
        let gd_rep = evaluate_metrics(
            &g,
            &m64,
            &[
                MetricTarget::GeodesicDistortion,
                MetricTarget::AngularResolution,
            ],
            &cfg,
        )
        .unwrap();
        assert_eq!(
            gd_rep.geodesic_distortion,
            Some(geodesic_distortion(&g, &m64).unwrap())
        );
        assert_eq!(
            gd_rep.angular_resolution,
            Some(angular_resolution(&g, &m64).unwrap())
        );

        let mut m32 = ManifoldLayout2d::<f32>::init(kind, 6, 42, 0.3).unwrap();
        m32.set_scale(1.25).unwrap();
        assert_general_targets_parity(&g, &m32, &cfg);
    }

    // Unweighted CsrGraph<()> works for GeodesicDistortion as well.
    let g_unit = CsrGraph::<()>::from_unweighted_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        false,
        false,
    )
    .unwrap();
    let sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 6, 42, 0.3).unwrap();
    assert_eq!(
        evaluate_metrics(&g_unit, &sph, &[MetricTarget::GeodesicDistortion], &cfg)
            .unwrap()
            .geodesic_distortion,
        Some(geodesic_distortion(&g, &sph).unwrap())
    );

    // Torus crossings and aspect sensitivity.
    let g_cross = CsrGraph::<f64>::from_edges(4, &[(0, 2), (1, 3)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.4, 0.4],
        vec![0.1, 0.4, 0.4, 0.1],
    )
    .unwrap();
    torus.set_torus_extents(3.0, 1.5).unwrap();
    let rep_tor = evaluate_metrics(
        &g_cross,
        &torus,
        &[
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
        ],
        &cfg,
    )
    .unwrap();
    let expected_pairs = crossing_edges_torus(&g_cross, &torus).unwrap();
    assert_eq!(rep_tor.torus_crossing_number, Some(1));
    assert_eq!(rep_tor.crossing_edges_torus, Some(expected_pairs));

    // Non-torus manifolds reject TorusCrossingNumber / CrossingEdgesTorus.
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
            assert!(matches!(
                evaluate_metrics(&g_cross, &ml, &[target], &cfg),
                Err(CoreError::UnsupportedOperation(_))
            ));
        }
    }
}

/// Custom external `MetricSpace` stub verifying `evaluate_metrics` remains genuinely open.
struct SurfaceLayoutStub {
    coords: Vec<[f64; 3]>,
}

impl LayoutStorage for SurfaceLayoutStub {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.coords.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self
            .coords
            .iter()
            .all(|p| p.iter().all(|coord| coord.is_finite()))
        {
            Ok(())
        } else {
            Err(CoreError::non_finite("SurfaceLayoutStub"))
        }
    }
}

impl MetricSpace for SurfaceLayoutStub {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.coords[i];
        let b = self.coords[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[test]
fn test_external_metric_space_impl_and_boundaries() {
    let g = make_sample_graph_6();
    let cfg = test_cfg();
    let surface = SurfaceLayoutStub {
        coords: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.4, 0.9, 0.1],
            [4.0, 4.0, 1.0],
            [5.0, 4.2, 1.0],
            [4.4, 4.9, 1.1],
        ],
    };

    assert_general_targets_parity(&g, &surface, &cfg);

    // Non-MetricSpace-only targets return UnsupportedOperation cleanly.
    for target in [
        MetricTarget::AngularResolution,
        MetricTarget::GeodesicDistortion,
        MetricTarget::TorusCrossingNumber,
        MetricTarget::CrossingNumber,
        MetricTarget::AspectRatio,
    ] {
        assert!(matches!(
            evaluate_metrics(&g, &surface, &[target], &cfg),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    // Empty (n=0), singleton (n=1), edgeless (n=4), mismatch, and NaN boundaries.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let s0 = SurfaceLayoutStub { coords: vec![] };
    let r0 = evaluate_metrics(&g0, &s0, &GENERAL_METRIC_TARGETS, &cfg).unwrap();
    assert_eq!(r0.scale_normalized_stress, Some(0.0));
    assert_eq!(r0.layout_degeneracy, Some(0.0));

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let s1 = SurfaceLayoutStub {
        coords: vec![[1.0, 2.0, 3.0]],
    };
    let r1 = evaluate_metrics(&g1, &s1, &GENERAL_METRIC_TARGETS, &cfg).unwrap();
    assert_eq!(r1.scale_normalized_stress, Some(0.0));
    assert_eq!(r1.layout_degeneracy, Some(0.0));

    let g4 = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = Layout3d::<f64>::init(4, 42, 1.0).unwrap();
    let r4 = evaluate_metrics(&g4, &l4, &GENERAL_METRIC_TARGETS, &cfg).unwrap();
    assert!(r4.scale_normalized_stress.unwrap().is_finite());

    let s_mismatch = SurfaceLayoutStub {
        coords: vec![[0.0, 0.0, 0.0]; 5],
    };
    assert!(matches!(
        evaluate_metrics(
            &g,
            &s_mismatch,
            &[MetricTarget::ScaleNormalizedStress],
            &cfg
        ),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let s_nan = SurfaceLayoutStub {
        coords: vec![[0.0, 0.0, f64::NAN]; 6],
    };
    assert!(matches!(
        evaluate_metrics(&g, &s_nan, &[MetricTarget::ScaleNormalizedStress], &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
