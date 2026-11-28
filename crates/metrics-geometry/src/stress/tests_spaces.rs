use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    ManifoldKind, ManifoldLayout2d, MetricSpace,
};

use super::*;

struct CustomMetricCloud {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for CustomMetricCloud {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self
            .pts
            .iter()
            .all(|p| p.iter().all(|coord| coord.is_finite()))
        {
            Ok(())
        } else {
            Err(CoreError::non_finite("CustomMetricCloud"))
        }
    }
}

impl MetricSpace for CustomMetricCloud {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.pts[i];
        let b = self.pts[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[test]
fn test_stress_2d_3d_nd_custom_metric_parity_and_unit_graph() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();
    let g_unit = CsrGraph::<()>::from_unweighted_edges(4, &edges, false, false).unwrap();
    let l2 = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let flat = l2.to_interleaved();
    let lnd2 = LayoutNd::from_flat(4, 2, &flat).unwrap();

    assert_eq!(
        scale_normalized_stress(&g, &l2).unwrap().to_bits(),
        scale_normalized_stress(&g, &lnd2).unwrap().to_bits()
    );
    assert_eq!(
        scale_normalized_stress(&g_unit, &l2).unwrap().to_bits(),
        scale_normalized_stress(&g, &l2).unwrap().to_bits()
    );
    assert_eq!(
        scale_normalized_stress_detail(&g, &l2).unwrap(),
        scale_normalized_stress_detail(&g, &lnd2).unwrap()
    );
    assert_eq!(
        sampled_scale_normalized_stress(&g, &l2, 2, 42)
            .unwrap()
            .to_bits(),
        sampled_scale_normalized_stress(&g, &lnd2, 2, 42)
            .unwrap()
            .to_bits()
    );

    // Layout3d vs LayoutNd(dims=3) vs CustomMetricCloud parity.
    let l3 = Layout3d::from_xyz(
        vec![0.0, 1.2, -0.5, 0.8],
        vec![0.0, -0.3, 1.1, 0.9],
        vec![0.2, 0.7, -0.4, 1.5],
    )
    .unwrap();
    let mut flat3 = Vec::with_capacity(12);
    let mut custom_pts = Vec::with_capacity(4);
    for i in 0..4 {
        flat3.extend_from_slice(&[l3.x()[i], l3.y()[i], l3.z()[i]]);
        custom_pts.push([l3.x()[i], l3.y()[i], l3.z()[i]]);
    }
    let lnd3 = LayoutNd::from_flat(4, 3, &flat3).unwrap();
    let custom = CustomMetricCloud { pts: custom_pts };

    let s_l3 = scale_normalized_stress(&g, &l3).unwrap();
    let s_lnd3 = scale_normalized_stress(&g, &lnd3).unwrap();
    let s_cust = scale_normalized_stress(&g_unit, &custom).unwrap();
    assert!((s_l3 - s_lnd3).abs() < 1e-14);
    assert!((s_l3 - s_cust).abs() < 1e-14);

    let dists = all_pairs_bfs(&g_unit, 1.0).unwrap();
    assert!(
        (scale_normalized_stress_from_distances(&l3, &dists, None).unwrap()
            - scale_normalized_stress_from_distances(&custom, &dists, None).unwrap())
        .abs()
            < 1e-14
    );
    let pairs = vec![(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (0, 2, 1.0, 1.0)];
    assert!(
        (scale_normalized_stress_from_pairs(&l3, &pairs)
            .unwrap()
            .normalized_stress
            - scale_normalized_stress_from_pairs(&custom, &pairs)
                .unwrap()
                .normalized_stress)
            .abs()
            < 1e-14
    );

    let flat32: Vec<f32> = flat.iter().map(|&v| v as f32).collect();
    let lnd32 = LayoutNd::from_flat(4, 2, &flat32).unwrap();
    let g32 = CsrGraph::<f32>::from_edges(4, &edges, None, false, false).unwrap();
    let d32 = scale_normalized_stress(&g32, &lnd32).unwrap();
    assert!((d32 - scale_normalized_stress(&g, &lnd2).unwrap()).abs() < 1e-6);
}

#[test]
fn test_stress_multi_dims_isometry_and_scale_invariance() {
    for dims in [1usize, 3, 4, 8, 10] {
        let g = CsrGraph::<f64>::from_edges(
            12,
            &(0..11).map(|i| (i, i + 1)).collect::<Vec<_>>(),
            None,
            false,
            false,
        )
        .unwrap();
        let l1 = LayoutNd::<f64>::init(12, dims, 77 + dims as u64, 1.0).unwrap();
        let mut scaled_coords = l1.coords().to_vec();
        for v in &mut scaled_coords {
            *v *= 37.5;
        }
        let l2 = LayoutNd::from_flat(12, dims, &scaled_coords).unwrap();
        let e1 = scale_normalized_stress_detail(&g, &l1).unwrap();
        let e2 = scale_normalized_stress_detail(&g, &l2).unwrap();
        assert!((e1.normalized_stress - e2.normalized_stress).abs() < 1e-12);
        assert!((e1.alpha_star - 37.5 * e2.alpha_star).abs() < 1e-10);

        let full = scale_normalized_stress(&g, &l1).unwrap();
        let sampled_all = sampled_scale_normalized_stress(&g, &l1, 12, 99).unwrap();
        assert!((full - sampled_all).abs() < 1e-12);

        let dists = all_pairs_bfs(&g, 1.0).unwrap();
        let from_d = scale_normalized_stress_from_distances(&l1, &dists, None).unwrap();
        assert!((full - from_d).abs() < 1e-12);
    }

    let path =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], None, false, false)
            .unwrap();
    let line = LayoutNd::from_flat(5, 1, &[0.0, 2.0, 4.0, 6.0, 8.0]).unwrap();
    let ev = scale_normalized_stress_detail(&path, &line).unwrap();
    assert!(ev.normalized_stress < 1e-12);
    assert!((ev.alpha_star - 0.5).abs() < 1e-12);

    let tri =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let tri3d = Layout3d::from_xyz(
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    )
    .unwrap();
    assert!(scale_normalized_stress(&tri, &tri3d).unwrap() < 1e-12);
}

#[test]
fn test_stress_spaces_boundaries_and_error_guards() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(scale_normalized_stress(&g, &l).unwrap(), 0.0);
        assert_eq!(sampled_scale_normalized_stress(&g, &l, 4, 1).unwrap(), 0.0);
        let d = vec![0.0_f64; n * n];
        assert_eq!(
            scale_normalized_stress_from_distances(&l, &d, None).unwrap(),
            0.0
        );
        assert_eq!(
            scale_normalized_stress_from_pairs(&l, &[])
                .unwrap()
                .normalized_stress,
            0.0
        );
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    assert_eq!(scale_normalized_stress(&g_edgeless, &l4).unwrap(), 0.0);
    assert_eq!(
        sampled_scale_normalized_stress(&g_edgeless, &l4, 2, 5).unwrap(),
        0.0
    );

    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l_col = LayoutNd::<f64>::zeros(2, 4).unwrap();
    assert_eq!(scale_normalized_stress(&g2, &l_col).unwrap(), 1.0);
    assert_eq!(
        sampled_scale_normalized_stress(&g2, &l_col, 2, 1).unwrap(),
        1.0
    );
    assert_eq!(
        scale_normalized_stress_from_pairs(&l_col, &[(0, 1, 2.0, 0.25)])
            .unwrap()
            .normalized_stress,
        1.0
    );

    let asym = [0.0, 1.0, 2.0, 0.0];
    assert!(matches!(
        scale_normalized_stress_from_distances(&LayoutNd::<f64>::zeros(2, 3).unwrap(), &asym, None),
        Err(CoreError::InvalidInput(_))
    ));

    let l_nan = LayoutNd::from_flat(2, 3, &[0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        scale_normalized_stress(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let l_inf = LayoutNd::from_flat(2, 3, &[0.0, 0.0, f64::INFINITY, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        sampled_scale_normalized_stress(&g2, &l_inf, 1, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        scale_normalized_stress(&g2, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sampled_scale_normalized_stress(&g2, &LayoutNd::<f64>::zeros(2, 3).unwrap(), 0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        scale_normalized_stress_from_pairs(&l4, &[(0, 4, 1.0, 1.0)]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    let mut rng = DeterministicRng::from_seed_u64(11);
    let coords32: Vec<f32> = (0..18).map(|_| rng.next_f64() as f32).collect();
    let l6_32 = LayoutNd::from_flat(6, 3, &coords32).unwrap();
    let g6_32 = CsrGraph::<f32>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let s1 = sampled_scale_normalized_stress(&g6_32, &l6_32, 3, 19).unwrap();
    let s2 = sampled_scale_normalized_stress(&g6_32, &l6_32, 3, 19).unwrap();
    assert_eq!(s1.to_bits(), s2.to_bits());
}

#[test]
fn geodesic_distortion_exact_and_no_alpha_rescaling() {
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 0.25, 0.5],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    torus.set_torus_extents(2.0, 2.0).unwrap();
    let gd_half = geodesic_distortion(&g, &torus).unwrap();
    assert!((gd_half - 0.5).abs() < 1e-12, "got {gd_half}");

    torus.set_torus_extents(4.0, 4.0).unwrap();
    let gd_exact = geodesic_distortion(&g, &torus).unwrap();
    assert!(gd_exact < 1e-12, "got {gd_exact}");
    assert!(scale_normalized_stress(&g, &torus).unwrap() < 1e-12);
}

#[test]
fn geodesic_distortion_weighted_directed_and_edge_cases() {
    let gw =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[2.0, 3.0]), false, false).unwrap();
    let mut lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![0.0, 0.5, 1.5],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    let d01 = lor.geodesic_distance(0, 1).unwrap();
    let d12 = lor.geodesic_distance(1, 2).unwrap();
    let d02 = lor.geodesic_distance(0, 2).unwrap();
    let expected =
        ((d01 - 2.0).abs() / 2.0 + (d12 - 3.0).abs() / 3.0 + (d02 - 5.0).abs() / 5.0) / 3.0;
    let got = geodesic_distortion(&gw, &lor).unwrap();
    assert!((got - expected).abs() < 1e-12);

    lor.set_scale(2.5).unwrap();
    let expected_s = ((2.5 * d01 - 2.0).abs() / 2.0
        + (2.5 * d12 - 3.0).abs() / 3.0
        + (2.5 * d02 - 5.0).abs() / 5.0)
        / 3.0;
    let got_s = geodesic_distortion(&gw, &lor).unwrap();
    assert!((got_s - expected_s).abs() < 1e-12);

    let gd = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (2, 0)],
        Some(&[1.0, 2.0, 4.0]),
        true,
        false,
    )
    .unwrap();
    let sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 3, 7, 1.0).unwrap();
    let got_dir = geodesic_distortion(&gd, &sph).unwrap();
    let d_true = [[0.0, 1.0, 3.0], [6.0, 0.0, 2.0], [4.0, 5.0, 0.0]];
    let mut manual = 0.0;
    for (i, row) in d_true.iter().enumerate() {
        for (j, &d_ij) in row.iter().enumerate() {
            if i != j {
                let r = sph.geodesic_distance(i, j).unwrap();
                manual += (r - d_ij).abs() / d_ij;
            }
        }
    }
    assert!((got_dir - manual / 6.0).abs() < 1e-12);

    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 0);
    assert_eq!(geodesic_distortion(&empty_g, &empty_l).unwrap(), 0.0);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 4, 1, 1.0).unwrap();
    assert_eq!(geodesic_distortion(&edgeless, &l4).unwrap(), 0.0);
}

#[test]
fn sns_manifold_scale_invariance_and_all_kinds() {
    let g = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        None,
        false,
        false,
    )
    .unwrap();
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 5, 11, 0.5).unwrap();
        let ev1 = scale_normalized_stress_detail(&g, &ml).unwrap();
        assert!(ev1.normalized_stress.is_finite());
        assert_eq!(ev1.pair_count, 10);

        ml.set_scale(3.5).unwrap();
        let ev2 = scale_normalized_stress_detail(&g, &ml).unwrap();
        assert!((ev1.normalized_stress - ev2.normalized_stress).abs() < 1e-12);
        assert!((ev1.alpha_star - 3.5 * ev2.alpha_star).abs() < 1e-10);

        let sampled_full = sampled_scale_normalized_stress_detail(&g, &ml, 5, 42).unwrap();
        assert!((sampled_full.normalized_stress - ev2.normalized_stress).abs() < 1e-12);
        assert_eq!(sampled_full.pair_count, ev2.pair_count);

        let sampled_sub = sampled_scale_normalized_stress(&g, &ml, 2, 42).unwrap();
        assert!(sampled_sub.is_finite() && (0.0..=1.0).contains(&sampled_sub));
    }

    let ml32 = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 5, 9, 1.0).unwrap();
    let sns = scale_normalized_stress(&g, &ml32).unwrap();
    let dists = all_pairs_bfs(&g, 1.0).unwrap();
    let sns_d = scale_normalized_stress_from_distances(&ml32, &dists, None).unwrap();
    assert!((sns - sns_d).abs() < 1e-12);
}
