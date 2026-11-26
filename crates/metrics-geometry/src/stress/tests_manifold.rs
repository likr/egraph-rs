use egraph_core::{CoreError, CsrGraph, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn geodesic_distortion_exact_and_no_alpha_rescaling() {
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    // On Torus2d with lx = 2.0, ly = 2.0: points at x = 0.0, 0.25, 0.5 have pairwise
    // distances d(0,1)=0.5, d(1,2)=0.5, d(0,2)=1.0 vs graph D = 1, 1, 2.
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 0.25, 0.5],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    torus.set_torus_extents(2.0, 2.0).unwrap();
    let gd_half = geodesic_distortion(&g, &torus).unwrap();
    assert!((gd_half - 0.5).abs() < 1e-12, "got {gd_half}");

    // Doubling scale to 4.0 makes d_phys == D exactly -> distortion drops to 0.0
    // (proving no post-hoc alpha* rescaling is applied, unlike SNS which is 0.0 at both scales).
    torus.set_torus_extents(4.0, 4.0).unwrap();
    let gd_exact = geodesic_distortion(&g, &torus).unwrap();
    assert!(gd_exact < 1e-12, "got {gd_exact}");
    assert!(scale_normalized_stress_manifold(&g, &torus).unwrap() < 1e-12);
}

#[test]
fn geodesic_distortion_weighted_directed_and_edge_cases() {
    // Weighted path 0 --(2.0)-- 1 --(3.0)-- 2 (D_02 = 5.0)
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
    assert!(
        (got - expected).abs() < 1e-12,
        "got {got}, expected {expected}"
    );

    lor.set_scale(2.5).unwrap();
    let d01_s = lor.geodesic_distance(0, 1).unwrap();
    assert!((d01_s - 2.5 * d01).abs() < 1e-12);
    let expected_s = ((2.5 * d01 - 2.0).abs() / 2.0
        + (2.5 * d12 - 3.0).abs() / 3.0
        + (2.5 * d02 - 5.0).abs() / 5.0)
        / 3.0;
    let got_s = geodesic_distortion(&gw, &lor).unwrap();
    assert!((got_s - expected_s).abs() < 1e-12);

    // Directed graph with asymmetric paths: 0 -> 1 (1), 1 -> 2 (2), 2 -> 0 (4)
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
    // Manually check all 6 ordered pairs (i != j)
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

    // Edgeless, singleton, empty -> 0.0
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
        let ev1 = scale_normalized_stress_detail_manifold(&g, &ml).unwrap();
        assert!(ev1.normalized_stress.is_finite());
        assert_eq!(ev1.pair_count, 10);

        ml.set_scale(3.5).unwrap();
        let ev2 = scale_normalized_stress_detail_manifold(&g, &ml).unwrap();
        assert!(
            (ev1.normalized_stress - ev2.normalized_stress).abs() < 1e-12,
            "{kind:?}: sns changed across conformal scale"
        );
        assert!(
            (ev1.alpha_star - 3.5 * ev2.alpha_star).abs() < 1e-10,
            "{kind:?}: alpha_star did not scale inversely"
        );

        let sampled_full = sampled_scale_normalized_stress_detail_manifold(&g, &ml, 5, 42).unwrap();
        assert!((sampled_full.normalized_stress - ev2.normalized_stress).abs() < 1e-12);
        assert_eq!(sampled_full.pair_count, ev2.pair_count);

        let sampled_sub = sampled_scale_normalized_stress_manifold(&g, &ml, 2, 42).unwrap();
        assert!(sampled_sub.is_finite() && (0.0..=1.0).contains(&sampled_sub));
    }
}

#[test]
fn sns_manifold_from_distances_pairs_and_f32() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let ml = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 4, 9, 1.0).unwrap();
    let sns = scale_normalized_stress_manifold(&g, &ml).unwrap();
    let dists = all_pairs_bfs(&g, 1.0).unwrap();
    let sns_d = scale_normalized_stress_from_distances_manifold(&ml, &dists, None).unwrap();
    assert!((sns - sns_d).abs() < 1e-12);

    let pairs = vec![(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (2, 3, 1.0, 1.0)];
    let ev_p = scale_normalized_stress_from_pairs_manifold(&ml, &pairs).unwrap();
    assert_eq!(ev_p.pair_count, 3);
    assert!(ev_p.normalized_stress.is_finite());

    let bad_pairs = vec![(0, 99, 1.0, 1.0)];
    assert!(matches!(
        scale_normalized_stress_from_pairs_manifold(&ml, &bad_pairs),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    let neg_w = vec![(0, 1, 1.0, -1.0)];
    assert!(matches!(
        scale_normalized_stress_from_pairs_manifold(&ml, &neg_w),
        Err(CoreError::InvalidInput { .. })
    ));
}
