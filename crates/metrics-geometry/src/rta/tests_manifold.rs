use egraph_core::{CoreError, CsrGraph, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn rta_manifold_all_kinds_and_from_landmarks() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
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
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 6, 42, 0.5).unwrap();
        let ev1 = landmark_stratified_rta_detail_manifold(&g, &ml, 4, 1000, 1, 42).unwrap();
        assert!(ev1.is_exact);
        assert!((0.0..=1.0).contains(&ev1.rta));

        ml.set_scale(3.0).unwrap();
        let rta2 = landmark_stratified_rta_manifold(&g, &ml, 4, 1000, 1, 42).unwrap();
        assert!((ev1.rta - rta2).abs() < 1e-12);
    }

    // Exact path on ordered line in HyperbolicLorentz: RTA == 1.0
    let lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![0.0, 0.5, 1.5, 3.0, 6.0, 10.0],
        vec![0.0; 6],
    )
    .unwrap();
    let ev = landmark_stratified_rta_detail_manifold(&g, &lor, 2, 1000, 1, 1).unwrap();
    assert!((ev.rta - 1.0).abs() < 1e-12);

    // From landmarks u32 & f64 parity
    let landmark_ids = vec![0, 5];
    let dists_u32 = vec![0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0];
    let dists_f64 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let ev_u32 =
        landmark_stratified_rta_from_landmarks_manifold(&lor, &landmark_ids, &dists_u32, 100, 7)
            .unwrap();
    let ev_f64 = landmark_stratified_rta_from_landmarks_f64_manifold(
        &lor,
        &landmark_ids,
        &dists_f64,
        100,
        7,
    )
    .unwrap();
    assert_eq!(ev_u32, ev_f64);
    assert!((ev_u32.rta - 1.0).abs() < 1e-12);
}

#[test]
fn rta_manifold_collapsed_and_error_cases() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let north_pole = ManifoldLayout2d::<f32>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(
        landmark_stratified_rta_manifold(&g, &north_pole, 2, 100, 1, 42).unwrap(),
        0.0
    );

    let lor = ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, 4, 5, 1.0).unwrap();
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64_manifold(
            &lor,
            &[0],
            &[0.0, 1.5, 2.0, 3.0],
            10,
            1
        ),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64_manifold(
            &lor,
            &[0],
            &[0.0, f64::NAN, 2.0, 3.0],
            10,
            1
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
