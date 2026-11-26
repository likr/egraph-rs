use egraph_core::{CoreError, CsrGraph, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn neighborhood_manifold_torus_aspect_and_exact_vs_sampled() {
    // 4-node graph: 0-1 and 2-3.
    // On Torus2d, place 0=(0.1, 0.1), 1=(0.1, 0.35), 2=(0.25, 0.1), 3=(0.25, 0.35).
    // With square (1,1): dx=0.15 < dy=0.25 -> nearest to 0 is 2 (wrong), NP1 = 0.0.
    // With tall torus (lx=4.0, ly=1.0): phys dx=0.60 > phys dy=0.25 -> nearest to 0 is 1 (exact!), NP1 = 1.0.
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.25, 0.25],
        vec![0.1, 0.35, 0.1, 0.35],
    )
    .unwrap();
    assert_eq!(
        neighborhood_preservation_1hop_manifold(&g, &torus).unwrap(),
        0.0
    );

    torus.set_torus_extents(4.0, 1.0).unwrap();
    assert_eq!(
        neighborhood_preservation_1hop_manifold(&g, &torus).unwrap(),
        1.0
    );
    assert_eq!(
        neighborhood_preservation_2hop_manifold(&g, &torus).unwrap(),
        1.0
    );

    // Sampled >= N matches exact bit-for-bit.
    assert_eq!(
        sampled_neighborhood_preservation_1hop_manifold(&g, &torus, 4, 42).unwrap(),
        1.0
    );
    assert_eq!(
        sampled_neighborhood_preservation_2hop_manifold(&g, &torus, 10, 42).unwrap(),
        1.0
    );
}

#[test]
fn neighborhood_manifold_all_kinds_and_boundaries() {
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
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 6, 42, 0.6).unwrap();
        let np1 = neighborhood_preservation_1hop_manifold(&g, &ml).unwrap();
        let np2 = neighborhood_preservation_2hop_manifold(&g, &ml).unwrap();
        assert!((0.0..=1.0).contains(&np1));
        assert!((0.0..=1.0).contains(&np2));

        // Conformal scaling preserves neighborhood order and exact NP1/NP2.
        ml.set_scale(3.0).unwrap();
        assert_eq!(
            neighborhood_preservation_1hop_manifold(&g, &ml).unwrap(),
            np1
        );
        assert_eq!(
            neighborhood_preservation_2hop_manifold(&g, &ml).unwrap(),
            np2
        );

        let s1 = sampled_neighborhood_preservation_1hop_manifold(&g, &ml, 6, 7).unwrap();
        let s2 = sampled_neighborhood_preservation_2hop_manifold(&g, &ml, 6, 7).unwrap();
        assert_eq!(s1, np1);
        assert_eq!(s2, np2);

        let sub1 = sampled_neighborhood_preservation_1hop_manifold(&g, &ml, 3, 7).unwrap();
        let sub2 = sampled_neighborhood_preservation_2hop_manifold(&g, &ml, 3, 7).unwrap();
        assert!((0.0..=1.0).contains(&sub1));
        assert!((0.0..=1.0).contains(&sub2));
    }

    // Error on num_samples == 0
    let ml = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 6, 42, 1.0).unwrap();
    assert!(matches!(
        sampled_neighborhood_preservation_1hop_manifold(&g, &ml, 0, 1),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        sampled_neighborhood_preservation_2hop_manifold(&g, &ml, 0, 1),
        Err(CoreError::InvalidInput { .. })
    ));
}

#[test]
fn neighborhood_manifold_tie_breaking_and_isolated() {
    // 4-node graph with tied distances from node 0 to {1, 2, 3}:
    // deterministic index tie-breaking must pick node 1 over 2, 3.
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let mut buf = Vec::new();
    k_nearest_manifold_excluding_self_into(&sph, 0, 1, &mut buf).unwrap();
    assert_eq!(buf.len(), 1);
    assert_eq!(buf[0].0, 1);
    let np1 = neighborhood_preservation_1hop_manifold(&g, &sph).unwrap();
    assert!((0.0..=1.0).contains(&np1));

    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 0);
    assert_eq!(
        neighborhood_preservation_1hop_manifold(&empty_g, &empty_l).unwrap(),
        0.0
    );
    assert_eq!(
        neighborhood_preservation_2hop_manifold(&empty_g, &empty_l).unwrap(),
        0.0
    );
}
