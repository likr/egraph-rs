use egraph_core::{CoreError, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn crossings_torus_layout_view_supports_layout2d_and_manifold() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let l2 = Layout2d::from_xy(vec![0.1, 0.4, 0.1, 0.4], vec![0.1, 0.4, 0.4, 0.1]).unwrap();
    let cx2 = crossing_edges_torus(&g, &l2).unwrap();
    assert_eq!(cx2.len(), 1);
    assert!(cx2[0].cos_sq < 1e-12);
    assert_eq!(torus_crossing_number(&g, &l2).unwrap(), 1);

    let mut ml = ManifoldLayout2d::from_layout(ManifoldKind::Torus2d, l2).unwrap();
    ml.set_torus_extents(2.0, 1.0).unwrap();
    let cx_m = crossing_edges_torus(&g, &ml).unwrap();
    assert_eq!(cx_m.len(), 1);
    assert!(cx_m[0].cos_sq > 0.1);
    assert_eq!(torus_crossing_number(&g, &ml).unwrap(), 1);

    let cx_edges = crossing_edges_torus_from_edges(&[(0, 1), (2, 3)], &ml).unwrap();
    assert_eq!(cx_edges.len(), 1);

    // Non-torus manifold rejected
    let sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 4, 1, 1.0).unwrap();
    assert!(matches!(
        crossing_edges_torus(&g, &sph),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[test]
fn crossings_spherical_and_hyperbolic_reexports() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let sph = ManifoldLayout2d::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 0.5, 0.5],
        vec![1.5, 1.5, 1.0, 2.0],
    )
    .unwrap();
    assert_eq!(spherical_crossing_number(&g, &sph).unwrap(), 1);
    assert_eq!(crossing_edges_spherical(&g, &sph).unwrap().len(), 1);

    let hyp = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![-0.5, 0.5, 0.0, 0.0],
        vec![0.0, 0.0, -0.5, 0.5],
    )
    .unwrap();
    assert_eq!(hyperbolic_crossing_number(&g, &hyp).unwrap(), 1);
    assert_eq!(crossing_edges_hyperbolic(&g, &hyp).unwrap().len(), 1);

    let lor = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-2.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, -2.0, 2.0],
    )
    .unwrap();
    assert_eq!(hyperbolic_crossing_number(&g, &lor).unwrap(), 1);
}
