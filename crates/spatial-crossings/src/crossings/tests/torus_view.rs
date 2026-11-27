use std::f64::consts::FRAC_PI_2;

use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, Layout2d, ManifoldKind, ManifoldLayout2d,
};

use super::super::{
    CrossingPair, CrossingSummary, SweepLineCrossings, TorusCrossing, TorusLayoutView, TorusView,
    crossing_edges_hyperbolic, crossing_edges_spherical, hyperbolic_crossing_number,
    sampled_crossing_number, spherical_crossing_number, torus_crossing_edges,
    torus_crossing_edges_from_edges, torus_crossing_number, torus_layout_edge_segments,
};

#[test]
fn test_torus_view_validation_and_extents() {
    let layout = Layout2d::from_xy(vec![0.2, 0.6, 0.2, 0.6], vec![0.2, 0.6, 0.6, 0.2]).unwrap();

    // Valid construction and accessors.
    let view = TorusView::new(&layout, 2.0, 1.0).unwrap();
    assert_eq!(view.extents(), (2.0, 1.0));
    assert_eq!(view.lx(), 2.0);
    assert_eq!(view.ly(), 1.0);
    assert_eq!(view.layout().len(), 4);
    assert_eq!(view.torus_extents().unwrap(), (2.0, 1.0));
    assert_eq!(view.torus_layout().unwrap().len(), 4);

    // Physical crossing angle matches ManifoldLayout2d with (2.0, 1.0) extents.
    let g = CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (2, 3)], false, false).unwrap();
    let crossings = torus_crossing_edges(&g, &view).unwrap();
    assert_eq!(crossings.len(), 1);
    assert!((crossings[0].cos_sq - 0.36).abs() < 1e-12);

    // Non-positive extents rejected as InvalidInput.
    for (lx, ly) in [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, -0.5),
        (-2.0, -3.0),
    ] {
        assert!(
            matches!(
                TorusView::new(&layout, lx, ly),
                Err(CoreError::InvalidInput(_))
            ),
            "expected InvalidInput for extents ({lx}, {ly})"
        );
    }

    // Non-finite extents rejected as NonFiniteValue.
    for (lx, ly) in [
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 1.0),
        (1.0, f64::NEG_INFINITY),
    ] {
        assert!(
            matches!(
                TorusView::new(&layout, lx, ly),
                Err(CoreError::NonFiniteValue { .. })
            ),
            "expected NonFiniteValue for extents ({lx}, {ly})"
        );
    }

    // Non-finite layout coordinates rejected on TorusView::new.
    let nan_layout = Layout2d::from_xy(vec![0.1, f64::NAN], vec![0.2, 0.3]).unwrap();
    assert!(matches!(
        TorusView::new(&nan_layout, 1.0, 1.0),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let inf_layout = Layout2d::from_xy(vec![0.1, 0.2], vec![f64::INFINITY, 0.3]).unwrap();
    assert!(matches!(
        TorusView::new(&inf_layout, 1.0, 1.0),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_unweighted_csr_unit_weight_and_clone_crossing_structs() {
    let edges = [(0, 1), (2, 3)];
    let g_unit = CsrGraph::<()>::from_unweighted_edges(4, &edges, false, false).unwrap();

    // 1. Euclidean SweepLineCrossings and sampling with CsrGraph<()>.
    let euc = Layout2d::from_xy(vec![-1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -1.0, 1.0]).unwrap();
    assert_eq!(SweepLineCrossings::count(&g_unit, &euc).unwrap(), 1);

    let summary: CrossingSummary = SweepLineCrossings::summarize(&g_unit, &euc).unwrap();
    let summary_cloned = summary.clone();
    assert_eq!(summary, summary_cloned);
    assert_eq!(summary.count, 1);
    assert!((summary.min_angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

    let pairs: Vec<CrossingPair> = SweepLineCrossings::find_crossings(&g_unit, &euc).unwrap();
    assert_eq!(pairs.len(), 1);
    let pair_cloned = pairs[0].clone();
    assert_eq!(pairs[0], pair_cloned);

    let mut rng = DeterministicRng::from_seed_u64(42);
    let sampled = sampled_crossing_number(&g_unit, &euc, 10, &mut rng).unwrap();
    assert_eq!(sampled, 1.0);

    // 2. Torus crossing functions with CsrGraph<()> and TorusView<f32>.
    let tor_layout32 =
        Layout2d::<f32>::from_xy(vec![0.9, 0.1, 0.9, 0.1], vec![0.1, 0.9, 0.9, 0.1]).unwrap();
    let tor_view32 = TorusView::new(&tor_layout32, 1.5, 1.5).unwrap();
    assert_eq!(torus_crossing_number(&g_unit, &tor_view32).unwrap(), 1);
    let tor_pairs: Vec<TorusCrossing> = torus_crossing_edges(&g_unit, &tor_view32).unwrap();
    assert_eq!(tor_pairs.len(), 1);
    assert_eq!(tor_pairs[0], tor_pairs[0].clone());
    assert_eq!(
        torus_crossing_edges_from_edges(&edges, &tor_view32)
            .unwrap()
            .len(),
        1
    );
    assert!(
        !torus_layout_edge_segments(&g_unit, &tor_view32)
            .unwrap()
            .is_empty()
    );

    // 3. Spherical & Hyperbolic crossing functions with CsrGraph<()>.
    let sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 0.5, 0.5],
        vec![FRAC_PI_2, FRAC_PI_2, 1.0, 2.0],
    )
    .unwrap();
    assert_eq!(spherical_crossing_number(&g_unit, &sph).unwrap(), 1);
    assert_eq!(crossing_edges_spherical(&g_unit, &sph).unwrap().len(), 1);

    let hyp = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![-0.5, 0.5, 0.0, 0.0],
        vec![0.0, 0.0, -0.5, 0.5],
    )
    .unwrap();
    assert_eq!(hyperbolic_crossing_number(&g_unit, &hyp).unwrap(), 1);
    assert_eq!(crossing_edges_hyperbolic(&g_unit, &hyp).unwrap().len(), 1);
}
