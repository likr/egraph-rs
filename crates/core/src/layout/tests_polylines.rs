use super::*;
use crate::csr::CsrGraph;
use crate::error::CoreError;

#[test]
fn test_edge_polylines_parts_validation_and_accessors() {
    let pts = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 2.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let mut poly = EdgePolylines::from_parts(&[0, 2, 4], pts).unwrap();
    assert_eq!(poly.num_edges(), 2);
    assert_eq!(poly.total_points(), 4);
    assert!(!poly.is_empty());
    assert_eq!(poly.edge_offsets(), &[0, 2, 4]);
    assert_eq!(poly.edge_point_range(0).unwrap(), 0..2);
    assert_eq!(poly.edge_point_range(1).unwrap(), 2..4);
    assert!(matches!(
        poly.edge_point_range(2),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    poly.validate_finite().unwrap();

    poly.points_mut().x_mut()[1] = 0.5;
    assert_eq!(poly.points().x()[1], 0.5);

    let (offsets, inner) = poly.clone().into_parts();
    assert_eq!(offsets, vec![0, 2, 4]);
    assert_eq!(inner.len(), 4);

    // Replace parts atomically.
    let new_pts = Layout2d::from_xy(vec![10.0, 20.0], vec![30.0, 40.0]).unwrap();
    poly.replace_parts(vec![0, 2], new_pts).unwrap();
    assert_eq!(poly.num_edges(), 1);
    assert_eq!(poly.total_points(), 2);

    // Invalid offsets rejected.
    assert!(matches!(
        EdgePolylines::from_parts(&[], Layout2d::<f64>::zeros(0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        EdgePolylines::from_parts(&[1, 1], Layout2d::<f64>::zeros(1)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        EdgePolylines::from_parts(&[0, 3, 2], Layout2d::<f64>::zeros(2)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        EdgePolylines::from_parts(&[0, 2], Layout2d::<f64>::zeros(3)),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_edge_polylines_straight_line_and_subdivision_2d_3d_nd() {
    let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();

    // 2D
    let l2 = Layout2d::from_xy(vec![0.0, 2.0, 4.0], vec![0.0, 2.0, 0.0]).unwrap();
    let p2 = EdgePolylines::straight_line_from_graph(&g, &l2).unwrap();
    assert_eq!(p2.num_edges(), 2);
    assert_eq!(p2.total_points(), 4);
    assert_eq!(p2.points().x(), &[0.0, 2.0, 2.0, 4.0]);
    assert_eq!(p2.points().y(), &[0.0, 2.0, 2.0, 0.0]);

    let sub2 = p2.with_subdivisions(1).unwrap();
    assert_eq!(sub2.edge_offsets(), &[0, 3, 6]);
    assert_eq!(sub2.points().x(), &[0.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
    assert_eq!(sub2.points().y(), &[0.0, 1.0, 2.0, 2.0, 1.0, 0.0]);
    assert_eq!(p2.with_subdivisions(0).unwrap(), p2);

    // 3D
    let l3 = Layout3d::from_xyz(
        vec![0.0, 3.0, 6.0],
        vec![0.0, 0.0, 3.0],
        vec![0.0, 6.0, 6.0],
    )
    .unwrap();
    let p3 = EdgePolylines::straight_line_from_graph(&g, &l3).unwrap();
    let sub3 = p3.with_subdivisions(2).unwrap();
    assert_eq!(sub3.edge_offsets(), &[0, 4, 8]);
    assert_eq!(sub3.points().get(1).unwrap(), [1.0, 0.0, 2.0]);
    assert_eq!(sub3.points().get(2).unwrap(), [2.0, 0.0, 4.0]);

    // ND (4D)
    let l4 = LayoutNd::from_flat(
        3,
        4,
        &[0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0],
    )
    .unwrap();
    let p4 = EdgePolylines::straight_line_from_graph(&g, &l4).unwrap();
    let sub4 = p4.with_subdivisions(1).unwrap();
    assert_eq!(sub4.points().node(1).unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    // Mismatched node count or non-finite coordinates rejected.
    let bad_len = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        EdgePolylines::straight_line_from_graph(&g, &bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let bad_nan = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        EdgePolylines::straight_line_from_graph(&g, &bad_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
