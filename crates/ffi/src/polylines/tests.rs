use egraph_core::{EdgePolylines, Layout2d, Layout3d};

use super::*;

#[test]
fn test_validate_edge_polylines_parts_and_offsets() {
    let valid_pts = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]).unwrap();
    validate_edge_polylines_parts(&[0, 2, 3], &valid_pts).unwrap();
    validate_edge_offsets(&[0, 0, 3], 3).unwrap();

    let empty_offsets = validate_edge_polylines_parts(&[], &Layout2d::<f64>::zeros(0)).unwrap_err();
    assert_eq!(empty_offsets.kind, ErrorKind::InvalidInput);

    let nonzero_start =
        validate_edge_polylines_parts(&[1, 2], &Layout2d::<f64>::zeros(2)).unwrap_err();
    assert_eq!(nonzero_start.kind, ErrorKind::InvalidInput);

    let non_monotonic =
        validate_edge_polylines_parts(&[0, 3, 2], &Layout2d::<f64>::zeros(2)).unwrap_err();
    assert_eq!(non_monotonic.kind, ErrorKind::InvalidInput);

    let mismatched_len =
        validate_edge_polylines_parts(&[0, 2], &Layout2d::<f64>::zeros(3)).unwrap_err();
    assert_eq!(mismatched_len.kind, ErrorKind::DimensionMismatch);

    let non_finite_pts =
        Layout3d::from_xyz(vec![0.0, f64::NAN], vec![0.0, 1.0], vec![0.0, 2.0]).unwrap();
    let err_nf = validate_edge_polylines_parts(&[0, 2], &non_finite_pts).unwrap_err();
    assert_eq!(err_nf.kind, ErrorKind::NonFiniteValue);
}

#[test]
fn test_interleaved_edge_polylines_2d_roundtrip_and_replace() {
    let empty = InterleavedEdgePolylines2d::<f64>::empty();
    assert!(empty.is_empty());
    assert_eq!(empty.num_edges(), 0);
    assert_eq!(empty.total_points(), 0);
    assert_eq!(empty.edge_offsets(), &[0]);

    let mut poly = InterleavedEdgePolylines2d::<f32>::from_parts(
        &[0, 2, 4],
        &[0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0],
    )
    .unwrap();
    assert_eq!(poly.num_edges(), 2);
    assert_eq!(poly.total_points(), 4);
    assert!(!poly.is_empty());
    assert_eq!(poly.offsets_len(), 3);
    assert_eq!(poly.points_len(), 8);
    assert_eq!(poly.offsets_ptr(), poly.edge_offsets().as_ptr());
    assert_eq!(poly.points_ptr(), poly.points_as_slice().as_ptr());
    assert_eq!(
        poly.points_mut_ptr(),
        poly.points_as_mut_slice().as_mut_ptr()
    );
    assert!(poly.is_all_finite());
    poly.validate_finite().unwrap();

    let mut core_poly = poly.to_core().unwrap();
    assert_eq!(core_poly.num_edges(), 2);
    assert_eq!(core_poly.total_points(), 4);

    // Same-length replacement writes in place without changing allocations.
    let ptr_offsets_before = poly.offsets_ptr();
    let ptr_points_before = poly.points_ptr();
    core_poly.points_mut().x_mut()[1] = 4.5;
    poly.replace_from_core(&core_poly);
    assert_eq!(poly.offsets_ptr(), ptr_offsets_before);
    assert_eq!(poly.points_ptr(), ptr_points_before);
    assert_eq!(poly.points().x(1).unwrap(), 4.5);

    // Subdivision changes total points and replaces cleanly.
    let sub = core_poly.with_subdivisions(1).unwrap();
    poly.replace_from_core(&sub);
    assert_eq!(poly.edge_offsets(), &[0, 3, 6]);
    assert_eq!(poly.total_points(), 6);
    assert_eq!(InterleavedEdgePolylines2d::from_core(&sub), poly);

    // Validation failures
    assert_eq!(
        InterleavedEdgePolylines2d::<f64>::from_parts(&[0, 1], &[1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        ErrorKind::DimensionMismatch
    );
    assert_eq!(
        InterleavedEdgePolylines2d::<f64>::from_parts(&[0, 1], &[1.0, f64::INFINITY])
            .unwrap_err()
            .kind,
        ErrorKind::NonFiniteValue
    );
    assert_eq!(
        InterleavedEdgePolylines2d::<f64>::from_parts(&[0, 2], &[1.0, 2.0])
            .unwrap_err()
            .kind,
        ErrorKind::DimensionMismatch
    );

    poly.points_as_mut_slice()[0] = f32::NAN;
    assert!(!poly.is_all_finite());
    assert_eq!(
        poly.validate_finite().unwrap_err().kind,
        ErrorKind::NonFiniteValue
    );
}

#[test]
fn test_interleaved_edge_polylines_3d_roundtrip_and_replace() {
    let empty = InterleavedEdgePolylines3d::<f32>::empty();
    assert!(empty.is_empty());
    assert_eq!(empty.num_edges(), 0);
    assert_eq!(empty.total_points(), 0);
    assert_eq!(empty.edge_offsets(), &[0]);

    let mut poly = InterleavedEdgePolylines3d::<f64>::from_parts(
        &[0, 2, 5],
        &[
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    assert_eq!(poly.num_edges(), 2);
    assert_eq!(poly.total_points(), 5);
    assert_eq!(poly.offsets_len(), 3);
    assert_eq!(poly.points_len(), 15);
    assert_eq!(poly.offsets_ptr(), poly.edge_offsets().as_ptr());
    assert_eq!(poly.points_ptr(), poly.points_as_slice().as_ptr());
    assert_eq!(
        poly.points_mut_ptr(),
        poly.points_as_mut_slice().as_mut_ptr()
    );
    poly.validate_finite().unwrap();

    let mut core_poly = poly.to_core().unwrap();
    let ptr_offsets_before = poly.offsets_ptr();
    let ptr_points_before = poly.points_ptr();
    core_poly.points_mut().z_mut()[0] = -2.5;
    poly.replace_from_core(&core_poly);
    assert_eq!(poly.offsets_ptr(), ptr_offsets_before);
    assert_eq!(poly.points_ptr(), ptr_points_before);
    assert_eq!(poly.points().z(0).unwrap(), -2.5);

    let sub = core_poly.with_subdivisions(1).unwrap();
    poly.replace_from_core(&sub);
    assert_eq!(poly.edge_offsets(), &[0, 3, 8]);
    assert_eq!(poly.total_points(), 8);
    assert_eq!(InterleavedEdgePolylines3d::from_core(&sub), poly);

    // Validation failures
    assert_eq!(
        InterleavedEdgePolylines3d::<f64>::from_parts(&[0, 1], &[1.0, 2.0])
            .unwrap_err()
            .kind,
        ErrorKind::DimensionMismatch
    );
    assert_eq!(
        InterleavedEdgePolylines3d::<f64>::from_parts(&[0, 1], &[1.0, 2.0, f64::NAN])
            .unwrap_err()
            .kind,
        ErrorKind::NonFiniteValue
    );
    assert_eq!(
        InterleavedEdgePolylines3d::<f64>::from_parts(&[1, 1], &[1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        ErrorKind::InvalidInput
    );

    poly.points_mut().set_z(0, f64::NEG_INFINITY).unwrap();
    assert!(!poly.is_all_finite());
    assert_eq!(
        poly.validate_finite().unwrap_err().kind,
        ErrorKind::NonFiniteValue
    );

    let direct_core = EdgePolylines::from_parts(
        &[0, 2],
        Layout3d::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let direct_ffi = InterleavedEdgePolylines3d::from_core(&direct_core);
    assert_eq!(direct_ffi.to_core().unwrap(), direct_core);
}
