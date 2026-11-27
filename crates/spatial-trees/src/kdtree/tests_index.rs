use egraph_core::{CoreError, DeterministicRng, Layout2d, Layout3d, LayoutNd};

use super::*;

fn test_index_generic<I: NearestNeighborIndex>(
    idx: &I,
    query: &[f64],
    expected_dims: usize,
    expected_len: usize,
) {
    assert_eq!(idx.dims(), expected_dims);
    assert_eq!(idx.len(), expected_len);
    assert_eq!(idx.is_empty(), expected_len == 0);

    let mut scratch = KdQueryScratch::with_capacity(4);
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    idx.query_knn(query, 3, &mut out_a).unwrap();
    idx.query_knn_with_scratch(query, 3, &mut scratch, &mut out_b)
        .unwrap();
    assert_eq!(out_a, out_b);

    idx.query_radius(query, 2.0, &mut out_a).unwrap();
    idx.query_radius_sq(query, 4.0, &mut out_b).unwrap();
    assert_eq!(out_a, out_b);
    idx.query_radius_sq_with_scratch(query, 4.0, &mut scratch, &mut out_b)
        .unwrap();
    assert_eq!(out_a, out_b);

    if expected_len > 1 {
        idx.query_knn_excluding_self(0, 2, &mut out_a).unwrap();
        idx.query_knn_excluding_self_with_scratch(0, 2, &mut scratch, &mut out_b)
            .unwrap();
        assert_eq!(out_a, out_b);

        let n1 = idx.nearest_excluding_self(0).unwrap();
        let n2 = idx
            .nearest_excluding_self_with_scratch(0, &mut scratch)
            .unwrap();
        assert_eq!(n1, n2);
        assert_eq!(n1, out_a.first().copied());
    }
}

#[test]
fn test_nearest_neighbor_index_trait_static_dynamic_and_blankets() {
    let mut rng = DeterministicRng::from_seed_u64(4242);
    let n = 40usize;
    let pts3: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            [
                rng.next_f64() * 6.0 - 3.0,
                rng.next_f64() * 6.0 - 3.0,
                rng.next_f64() * 6.0 - 3.0,
            ]
        })
        .collect();
    let l3 = Layout3d::from_xyz(
        pts3.iter().map(|p| p[0]).collect(),
        pts3.iter().map(|p| p[1]).collect(),
        pts3.iter().map(|p| p[2]).collect(),
    )
    .unwrap();
    let l3_f32 = Layout3d::from_xyz(
        pts3.iter().map(|p| p[0] as f32).collect(),
        pts3.iter().map(|p| p[1] as f32).collect(),
        pts3.iter().map(|p| p[2] as f32).collect(),
    )
    .unwrap();

    let static_3d = StaticKdTree::<3, f64>::from_layout_3d(&l3).unwrap();
    let static_3d_f32 = StaticKdTree::<3, f32>::from_layout_3d(&l3_f32).unwrap();
    let dyn_3d = DynamicKdTree::<f64>::from_euclidean(&l3).unwrap();
    let dyn_3d_f32 = DynamicKdTree::<f32>::from_euclidean(&l3_f32).unwrap();

    let query = [0.5_f64, -0.5, 0.25];
    test_index_generic(&static_3d, &query, 3, n);
    test_index_generic(&dyn_3d, &query, 3, n);
    test_index_generic(&static_3d_f32, &query, 3, n);
    test_index_generic(&dyn_3d_f32, &query, 3, n);

    // Parity between StaticKdTree<3, f64> and DynamicKdTree<f64> through SpatialIndex trait objects.
    let boxed: Vec<Box<dyn SpatialIndex>> = vec![Box::new(static_3d.clone()), Box::new(dyn_3d)];
    let mut out_s = Vec::new();
    let mut out_d = Vec::new();
    boxed[0].query_knn(&query, 5, &mut out_s).unwrap();
    boxed[1].query_knn(&query, 5, &mut out_d).unwrap();
    assert_eq!(out_s, out_d);

    boxed[0].query_radius(&query, 2.5, &mut out_s).unwrap();
    boxed[1].query_radius(&query, 2.5, &mut out_d).unwrap();
    assert_eq!(out_s, out_d);

    // Blanket impls for &T, &mut T, and Box<T>.
    let mut owned = static_3d;
    let ref_idx: &StaticKdTree<3, f64> = &owned;
    test_index_generic(&ref_idx, &query, 3, n);
    let mut_idx: &mut StaticKdTree<3, f64> = &mut owned;
    test_index_generic(&mut_idx, &query, 3, n);
    test_index_generic(&boxed[0], &query, 3, n);

    // Error paths through NearestNeighborIndex.
    let mut err_out = Vec::new();
    assert!(matches!(
        boxed[0].query_knn(&[0.0, 1.0], 2, &mut err_out),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        boxed[1].query_radius(&[0.0, f64::NAN, 0.0], 1.0, &mut err_out),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        boxed[0].query_radius(&[0.0, 0.0, 0.0], -0.5, &mut err_out),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        boxed[1].query_radius_sq(&[0.0, 0.0, 0.0], -1.0, &mut err_out),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        boxed[0].query_knn_excluding_self(n, 1, &mut err_out),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_kdtree_from_euclidean_2d_3d_nd_and_mismatch() {
    let l2 = Layout2d::from_xy(vec![0.0_f64, 1.0, 2.0], vec![1.0, 0.0, -1.0]).unwrap();
    let s2 = StaticKdTree::<2, f64>::from_euclidean(&l2).unwrap();
    let d2 = DynamicKdTree::<f64>::from_euclidean(&l2).unwrap();
    assert_eq!(s2.len(), 3);
    assert_eq!(d2.dims(), 2);
    assert_eq!(
        s2.nearest_excluding_self(0).unwrap(),
        d2.nearest_excluding_self(0).unwrap()
    );

    let l_nd = LayoutNd::from_flat(3, 4, &[0.0_f64; 12]).unwrap();
    let s4 = StaticKdTree::<4, f64>::from_euclidean(&l_nd).unwrap();
    assert_eq!(s4.dims(), 4);
    assert_eq!(s4.len(), 3);

    assert!(matches!(
        StaticKdTree::<3, f64>::from_euclidean(&l2),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        StaticKdTree::<2, f64>::from_euclidean(&l_nd),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        StaticKdTree::<0, f64>::from_euclidean(&l_nd),
        Err(CoreError::InvalidInput(_))
    ));

    let bad_l3 = Layout3d::from_xyz(vec![f64::NAN], vec![0.0], vec![0.0]).unwrap();
    assert!(matches!(
        StaticKdTree::<3, f64>::from_layout_3d(&bad_l3),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        DynamicKdTree::<f64>::from_euclidean(&bad_l3),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
