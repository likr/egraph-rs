use egraph_core::{CoreError, CsrGraph, Layout2d, LayoutNd};

use super::*;

#[test]
fn test_rta_nd_2d_bit_exact_parity_and_f32() {
    let edges: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
    let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();
    let l2 = Layout2d::<f64>::init(12, 42, 10.0).unwrap();
    let flat = l2.to_interleaved();
    let lnd = LayoutNd::from_flat(12, 2, flat.clone()).unwrap();

    assert_eq!(
        landmark_stratified_rta(&g, &l2, 4, 200, 1, 7)
            .unwrap()
            .to_bits(),
        landmark_stratified_rta_nd(&g, &lnd, 4, 200, 1, 7)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        landmark_stratified_rta_detail(&g, &l2, 4, 200, 1, 7).unwrap(),
        landmark_stratified_rta_detail_nd(&g, &lnd, 4, 200, 1, 7).unwrap()
    );

    let landmark_ids = vec![0u32, 11];
    let dists_u32: Vec<u32> = (0..12)
        .map(|i| i as u32)
        .chain((0..12).map(|i| (11 - i) as u32))
        .collect();
    let dists_f64: Vec<f64> = dists_u32.iter().map(|&d| d as f64).collect();
    assert_eq!(
        landmark_stratified_rta_from_landmarks(&l2, &landmark_ids, &dists_u32, 100, 5).unwrap(),
        landmark_stratified_rta_from_landmarks_nd(&lnd, &landmark_ids, &dists_u32, 100, 5).unwrap()
    );
    assert_eq!(
        landmark_stratified_rta_from_landmarks_f64(&l2, &landmark_ids, &dists_f64, 100, 5).unwrap(),
        landmark_stratified_rta_from_landmarks_f64_nd(&lnd, &landmark_ids, &dists_f64, 100, 5)
            .unwrap()
    );

    let g32 = CsrGraph::<f32>::from_edges(12, &edges, None, false, false).unwrap();
    let lnd32 = LayoutNd::from_flat(12, 2, flat.iter().map(|&v| v as f32).collect()).unwrap();
    let rta32 = landmark_stratified_rta_nd(&g32, &lnd32, 4, 200, 1, 7).unwrap();
    assert!((rta32 - landmark_stratified_rta_nd(&g, &lnd, 4, 200, 1, 7).unwrap()).abs() < 1e-6);
}

#[test]
fn test_rta_nd_multi_dims_exact_path_and_from_landmarks() {
    let n = 10usize;
    let edges: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    // Monotone embedding along first axis in dims in [1, 3, 4, 8, 10] achieves RTA == 1.0.
    for dims in [1usize, 3, 4, 8, 10] {
        let mut coords = vec![0.0_f64; n * dims];
        for i in 0..n {
            coords[i * dims] = i as f64;
        }
        let layout = LayoutNd::from_flat(n, dims, coords).unwrap();
        let ev = landmark_stratified_rta_detail_nd(&g, &layout, 3, 500, 1, 13).unwrap();
        assert!((ev.rta - 1.0).abs() < 1e-12);
        assert!(ev.is_exact);

        // Stratified sampling mode when num_triplets is smaller than comparable pairs.
        let ev_sampled = landmark_stratified_rta_detail_nd(&g, &layout, 3, 12, 1, 13).unwrap();
        assert!((ev_sampled.rta - 1.0).abs() < 1e-12);
        assert!(!ev_sampled.is_exact);
    }

    // f64 landmark table with +inf unreachable sentinel.
    let layout3d = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0],
    )
    .unwrap();
    let eval_inf = landmark_stratified_rta_from_landmarks_f64_nd(
        &layout3d,
        &[0],
        &[0.0, 1.0, 2.0, f64::INFINITY],
        10,
        1,
    )
    .unwrap();
    assert!((eval_inf.rta - 1.0).abs() < 1e-12);
    assert_eq!(eval_inf.num_triplets, 1);
}

#[test]
fn test_rta_nd_boundaries_and_errors() {
    for n in [0usize, 1, 2] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(
            landmark_stratified_rta_nd(&g, &l, 2, 10, 1, 1).unwrap(),
            0.0
        );
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    assert_eq!(
        landmark_stratified_rta_nd(&g_edgeless, &l4, 2, 10, 1, 1).unwrap(),
        0.0
    );

    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let l_col = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert_eq!(
        landmark_stratified_rta_nd(&g4, &l_col, 2, 10, 1, 1).unwrap(),
        0.0
    );

    // Invalid inputs and non-finite coordinates.
    let l_nan =
        LayoutNd::from_flat(4, 2, vec![0.0, f64::NAN, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
    assert!(matches!(
        landmark_stratified_rta_nd(&g4, &l_nan, 2, 10, 1, 1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta_nd(&g4, &l4, 0, 10, 1, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta_nd(&g4, &l4, 2, 0, 1, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta_nd(&g4, &LayoutNd::<f64>::zeros(3, 3).unwrap(), 2, 10, 1, 1),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64_nd(&l4, &[0], &[0.0, -1.0, 2.0, 3.0], 10, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64_nd(&l4, &[4], &[0.0, 1.0, 2.0, 3.0], 10, 1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}
