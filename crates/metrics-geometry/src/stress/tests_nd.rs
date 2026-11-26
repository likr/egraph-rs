use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, LayoutNd};

use super::*;

#[test]
fn test_stress_nd_2d_parity_and_f32() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();
    let l2 = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let flat = l2.to_interleaved();
    let lnd = LayoutNd::from_flat(4, 2, flat.clone()).unwrap();

    assert_eq!(
        scale_normalized_stress(&g, &l2).unwrap().to_bits(),
        scale_normalized_stress_nd(&g, &lnd).unwrap().to_bits()
    );
    assert_eq!(
        scale_normalized_stress_detail(&g, &l2).unwrap(),
        scale_normalized_stress_detail_nd(&g, &lnd).unwrap()
    );
    assert_eq!(
        sampled_scale_normalized_stress(&g, &l2, 2, 42)
            .unwrap()
            .to_bits(),
        sampled_scale_normalized_stress_nd(&g, &lnd, 2, 42)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_scale_normalized_stress_detail(&g, &l2, 2, 42).unwrap(),
        sampled_scale_normalized_stress_detail_nd(&g, &lnd, 2, 42).unwrap()
    );

    let dists = all_pairs_bfs(&g, 1.0).unwrap();
    assert_eq!(
        scale_normalized_stress_from_distances(&l2, &dists, None)
            .unwrap()
            .to_bits(),
        scale_normalized_stress_from_distances_nd(&lnd, &dists, None)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        scale_normalized_stress_from_distances_detail(&l2, &dists, None).unwrap(),
        scale_normalized_stress_from_distances_detail_nd(&lnd, &dists, None).unwrap()
    );

    let pairs = vec![(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (0, 2, 1.0, 1.0)];
    assert_eq!(
        scale_normalized_stress_from_pairs(&l2, &pairs).unwrap(),
        scale_normalized_stress_from_pairs_nd(&lnd, &pairs).unwrap()
    );

    let flat32: Vec<f32> = flat.iter().map(|&v| v as f32).collect();
    let lnd32 = LayoutNd::from_flat(4, 2, flat32).unwrap();
    let g32 = CsrGraph::<f32>::from_edges(4, &edges, None, false, false).unwrap();
    let d32 = scale_normalized_stress_nd(&g32, &lnd32).unwrap();
    assert!((d32 - scale_normalized_stress_nd(&g, &lnd).unwrap()).abs() < 1e-6);
}

#[test]
fn test_stress_nd_multi_dims_isometry_and_scale_invariance() {
    // A regular simplex on N = D + 1 vertices in D dimensions is isometric to K_{D+1}.
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
        let l2 = LayoutNd::from_flat(12, dims, scaled_coords).unwrap();
        let e1 = scale_normalized_stress_detail_nd(&g, &l1).unwrap();
        let e2 = scale_normalized_stress_detail_nd(&g, &l2).unwrap();
        assert!((e1.normalized_stress - e2.normalized_stress).abs() < 1e-12);
        assert!((e1.alpha_star - 37.5 * e2.alpha_star).abs() < 1e-10);

        let full = scale_normalized_stress_nd(&g, &l1).unwrap();
        let sampled_all = sampled_scale_normalized_stress_nd(&g, &l1, 12, 99).unwrap();
        assert!((full - sampled_all).abs() < 1e-12);

        let dists = all_pairs_bfs(&g, 1.0).unwrap();
        let from_d = scale_normalized_stress_from_distances_nd(&l1, &dists, None).unwrap();
        assert!((full - from_d).abs() < 1e-12);
    }

    // 1D exact arithmetic progression of path graph has SNS == 0.0.
    let path =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], None, false, false)
            .unwrap();
    let line = LayoutNd::from_flat(5, 1, vec![0.0, 2.0, 4.0, 6.0, 8.0]).unwrap();
    let ev = scale_normalized_stress_detail_nd(&path, &line).unwrap();
    assert!(ev.normalized_stress < 1e-12);
    assert!((ev.alpha_star - 0.5).abs() < 1e-12);

    // 3D orthogonal pair (0,1), (0,2), (1,2) in equilateral triangle -> SNS == 0.0.
    let tri =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let tri3d =
        LayoutNd::from_flat(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    assert!(scale_normalized_stress_nd(&tri, &tri3d).unwrap() < 1e-12);
}

#[test]
fn test_stress_nd_boundaries_and_error_guards() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(scale_normalized_stress_nd(&g, &l).unwrap(), 0.0);
        assert_eq!(
            sampled_scale_normalized_stress_nd(&g, &l, 4, 1).unwrap(),
            0.0
        );
        let d = vec![0.0_f64; n * n];
        assert_eq!(
            scale_normalized_stress_from_distances_nd(&l, &d, None).unwrap(),
            0.0
        );
        assert_eq!(
            scale_normalized_stress_from_pairs_nd(&l, &[])
                .unwrap()
                .normalized_stress,
            0.0
        );
    }

    // Edgeless N = 4 graph.
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    assert_eq!(scale_normalized_stress_nd(&g_edgeless, &l4).unwrap(), 0.0);
    assert_eq!(
        sampled_scale_normalized_stress_nd(&g_edgeless, &l4, 2, 5).unwrap(),
        0.0
    );

    // Collapsed layout with edges -> normalized_stress == 1.0.
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l_col = LayoutNd::<f64>::zeros(2, 4).unwrap();
    assert_eq!(scale_normalized_stress_nd(&g2, &l_col).unwrap(), 1.0);
    assert_eq!(
        sampled_scale_normalized_stress_nd(&g2, &l_col, 2, 1).unwrap(),
        1.0
    );
    assert_eq!(
        scale_normalized_stress_from_pairs_nd(&l_col, &[(0, 1, 2.0, 0.25)])
            .unwrap()
            .normalized_stress,
        1.0
    );

    // Non-symmetric distance matrix rejection (C14).
    let asym = [0.0, 1.0, 2.0, 0.0];
    assert!(matches!(
        scale_normalized_stress_from_distances_nd(
            &LayoutNd::<f64>::zeros(2, 3).unwrap(),
            &asym,
            None
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // NaN / Inf rejection, node-count mismatch, num_pivots == 0, pair index OOB.
    let l_nan = LayoutNd::from_flat(2, 3, vec![0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        scale_normalized_stress_nd(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let l_inf = LayoutNd::from_flat(2, 3, vec![0.0, 0.0, f64::INFINITY, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        sampled_scale_normalized_stress_nd(&g2, &l_inf, 1, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        scale_normalized_stress_nd(&g2, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sampled_scale_normalized_stress_nd(&g2, &LayoutNd::<f64>::zeros(2, 3).unwrap(), 0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        scale_normalized_stress_from_pairs_nd(&l4, &[(0, 4, 1.0, 1.0)]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    // f32 sampled stress determinism.
    let mut rng = DeterministicRng::from_seed_u64(11);
    let coords32: Vec<f32> = (0..18).map(|_| rng.next_f64() as f32).collect();
    let l6_32 = LayoutNd::from_flat(6, 3, coords32).unwrap();
    let g6_32 = CsrGraph::<f32>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let s1 = sampled_scale_normalized_stress_nd(&g6_32, &l6_32, 3, 19).unwrap();
    let s2 = sampled_scale_normalized_stress_nd(&g6_32, &l6_32, 3, 19).unwrap();
    assert_eq!(s1.to_bits(), s2.to_bits());
}
