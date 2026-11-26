use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, LayoutNd};

use crate::nd_index::build_nd_spatial_index;

use super::*;

#[test]
fn test_neighborhood_nd_2d_bit_exact_parity_and_f32() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let l2 = Layout2d::from_xy(
        vec![0.0, 1.0, 2.1, 3.3, 4.4, 5.6],
        vec![0.0, 0.5, -0.3, 0.2, 0.8, 0.1],
    )
    .unwrap();
    let flat = l2.to_interleaved();
    let lnd = LayoutNd::from_flat(6, 2, flat.clone()).unwrap();
    let idx = build_nd_spatial_index(&lnd).unwrap();

    assert_eq!(
        neighborhood_preservation_1hop(&g, &l2).unwrap().to_bits(),
        neighborhood_preservation_1hop_nd(&g, &lnd)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_1hop_nd(&g, &lnd)
            .unwrap()
            .to_bits(),
        neighborhood_preservation_1hop_nd_with_index(&g, &lnd, &idx)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_2hop(&g, &l2).unwrap().to_bits(),
        neighborhood_preservation_2hop_nd(&g, &lnd)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_2hop_nd(&g, &lnd)
            .unwrap()
            .to_bits(),
        neighborhood_preservation_2hop_nd_with_index(&g, &lnd, &idx)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_1hop(&g, &l2, 4, 42)
            .unwrap()
            .to_bits(),
        sampled_neighborhood_preservation_1hop_nd(&g, &lnd, 4, 42)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_1hop_nd(&g, &lnd, 6, 42)
            .unwrap()
            .to_bits(),
        neighborhood_preservation_1hop_nd(&g, &lnd)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_2hop(&g, &l2, 4, 42)
            .unwrap()
            .to_bits(),
        sampled_neighborhood_preservation_2hop_nd(&g, &lnd, 4, 42)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_2hop_nd(&g, &lnd, 6, 42)
            .unwrap()
            .to_bits(),
        neighborhood_preservation_2hop_nd(&g, &lnd)
            .unwrap()
            .to_bits()
    );

    let g32 = CsrGraph::<f32>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let lnd32 = LayoutNd::from_flat(6, 2, flat.iter().map(|&v| v as f32).collect()).unwrap();
    assert!(
        (neighborhood_preservation_1hop_nd(&g32, &lnd32).unwrap()
            - neighborhood_preservation_1hop_nd(&g, &lnd).unwrap())
        .abs()
            < 1e-12
    );
}

#[test]
fn test_neighborhood_nd_multi_dims_and_brute_force_oracle() {
    let edges = [
        (0u32, 1u32),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 3),
        (2, 5),
    ];
    let g = CsrGraph::<f64>::from_edges(8, &edges, None, false, false).unwrap();

    // Perfect 1D line layout of P8 -> NP1 == 1.0, NP2 == 1.0.
    let g_p8 = CsrGraph::<f64>::from_edges(
        8,
        &(0..7).map(|i| (i, i + 1)).collect::<Vec<_>>(),
        None,
        false,
        false,
    )
    .unwrap();
    let l1d = LayoutNd::from_flat(8, 1, (0..8).map(|i| i as f64).collect()).unwrap();
    assert!((neighborhood_preservation_1hop_nd(&g_p8, &l1d).unwrap() - 1.0).abs() < 1e-12);
    assert!((neighborhood_preservation_2hop_nd(&g_p8, &l1d).unwrap() - 1.0).abs() < 1e-12);

    // Random layouts across dims in [1, 3, 4, 8, 10] match brute-force k-NN oracle.
    let mut rng = DeterministicRng::from_seed_u64(515);
    for dims in [1usize, 3, 4, 8, 10] {
        let coords: Vec<f64> = (0..8 * dims).map(|_| rng.next_f64() * 5.0).collect();
        let layout = LayoutNd::from_flat(8, dims, coords).unwrap();
        let idx = build_nd_spatial_index(&layout).unwrap();

        let np1 = neighborhood_preservation_1hop_nd(&g, &layout).unwrap();
        let np2 = neighborhood_preservation_2hop_nd(&g, &layout).unwrap();
        assert!((0.0..=1.0).contains(&np1));
        assert!((0.0..=1.0).contains(&np2));
        assert_eq!(
            np1.to_bits(),
            neighborhood_preservation_1hop_nd_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            np2.to_bits(),
            neighborhood_preservation_2hop_nd_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );

        let s1 = sampled_neighborhood_preservation_1hop_nd(&g, &layout, 5, 17).unwrap();
        let s2 = sampled_neighborhood_preservation_2hop_nd(&g, &layout, 5, 17).unwrap();
        assert_eq!(
            s1.to_bits(),
            sampled_neighborhood_preservation_1hop_nd_with_index(&g, &layout, &idx, 5, 17)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            s2.to_bits(),
            sampled_neighborhood_preservation_2hop_nd_with_index(&g, &layout, &idx, 5, 17)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            sampled_neighborhood_preservation_1hop_nd(&g, &layout, 10, 17)
                .unwrap()
                .to_bits(),
            np1.to_bits()
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop_nd(&g, &layout, 10, 17)
                .unwrap()
                .to_bits(),
            np2.to_bits()
        );
    }
}

#[test]
fn test_neighborhood_nd_boundaries_and_errors() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(neighborhood_preservation_1hop_nd(&g, &l).unwrap(), 0.0);
        assert_eq!(
            sampled_neighborhood_preservation_1hop_nd(&g, &l, 2, 1).unwrap(),
            0.0
        );
        let expected_np2 = if n == 0 { 0.0 } else { 1.0 };
        assert_eq!(
            neighborhood_preservation_2hop_nd(&g, &l).unwrap(),
            expected_np2
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop_nd(&g, &l, 2, 1).unwrap(),
            expected_np2
        );
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 10, 9, 1.0).unwrap();
    assert_eq!(
        neighborhood_preservation_1hop_nd(&g_edgeless, &l4).unwrap(),
        0.0
    );
    assert_eq!(
        neighborhood_preservation_2hop_nd(&g_edgeless, &l4).unwrap(),
        1.0
    );

    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l_nan = LayoutNd::from_flat(2, 3, vec![0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        neighborhood_preservation_1hop_nd(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        neighborhood_preservation_2hop_nd(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        neighborhood_preservation_1hop_nd(&g2, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sampled_neighborhood_preservation_1hop_nd(
            &g2,
            &LayoutNd::<f64>::zeros(2, 3).unwrap(),
            0,
            1
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sampled_neighborhood_preservation_2hop_nd(
            &g2,
            &LayoutNd::<f64>::zeros(2, 3).unwrap(),
            0,
            1
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // Index dimension mismatch check.
    let l4_d3 = LayoutNd::<f64>::zeros(4, 3).unwrap();
    let idx_d10 = build_nd_spatial_index(&l4).unwrap();
    assert!(matches!(
        neighborhood_preservation_1hop_nd_with_index(&g_edgeless, &l4_d3, &idx_d10),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
