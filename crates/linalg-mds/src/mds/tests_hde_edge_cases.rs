use egraph_core::{CoreError, CsrGraph};

use super::tests_hde_oracle::{hde_naive_oracle, pairwise_euclidean_nd};
use super::tests_mds::isotropic_std;
use super::*;

#[test]
fn test_hde_disconnected_and_isolated_nodes() {
    // Mixed components: triangle {0,1,2} + edge {3,4} + isolated {5,6}
    let mixed =
        CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (2, 0), (3, 4)], None, false, false)
            .unwrap();
    let pivots = vec![0, 3, 5];
    let cfg_unnorm = HdeConfig {
        tol: 1e-12,
        max_sweeps: 200,
        normalize: false,
        ..HdeConfig::default()
    };
    let lay_unnorm = hde_with_pivots::<f64, f64>(&mixed, &pivots, &cfg_unnorm).unwrap();
    let oracle = hde_naive_oracle(&mixed, &pivots, 2);
    let coords: Vec<Vec<f64>> = (0..7)
        .map(|i| vec![lay_unnorm.x()[i], lay_unnorm.y()[i]])
        .collect();
    for (a, b) in pairwise_euclidean_nd(&coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }

    let lay_norm = hde::<f64, f64>(&mixed, &HdeConfig::default()).unwrap();
    assert!(lay_norm.is_all_finite());
    assert!((isotropic_std(&lay_norm) - 1.0).abs() < 1e-12);

    // All-isolated edgeless graph (n = 5)
    let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let edge_unnorm = hde_with_pivots::<f64, f64>(&edgeless, &[0, 2, 4], &cfg_unnorm).unwrap();
    let edge_oracle = hde_naive_oracle(&edgeless, &[0, 2, 4], 2);
    let edge_coords: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![edge_unnorm.x()[i], edge_unnorm.y()[i]])
        .collect();
    for (a, b) in pairwise_euclidean_nd(&edge_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&edge_oracle).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }
    let edge_norm = hde::<f64, f64>(&edgeless, &HdeConfig::default()).unwrap();
    assert!(edge_norm.is_all_finite());
    assert!((isotropic_std(&edge_norm) - 1.0).abs() < 1e-12);
}

#[test]
fn test_hde_directed_and_reciprocal_edges() {
    let undir_edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)];
    let mut recip_edges = Vec::new();
    let mut recip_weights = Vec::new();
    let undir_weights = vec![1.0, 2.5, 1.5, 3.0, 4.0, 0.75];
    for (&(u, v), &w) in undir_edges.iter().zip(undir_weights.iter()) {
        recip_edges.push((u, v));
        recip_edges.push((v, u));
        recip_weights.push(w);
        recip_weights.push(w);
    }

    // Unweighted reciprocal vs undirected: bit-exact
    let gu = CsrGraph::<f64>::from_edges(5, &undir_edges, None, false, false).unwrap();
    let gr = CsrGraph::<f64>::from_edges(5, &recip_edges, None, true, true).unwrap();
    let cfg = HdeConfig::default();
    let lu = hde::<f64, f64>(&gu, &cfg).unwrap();
    let lr = hde::<f64, f64>(&gr, &cfg).unwrap();
    assert_eq!(
        lu.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lr.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        lu.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lr.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // Weighted reciprocal vs undirected: bit-exact
    let gwu =
        CsrGraph::<f64>::from_edges(5, &undir_edges, Some(&undir_weights), false, false).unwrap();
    let gwr =
        CsrGraph::<f64>::from_edges(5, &recip_edges, Some(&recip_weights), true, true).unwrap();
    let lwu = hde_with_pivots::<f64, f64>(&gwu, &[0, 3, 4], &cfg).unwrap();
    let lwr = hde_with_pivots::<f64, f64>(&gwr, &[0, 3, 4], &cfg).unwrap();
    assert_eq!(
        lwu.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lwr.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        lwu.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lwr.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // One-way DAG 0 -> 1 -> 2 -> 3 with pivots = &[3, 0]
    let dag = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, true, false).unwrap();
    let dag_cfg = HdeConfig {
        tol: 1e-12,
        max_sweeps: 200,
        normalize: false,
        ..HdeConfig::default()
    };
    let dag_lay = hde_with_pivots::<f64, f64>(&dag, &[3, 0], &dag_cfg).unwrap();
    assert!(dag_lay.is_all_finite());
    let dag_oracle = hde_naive_oracle(&dag, &[3, 0], 2);
    let dag_coords: Vec<Vec<f64>> = (0..4)
        .map(|i| vec![dag_lay.x()[i], dag_lay.y()[i]])
        .collect();
    for (a, b) in pairwise_euclidean_nd(&dag_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&dag_oracle).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_hde_weighted_positive_and_invalid() {
    // Shortcut 0-2 (w=10.0) longer than 0-1-2 (w=1.0, 2.0) -> Dijkstra distance = 3.0
    let wg = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (0, 2), (2, 3)],
        Some(&[1.0, 2.0, 10.0, 1.5]),
        false,
        false,
    )
    .unwrap();
    let cfg_unnorm = HdeConfig {
        tol: 1e-12,
        max_sweeps: 200,
        normalize: false,
        ..HdeConfig::default()
    };
    let lay = hde_with_pivots::<f64, f64>(&wg, &[0, 1, 3], &cfg_unnorm).unwrap();
    let oracle = hde_naive_oracle(&wg, &[0, 1, 3], 2);
    let coords: Vec<Vec<f64>> = (0..4).map(|i| vec![lay.x()[i], lay.y()[i]]).collect();
    for (a, b) in pairwise_euclidean_nd(&coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }

    // Tiny weights (1e-8) with normalize=false preserve rank-2 axes via relative threshold
    let tiny = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        Some(&[1e-8, 2e-8, 1.5e-8, 2.5e-8, 3e-8]),
        false,
        false,
    )
    .unwrap();
    let tiny_lay = hde_with_pivots::<f64, f64>(&tiny, &[0, 1, 2], &cfg_unnorm).unwrap();
    assert!(tiny_lay.is_all_finite());
    assert!(tiny_lay.x().iter().any(|&v| v.abs() > 1e-12));
    assert!(tiny_lay.y().iter().any(|&v| v.abs() > 1e-12));

    // Huge weights (1e150) test covariance scaling overflow guard
    let huge = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        Some(&[1e150, 2e150, 1.5e150, 2.5e150, 3e150]),
        false,
        false,
    )
    .unwrap();
    let huge_norm = hde::<f64, f64>(&huge, &HdeConfig::default()).unwrap();
    assert!(huge_norm.is_all_finite());
    assert!((isotropic_std(&huge_norm) - 1.0).abs() < 1e-12);
    let huge_unnorm = hde_with_pivots::<f64, f64>(&huge, &[0, 1, 2], &cfg_unnorm).unwrap();
    assert!(huge_unnorm.is_all_finite());
    assert!(huge_unnorm.x().iter().any(|&v| v.abs() > 1e140));

    // Zero and negative weights rejected with InvalidInput
    for bad_w in [0.0_f64, -1.0_f64] {
        let bad = CsrGraph::<f64>::from_edges(3, &[(0, 1)], Some(&[bad_w]), false, false).unwrap();
        assert!(matches!(
            hde::<f64, f64>(&bad, &HdeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            hde_with_pivots::<f64, f64>(&bad, &[0, 1], &HdeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            hde_nd::<f64, f64>(&bad, &HdeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            hde_nd_with_pivots::<f64, f64>(&bad, &[0, 1], &HdeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
    }

    // CsrGraph rejects NaN/Inf weights at construction
    assert!(matches!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::INFINITY]), false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_hde_degenerate_and_validation_errors() {
    let cfg = HdeConfig::default();
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(hde::<f64, f64>(&empty, &cfg).unwrap().len(), 0);
    assert_eq!(
        hde_with_pivots::<f64, f64>(&empty, &[], &cfg)
            .unwrap()
            .len(),
        0
    );
    assert!(matches!(
        hde_with_pivots::<f64, f64>(&empty, &[0], &cfg),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    for normalize in [true, false] {
        let s_cfg = HdeConfig { normalize, ..cfg };
        let s1 = hde::<f64, f64>(&single, &s_cfg).unwrap();
        assert_eq!(s1.x(), &[0.0]);
        assert_eq!(s1.y(), &[0.0]);
        let s2 = hde_with_pivots::<f64, f64>(&single, &[0], &s_cfg).unwrap();
        assert_eq!(s2.x(), &[0.0]);
        assert_eq!(s2.y(), &[0.0]);
    }

    // N = 2 edge 0-1
    let two = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l2 = hde::<f64, f64>(&two, &cfg).unwrap();
    assert_eq!(l2.y(), &[0.0, 0.0]);
    assert!((l2.x()[0] + l2.x()[1]).abs() < 1e-12);
    assert!(l2.x()[0].abs() > 0.5);

    // dimensions > n
    let g5 = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        None,
        false,
        false,
    )
    .unwrap();
    let l5 = hde::<f64, f64>(&g5, &cfg).unwrap();
    assert!(l5.is_all_finite());
    assert!((isotropic_std(&l5) - 1.0).abs() < 1e-12);

    // Duplicate pivots produce bit-exact results to deduplicated pivots
    let l_dup = hde_with_pivots::<f64, f64>(&g5, &[0, 2, 0, 4, 2], &cfg).unwrap();
    let l_uniq = hde_with_pivots::<f64, f64>(&g5, &[0, 2, 4], &cfg).unwrap();
    assert_eq!(
        l_dup.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l_uniq.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l_dup.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l_uniq.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // Single pivot m_eff = 1 -> y identically 0.0
    for normalize in [false, true] {
        let s_cfg = HdeConfig { normalize, ..cfg };
        let l1a = hde_with_pivots::<f64, f64>(&g5, &[1], &s_cfg).unwrap();
        let l1b = hde_with_pivots::<f64, f64>(&g5, &[1, 1, 1], &s_cfg).unwrap();
        assert!(l1a.y().iter().all(|&y| y == 0.0));
        assert_eq!(l1a.x(), l1b.x());
        assert_eq!(l1a.y(), l1b.y());
    }

    // Empty pivots on n >= 1 and out-of-range pivot
    assert!(matches!(
        hde_with_pivots::<f64, f64>(&g5, &[], &cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        hde_with_pivots::<f64, f64>(&g5, &[0, 5], &cfg),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    // Config validation errors even when n == 0
    for graph in [&empty, &g5] {
        assert!(matches!(
            hde::<f64, f64>(
                graph,
                &HdeConfig {
                    dimensions: 0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        for bad_out in [0usize, 1, 3] {
            assert!(matches!(
                hde::<f64, f64>(
                    graph,
                    &HdeConfig {
                        output_dims: bad_out,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
            assert!(matches!(
                hde_with_pivots::<f64, f64>(
                    graph,
                    &[0],
                    &HdeConfig {
                        output_dims: bad_out,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            hde_nd::<f64, f64>(
                graph,
                &HdeConfig {
                    output_dims: 0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        for bad_tol in [0.0, -1.0] {
            assert!(matches!(
                hde::<f64, f64>(
                    graph,
                    &HdeConfig {
                        tol: bad_tol,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
        }
        for nf_tol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                hde::<f64, f64>(graph, &HdeConfig { tol: nf_tol, ..cfg }),
                Err(CoreError::NonFiniteValue { .. })
            ));
        }
        assert!(matches!(
            hde::<f64, f64>(
                graph,
                &HdeConfig {
                    max_sweeps: 0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }
}

#[test]
fn test_hde_f32_f64_and_determinism() {
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 3),
        (1, 4),
    ];
    let w64 = vec![1.0_f64, 1.5, 2.0, 1.2, 0.8, 2.5, 1.7, 1.9];
    let w32: Vec<f32> = w64.iter().map(|&v| v as f32).collect();

    let g64 = CsrGraph::<f64>::from_edges(6, &edges, Some(&w64), false, false).unwrap();
    let g32 = CsrGraph::<f32>::from_edges(6, &edges, Some(&w32), false, false).unwrap();
    let cfg = HdeConfig {
        dimensions: 4,
        seed: 99,
        ..HdeConfig::default()
    };

    let l64_64_a = hde::<f64, f64>(&g64, &cfg).unwrap();
    let l64_64_b = hde::<f64, f64>(&g64, &cfg).unwrap();
    assert_eq!(
        l64_64_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l64_64_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l64_64_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l64_64_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(l64_64_a.is_all_finite());
    assert!((isotropic_std(&l64_64_a) - 1.0).abs() < 1e-12);

    let l64_32 = hde::<f64, f32>(&g64, &cfg).unwrap();
    let l32_64 = hde::<f32, f64>(&g32, &cfg).unwrap();
    let l32_32 = hde::<f32, f32>(&g32, &cfg).unwrap();
    assert!(l64_32.is_all_finite());
    assert!(l32_64.is_all_finite());
    assert!(l32_32.is_all_finite());
    assert!((isotropic_std(&l32_64) - 1.0).abs() < 1e-6);

    for i in 0..6 {
        assert!((f64::from(l64_32.x()[i]) - l64_64_a.x()[i]).abs() < 1e-5);
        assert!((f64::from(l64_32.y()[i]) - l64_64_a.y()[i]).abs() < 1e-5);
        assert!((f64::from(l32_32.x()[i]) - l32_64.x()[i]).abs() < 1e-5);
        assert!((f64::from(l32_32.y()[i]) - l32_64.y()[i]).abs() < 1e-5);
    }
}
