use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn test_crossings_exact_summary_torus_and_sampled() {
    // Right-angle crossing between (0, 1) and (2, 3).
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let layout = Layout2d::from_xy(vec![-1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -1.0, 1.0]).unwrap();
    assert_eq!(crossing_number(&g, &layout).unwrap(), 1);
    assert!(crossing_angle(&g, &layout).unwrap().abs() < 1e-12);
    let pairs = crossing_edges(&g, &layout).unwrap();
    assert_eq!(crossing_number_from_pairs(&pairs), 1);
    assert!(crossing_angle_from_pairs(&pairs).abs() < 1e-12);
    assert!((pairs[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

    // K5 pentagram has 5 crossings.
    let mut k5_edges = Vec::new();
    for u in 0..5u32 {
        for v in (u + 1)..5u32 {
            k5_edges.push((u, v));
        }
    }
    let k5 = CsrGraph::<f64>::from_edges(5, &k5_edges, None, false, false).unwrap();
    let k5_x: Vec<f64> = (0..5)
        .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).cos())
        .collect();
    let k5_y: Vec<f64> = (0..5)
        .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).sin())
        .collect();
    let k5_layout = Layout2d::from_xy(k5_x, k5_y).unwrap();
    assert_eq!(crossing_number(&k5, &k5_layout).unwrap(), 5);
    assert_eq!(
        sampled_crossing_number(&k5, &k5_layout, 100, 42).unwrap(),
        5.0
    );

    // Torus wrap crossing via explicit TorusView.
    let torus_layout =
        Layout2d::from_xy(vec![0.9, 0.1, 0.9, 0.1], vec![0.1, 0.9, 0.9, 0.1]).unwrap();
    let torus_view = TorusView::new(&torus_layout, 1.0, 1.0).unwrap();
    let torus_crossings = crossing_edges_torus(&g, &torus_view).unwrap();
    assert!(!torus_crossings.is_empty());

    // Unweighted CsrGraph<()> works for crossing metrics.
    let g_unit = CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (2, 3)], false, false).unwrap();
    assert_eq!(crossing_number(&g_unit, &layout).unwrap(), 1);
}

#[test]
fn test_crossings_validation_isolated_mismatch_and_nan() {
    // Node 4 is isolated, so SweepLineCrossings edge bounds alone would miss len mismatch.
    let g = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let short_layout = Layout2d::<f64>::zeros(4);
    let short_view = TorusView::new(&short_layout, 1.0, 1.0).unwrap();
    assert!(matches!(
        crossing_number(&g, &short_layout),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        crossing_edges_torus(&g, &short_view),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sampled_crossing_number(&g, &short_layout, 10, 42),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let bad_layout = Layout2d::from_xy(
        vec![0.0, 1.0, 0.0, 1.0, f64::NAN],
        vec![0.0, 0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    assert!(matches!(
        crossing_angle(&g, &bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        crosslessness(&g, &bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let ok_layout = Layout2d::<f64>::zeros(5);
    assert!(matches!(
        sampled_crosslessness(&g, &ok_layout, 0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        crosslessness_from_count(&g, -1.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        crosslessness_from_count(&g, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

fn brute_cmax_and_crossings<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> (u128, usize) {
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    let mut c_max = 0_u128;
    for i in 0..m {
        let (u1, v1) = edges[i];
        if u1 == v1 {
            continue;
        }
        for &(u2, v2) in &edges[(i + 1)..] {
            if u2 == v2 {
                continue;
            }
            if u1 != u2 && u1 != v2 && v1 != u2 && v1 != v2 {
                c_max += 1;
            }
        }
    }
    let exact_crossings = crossing_number(graph, layout).unwrap();
    (c_max, exact_crossings)
}

#[test]
fn test_crosslessness_brute_force_oracle_and_degeneracies() {
    // 1. Star K_{1,5} and triangle K3 have c_max == 0 -> crosslessness == 1.0.
    let star = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let star_layout = Layout2d::from_xy(
        vec![0.0, 1.0, 0.0, -1.0, 0.5, -0.5],
        vec![0.0, 0.0, 1.0, 0.0, -1.0, -1.0],
    )
    .unwrap();
    assert_eq!(max_crossing_pairs(&star), 0);
    assert_eq!(crosslessness(&star, &star_layout).unwrap(), 1.0);
    assert_eq!(
        sampled_crosslessness(&star, &star_layout, 16, 42).unwrap(),
        1.0
    );

    let k3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let k3_layout = Layout2d::from_xy(vec![0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0]).unwrap();
    assert_eq!(max_crossing_pairs(&k3), 0);
    assert_eq!(crosslessness(&k3, &k3_layout).unwrap(), 1.0);

    // 2. K4 square with crossing diagonals: c = 1, c_max = C(6,2) - 4*C(3,2) = 15 - 12 = 3.
    let k4 = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let sq = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    assert_eq!(max_crossing_pairs(&k4), 3);
    let expected_k4 = 1.0 - (1.0_f64 / 3.0).sqrt();
    assert!((crosslessness(&k4, &sq).unwrap() - expected_k4).abs() < 1e-12);
    assert!((sampled_crosslessness(&k4, &sq, 100, 42).unwrap() - expected_k4).abs() < 1e-12);
    assert!((crosslessness_from_count(&k4, 1.0).unwrap() - expected_k4).abs() < 1e-12);

    // 3. Directed graph with reciprocal pair (0,1)/(1,0) and disjoint (2,3) (plus self-loop stripped via sanitize=true).
    let dir = CsrGraph::<f64>::from_edges(4, &[(0, 0), (0, 1), (1, 0), (2, 3)], None, true, true)
        .unwrap();
    let (brute_cmax, brute_c) = brute_cmax_and_crossings(&dir, &sq);
    assert_eq!(brute_cmax, 2);
    assert_eq!(max_crossing_pairs(&dir), brute_cmax);
    assert_eq!(brute_c, 0);
    assert_eq!(crosslessness(&dir, &sq).unwrap(), 1.0);

    // 4. Multi-seed random graphs (undirected & directed, f32 & f64) vs O(M^2) oracle.
    for seed in 1..=15u64 {
        let mut rng = DeterministicRng::from_seed_u64(400 + seed);
        let n = 12usize;
        let directed = seed.is_multiple_of(3);
        let mut edges = Vec::new();
        for _ in 0..24 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, directed, true).unwrap();
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let l64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let l32 = Layout2d::from_xy(
            xs.iter().map(|&v| v as f32).collect(),
            ys.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();

        let (oracle_cmax64, oracle_c64) = brute_cmax_and_crossings(&g, &l64);
        assert_eq!(max_crossing_pairs(&g), oracle_cmax64);
        let expected64 = if oracle_cmax64 == 0 {
            1.0
        } else {
            1.0 - (oracle_c64 as f64 / oracle_cmax64 as f64).sqrt()
        };
        assert!((crosslessness(&g, &l64).unwrap() - expected64).abs() < 1e-12);
        // Total pairs <= 1000 so sampled_crosslessness hits exact branch.
        assert!((sampled_crosslessness(&g, &l64, 1000, 42).unwrap() - expected64).abs() < 1e-12);

        let (oracle_cmax32, oracle_c32) = brute_cmax_and_crossings(&g, &l32);
        let expected32 = if oracle_cmax32 == 0 {
            1.0
        } else {
            1.0 - (oracle_c32 as f64 / oracle_cmax32 as f64).sqrt()
        };
        assert!((crosslessness(&g, &l32).unwrap() - expected32).abs() < 1e-12);
    }
}

#[test]
fn test_max_crossing_pairs_directed_reciprocal_and_parallel_oracles() {
    // 1. Single reciprocal pair: c_max == 0, crosslessness == 1.0.
    let recip_pair = CsrGraph::<f64>::from_edges(2, &[(0, 1), (1, 0)], None, true, true).unwrap();
    let l2 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]).unwrap();
    assert_eq!(max_crossing_pairs(&recip_pair), 0);
    assert_eq!(crosslessness(&recip_pair, &l2).unwrap(), 1.0);
    assert_eq!(crosslessness_from_count(&recip_pair, 0.0).unwrap(), 1.0);

    // 2. Reciprocal 3-leaf star: adj_pairs > total_pairs, yet c_max == 0.
    let recip_star = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
        None,
        true,
        true,
    )
    .unwrap();
    let l4 = Layout2d::from_xy(vec![0.0, 1.0, 0.0, -1.0], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(max_crossing_pairs(&recip_star), 0);
    assert_eq!(crosslessness(&recip_star, &l4).unwrap(), 1.0);
    assert_eq!(
        sampled_crosslessness(&recip_star, &l4, 64, 42).unwrap(),
        1.0
    );
    assert_eq!(crosslessness_from_count(&recip_star, 1.0).unwrap(), 1.0);

    // 3. 4-node 6-edge simple directed graph with adj_pairs (16) > total_pairs (15):
    //    only {(0,1), (3,2)} are vertex-disjoint -> c_max = 1, c = 1, crosslessness = 0.0.
    let dir_4 = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)],
        None,
        true,
        true,
    )
    .unwrap();
    let l_cross = Layout2d::from_xy(vec![0.0, 2.0, 0.0, 2.0], vec![0.0, 2.0, 2.0, 0.0]).unwrap();
    let (brute_cmax, brute_c) = brute_cmax_and_crossings(&dir_4, &l_cross);
    assert_eq!(brute_cmax, 1);
    assert_eq!(brute_c, 1);
    assert_eq!(max_crossing_pairs(&dir_4), 1);
    assert_eq!(crosslessness(&dir_4, &l_cross).unwrap(), 0.0);
    assert_eq!(
        sampled_crosslessness(&dir_4, &l_cross, 100, 42).unwrap(),
        0.0
    );

    // 4. Raw edge list with parallel multi-edges, reciprocal edges, and self-loops vs O(M^2).
    let raw_edges = [
        (0_u32, 1),
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (2, 3),
        (2, 3),
        (1, 1),
        (3, 3),
    ];
    let mut expected_raw = 0_u128;
    for i in 0..raw_edges.len() {
        let (u1, v1) = raw_edges[i];
        if u1 == v1 {
            continue;
        }
        for &(u2, v2) in &raw_edges[(i + 1)..] {
            if u2 == v2 {
                continue;
            }
            if u1 != u2 && u1 != v2 && v1 != u2 && v1 != v2 {
                expected_raw += 1;
            }
        }
    }
    assert_eq!(max_crossing_pairs_from_edges(4, &raw_edges), expected_raw);
}

#[test]
fn crossings_torus_view_and_non_euclidean_geodesic_crossings() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let l2 = Layout2d::from_xy(vec![0.1, 0.4, 0.1, 0.4], vec![0.1, 0.4, 0.4, 0.1]).unwrap();
    let view = TorusView::new(&l2, 1.0, 1.0).unwrap();
    let cx2 = crossing_edges_torus(&g, &view).unwrap();
    assert_eq!(cx2.len(), 1);
    assert!(cx2[0].cos_sq < 1e-12);
    assert_eq!(torus_crossing_number(&g, &view).unwrap(), 1);

    let mut ml = ManifoldLayout2d::from_layout(ManifoldKind::Torus2d, l2).unwrap();
    ml.set_torus_extents(2.0, 1.0).unwrap();
    let cx_m = crossing_edges_torus(&g, &ml).unwrap();
    assert_eq!(cx_m.len(), 1);
    assert!(cx_m[0].cos_sq > 0.1);
    assert_eq!(torus_crossing_number(&g, &ml).unwrap(), 1);

    let cx_edges = crossing_edges_torus_from_edges(&[(0, 1), (2, 3)], &ml).unwrap();
    assert_eq!(cx_edges.len(), 1);

    let sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 4, 1, 1.0).unwrap();
    assert!(matches!(
        crossing_edges_torus(&g, &sph),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert_eq!(spherical_crossing_number(&g, &sph).unwrap(), 0);

    let sph_cross = ManifoldLayout2d::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 0.5, 0.5],
        vec![1.5, 1.5, 1.0, 2.0],
    )
    .unwrap();
    assert_eq!(spherical_crossing_number(&g, &sph_cross).unwrap(), 1);
    assert_eq!(crossing_edges_spherical(&g, &sph_cross).unwrap().len(), 1);

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
