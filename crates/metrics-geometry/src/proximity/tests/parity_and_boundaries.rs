use std::f64::consts::TAU;

use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d};

use super::super::exact_predicates::euclidean_diameter_2d;
use super::super::*;
use super::{
    assert_proximity_determinism_and_tree_parity, brute_gabriel_graph_property, brute_gg_edges,
    brute_node_resolution, brute_rng_edges,
};

#[test]
fn test_planar_crystalline_s_rng_and_s_gg_equal_one() {
    // 1. 4x4 square lattice (planar crystalline grid).
    let mut grid_edges = Vec::new();
    let mut gx = Vec::new();
    let mut gy = Vec::new();
    for r in 0..4u32 {
        for c in 0..4u32 {
            let id = r * 4 + c;
            gx.push(c as f64);
            gy.push(r as f64);
            if c + 1 < 4 {
                grid_edges.push((id, id + 1));
            }
            if r + 1 < 4 {
                grid_edges.push((id, id + 4));
            }
        }
    }
    let grid = CsrGraph::<f64>::from_edges(16, &grid_edges, None, false, false).unwrap();
    let grid_layout = Layout2d::from_xy(gx, gy).unwrap();
    assert!((relative_neighborhood_preservation(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12);
    assert!((gabriel_graph_preservation(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(gabriel_graph_property(&grid, &grid_layout).unwrap(), 0.0);
    // On a 4x4 unit grid, r * d_max = (3 * sqrt(2)) / 4 > 1.0, so the 24 unit edges violate slightly.
    let thresh_4x4 = 3.0 * 2.0_f64.sqrt() / 4.0;
    let expected_vr_4x4 = 24.0 * (1.0 - 1.0 / thresh_4x4).powi(2);
    assert!((node_resolution(&grid_layout).unwrap() - expected_vr_4x4).abs() < 1e-12);

    // On a 3x3 unit grid, r * d_max = (2 * sqrt(2)) / 3 < 1.0, so node_resolution == 0.0.
    let mut g3x = Vec::new();
    let mut g3y = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            g3x.push(c as f64);
            g3y.push(r as f64);
        }
    }
    let grid3_layout = Layout2d::from_xy(g3x, g3y).unwrap();
    assert_eq!(node_resolution(&grid3_layout).unwrap(), 0.0);

    // Rotated 4x4 grid preserves exact S_RNG == 1.0 and S_GG == 1.0 despite floating-point rotation.
    let (st, ct) = (0.37_f64).sin_cos();
    let rx: Vec<f64> = grid_layout
        .x()
        .iter()
        .zip(grid_layout.y().iter())
        .map(|(&x, &y)| ct * x - st * y)
        .collect();
    let ry: Vec<f64> = grid_layout
        .x()
        .iter()
        .zip(grid_layout.y().iter())
        .map(|(&x, &y)| st * x + ct * y)
        .collect();
    let rot_grid_layout = Layout2d::from_xy(rx, ry).unwrap();
    assert!(
        (relative_neighborhood_preservation(&grid, &rot_grid_layout).unwrap() - 1.0).abs() < 1e-12
    );
    assert!((gabriel_graph_preservation(&grid, &rot_grid_layout).unwrap() - 1.0).abs() < 1e-12);

    // 2. Triangular crystalline lattice (equilateral triangles).
    let rows = 3usize;
    let cols = 4usize;
    let mut tx = Vec::new();
    let mut ty = Vec::new();
    let mut tri_edges = Vec::new();
    let dy = 0.5 * 3.0_f64.sqrt();
    for r in 0..rows {
        for c in 0..cols {
            let id = (r * cols + c) as u32;
            let shift = if r % 2 == 1 { 0.5 } else { 0.0 };
            tx.push(c as f64 + shift);
            ty.push(r as f64 * dy);
            if c + 1 < cols {
                tri_edges.push((id, id + 1));
            }
            if r + 1 < rows {
                tri_edges.push((id, id + cols as u32));
                if r % 2 == 0 && c > 0 {
                    tri_edges.push((id, id + cols as u32 - 1));
                } else if r % 2 == 1 && c + 1 < cols {
                    tri_edges.push((id, id + cols as u32 + 1));
                }
            }
        }
    }
    let tri = CsrGraph::<f64>::from_edges(rows * cols, &tri_edges, None, false, false).unwrap();
    let tri_layout = Layout2d::from_xy(tx, ty).unwrap();
    assert!((relative_neighborhood_preservation(&tri, &tri_layout).unwrap() - 1.0).abs() < 1e-12);
    assert!((gabriel_graph_preservation(&tri, &tri_layout).unwrap() - 1.0).abs() < 1e-12);

    // 3. Regular 10-cycle.
    let c_edges: Vec<(u32, u32)> = (0..10u32).map(|i| (i, (i + 1) % 10)).collect();
    let cycle = CsrGraph::<f64>::from_edges(10, &c_edges, None, false, false).unwrap();
    let cx: Vec<f64> = (0..10).map(|i| (i as f64 * TAU / 10.0).cos()).collect();
    let cy: Vec<f64> = (0..10).map(|i| (i as f64 * TAU / 10.0).sin()).collect();
    let cycle_layout = Layout2d::from_xy(cx, cy).unwrap();
    assert!(
        (relative_neighborhood_preservation(&cycle, &cycle_layout).unwrap() - 1.0).abs() < 1e-12
    );
    assert!((gabriel_graph_preservation(&cycle, &cycle_layout).unwrap() - 1.0).abs() < 1e-12);

    // Scrambled random positions drop preservation strictly below 1.0.
    let mut rng = DeterministicRng::from_seed_u64(123);
    let rand_layout = Layout2d::from_xy(
        (0..16).map(|_| rng.next_f64() * 10.0).collect(),
        (0..16).map(|_| rng.next_f64() * 10.0).collect(),
    )
    .unwrap();
    assert!(relative_neighborhood_preservation(&grid, &rand_layout).unwrap() < 0.7);
    assert!(gabriel_graph_preservation(&grid, &rand_layout).unwrap() < 0.7);
}

#[test]
fn test_kdtree_gg_rng_and_diameter_exact_brute_force_parity() {
    // Test random, clustered+outlier, collinear, cocircular, and duplicate point sets.
    for seed in 1..=10u64 {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let n = 80usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            if i < 65 {
                xs.push((rng.next_f64() - 0.5) * 4.0);
                ys.push((rng.next_f64() - 0.5) * 4.0);
            } else {
                // Distant outlier ring testing sector bounds across long edges.
                xs.push((rng.next_f64() - 0.5) * 200.0);
                ys.push((rng.next_f64() - 0.5) * 200.0);
            }
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        assert_eq!(
            gabriel_graph_edges(&layout).unwrap(),
            brute_gg_edges(&layout)
        );
        assert_eq!(
            relative_neighborhood_edges(&layout).unwrap(),
            brute_rng_edges(&layout)
        );

        // Verify O(N log N) rotating calipers diameter equals O(N^2) brute force.
        let got_diam = euclidean_diameter_2d(&layout);
        let mut expected_sq = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                expected_sq = expected_sq.max(dx * dx + dy * dy);
            }
        }
        assert!((got_diam - expected_sq.sqrt()).abs() < 1e-10);
    }

    // Duplicate points & collapsed layout checks.
    let dup = Layout2d::from_xy(vec![0.0, 0.0, 2.0, 2.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(gabriel_graph_edges(&dup).unwrap(), brute_gg_edges(&dup));
    assert_eq!(
        relative_neighborhood_edges(&dup).unwrap(),
        brute_rng_edges(&dup)
    );

    let collapsed = Layout2d::<f64>::zeros(6);
    let k6 = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    assert_eq!(
        relative_neighborhood_preservation(&k6, &collapsed).unwrap(),
        0.0
    );
    assert_eq!(gabriel_graph_preservation(&k6, &collapsed).unwrap(), 0.0);
    assert_eq!(node_resolution(&collapsed).unwrap(), 15.0);

    // Large (> 64) collinear, coincident, and skewed parallelogram diameter tests.
    let col100 =
        Layout2d::from_xy((0..100).map(|i| i as f64 * 0.5).collect(), vec![2.0; 100]).unwrap();
    assert!((euclidean_diameter_2d(&col100) - 49.5).abs() < 1e-12);

    let dup100 = Layout2d::<f64>::zeros(100);
    assert_eq!(euclidean_diameter_2d(&dup100), 0.0);

    let mut px = vec![0.0, 10.0, 14.0, 4.0];
    let mut py = vec![0.0, 0.0, 3.0, 3.0];
    let mut rng_p = DeterministicRng::from_seed_u64(99);
    for _ in 0..96 {
        let a = rng_p.next_f64();
        let b = rng_p.next_f64();
        px.push(a * 10.0 + b * 4.0);
        py.push(b * 3.0);
    }
    let para = Layout2d::from_xy(px, py).unwrap();
    assert!((euclidean_diameter_2d(&para) - 14.0_f64.hypot(3.0)).abs() < 1e-12);
}

#[test]
fn test_gabriel_property_and_node_resolution_brute_force_parity() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 2)], None, false, false)
        .unwrap();
    let layout = Layout2d::from_xy(vec![0.0, 4.0, 2.0, 1.9], vec![0.0, 0.0, 1.0, 0.2]).unwrap();
    let gg_prop = gabriel_graph_property(&g, &layout).unwrap();
    // Verify exact match against naive O(|E| * N) sum.
    let mut expected_gg = 0.0_f64;
    for (u, v) in extract_unique_edges(&g) {
        let ui = u as usize;
        let vi = v as usize;
        let cx = 0.5 * (layout.x()[ui] + layout.x()[vi]);
        let cy = 0.5 * (layout.y()[ui] + layout.y()[vi]);
        let r = 0.5 * (layout.x()[ui] - layout.x()[vi]).hypot(layout.y()[ui] - layout.y()[vi]);
        for i in 0..4 {
            let d: f64 = (layout.x()[i] - cx).hypot(layout.y()[i] - cy);
            expected_gg += (r - d).max(0.0).powi(2);
        }
    }
    assert!((gg_prop - expected_gg).abs() < 1e-12);

    let vr = node_resolution(&layout).unwrap();
    let d_max = euclidean_diameter_2d(&layout);
    let thresh = d_max / 2.0;
    let mut expected_vr = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
            expected_vr += (1.0 - d / thresh).max(0.0).powi(2);
        }
    }
    assert!((vr - expected_vr).abs() < 1e-12);

    let bad = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        node_resolution(&bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_empty_and_singleton_boundaries_all_12_functions() {
    // N = 0 (empty graph & empty layout)
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let l0 = Layout2d::<f64>::zeros(0);
    let t0 = build_kdtree_f64(&l0).unwrap();
    assert_eq!(gabriel_graph_property(&g0, &l0).unwrap(), 0.0);
    assert_eq!(
        gabriel_graph_property_with_tree(&g0, &l0, &t0).unwrap(),
        0.0
    );
    assert_eq!(node_resolution(&l0).unwrap(), 0.0);
    assert_eq!(node_resolution_with_tree(&l0, &t0).unwrap(), 0.0);
    assert!(relative_neighborhood_edges(&l0).unwrap().is_empty());
    assert!(
        relative_neighborhood_edges_with_tree(&l0, &t0)
            .unwrap()
            .is_empty()
    );
    assert!(gabriel_graph_edges(&l0).unwrap().is_empty());
    assert!(gabriel_graph_edges_with_tree(&l0, &t0).unwrap().is_empty());
    assert_eq!(relative_neighborhood_preservation(&g0, &l0).unwrap(), 0.0);
    assert_eq!(
        relative_neighborhood_preservation_with_tree(&g0, &l0, &t0).unwrap(),
        0.0
    );
    assert_eq!(gabriel_graph_preservation(&g0, &l0).unwrap(), 0.0);
    assert_eq!(
        gabriel_graph_preservation_with_tree(&g0, &l0, &t0).unwrap(),
        0.0
    );
    assert_proximity_determinism_and_tree_parity(&g0, &l0);

    // N = 1 (singleton graph & layout): S_RNG = 1.0, S_GG = 1.0 (0/0 convention).
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let l1 = Layout2d::from_xy(vec![3.5], vec![-2.25]).unwrap();
    let t1 = build_kdtree_f64(&l1).unwrap();
    assert_eq!(gabriel_graph_property(&g1, &l1).unwrap(), 0.0);
    assert_eq!(
        gabriel_graph_property_with_tree(&g1, &l1, &t1).unwrap(),
        0.0
    );
    assert_eq!(node_resolution(&l1).unwrap(), 0.0);
    assert_eq!(node_resolution_with_tree(&l1, &t1).unwrap(), 0.0);
    assert!(relative_neighborhood_edges(&l1).unwrap().is_empty());
    assert!(
        relative_neighborhood_edges_with_tree(&l1, &t1)
            .unwrap()
            .is_empty()
    );
    assert!(gabriel_graph_edges(&l1).unwrap().is_empty());
    assert!(gabriel_graph_edges_with_tree(&l1, &t1).unwrap().is_empty());
    assert_eq!(relative_neighborhood_preservation(&g1, &l1).unwrap(), 1.0);
    assert_eq!(
        relative_neighborhood_preservation_with_tree(&g1, &l1, &t1).unwrap(),
        1.0
    );
    assert_eq!(gabriel_graph_preservation(&g1, &l1).unwrap(), 1.0);
    assert_eq!(
        gabriel_graph_preservation_with_tree(&g1, &l1, &t1).unwrap(),
        1.0
    );
    assert_proximity_determinism_and_tree_parity(&g1, &l1);

    // f32 singleton check.
    let g1_32 = CsrGraph::<f32>::from_edges(1, &[], None, false, false).unwrap();
    let l1_32 = Layout2d::from_xy(vec![1.25_f32], vec![-0.5_f32]).unwrap();
    assert_proximity_determinism_and_tree_parity(&g1_32, &l1_32);
}

#[test]
fn test_edgeless_disconnected_and_isolated_vertex_cases() {
    // 1. Edgeless graph (|E| = 0, N = 4) with non-collocated layout.
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l_square = Layout2d::from_xy(vec![0.0, 2.0, 2.0, 0.0], vec![0.0, 0.0, 2.0, 2.0]).unwrap();
    assert_eq!(gabriel_graph_property(&g_edgeless, &l_square).unwrap(), 0.0);
    assert_eq!(
        relative_neighborhood_preservation(&g_edgeless, &l_square).unwrap(),
        0.0
    );
    assert_eq!(
        gabriel_graph_preservation(&g_edgeless, &l_square).unwrap(),
        0.0
    );
    assert!((node_resolution(&l_square).unwrap() - brute_node_resolution(&l_square)).abs() < 1e-12);
    assert_eq!(
        gabriel_graph_edges(&l_square).unwrap(),
        brute_gg_edges(&l_square)
    );
    assert_eq!(
        relative_neighborhood_edges(&l_square).unwrap(),
        brute_rng_edges(&l_square)
    );
    assert_proximity_determinism_and_tree_parity(&g_edgeless, &l_square);

    // 2. Two disjoint triangles: inter-component GG/RNG bridges lower S_RNG/S_GG < 1.0,
    // and an inter-component node placed inside a Gabriel disk contributes to gabriel_graph_property.
    let g_disjoint = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let l_disjoint = Layout2d::from_xy(
        vec![0.0, 4.0, 2.0, 2.0, 7.0, 7.0],
        vec![0.0, 0.0, 3.0, 0.5, 0.0, 2.0],
    )
    .unwrap();
    let gg_edges = gabriel_graph_edges(&l_disjoint).unwrap();
    let rng_edges = relative_neighborhood_edges(&l_disjoint).unwrap();
    assert_eq!(gg_edges, brute_gg_edges(&l_disjoint));
    assert_eq!(rng_edges, brute_rng_edges(&l_disjoint));
    let expected_s_gg = mean_jaccard_graph_vs_edges(&g_disjoint, &gg_edges);
    let expected_s_rng = mean_jaccard_graph_vs_edges(&g_disjoint, &rng_edges);
    assert!(expected_s_gg < 1.0 && expected_s_rng < 1.0);
    assert!(
        (gabriel_graph_preservation(&g_disjoint, &l_disjoint).unwrap() - expected_s_gg).abs()
            < 1e-12
    );
    assert!(
        (relative_neighborhood_preservation(&g_disjoint, &l_disjoint).unwrap() - expected_s_rng)
            .abs()
            < 1e-12
    );
    let expected_gg_prop = brute_gabriel_graph_property(&g_disjoint, &l_disjoint);
    assert!(expected_gg_prop > 0.0);
    assert!(
        (gabriel_graph_property(&g_disjoint, &l_disjoint).unwrap() - expected_gg_prop).abs()
            < 1e-12
    );
    assert_proximity_determinism_and_tree_parity(&g_disjoint, &l_disjoint);

    // 3. Connected triangle (0,1,2) + isolated degree-0 vertex (3):
    // (a) Separated isolated vertex has spatial neighbors (deg_p >= 1) so it scores 0.0.
    let g_iso =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let l_iso_sep = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 10.0],
        vec![0.0, 0.0, 3.0_f64.sqrt(), 0.0],
    )
    .unwrap();
    let sep_gg = brute_gg_edges(&l_iso_sep);
    let sep_rng = brute_rng_edges(&l_iso_sep);
    assert_eq!(gabriel_graph_edges(&l_iso_sep).unwrap(), sep_gg);
    assert_eq!(relative_neighborhood_edges(&l_iso_sep).unwrap(), sep_rng);
    assert_eq!(
        gabriel_graph_preservation(&g_iso, &l_iso_sep)
            .unwrap()
            .to_bits(),
        mean_jaccard_graph_vs_edges(&g_iso, &sep_gg).to_bits()
    );
    assert_eq!(
        relative_neighborhood_preservation(&g_iso, &l_iso_sep)
            .unwrap()
            .to_bits(),
        mean_jaccard_graph_vs_edges(&g_iso, &sep_rng).to_bits()
    );
    assert!(gabriel_graph_preservation(&g_iso, &l_iso_sep).unwrap() < 1.0);
    assert!(relative_neighborhood_preservation(&g_iso, &l_iso_sep).unwrap() < 1.0);

    // (b) Isolated vertex co-located onto vertex 0 at (0.0, 0.0).
    let l_iso_coloc = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 0.0],
        vec![0.0, 0.0, 3.0_f64.sqrt(), 0.0],
    )
    .unwrap();
    let coloc_gg = brute_gg_edges(&l_iso_coloc);
    let coloc_rng = brute_rng_edges(&l_iso_coloc);
    assert_eq!(gabriel_graph_edges(&l_iso_coloc).unwrap(), coloc_gg);
    assert_eq!(
        relative_neighborhood_edges(&l_iso_coloc).unwrap(),
        coloc_rng
    );
    assert!(
        !coloc_gg
            .iter()
            .any(|&(u, v)| u == 0 || v == 0 || u == 3 || v == 3)
    );
    assert!(coloc_rng.iter().any(|&(u, v)| u == 3 || v == 3));
    assert!((gabriel_graph_preservation(&g_iso, &l_iso_coloc).unwrap() - 0.5).abs() < 1e-12);
    assert_eq!(
        relative_neighborhood_preservation(&g_iso, &l_iso_coloc)
            .unwrap()
            .to_bits(),
        mean_jaccard_graph_vs_edges(&g_iso, &coloc_rng).to_bits()
    );
    assert_proximity_determinism_and_tree_parity(&g_iso, &l_iso_coloc);
}
