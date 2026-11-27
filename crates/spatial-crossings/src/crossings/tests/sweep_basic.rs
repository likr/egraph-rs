use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d};

use super::super::{
    SweepLineCrossings, TorusView, sampled_crossing_number_edges, torus_crossing_edges,
};
use super::helpers::assert_matches_brute_force;

#[test]
fn test_sweep_line_planar_nonplanar_grid_and_star() {
    // 1. Planar 4x4 grid (0 crossings).
    let mut grid_x = Vec::new();
    let mut grid_y = Vec::new();
    let mut grid_edges = Vec::new();
    for r in 0..4u32 {
        for c in 0..4u32 {
            let id = r * 4 + c;
            grid_x.push(c as f64);
            grid_y.push(r as f64);
            if c + 1 < 4 {
                grid_edges.push((id, id + 1));
            }
            if r + 1 < 4 {
                grid_edges.push((id, id + 4));
            }
        }
    }
    let grid_layout = Layout2d::from_xy(grid_x, grid_y).unwrap();
    assert_matches_brute_force("grid_4x4", &grid_edges, &grid_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&grid_edges, &grid_layout).unwrap(),
        0
    );

    // 2. K5 pentagon + pentagram (non-planar, 5 crossings).
    let k5_x: Vec<f64> = (0..5)
        .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).cos())
        .collect();
    let k5_y: Vec<f64> = (0..5)
        .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).sin())
        .collect();
    let mut k5_edges = Vec::new();
    for u in 0..5u32 {
        for v in (u + 1)..5u32 {
            k5_edges.push((u, v));
        }
    }
    let k5_layout = Layout2d::from_xy(k5_x, k5_y).unwrap();
    assert_matches_brute_force("K5", &k5_edges, &k5_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&k5_edges, &k5_layout).unwrap(),
        5
    );

    // 3. K_{3,3} bipartite layout (9 crossings).
    let k33_layout = Layout2d::from_xy(
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let mut k33_edges = Vec::new();
    for u in 0..3u32 {
        for v in 3..6u32 {
            k33_edges.push((u, v));
        }
    }
    assert_matches_brute_force("K3,3", &k33_edges, &k33_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&k33_edges, &k33_layout).unwrap(),
        9
    );

    // 4. Star graph with 64 rays sharing center vertex 0 (0 crossings).
    let mut star_x = vec![0.0];
    let mut star_y = vec![0.0];
    let mut star_edges = Vec::new();
    for i in 1..=64u32 {
        let ang = (i as f64) * std::f64::consts::TAU / 64.0;
        star_x.push(ang.cos());
        star_y.push(ang.sin());
        star_edges.push((0, i));
    }
    let star_layout = Layout2d::from_xy(star_x, star_y).unwrap();
    assert_matches_brute_force("star_64", &star_edges, &star_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&star_edges, &star_layout).unwrap(),
        0
    );
}

#[test]
fn test_sweep_line_degeneracies_horizontal_vertical_collinear_and_shared_coords() {
    // Horizontal, vertical, collinear overlapping, collinear disjoint, shared endpoints,
    // coincident distinct-ID endpoints, and multi-way intersection at (1.0, 1.0).
    let layout = Layout2d::from_xy(
        vec![
            0.0, 3.0, 1.0, 4.0, 5.0, 6.0, // horizontal / collinear on y=0
            1.0, 1.0, 1.0, 1.0, // vertical overlapping on x=1
            0.0, 2.0, 0.0, 2.0, 0.0, 2.0, // multi-way X crossing at (1, 1)
            0.0, 1.0, // coincident distinct vertex at (0, 0) & T-junction at (1, 0)
        ],
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            -1.0, 2.0, 0.5, 3.0, //
            0.0, 2.0, 2.0, 0.0, 1.0, 1.0, //
            0.0, 0.0, //
        ],
    )
    .unwrap();
    let edges = vec![
        (0, 1),   // (0,0)-(3,0)
        (2, 3),   // (1,0)-(4,0) collinear overlap with (0,1)
        (4, 5),   // (5,0)-(6,0) collinear disjoint
        (6, 7),   // vertical (1,-1)-(1,2)
        (8, 9),   // vertical (1,0.5)-(1,3) overlaps (6,7)
        (10, 11), // diag (0,0)-(2,2)
        (12, 13), // diag (0,2)-(2,0)
        (14, 15), // horiz (0,1)-(2,1) multi-way through (1,1)
        (16, 17), // coincident vertex (0,0) to T-junction (1,0)
        (0, 10),  // self-loop-like shared coord with distinct IDs
        (1, 2),   // shared endpoint test
    ];
    assert_matches_brute_force("degenerate_suite", &edges, &layout);

    // Bridged-hub cluster at (1.5, 0.75): a star at `hub`, an edge bridging across to
    // a non-incident chord, plus passing/ending/starting edges meeting at the exact hub point.
    let bridge_layout = Layout2d::from_xy(
        vec![
            1.5, 2.5, 2.5, 2.25, // hub=0 and 3 rays extending right
            0.5, 2.5, // 4->5 passes through (1.5, 0.75) without sharing hub
            0.75, 1.5, // 6->7 ends at (1.5, 0.75) on distinct vertex
            1.0, 2.0, // 8->9 passes through (1.5, 0.75) without sharing hub
            1.5, 2.5, // 10->11 starts at (1.5, 0.75) on distinct vertex
        ],
        vec![
            0.75, 1.75, -0.25, 0.75, //
            0.25, 1.25, //
            0.0, 0.75, //
            1.25, 0.25, //
            0.75, 1.25, //
        ],
    )
    .unwrap();
    let bridge_edges = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (3, 8), // shares vertex 3 with (0, 3) and vertex 8 with (8, 9)
    ];
    assert_matches_brute_force("bridged_hub_cluster", &bridge_edges, &bridge_layout);

    // Random dense stress test across 24 quantized seeds forcing dense multi-way ties.
    for seed in 1..=24u64 {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let n = 25usize;
        // Include quantized grid coordinates to force frequent collinear/vertical/shared ties.
        let xs: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 0.25)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 0.25)
            .collect();
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let mut es = Vec::new();
        for _ in 0..45 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            es.push((u, v));
        }
        assert_matches_brute_force(&format!("random_quantized_seed_{seed}"), &es, &l);
    }

    // Large 1024-ray star centered at (0, 0) verifying O(|E| log |E|) Misra-Gries hub pruning.
    let mut big_x = vec![0.0];
    let mut big_y = vec![0.0];
    let mut big_edges = Vec::with_capacity(1024);
    for i in 1..=1024u32 {
        let theta = (i as f64) * (2.0 * std::f64::consts::PI / 1024.0);
        big_x.push(theta.cos());
        big_y.push(theta.sin());
        big_edges.push((0, i));
    }
    let big_star = Layout2d::from_xy(big_x, big_y).unwrap();
    assert_eq!(
        SweepLineCrossings::count_edges(&big_edges, &big_star).unwrap(),
        0
    );
}

#[test]
fn test_sweep_line_angles_csr_torus_and_sampling() {
    // Right-angle crossing gives cos_sq == 0 and min_angle == PI/2.
    let layout = Layout2d::from_xy(vec![-1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -1.0, 1.0]).unwrap();
    let csr = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let summary = SweepLineCrossings::summarize(&csr, &layout).unwrap();
    assert_eq!(summary.count, 1);
    assert!(summary.angle_cos_sq_sum.abs() < 1e-12);
    assert!((summary.min_angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

    // Torus crossing test: two wrapping edges that cross after periodic boundary split.
    let torus_layout =
        Layout2d::from_xy(vec![0.9, 0.1, 0.9, 0.1], vec![0.1, 0.9, 0.9, 0.1]).unwrap();
    let torus_view = TorusView::new(&torus_layout, 1.0, 1.0).unwrap();
    let torus_csr = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let torus_crossings = torus_crossing_edges(&torus_csr, &torus_view).unwrap();
    assert!(!torus_crossings.is_empty());

    // Sampled crossing number determinism and unbiased convergence on a dense bipartite layout.
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bedges = Vec::new();
    for i in 0..10u32 {
        bx.push(i as f64);
        by.push(0.0);
    }
    for j in 0..10u32 {
        bx.push(j as f64);
        by.push(1.0);
    }
    for u in 0..10u32 {
        for v in 10..20u32 {
            bedges.push((u, v));
        }
    }
    let blayout = Layout2d::from_xy(bx, by).unwrap();
    let exact = SweepLineCrossings::count_edges(&bedges, &blayout).unwrap() as f64;
    let mut r1 = DeterministicRng::from_seed_u64(77);
    let mut r2 = DeterministicRng::from_seed_u64(77);
    let est1 = sampled_crossing_number_edges(&bedges, &blayout, 4000, &mut r1).unwrap();
    let est2 = sampled_crossing_number_edges(&bedges, &blayout, 4000, &mut r2).unwrap();
    assert_eq!(est1, est2);
    assert!(
        (est1 - exact).abs() / exact < 0.08,
        "est1={est1}, exact={exact}"
    );

    // Validation errors.
    let bad = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        SweepLineCrossings::count_edges(&[(0, 1)], &bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        sampled_crossing_number_edges(&[(0, 1)], &layout, 0, &mut r1),
        Err(CoreError::InvalidInput(_))
    ));
}
