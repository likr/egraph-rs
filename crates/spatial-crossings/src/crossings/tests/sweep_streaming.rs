use egraph_core::{CsrGraph, DeterministicRng, Layout2d};

use super::super::SweepLineCrossings;
use super::super::geometry::build_segments;
use super::super::sweep::sweep_segments_with_stats;
use super::helpers::{assert_matches_brute_force, assert_streaming_matches_find};

#[test]
fn test_streaming_sweep_collinear_zero_length_shared_tips_and_eps_bands() {
    // 1. Collinear overlaps (horizontal, vertical, diagonal), touching tips (a.x2 == b.x1),
    // containment, duplicate coordinates with distinct vertex IDs, zero-length segments,
    // self-loops, and shared right endpoints.
    let layout = Layout2d::from_xy(
        vec![
            0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 0.0, 5.0, // horizontal collinear on y=0
            0.0, 2.0, 1.0, 3.0, 2.0, 4.0, // diagonal collinear on y=x
            1.5, 1.5, 1.5, 1.5, // zero-length segment (14,15) and vertical (16,17)
            0.5, 1.5, // zero-length at (0.5, 0.5) on y=x
            3.0, 5.0, 3.0, 5.0, // shared right endpoint at (5, 2)
            0.0, 4.0, // duplicate diagonal with distinct vertex IDs
        ],
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 1.0, 3.0, 2.0, 4.0, //
            0.5, 0.5, -1.0, 4.0, //
            0.5, 0.5, //
            1.0, 2.0, 3.0, 2.0, //
            0.0, 4.0, //
        ],
    )
    .unwrap();
    let edges = vec![
        (0, 1),   // [0,2] on y=0
        (2, 3),   // [2,4] touching tip on y=0
        (4, 5),   // [1,3] partial overlap with (0,1) and (2,3)
        (6, 7),   // [0,5] containment on y=0
        (8, 9),   // [0,2] on y=x
        (10, 11), // [1,3] on y=x
        (12, 13), // [2,4] touching tip on y=x
        (14, 15), // zero-length at (1.5, 0.5)
        (16, 17), // vertical x=1.5 crossing y=0, y=x, and zero-length (14,15)
        (18, 19), // zero-length at (0.5, 0.5) on y=x
        (20, 21), // ends at (5, 2)
        (22, 23), // shares right endpoint (5, 2) with (20, 21) via distinct IDs
        (24, 25), // [0,4] duplicate diagonal
        (0, 0),   // self-loop
        (14, 14), // self-loop on zero-length vertex
    ];
    assert_matches_brute_force("collinear_zero_shared_tips", &edges, &layout);
    assert_streaming_matches_find("collinear_zero_shared_tips_find", &edges, &layout);

    // 2. Near-parallel eps_y band stress at y = 100.0 (|slope_a - slope_b| ~ 1e-15)
    // plus crossing transverse segments and identical x_int ties.
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bedges = Vec::new();
    for i in 0..8u32 {
        let x0 = 0.0;
        let x1 = 10.0;
        let dy = (i as f64) * 1e-13;
        let y0 = 100.0 + dy;
        let y1 = 100.0 + (7 - i) as f64 * 1e-13;
        let u = bx.len() as u32;
        bx.push(x0);
        by.push(y0);
        bx.push(x1);
        by.push(y1);
        bedges.push((u, u + 1));
    }
    // Add transverse segments crossing the band at distinct and shared x positions.
    for k in 0..6u32 {
        let u = bx.len() as u32;
        let xc = 1.0 + (k as f64) * 1.5;
        bx.push(xc - 0.5);
        by.push(99.0);
        bx.push(xc + 0.5);
        by.push(101.0);
        bedges.push((u, u + 1));
    }
    let band_layout = Layout2d::from_xy(bx, by).unwrap();
    assert_matches_brute_force("near_parallel_eps_band_y100", &bedges, &band_layout);
    assert_streaming_matches_find("near_parallel_eps_band_y100_find", &bedges, &band_layout);

    // 3. Deferred crossings (x_int > sweep_x with intermediate segments starting/ending)
    // and early dt_probe swap stress.
    for seed in 1..=20u64 {
        let mut rng = DeterministicRng::from_seed_u64(9000 + seed);
        let n = 32usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| rng.gen_range_usize(1000) as f64 * 1e-3)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| 50.0 + rng.gen_range_usize(1000) as f64 * 1e-12)
            .collect();
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let mut es = Vec::new();
        for _ in 0..60 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            es.push((u, v));
        }
        assert_matches_brute_force(&format!("eps_band_random_seed_{seed}"), &es, &l);
    }
}

#[test]
fn test_streaming_sweep_dense_high_crossing_and_bounded_memory() {
    // 1. K_{40,40} bipartite layout on parallel lines x=0 and x=1 (many identical x_int ties).
    let mut bx = Vec::with_capacity(80);
    let mut by = Vec::with_capacity(80);
    for i in 0..40u32 {
        bx.push(0.0);
        by.push(i as f64);
    }
    for j in 0..40u32 {
        bx.push(1.0);
        by.push(j as f64);
    }
    let mut bedges = Vec::with_capacity(1600);
    for u in 0..40u32 {
        for v in 40..80u32 {
            bedges.push((u, v));
        }
    }
    let blayout = Layout2d::from_xy(bx, by).unwrap();
    assert_streaming_matches_find("K_40_40", &bedges, &blayout);

    let bsegs = build_segments(&bedges, &blayout).unwrap();
    let mut bcount = 0usize;
    let b_peak = sweep_segments_with_stats(&bsegs, |_, _| {
        bcount += 1;
    });
    // Exact analytical crossing count for K_{40,40}: C(40,2)^2 = 780^2 = 608,400.
    assert_eq!(bcount, 780 * 780);
    assert!(
        b_peak * 100 < bcount,
        "K_{{40,40}} active band tracking ({b_peak}) must stay << total crossings ({bcount})"
    );

    // 2. K_50 convex circle layout (C(50, 4) = 230,300 crossings).
    let k50_x: Vec<f64> = (0..50)
        .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).cos())
        .collect();
    let k50_y: Vec<f64> = (0..50)
        .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin())
        .collect();
    let mut k50_edges = Vec::with_capacity(50 * 49 / 2);
    for u in 0..50u32 {
        for v in (u + 1)..50u32 {
            k50_edges.push((u, v));
        }
    }
    let k50_layout = Layout2d::from_xy(k50_x, k50_y).unwrap();
    assert_streaming_matches_find("K_50_circle", &k50_edges, &k50_layout);

    let k50_segs = build_segments(&k50_edges, &k50_layout).unwrap();
    let mut k50_count = 0usize;
    let k50_peak = sweep_segments_with_stats(&k50_segs, |_, _| {
        k50_count += 1;
    });
    assert_eq!(k50_count, 230_300);
    assert!(
        k50_peak * 20 < k50_count,
        "K_50 active band tracking ({k50_peak}) must stay << total crossings ({k50_count})"
    );

    // 3. Random dense graphs (N=80, M=600) verifying CSR count/summarize parity and tiny active band.
    for seed in 1..=5u64 {
        let mut rng = DeterministicRng::from_seed_u64(7000 + seed);
        let n = 80usize;
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::with_capacity(600);
        for _ in 0..600 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        assert_streaming_matches_find(&format!("random_dense_seed_{seed}"), &edges, &layout);

        let segs = build_segments(&edges, &layout).unwrap();
        let mut c = 0usize;
        let peak = sweep_segments_with_stats(&segs, |_, _| {
            c += 1;
        });
        assert!(c > 10_000, "expected high-crossing dense layout, got {c}");
        assert!(
            peak * 100 < c,
            "random_dense_seed_{seed}: active band ({peak}) must stay << total crossings ({c})"
        );

        // Also verify CsrGraph entrypoints count() and summarize() match find_crossings().
        let csr = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();
        let csr_pairs = SweepLineCrossings::find_crossings(&csr, &layout).unwrap();
        let csr_count = SweepLineCrossings::count(&csr, &layout).unwrap();
        let csr_summary = SweepLineCrossings::summarize(&csr, &layout).unwrap();
        assert_eq!(csr_count, csr_pairs.len());
        assert_eq!(csr_summary.count, csr_pairs.len());
        assert_eq!(
            csr_summary.min_angle_rad,
            csr_pairs
                .iter()
                .filter_map(|p| p.angle_rad)
                .reduce(f64::min)
        );
    }
}
