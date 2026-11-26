use egraph_core::{DeterministicRng, Layout2d};

use super::super::SweepLineCrossings;
use super::helpers::{assert_matches_brute_force, assert_streaming_matches_find};

#[test]
fn test_sweep_line_large_tiny_steep_scales_and_repros() {
    // 1. Large 1e8 scale: multi-way x_probe must not underflow to x_evt.
    let large_layout = Layout2d::from_xy(
        vec![
            1e8 + 1.0,
            1e8 + 5.0,
            1e8,
            1e8 + 5.0,
            1e8 + 3.0,
            1e8 + 4.0,
            1e8 + 5.0,
            1e8 + 4.0,
        ],
        vec![
            1e8 + 3.0,
            1e8 + 5.0,
            1e8,
            1e8 + 6.0,
            1e8 + 6.0,
            1e8 + 3.0,
            1e8 + 2.0,
            1e8 + 3.0,
        ],
    )
    .unwrap();
    let large_edges = vec![(0, 1), (2, 3), (4, 5), (6, 7)];
    assert_matches_brute_force("large_1e8_repro", &large_edges, &large_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&large_edges, &large_layout).unwrap(),
        4
    );

    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(1000 + seed);
        let n = 25usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| 1e8 + rng.gen_range_usize(11) as f64)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| 1e8 + rng.gen_range_usize(11) as f64)
            .collect();
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let mut es = Vec::new();
        for _ in 0..45 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            es.push((u, v));
        }
        assert_matches_brute_force(&format!("large_1e8_seed_{seed}"), &es, &l);
    }

    // 2. Tiny 1e-8 scale: co-located Start/End ULP tie and post-end_batch slope ordering.
    let mut tx = vec![0.0; 29];
    let mut ty = vec![0.0; 29];
    let coords = [
        (10usize, (2e-8, 6e-8)),
        (14, (5e-8, 0.0)),
        (3, (0.0, 3e-8)),
        (24, (3e-8, 4e-8)),
        (25, (2e-8, 2e-8)),
        (6, (4e-8, 5e-8)),
        (13, (6e-8, 3e-8)),
        (28, (3e-8, 4e-8)),
        (2, (5e-8, 2e-8)),
        (9, (1e-8, 3e-8)),
        (1, (4e-8, 6e-8)),
        (11, (1e-8, 3e-8)),
    ];
    for (idx, (x, y)) in coords {
        tx[idx] = x;
        ty[idx] = y;
    }
    let tiny_layout = Layout2d::from_xy(tx, ty).unwrap();
    let tiny_edges = vec![
        (10, 14),
        (3, 24),
        (24, 25),
        (3, 6),
        (13, 28),
        (2, 9),
        (1, 11),
    ];
    assert_matches_brute_force("tiny_1e-8_repro", &tiny_edges, &tiny_layout);

    // Also test `(k as f64) * 1e-8` construction (1-ULP difference on 6e-8) covering the 1-ULP quantization boundary at 6e-8.
    let mut sx = vec![0.0; 29];
    let mut sy = vec![0.0; 29];
    let scaled = [
        (10usize, (2, 6)),
        (14, (5, 0)),
        (3, (0, 3)),
        (24, (3, 4)),
        (25, (2, 2)),
        (6, (4, 5)),
        (13, (6, 3)),
        (28, (3, 4)),
        (2, (5, 2)),
        (9, (1, 3)),
        (1, (4, 6)),
        (11, (1, 3)),
    ];
    for (idx, (ix, iy)) in scaled {
        sx[idx] = (ix as f64) * 1e-8;
        sy[idx] = (iy as f64) * 1e-8;
    }
    let tiny_scaled_layout = Layout2d::from_xy(sx, sy).unwrap();
    assert_matches_brute_force("tiny_1e-8_scaled_repro", &tiny_edges, &tiny_scaled_layout);

    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(2000 + seed);
        let n = 30usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 1e-8)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 1e-8)
            .collect();
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let mut es = Vec::new();
        for _ in 0..50 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            es.push((u, v));
        }
        assert_matches_brute_force(&format!("tiny_1e-8_seed_{seed}"), &es, &l);
    }

    // 3. Steep-slope x=100, dx=1e-8 ULP drift repro + stress seeds.
    let steep_layout = Layout2d::from_xy(
        vec![
            100.0 + 2e-8,
            100.0 + 3e-8,
            100.0,
            100.0 + 2e-8,
            100.0 + 1e-8,
            100.0 + 3e-8,
        ],
        vec![3.5, 9.0, 6.0, 3.5, 1.5, 5.5],
    )
    .unwrap();
    let steep_edges = vec![(0, 1), (2, 3), (4, 5), (5, 4)];
    assert_matches_brute_force("steep_slope_repro", &steep_edges, &steep_layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&steep_edges, &steep_layout).unwrap(),
        5
    );

    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(3000 + seed);
        let n = 20usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| 100.0 + (rng.gen_range_usize(7) as f64) * 1e-8)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(20) as f64) * 0.5)
            .collect();
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let mut es = Vec::new();
        for _ in 0..40 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            es.push((u, v));
        }
        assert_matches_brute_force(&format!("steep_slope_seed_{seed}"), &es, &l);
    }
}

#[test]
fn test_randomized_clustered_and_steep_stress_regression() {
    // 1. 40 clustered seeds (4 tight clusters of radius 1e-4 with inter-cluster chords).
    for seed in 1..=40u64 {
        let mut rng = DeterministicRng::from_seed_u64(10_000 + seed);
        let centers = [(0.0, 0.0), (50.0, 20.0), (50.0, 80.0), (100.0, 0.0)];
        let n = 28usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let (cx, cy) = centers[i % 4];
            xs.push(cx + (rng.next_f64() - 0.5) * 1e-4);
            ys.push(cy + (rng.next_f64() - 0.5) * 1e-4);
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..80 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("clustered_seed_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("clustered_seed_{seed}_stream"), &edges, &layout);
    }

    // 2. 40 steep seeds at x_off = 100 and x_off = 1e6 with dx ~ 1e-8.
    for seed in 1..=40u64 {
        let mut rng = DeterministicRng::from_seed_u64(20_000 + seed);
        let x_off = if seed <= 20 { 100.0 } else { 1e6 };
        let n = 24usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| x_off + (rng.gen_range_usize(8) as f64) * 1e-8)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(24) as f64) * 0.5)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..55 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("steep_stress_seed_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("steep_stress_seed_{seed}_stream"), &edges, &layout);
    }

    // 3. 15 tiny 1e-165 scale seeds (det underflows to 0.0) verifying det == 0.0 fallback.
    for seed in 1..=15u64 {
        let mut rng = DeterministicRng::from_seed_u64(30_000 + seed);
        let n = 20usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 1e-165)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(11) as f64) * 1e-165)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..40 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("tiny_1e-165_seed_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("tiny_1e-165_seed_{seed}_stream"), &edges, &layout);
    }
}
