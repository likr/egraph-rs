use egraph_core::{DeterministicRng, Layout2d};

use super::super::torus::{TorusView, torus_crossing_edges_from_edges, torus_edge_segments};
use super::helpers::{assert_matches_brute_force, assert_torus_matches_oracle};

#[test]
fn test_torus_edge_segments_bounds_and_crossing_parity() {
    // Explicit check that shared-vertex wrapped pieces swap inside (0, 1)^2 and never distort
    // subsequent crossings with unrelated edges, and all split coordinates stay in [0, 1)^2.
    let adj_wrap_layout = Layout2d::from_xy(
        vec![0.1, 0.9, 0.95, 0.92, 0.92],
        vec![0.2, 0.85, 0.15, 0.4, 0.89],
    )
    .unwrap();
    let adj_wrap_edges = vec![(0, 1), (0, 2), (3, 4)];
    for extreme in [
        (0.48_f64, 0.95, 0.99, 0.01),
        (0.02, 0.98, 0.99, 0.48),
        (0.45, 0.05, 0.96, 0.98),
    ] {
        for p in torus_edge_segments(extreme.0, extreme.1, extreme.2, extreme.3).unwrap() {
            for coord in p {
                assert!(
                    coord.is_finite() && (0.0..1.0).contains(&coord),
                    "extreme torus segment coord out of [0, 1): {p:?}"
                );
            }
        }
    }
    for &(u, v) in &adj_wrap_edges {
        let pieces = torus_edge_segments(
            adj_wrap_layout.x()[u as usize],
            adj_wrap_layout.y()[u as usize],
            adj_wrap_layout.x()[v as usize],
            adj_wrap_layout.y()[v as usize],
        )
        .unwrap();
        for p in pieces {
            for coord in p {
                assert!(
                    coord.is_finite() && (0.0..1.0).contains(&coord),
                    "torus segment coord out of [0, 1): {p:?}"
                );
            }
        }
    }
    let adj_wrap_view = TorusView::new(&adj_wrap_layout, 1.0, 1.0).unwrap();
    let got_adj = torus_crossing_edges_from_edges(&adj_wrap_edges, &adj_wrap_view).unwrap();
    assert!(!got_adj.is_empty());

    // Compare torus_crossing_edges_from_edges against O(|S|^2) brute force across 30 random seeds.
    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(4000 + seed);
        let n = 18usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(33) as f64) / 33.0)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(33) as f64) / 33.0)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let view = TorusView::new(&layout, 1.0, 1.0).unwrap();
        let mut edges = Vec::new();
        for _ in 0..32 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }

        assert_torus_matches_oracle(&format!("seed {seed}"), &edges, &view);
    }
}

#[test]
fn test_torus_steep_and_clustered_parity() {
    // Steep wrapped torus sub-segments (dx ~ 1e-8 across periodic boundary).
    for seed in 1..=25u64 {
        let mut rng = DeterministicRng::from_seed_u64(5000 + seed);
        let n = 20usize;
        let base_x = 0.99999998;
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    (base_x + (rng.gen_range_usize(10) as f64) * 1e-9).rem_euclid(1.0)
                } else {
                    ((rng.gen_range_usize(10) as f64) * 1e-9).rem_euclid(1.0)
                }
            })
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(20) as f64) / 20.0)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let view = TorusView::new(&layout, 1.0, 1.0).unwrap();
        let mut edges = Vec::new();
        for _ in 0..35 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        assert_torus_matches_oracle(&format!("torus steep seed {seed}"), &edges, &view);
    }

    // Clustered torus layout (4 tight clusters of radius 1e-4 on [0, 1)^2).
    for seed in 1..=25u64 {
        let mut rng = DeterministicRng::from_seed_u64(6000 + seed);
        let centers = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)];
        let n = 24usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let (cx, cy) = centers[i % 4];
            xs.push((cx + (rng.next_f64() - 0.5) * 1e-4).rem_euclid(1.0));
            ys.push((cy + (rng.next_f64() - 0.5) * 1e-4).rem_euclid(1.0));
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let view = TorusView::new(&layout, 1.0, 1.0).unwrap();
        let mut edges = Vec::new();
        for _ in 0..45 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        assert_torus_matches_oracle(&format!("torus clustered seed {seed}"), &edges, &view);
    }
}

#[test]
fn test_torus_corner_wrap_sort_tie_break() {
    // Degenerate corner-wrap segment `(0.0, 0.0) -> (1 - eps, 1 - eps)` produces sub-segments
    // sharing left endpoint `(0.0, 0.0)` and previously exposed `-0.0` / sort tie-break gaps.
    let eps_max = 1.0 - f64::EPSILON;
    let layout = Layout2d::from_xy(
        vec![0.0, eps_max, 0.05, 0.95, 0.25, 0.75, 0.1, 0.9],
        vec![0.0, eps_max, 0.95, 0.05, 0.01, 0.99, 0.5, 0.5],
    )
    .unwrap();
    let view = TorusView::new(&layout, 1.0, 1.0).unwrap();
    let edges = vec![(0, 1), (2, 3), (4, 5), (6, 7), (0, 3), (1, 2)];
    assert_torus_matches_oracle("torus corner wrap", &edges, &view);
}

#[test]
fn test_f32_sweep_and_out_of_unit_torus_oracles() {
    // 1. Multi-seed Layout2d<f32> randomized & quantized degenerate sweep-line vs O(M^2) oracle.
    for seed in 1..=20u64 {
        let mut rng = DeterministicRng::from_seed_u64(95_000 + seed);
        let n = 22usize;
        let xs: Vec<f32> = (0..n)
            .map(|_| ((rng.gen_range_usize(11) as f32) - 5.0) * 0.25)
            .collect();
        let ys: Vec<f32> = (0..n)
            .map(|_| ((rng.gen_range_usize(11) as f32) - 5.0) * 0.25)
            .collect();
        let layout32 = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..48 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("f32_quantized_{seed}"), &edges, &layout32);
    }

    // 2. Out-of-[0, 1) periodic torus wrap tests (negative and >= 1.0 coordinates) in f64 & f32.
    for seed in 1..=12u64 {
        let mut rng = DeterministicRng::from_seed_u64(96_000 + seed);
        let n = 12usize;
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 5.0 - 2.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 5.0 - 2.0).collect();
        let layout64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let view64 = TorusView::new(&layout64, 1.0, 1.0).unwrap();
        let mut edges = Vec::new();
        for _ in 0..16 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        assert_torus_matches_oracle(&format!("out_of_unit_torus_f64_{seed}"), &edges, &view64);

        let layout32 = Layout2d::from_xy(
            xs.iter().map(|&v| v as f32).collect(),
            ys.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let view32 = TorusView::new(&layout32, 1.0, 1.0).unwrap();
        assert_torus_matches_oracle(&format!("out_of_unit_torus_f32_{seed}"), &edges, &view32);
    }
}
