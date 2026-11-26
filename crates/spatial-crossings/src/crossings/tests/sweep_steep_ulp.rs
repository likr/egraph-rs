use egraph_core::{DeterministicRng, Layout2d};

use super::super::SweepLineCrossings;
use super::helpers::{assert_matches_brute_force, assert_streaming_matches_find};

#[test]
#[allow(clippy::excessive_precision)]
fn test_steep_ulp_and_multiway_cluster_crossings() {
    // 1. Two-edge ULP drift at xi + past-crossing on_band_pair double-count guard.
    let layout_a = Layout2d::from_xy(
        vec![
            -3.04805189247130648e-5,
            5.00000097576867404e1,
            3.31165387303849953e-5,
            4.99999652101946026e1,
        ],
        vec![
            -7.55507051547136343e-6,
            2.00000408856693603e1,
            3.83695954409438164e-5,
            1.99999834070688145e1,
        ],
    )
    .unwrap();
    let edges_a = vec![(0, 1), (2, 3)];
    assert_matches_brute_force("steep_ulp_drift_pair", &edges_a, &layout_a);
    assert_streaming_matches_find("steep_ulp_drift_pair_stream", &edges_a, &layout_a);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_a, &layout_a).unwrap(),
        1
    );

    // 2. Four-edge cluster where reverting completed swap blocked (0, 2).
    let layout_b = Layout2d::from_xy(
        vec![
            4.99999832431687921e1,
            9.99999952436454294e1,
            4.99999784950573343e1,
            1.00000017056024674e2,
            -9.83780916952248230e-7,
            4.99999681576062898e1,
        ],
        vec![
            8.00000140893703815e1,
            1.69716668579220089e-5,
            7.99999695116295868e1,
            4.72816566238181744e-5,
            4.90065288854551163e-5,
            1.99999520082606139e1,
        ],
    )
    .unwrap();
    let edges_b = vec![(0, 1), (2, 3), (4, 3), (5, 1)];
    assert_matches_brute_force("four_edge_completed_swap", &edges_b, &layout_b);
    assert_streaming_matches_find("four_edge_completed_swap_stream", &edges_b, &layout_b);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_b, &layout_b).unwrap(),
        3
    );

    // 3. Eight-edge steep cluster at x ~ 1e6, dx ~ 4e-8.
    let mut xc = vec![0.0; 20];
    let mut yc = vec![0.0; 20];
    let pts_c = [
        (9usize, 1.00000000000005995e6, 2.5),
        (6, 1.00000000000002002e6, 9.5),
        (18, 1.00000000000001001e6, 4.0),
        (3, 1.00000000000004994e6, 5.5),
        (2, 1.00000000000000000e6, 0.0),
        (8, 1.00000000000004994e6, 4.0),
        (10, 1.00000000000003004e6, 4.0),
        (14, 1.00000000000002002e6, 4.0),
        (5, 1.00000000000001001e6, 7.0),
        (11, 1.00000000000004994e6, 8.0),
        (7, 1.00000000000003004e6, 7.5),
        (19, 1.00000000000004005e6, 8.0),
        (0, 1.00000000000003004e6, 2.5),
    ];
    for (idx, px, py) in pts_c {
        xc[idx] = px;
        yc[idx] = py;
    }
    let layout_c = Layout2d::from_xy(xc, yc).unwrap();
    let edges_c = vec![
        (9, 6),
        (18, 3),
        (2, 8),
        (6, 10),
        (14, 5),
        (18, 6),
        (11, 7),
        (6, 19),
    ];
    assert_matches_brute_force("eight_edge_steep_cluster", &edges_c, &layout_c);
    assert_streaming_matches_find("eight_edge_steep_cluster_stream", &edges_c, &layout_c);

    // 4. Five-edge steep cluster with independent vertices at x ~ 1e6.
    let layout_d = Layout2d::from_xy(
        vec![
            1.00000000000001001e6,
            1.00000000000004994e6,
            1.00000000000001001e6,
            1.00000000000004005e6,
            1.00000000000001001e6,
            1.00000000000003004e6,
            1.00000000000001001e6,
            1.00000000000002002e6,
            1.00000000000002002e6,
            1.00000000000003004e6,
        ],
        vec![2.0, 8.0, 7.0, 8.0, 4.0, 2.5, 4.0, 9.5, 4.0, 4.0],
    )
    .unwrap();
    let edges_d = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    assert_matches_brute_force("five_edge_steep_cluster", &edges_d, &layout_d);
    assert_streaming_matches_find("five_edge_steep_cluster_stream", &edges_d, &layout_d);

    // Large non-hub multi-way intersection (70 segments crossing at (1.0, 1.0)) exercising
    // the k > 64 merge-sort inversion path in `emit_flipped_pairs`.
    let k_multi = 70usize;
    let mut mx = Vec::with_capacity(k_multi * 2);
    let mut my = Vec::with_capacity(k_multi * 2);
    let mut medges = Vec::with_capacity(k_multi);
    for i in 0..k_multi {
        let slope = -1.0 + 2.0 * (i as f64) / ((k_multi - 1) as f64);
        mx.push(0.0);
        my.push(1.0 - slope);
        mx.push(2.0);
        my.push(1.0 + slope);
        let u = (2 * i) as u32;
        medges.push((u, u + 1));
    }
    let mlayout = Layout2d::from_xy(mx, my).unwrap();
    assert_matches_brute_force("multiway_70_non_hub", &medges, &mlayout);
    assert_streaming_matches_find("multiway_70_non_hub_stream", &medges, &mlayout);
}

#[test]
#[allow(clippy::excessive_precision)]
fn test_negative_steep_ulp_and_near_vertical_crossings() {
    // Minimal 4-edge steep negative cluster at x ~ -1e6, dx ~ 1e-8 where `xi <= x_evt` must agree
    // with `cmp_post_x` so completed swaps are not reverted (expected count = 4).
    let layout_f = Layout2d::from_xy(
        vec![
            -9.99999999999950058e5,
            -1.00000000000000000e6,
            -9.99999999999979977e5,
            -9.99999999999989988e5,
            -9.99999999999950058e5,
            -9.99999999999940046e5,
        ],
        vec![-46.0, -46.0, -50.0, -42.0, -50.0, -47.5],
    )
    .unwrap();
    let edges_f = vec![(0, 2), (1, 4), (1, 5), (3, 2)];
    assert_matches_brute_force("neg_steep_four_edge_cluster", &edges_f, &layout_f);
    assert_streaming_matches_find("neg_steep_four_edge_cluster_stream", &edges_f, &layout_f);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_f, &layout_f).unwrap(),
        4
    );

    // `neg_steep_11` (`seed = 90_011`, `x_off = -1e6`, `dx ~ 1e-8`).
    {
        let mut rng = DeterministicRng::from_seed_u64(90_011);
        let n = 24usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| -1e6 + (rng.gen_range_usize(8) as f64) * 1e-8)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| -50.0 + (rng.gen_range_usize(24) as f64) * 0.5)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..55 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force("neg_steep_11", &edges, &layout);
        assert_streaming_matches_find("neg_steep_11_stream", &edges, &layout);
    }

    // 30 negative steep seeds at x_off = -1e6 (`dx ~ 1e-8`) and 30 at x_off = -1e8 (`dx ~ 1e-6`).
    for seed in 1..=60u64 {
        let mut rng = DeterministicRng::from_seed_u64(90_000 + seed);
        let (x_off, dx_step) = if seed <= 30 {
            (-1e6, 1e-8)
        } else {
            (-1e8, 1e-6)
        };
        let n = 24usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| x_off + (rng.gen_range_usize(8) as f64) * dx_step)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| -50.0 + (rng.gen_range_usize(24) as f64) * 0.5)
            .collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..55 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("neg_steep_batch_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("neg_steep_batch_{seed}_stream"), &edges, &layout);
    }

    // 30 positive steep seeds at x_off = 1e8 (`dx ~ 1e-6`) + 30 near-vertical 1..16 ULP seeds.
    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(91_000 + seed);
        let n = 24usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| 1e8 + (rng.gen_range_usize(8) as f64) * 1e-6)
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
        assert_matches_brute_force(&format!("pos_steep_1e8_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("pos_steep_1e8_{seed}_stream"), &edges, &layout);
    }

    for seed in 1..=30u64 {
        let mut rng = DeterministicRng::from_seed_u64(92_000 + seed);
        let base = if seed <= 10 {
            100.0_f64
        } else if seed <= 20 {
            -1e6_f64
        } else {
            1e6_f64
        };
        let n = 20usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let ulps = rng.gen_range_usize(17) as u64;
            let x = if base >= 0.0 {
                f64::from_bits(base.to_bits() + ulps)
            } else {
                f64::from_bits(base.to_bits() - ulps)
            };
            xs.push(x);
            ys.push((rng.gen_range_usize(20) as f64) * 0.5);
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let mut edges = Vec::new();
        for _ in 0..45 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        assert_matches_brute_force(&format!("near_vert_ulp_{seed}"), &edges, &layout);
        assert_streaming_matches_find(&format!("near_vert_ulp_{seed}_stream"), &edges, &layout);
    }
}
