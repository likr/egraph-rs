use std::cmp::Ordering;

use egraph_core::{DeterministicRng, Layout2d};

use super::super::SweepLineCrossings;
use super::super::euclidean_geom::cmp_slope_euclidean;
use super::super::exact_limbs::{ExactFactor, exact_deg3_sign};
use super::super::geometry::{Seg, seg_y_err};
use super::super::torus_exact::{ShiftedPt, exact_xi_cmp, intersection_x_dd_f64};
use super::helpers::{assert_matches_brute_force, assert_streaming_matches_find};

fn prev_down(v: f64) -> f64 {
    if v > 0.0 {
        f64::from_bits(v.to_bits() - 1)
    } else if v == 0.0 {
        -f64::from_bits(1)
    } else {
        f64::from_bits(v.to_bits() + 1)
    }
}

#[test]
fn test_wide_range_1e100_three_edge_crossing() {
    // Minimal 3-edge layout exceeding 1e100 where an exact intersection lies in [0.0, 5e-324].
    let edges = vec![(5, 10), (6, 10), (8, 1)];
    let mut xs = vec![0.0; 11];
    let mut ys = vec![0.0; 11];
    xs[1] = 5e-324;
    ys[1] = 5e-324;
    xs[5] = 1.000_000_000_000_001e100;
    ys[5] = 1e50;
    xs[6] = 5e-324;
    ys[6] = -1e100;
    xs[8] = 0.0;
    ys[8] = -1e100;
    xs[10] = -1e100;
    ys[10] = -1e100;

    let layout = Layout2d::from_xy(xs, ys).unwrap();
    assert_matches_brute_force("wide_1e100_three_edge", &edges, &layout);
    assert_streaming_matches_find("wide_1e100_three_edge_stream", &edges, &layout);
    assert_eq!(SweepLineCrossings::count_edges(&edges, &layout).unwrap(), 2);
    let pairs = SweepLineCrossings::find_edge_crossings(&edges, &layout).unwrap();
    let indices: Vec<(usize, usize)> = pairs.iter().map(|p| (p.edge_a, p.edge_b)).collect();
    assert_eq!(indices, vec![(0, 2), (1, 2)]);

    let rev_edges: Vec<(u32, u32)> = edges.iter().copied().rev().collect();
    assert_matches_brute_force("wide_1e100_three_edge_rev", &rev_edges, &layout);
    assert_eq!(
        SweepLineCrossings::count_edges(&rev_edges, &layout).unwrap(),
        2
    );
}

fn run_wide_pool_matrix(label: &str, seed_base: u64, pool: &[f64], cases: u64) {
    for seed in 1..=cases {
        let mut rng = DeterministicRng::from_seed_u64(seed_base + seed);
        let n = 12usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| pool[rng.gen_range_usize(pool.len())])
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| pool[rng.gen_range_usize(pool.len())])
            .collect();
        let mut edges = Vec::with_capacity(22);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..48 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v && seen.insert((u.min(v), u.max(v))) {
                edges.push((u, v));
                if edges.len() == 22 {
                    break;
                }
            }
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let case_label = format!("{label}_{seed}");
        assert_matches_brute_force(&case_label, &edges, &layout);
        assert_streaming_matches_find(&case_label, &edges, &layout);
        let summary = SweepLineCrossings::summarize_edges(&edges, &layout).unwrap();
        assert!(
            summary.angle_cos_sq_sum.is_finite(),
            "{case_label}: non-finite angle_cos_sq_sum"
        );
        if let Some(min_ang) = summary.min_angle_rad {
            assert!(
                min_ang.is_finite(),
                "{case_label}: non-finite min_angle_rad"
            );
        }

        let rev_edges: Vec<(u32, u32)> = edges.iter().copied().rev().collect();
        assert_eq!(
            SweepLineCrossings::count_edges(&rev_edges, &layout).unwrap(),
            summary.count,
            "{case_label}: reversed edge order count mismatch"
        );
    }
}

#[test]
fn test_wide_range_randomized_1e100_1e200_1e307_oracle_parity() {
    let pool_1e100 = [
        0.0,
        f64::from_bits(1),
        2.0_f64.powi(-500),
        1.0,
        1e50,
        1e100,
        1e100 + 1e85,
        -1e100,
    ];
    run_wide_pool_matrix("wide_1e100", 91_000, &pool_1e100, 200);

    let pool_1e200 = [0.0, 1.0, 1e150, 1e200, -1e200, 1e200 + 1e185];
    run_wide_pool_matrix("wide_1e200", 92_000, &pool_1e200, 200);

    let pool_1e307 = [
        0.0,
        f64::from_bits(1),
        1.0,
        1e150,
        1e307,
        -1e307,
        1e307 + 1e292,
    ];
    run_wide_pool_matrix("wide_1e307", 93_000, &pool_1e307, 200);
}

#[test]
fn test_wide_range_sign_spanning_max_f64() {
    // Sign-spanning segments near +/- f64::MAX whose raw endpoint deltas overflow f64.
    let big = 1.7e308;
    let tiny = f64::from_bits(1);
    let edges = vec![(0, 1), (2, 3), (4, 5), (6, 7)];
    let xs = vec![-big, big, -big, big, 0.0, 0.0, -big, tiny];
    let ys = vec![-big, big, big, -big, -big, big, tiny, -big];
    let layout = Layout2d::from_xy(xs, ys).unwrap();

    assert_matches_brute_force("sign_spanning_max_f64", &edges, &layout);
    assert_streaming_matches_find("sign_spanning_max_f64_stream", &edges, &layout);

    let summary = SweepLineCrossings::summarize_edges(&edges, &layout).unwrap();
    assert_eq!(summary.count, 5);
    assert!(summary.angle_cos_sq_sum.is_finite());
    assert!(summary.min_angle_rad.is_some_and(f64::is_finite));

    let seg = Seg::from_endpoints(0, 0, 1, -big, -big, big, big);
    assert!(seg.slope.is_finite() && (seg.slope - 1.0).abs() <= 1e-15);
    assert!(seg.y_at(0.5 * big).is_finite());
    assert!(seg_y_err(&seg, 0.0).is_finite());
}

#[test]
fn test_exact_deg3_xi_cmp_and_subnormal_slope_full_f64_span() {
    let f_max = ExactFactor::from_pair(f64::MAX, 0.0);
    let f_prev = ExactFactor::from_pair(prev_down(f64::MAX), 0.0);
    let f_sub = ExactFactor::from_pair(f64::from_bits(1), 0.0);

    // High-limb cancellation at f64::MAX^3 alongside low-limb 5e-324^3 term (~6294-bit span).
    assert_eq!(
        exact_deg3_sign(&[
            (f_max, f_max, f_max, 1),
            (f_max, f_max, f_max, -1),
            (f_sub, f_sub, f_sub, 1),
        ]),
        Ordering::Greater
    );
    assert_eq!(
        exact_deg3_sign(&[
            (f_max, f_max, f_max, 1),
            (f_max, f_max, f_prev, -1),
            (f_sub, f_sub, f_sub, -1),
        ]),
        Ordering::Greater
    );
    assert_eq!(
        exact_deg3_sign(&[
            (f_max, f_max, f_max, 1),
            (f_max, f_max, f_max, -1),
            (f_sub, f_sub, f_sub, -1),
        ]),
        Ordering::Less
    );

    // Exact ceiling check for intersection_x_dd_f64 across huge and tiny scales.
    let cases: [([f64; 4], [f64; 4]); 4] = [
        (
            [-1e308, -1e308, 1e308, 1e308],
            [-1e308, 1e308, 1e308, -1e308],
        ),
        ([0.0, -1e200, 1e200, 1e200], [0.0, 1e150, 1e200, -1e200]),
        ([0.0, 0.0, 1e-250, 1e-250], [0.0, 1e-250, 1e-250, 0.0]),
        ([0.0, -1e308, 5e-324, 5e-324], [-1e308, -1e308, 1e308, 1e50]),
    ];
    for (idx, (a, b)) in cases.into_iter().enumerate() {
        let lo = a[0].min(a[2]).max(b[0].min(b[2]));
        let hi = a[0].max(a[2]).min(b[0].max(b[2]));
        let xi = intersection_x_dd_f64(a, b, lo, hi).expect("non-parallel intersection");
        assert!(xi.is_finite(), "case {idx}: non-finite xi");
        let pa1 = ShiftedPt::from_f64(a[0], a[1]);
        let pa2 = ShiftedPt::from_f64(a[2], a[3]);
        let pb1 = ShiftedPt::from_f64(b[0], b[1]);
        let pb2 = ShiftedPt::from_f64(b[2], b[3]);
        assert_ne!(
            exact_xi_cmp(pa1, pa2, pb1, pb2, xi),
            Some(Ordering::Greater),
            "case {idx}: xi underestimated"
        );
        if xi > lo {
            let prev = prev_down(xi).max(lo);
            assert_eq!(
                exact_xi_cmp(pa1, pa2, pb1, pb2, prev),
                Some(Ordering::Greater),
                "case {idx}: xi is not minimal ceil"
            );
        }
    }

    // Subnormal slopes (< 1e-280) with large aspect ratio must order via exact cross product.
    let s1 = Seg::from_endpoints(0, 0, 1, 0.0, 0.0, 1e165, 1e-145);
    let s2 = Seg::from_endpoints(1, 2, 3, 0.0, 0.0, 2e165, 2e-145 + 1e-159);
    assert_eq!(cmp_slope_euclidean(&s1, &s2), Ordering::Less);
    assert_eq!(cmp_slope_euclidean(&s2, &s1), Ordering::Greater);
}

#[test]
fn test_band_seed_coverage_and_delayed_band_entry() {
    // 1. Overlapping y-error band seeds at distinct y centers in [0, 1).
    let edges_a = vec![(13, 9), (5, 0), (13, 12), (7, 1)];
    let mut xs_a = vec![0.0; 14];
    let mut ys_a = vec![0.0; 14];
    xs_a[0] = 0.0;
    ys_a[0] = 0.500_000_000_000_000_1;
    xs_a[1] = 0.75;
    ys_a[1] = 0.999_999_999_999_999;
    xs_a[5] = 0.999_999_999_999_999_9;
    ys_a[5] = 0.999_999_999_999_999_9;
    xs_a[7] = 0.25;
    ys_a[7] = 0.500_000_000_000_000_1;
    xs_a[9] = 0.5;
    ys_a[9] = 1e-16;
    xs_a[12] = 0.5;
    ys_a[12] = 0.999_999_999_999_999;
    xs_a[13] = 0.500_000_000_000_000_1;
    ys_a[13] = 0.500_000_000_000_000_1;
    let layout_a = Layout2d::from_xy(xs_a, ys_a).unwrap();
    assert_matches_brute_force("band_seed_distinct_center_fwd", &edges_a, &layout_a);
    assert_streaming_matches_find("band_seed_distinct_center_fwd_stream", &edges_a, &layout_a);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_a, &layout_a).unwrap(),
        3
    );
    let rev_a: Vec<(u32, u32)> = edges_a.iter().copied().rev().collect();
    assert_matches_brute_force("band_seed_distinct_center_rev", &rev_a, &layout_a);
    assert_streaming_matches_find("band_seed_distinct_center_rev_stream", &rev_a, &layout_a);

    // 2. Second [0, 1) multi-seed band layout with subnormal endpoint coordinate.
    let edges_b = vec![(2, 12), (5, 8), (3, 13), (0, 6)];
    let mut xs_b = vec![0.0; 14];
    let mut ys_b = vec![0.0; 14];
    xs_b[0] = 0.75;
    ys_b[0] = 0.75;
    xs_b[2] = 0.500_000_000_000_000_1;
    ys_b[2] = 0.75;
    xs_b[3] = 0.5;
    ys_b[3] = 0.999_999_999_999_999;
    xs_b[5] = 0.999_999_999_999_999_9;
    ys_b[5] = 0.999_999_999_999_999;
    xs_b[6] = 0.25;
    ys_b[6] = 0.75;
    xs_b[8] = f64::from_bits(1);
    ys_b[8] = 0.75;
    xs_b[12] = 0.5;
    ys_b[12] = 1e-16;
    xs_b[13] = 0.500_000_000_000_000_1;
    ys_b[13] = 0.500_000_000_000_000_1;
    let layout_b = Layout2d::from_xy(xs_b, ys_b).unwrap();
    assert_matches_brute_force("band_seed_subnormal_fwd", &edges_b, &layout_b);
    assert_streaming_matches_find("band_seed_subnormal_fwd_stream", &edges_b, &layout_b);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_b, &layout_b).unwrap(),
        4
    );
    let rev_b: Vec<(u32, u32)> = edges_b.iter().copied().rev().collect();
    assert_matches_brute_force("band_seed_subnormal_rev", &rev_b, &layout_b);
    assert_streaming_matches_find("band_seed_subnormal_rev_stream", &rev_b, &layout_b);

    // 3. Delayed band entry where segments cross before entering each other's band.
    let edges_c = vec![(9, 8), (12, 3), (0, 8), (10, 5)];
    let mut xs_c = vec![0.0; 13];
    let mut ys_c = vec![0.0; 13];
    xs_c[0] = -1e-200;
    ys_c[0] = -1e307;
    xs_c[3] = 1e-300;
    ys_c[3] = -f64::from_bits(1);
    xs_c[5] = -1e308;
    ys_c[5] = -1e307;
    xs_c[8] = 1e-300;
    ys_c[8] = 0.0;
    xs_c[9] = -1e-200;
    ys_c[9] = f64::from_bits(1);
    xs_c[10] = 0.0;
    ys_c[10] = -1e308;
    xs_c[12] = 0.0;
    ys_c[12] = 1e308;
    let layout_c = Layout2d::from_xy(xs_c, ys_c).unwrap();
    assert_matches_brute_force("delayed_band_entry_fwd", &edges_c, &layout_c);
    assert_streaming_matches_find("delayed_band_entry_fwd_stream", &edges_c, &layout_c);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_c, &layout_c).unwrap(),
        2
    );
    let rev_c: Vec<(u32, u32)> = edges_c.iter().copied().rev().collect();
    assert_matches_brute_force("delayed_band_entry_rev", &rev_c, &layout_c);
    assert_streaming_matches_find("delayed_band_entry_rev_stream", &rev_c, &layout_c);
}

#[test]
fn test_exact_coord_cmp_axis_mixed_exponent_underflow() {
    // 1. Mixed tiny-x / extreme-y triple-product layout where 1e-300 * 1e-200 * 1e308 = 1e-192.
    let edges_a = vec![(9, 7), (11, 8), (1, 6)];
    let mut xs_a = vec![0.0; 12];
    let mut ys_a = vec![0.0; 12];
    xs_a[1] = 1e-300;
    ys_a[1] = f64::from_bits(1);
    xs_a[6] = f64::from_bits(1);
    ys_a[6] = -1e308;
    xs_a[7] = 1e-300;
    ys_a[7] = -1e307;
    xs_a[8] = 1e-300;
    ys_a[8] = 0.0;
    xs_a[9] = -1e-200;
    ys_a[9] = 0.0;
    xs_a[11] = -1e-200;
    ys_a[11] = 1e307;
    let layout_a = Layout2d::from_xy(xs_a, ys_a).unwrap();
    assert_matches_brute_force("mixed_exp_underflow_fwd", &edges_a, &layout_a);
    assert_streaming_matches_find("mixed_exp_underflow_fwd_stream", &edges_a, &layout_a);
    let pairs_a = SweepLineCrossings::find_edge_crossings(&edges_a, &layout_a).unwrap();
    let idx_a: Vec<(usize, usize)> = pairs_a.iter().map(|p| (p.edge_a, p.edge_b)).collect();
    assert_eq!(idx_a, vec![(0, 2), (1, 2)]);
    let rev_a: Vec<(u32, u32)> = edges_a.iter().copied().rev().collect();
    assert_matches_brute_force("mixed_exp_underflow_rev", &rev_a, &layout_a);
    assert_streaming_matches_find("mixed_exp_underflow_rev_stream", &rev_a, &layout_a);

    // Direct exact_xi_cmp check on 1e-300 * 1e-200 * 1e308 cancellation.
    let a1 = ShiftedPt::from_f64(-1e-200, 0.0);
    let a2 = ShiftedPt::from_f64(1e-300, -1e307);
    let b1 = ShiftedPt::from_f64(f64::from_bits(1), -1e308);
    let b2 = ShiftedPt::from_f64(1e-300, f64::from_bits(1));
    assert_eq!(
        exact_xi_cmp(a1, a2, b1, b2, f64::from_bits(1)),
        Some(Ordering::Greater)
    );
    assert_eq!(exact_xi_cmp(a1, a2, b1, b2, 1e-300), Some(Ordering::Less));

    // 2. Mixed subnormal-x / 1e307-y three-edge layout.
    let edges_b = vec![(8, 9), (4, 12), (12, 2)];
    let mut xs_b = vec![0.0; 13];
    let mut ys_b = vec![0.0; 13];
    xs_b[2] = f64::from_bits(1);
    ys_b[2] = 9.332_636_185_032_189e-302;
    xs_b[4] = 0.0;
    ys_b[4] = 1.000_000_000_000_001e307;
    xs_b[8] = 3.054_936_363_499_605e-151;
    ys_b[8] = 1.0;
    xs_b[9] = f64::from_bits(1);
    ys_b[9] = 9.332_636_185_032_189e-302;
    xs_b[12] = f64::from_bits(3);
    ys_b[12] = -1e307;
    let layout_b = Layout2d::from_xy(xs_b, ys_b).unwrap();
    assert_matches_brute_force("mixed_subnormal_1e307_fwd", &edges_b, &layout_b);
    assert_streaming_matches_find("mixed_subnormal_1e307_fwd_stream", &edges_b, &layout_b);
    assert_eq!(
        SweepLineCrossings::count_edges(&edges_b, &layout_b).unwrap(),
        2
    );
    let rev_b: Vec<(u32, u32)> = edges_b.iter().copied().rev().collect();
    assert_matches_brute_force("mixed_subnormal_1e307_rev", &rev_b, &layout_b);
    assert_streaming_matches_find("mixed_subnormal_1e307_rev_stream", &rev_b, &layout_b);
}

#[test]
fn test_snapped_unit_and_mixed_exponent_band_matrices() {
    let pool_snapped_unit = [
        0.0,
        f64::from_bits(1),
        1e-16,
        0.25,
        0.5,
        0.5 + f64::EPSILON * 0.5,
        0.75,
        1.0 - 1e-15,
        1.0 - f64::EPSILON * 0.5,
    ];
    run_wide_pool_matrix("snapped_unit_band", 94_000, &pool_snapped_unit, 150);

    let pool_mixed_bands = [
        0.0,
        f64::from_bits(1),
        1e-300,
        -1e-200,
        1e-150,
        1.0,
        1e307,
        -1e307,
        1e308,
        -1e308,
    ];
    run_wide_pool_matrix("mixed_extreme_bands", 95_000, &pool_mixed_bands, 150);
}
