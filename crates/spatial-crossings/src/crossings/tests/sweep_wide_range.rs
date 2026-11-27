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
