use std::cmp::Ordering;

use egraph_core::{CsrGraph, DeterministicRng, Layout2d, ManifoldKind, ManifoldLayout2d};

use super::super::SweepLineCrossings;
use super::super::euclidean_geom::cmp_slope_euclidean;
use super::super::geometry::Seg;
use super::super::torus_clip::TORUS_MAX;
use super::super::torus_exact::{
    ShiftedF64, ShiftedPt, exact_intersection_in_torus_max_box, exact_intersection_in_unit_box,
    exact_xi_cmp, intersection_x_dd_f64,
};
use super::super::{torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number};
use super::helpers::{
    assert_matches_brute_force, assert_streaming_matches_find, assert_torus_matches_oracle,
    assert_torus_pair_intersects,
};

fn has_non_gap_copy(tor: &ManifoldLayout2d<f64>, e1: (u32, u32), e2: (u32, u32)) -> bool {
    let shift_of = |p0: f64, p1: f64| -> i32 {
        let d = p1 - p0;
        if d > 0.5 {
            -1
        } else if d < -0.5 {
            1
        } else {
            0
        }
    };
    let coords = tor.as_layout();
    let (ax0, ay0) = (coords.x()[e1.0 as usize], coords.y()[e1.0 as usize]);
    let (ax1, ay1) = (coords.x()[e1.1 as usize], coords.y()[e1.1 as usize]);
    let (bx0, by0) = (coords.x()[e2.0 as usize], coords.y()[e2.0 as usize]);
    let (bx1, by1) = (coords.x()[e2.1 as usize], coords.y()[e2.1 as usize]);
    let (adx, ady) = (shift_of(ax0, ax1), shift_of(ay0, ay1));
    let (bdx, bdy) = (shift_of(bx0, bx1), shift_of(by0, by1));

    for &acy in &[0, ady] {
        for &acx in &[0, adx] {
            let ap0 = ShiftedPt::new(
                ShiftedF64::new(ax0, -acx as i8),
                ShiftedF64::new(ay0, -acy as i8),
            );
            let ap1 = ShiftedPt::new(
                ShiftedF64::new(ax1, (adx - acx) as i8),
                ShiftedF64::new(ay1, (ady - acy) as i8),
            );
            for &bcy in &[0, bdy] {
                for &bcx in &[0, bdx] {
                    let bp0 = ShiftedPt::new(
                        ShiftedF64::new(bx0, -bcx as i8),
                        ShiftedF64::new(by0, -bcy as i8),
                    );
                    let bp1 = ShiftedPt::new(
                        ShiftedF64::new(bx1, (bdx - bcx) as i8),
                        ShiftedF64::new(by1, (bdy - bcy) as i8),
                    );
                    if exact_intersection_in_unit_box(ap0, ap1, bp0, bp1)
                        && exact_intersection_in_torus_max_box(ap0, ap1, bp0, bp1, TORUS_MAX)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn assert_torus_all_apis_match_oracle(label: &str, edges: &[(u32, u32)], xs: &[f64], ys: &[f64]) {
    let n = xs.len();
    let g = CsrGraph::<f64>::from_edges(n, edges, None, false, false).unwrap();
    let lay =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs.to_vec(), ys.to_vec()).unwrap();
    let pairs = assert_torus_matches_oracle(label, edges, &lay);
    let cnt = torus_crossing_number(&g, &lay).unwrap();
    let graph_edges = torus_crossing_edges(&g, &lay).unwrap();
    assert_eq!(cnt, pairs.len(), "{label}: count vs from_edges");
    assert_eq!(
        graph_edges.len(),
        pairs.len(),
        "{label}: graph edges vs from_edges"
    );
    for p in &pairs {
        let in_unit = |v: &f64| v.is_finite() && (0.0..1.0).contains(v);
        assert!(p.seg_a.iter().all(in_unit) && p.seg_b.iter().all(in_unit));
        if has_non_gap_copy(&lay, edges[p.edge_a], edges[p.edge_b]) {
            assert_torus_pair_intersects(label, p);
        }
    }

    let rev_edges: Vec<(u32, u32)> = edges.iter().copied().rev().collect();
    let rev_pairs = assert_torus_matches_oracle(&format!("{label}_rev"), &rev_edges, &lay);
    assert_eq!(
        rev_pairs.len(),
        pairs.len(),
        "{label}: reversed edge order count mismatch"
    );
}

#[test]
fn test_steep_slope_and_vertical_segment_torus_and_euclidean() {
    // Steep segment (0, 0) -> (2^-1010, 0.4) (slope ~ 1.8e304) crossing vertical [0.1, 0.3].
    let edges = vec![(0, 1), (2, 3)];
    let xs = vec![
        0.0,
        2.0_f64.powi(-1010),
        2.0_f64.powi(-1011),
        2.0_f64.powi(-1011),
    ];
    let ys = vec![0.0, 0.4, 0.1, 0.3];

    assert_torus_all_apis_match_oracle("steep_vs_vertical_minimal", &edges, &xs, &ys);
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();
    let tor =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs.clone(), ys.clone()).unwrap();
    assert_eq!(torus_crossing_number(&g, &tor).unwrap(), 1);

    let euc = Layout2d::from_xy(xs, ys).unwrap();
    assert_matches_brute_force("steep_vs_vertical_euclidean", &edges, &euc);
    assert_streaming_matches_find("steep_vs_vertical_euclidean_stream", &edges, &euc);
    assert_eq!(SweepLineCrossings::count_edges(&edges, &euc).unwrap(), 1);
    assert_eq!(
        SweepLineCrossings::find_edge_crossings(&edges, &euc)
            .unwrap()
            .len(),
        1
    );

    // Multi-segment Euclidean layout where clamped-slope segments sit as internal AVL nodes
    // and cross both another steep segment (exact cross-product slope comparison) and verticals.
    let multi_edges = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let multi_xs = vec![
        0.0,
        2.0_f64.powi(-1010),
        0.0,
        2.0_f64.powi(-1009),
        2.0_f64.powi(-1011),
        2.0_f64.powi(-1011),
        0.0,
        2.0_f64.powi(-1008),
        0.0,
        2.0_f64.powi(-1008),
    ];
    let multi_ys = vec![0.0, 0.4, 0.0, 0.9, 0.15, 0.55, -0.2, -0.1, 0.8, 0.95];
    let multi_euc = Layout2d::from_xy(multi_xs, multi_ys).unwrap();
    assert_matches_brute_force("steep_internal_avl_multi_euc", &multi_edges, &multi_euc);
    assert_streaming_matches_find(
        "steep_internal_avl_multi_euc_stream",
        &multi_edges,
        &multi_euc,
    );
}

#[test]
fn test_final_ulp_steep_crossing_count_matches_edges() {
    // Case 1: crossing strictly inside final ULP before 2^-1010.
    let x2_a = 2.0_f64.powi(-1010);
    let x1_a = f64::from_bits(x2_a.to_bits() - 1);
    let edges_a = vec![(1, 2), (0, 2), (3, 4)];
    let xs_a = vec![0.5 - 2.0_f64.powi(-54), 0.1, x1_a, x1_a, x2_a];
    let ys_a = vec![0.1, 2.0_f64.powi(-1011), 0.1, 2.0_f64.powi(-998), 0.25];
    assert_torus_all_apis_match_oracle("final_ulp_steep_norm", &edges_a, &xs_a, &ys_a);
    let g_a = CsrGraph::<f64>::from_edges(5, &edges_a, None, false, false).unwrap();
    let tor_a = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs_a, ys_a).unwrap();
    assert_eq!(torus_crossing_number(&g_a, &tor_a).unwrap(), 2);
    assert_eq!(torus_crossing_edges(&g_a, &tor_a).unwrap().len(), 2);
    assert_eq!(
        torus_crossing_edges_from_edges(&edges_a, &tor_a)
            .unwrap()
            .len(),
        2
    );

    // Case 2: crossing inside final ULP from 0.0 to smallest positive subnormal 5e-324.
    let x2_b = f64::from_bits(1);
    let edges_b = vec![(1, 2), (0, 2), (3, 4)];
    let xs_b = vec![0.5 - 2.0_f64.powi(-54), 0.1, 0.0, 0.0, x2_b];
    let ys_b = vec![0.1, 2.0_f64.powi(-1011), 0.1, 2.0_f64.powi(-998), 0.25];
    assert_torus_all_apis_match_oracle("final_ulp_steep_subnormal", &edges_b, &xs_b, &ys_b);
    let g_b = CsrGraph::<f64>::from_edges(5, &edges_b, None, false, false).unwrap();
    let tor_b = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs_b, ys_b).unwrap();
    assert_eq!(torus_crossing_number(&g_b, &tor_b).unwrap(), 2);
    assert_eq!(torus_crossing_edges(&g_b, &tor_b).unwrap().len(), 2);
}

#[test]
fn test_deep_exponent_gap_intersection_ordering_and_commutativity() {
    // 447-bit exponent gap ({0.0, 2^-500, 2^-200, 0.5, TORUS_MAX}) where 106-bit DD
    // intersection collapses to 0.0 while true intersection is near 2^-948.
    let p_m500 = 2.0_f64.powi(-500);
    let p_m200 = 2.0_f64.powi(-200);
    let edges = vec![(3, 6), (0, 1), (3, 5), (2, 4)];
    let xs = vec![p_m500, TORUS_MAX, p_m500, p_m200, 0.0, p_m500, TORUS_MAX];
    let ys = vec![p_m200, 0.5, 0.5, TORUS_MAX, p_m200, p_m200, p_m500];

    assert_torus_all_apis_match_oracle("deep_exponent_447bit_gap", &edges, &xs, &ys);
    let g = CsrGraph::<f64>::from_edges(7, &edges, None, false, false).unwrap();
    let tor = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs, ys).unwrap();
    assert_eq!(torus_crossing_number(&g, &tor).unwrap(), 2);
    assert_eq!(
        torus_crossing_edges_from_edges(&edges, &tor).unwrap().len(),
        2
    );
    let rev_edges: Vec<(u32, u32)> = edges.iter().copied().rev().collect();
    assert_eq!(
        torus_crossing_edges_from_edges(&rev_edges, &tor)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn test_randomized_steep_vertical_and_deep_exponent_matrices() {
    let x_steep_pool = [
        0.0,
        f64::from_bits(1),
        f64::from_bits(2),
        2.0_f64.powi(-1011),
        f64::from_bits(2.0_f64.powi(-1010).to_bits() - 1),
        2.0_f64.powi(-1010),
        2.0_f64.powi(-998),
        0.1,
        0.5 - 2.0_f64.powi(-54),
        TORUS_MAX,
    ];
    let y_steep_pool = [
        0.0,
        2.0_f64.powi(-1011),
        2.0_f64.powi(-998),
        0.05,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.8,
        TORUS_MAX,
    ];

    for seed in 1..=40u64 {
        let mut rng = DeterministicRng::from_seed_u64(81_000 + seed);
        let n = 14usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(x_steep_pool[rng.gen_range_usize(x_steep_pool.len())]);
            ys.push(y_steep_pool[rng.gen_range_usize(y_steep_pool.len())]);
        }
        let mut edges = Vec::with_capacity(24);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..48 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v && seen.insert((u.min(v), u.max(v))) {
                edges.push((u, v));
                if edges.len() == 24 {
                    break;
                }
            }
        }
        assert_torus_all_apis_match_oracle(&format!("steep_vert_matrix_{seed}"), &edges, &xs, &ys);
        let euc = Layout2d::from_xy(xs, ys).unwrap();
        assert_matches_brute_force(&format!("steep_vert_euc_{seed}"), &edges, &euc);
        assert_streaming_matches_find(&format!("steep_vert_euc_stream_{seed}"), &edges, &euc);
    }

    let deep_pool = [
        0.0,
        2.0_f64.powi(-998),
        2.0_f64.powi(-500),
        2.0_f64.powi(-200),
        0.5 * f64::EPSILON,
        0.5,
        1.0 - f64::EPSILON,
        TORUS_MAX,
    ];
    for seed in 1..=40u64 {
        let mut rng = DeterministicRng::from_seed_u64(82_000 + seed);
        let n = 12usize;
        let xs: Vec<f64> = (0..n)
            .map(|_| deep_pool[rng.gen_range_usize(deep_pool.len())])
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| deep_pool[rng.gen_range_usize(deep_pool.len())])
            .collect();
        let mut edges = Vec::with_capacity(20);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..40 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v && seen.insert((u.min(v), u.max(v))) {
                edges.push((u, v));
                if edges.len() == 20 {
                    break;
                }
            }
        }
        assert_torus_all_apis_match_oracle(&format!("deep_exp_matrix_{seed}"), &edges, &xs, &ys);
        let euc = Layout2d::from_xy(xs, ys).unwrap();
        assert_matches_brute_force(&format!("deep_exp_euc_{seed}"), &edges, &euc);
        assert_streaming_matches_find(&format!("deep_exp_euc_stream_{seed}"), &edges, &euc);
    }
}

#[test]
fn test_colocated_multi_ending_sweep_fixed_point_torus_and_euclidean() {
    // 1. Torus layout with co-located ending segments and vertical wrapping:
    // fixed-point end-batch reordering must resolve blocked pairs before band checks.
    let edges_tor = vec![(10, 6), (15, 1), (3, 12), (2, 5), (1, 10)];
    let mut xs_tor = vec![0.0; 16];
    let mut ys_tor = vec![0.0; 16];
    xs_tor[1] = 1.5e-323;
    ys_tor[1] = 0.8;
    xs_tor[2] = 5e-324;
    ys_tor[2] = 0.7;
    xs_tor[3] = 9.113902524445496e-305;
    ys_tor[3] = 0.1;
    xs_tor[5] = 9.1139025244455e-305;
    ys_tor[5] = 0.6;
    xs_tor[6] = 1.5e-323;
    ys_tor[6] = 0.8;
    xs_tor[10] = 1e-323;
    ys_tor[10] = 0.6;
    xs_tor[12] = 1e-323;
    ys_tor[12] = 0.7;
    xs_tor[15] = 1e-323;
    ys_tor[15] = 0.1;
    assert_torus_all_apis_match_oracle(
        "colocated_multi_ending_torus",
        &edges_tor,
        &xs_tor,
        &ys_tor,
    );
    let g_tor = CsrGraph::<f64>::from_edges(16, &edges_tor, None, false, false).unwrap();
    let tor = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs_tor, ys_tor).unwrap();
    assert_eq!(torus_crossing_number(&g_tor, &tor).unwrap(), 5);
    assert_eq!(torus_crossing_edges(&g_tor, &tor).unwrap().len(), 5);
    assert_eq!(
        torus_crossing_edges_from_edges(&edges_tor, &tor)
            .unwrap()
            .len(),
        5
    );

    // 2. Euclidean layout where two parallel segments end together ahead of a steeper segment
    // and a vertical query sits at the shared end coordinate.
    let x_evt = f64::from_bits(1);
    let x2_c = f64::from_bits(2);
    let edges_euc1 = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let xs_euc1 = vec![0.0, x_evt, 0.0, x_evt, 0.0, x_evt, 0.0, x2_c, x_evt, x_evt];
    let ys_euc1 = vec![0.45, 0.45, 0.50, 0.50, 0.55, 0.55, 0.70, 0.10, 0.35, 0.42];
    let euc1 = Layout2d::from_xy(xs_euc1, ys_euc1).unwrap();
    for (label, e) in [
        ("multi_ending_vert_fwd", edges_euc1.clone()),
        (
            "multi_ending_vert_rev",
            edges_euc1.iter().copied().rev().collect(),
        ),
    ] {
        assert_matches_brute_force(label, &e, &euc1);
        assert_streaming_matches_find(label, &e, &euc1);
        assert_eq!(SweepLineCrossings::count_edges(&e, &euc1).unwrap(), 4);
    }

    // 3. Steep multi-ending Euclidean layout with concurrent starts at the end coordinate.
    let edges_euc2 = vec![(1, 6), (6, 13), (11, 9), (7, 6), (7, 10), (0, 11)];
    let mut xs_euc2 = vec![0.0; 14];
    let mut ys_euc2 = vec![0.0; 14];
    xs_euc2[0] = 9.113_902_524_445_497e-305;
    ys_euc2[0] = 0.4;
    xs_euc2[1] = 9.113_902_524_445_496e-305;
    ys_euc2[1] = 0.3;
    xs_euc2[6] = 5e-324;
    ys_euc2[6] = 0.2;
    xs_euc2[7] = 0.0;
    ys_euc2[7] = 0.999_999_999_999_999_9;
    xs_euc2[9] = 1e-323;
    ys_euc2[9] = 0.0;
    xs_euc2[10] = 5e-324;
    ys_euc2[10] = 3.733_054_474_012_875_5e-301;
    xs_euc2[11] = 0.0;
    ys_euc2[11] = 0.8;
    xs_euc2[13] = 3.733_054_474_012_875_5e-301;
    ys_euc2[13] = 0.3;
    let euc2 = Layout2d::from_xy(xs_euc2, ys_euc2).unwrap();
    for (label, e) in [
        ("steep_multi_ending_fwd", edges_euc2.clone()),
        (
            "steep_multi_ending_rev",
            edges_euc2.iter().copied().rev().collect(),
        ),
    ] {
        assert_matches_brute_force(label, &e, &euc2);
        assert_streaming_matches_find(label, &e, &euc2);
        assert_eq!(SweepLineCrossings::count_edges(&e, &euc2).unwrap(), 6);
    }
}

#[test]
fn test_euclidean_error_bound_filter_and_exact_intersection_ceil() {
    // 1. Disjoint vertical vs steep-diagonal pair with 500-bit cancellation in fast det.
    let e_deep = vec![(0, 1), (2, 3)];
    let lay_deep = Layout2d::from_xy(
        vec![
            0.9999999999999999,
            0.9999999999999999,
            9.113902524445497e-305,
            0.9999999999999999,
        ],
        vec![0.75, 3.054936363499605e-151, 0.9999999999999999, 8e-323],
    )
    .unwrap();
    assert_eq!(
        SweepLineCrossings::count_edges(&e_deep, &lay_deep).unwrap(),
        0
    );
    assert_matches_brute_force("disjoint_deep_cancellation", &e_deep, &lay_deep);

    // 2. Disjoint near-collinear [0, 1) pair 2 ULPs apart near (0.5, 0.5).
    let e_near = vec![(0, 1), (2, 3)];
    let lay_near = Layout2d::from_xy(
        vec![
            0.499_999_999_999_999_56,
            0.640_624_278_193_717_1,
            0.076_888_310_455_705_64,
            0.499_999_999_999_999_8,
        ],
        vec![
            0.5,
            0.074_631_134_113_468_65,
            0.088_590_192_549_458_87,
            0.500_000_000_000_000_2,
        ],
    )
    .unwrap();
    assert_eq!(
        SweepLineCrossings::count_edges(&e_near, &lay_near).unwrap(),
        0
    );
    assert_matches_brute_force("disjoint_near_collinear_ulp", &e_near, &lay_near);

    // 3. Near-parallel slope comparison within 2 ULPs must match exact cross-product sign.
    let seg_a = Seg::from_endpoints(0, 0, 1, 0.0, 0.0, 4.0, 4.0 + 5.0 * 2.0_f64.powi(-50));
    let seg_b = Seg::from_endpoints(
        1,
        2,
        3,
        2.0_f64.powi(-53),
        -2.0_f64.powi(-53),
        1.0 + 2.0_f64.powi(-52),
        1.0 + 2.0_f64.powi(-52),
    );
    // Exact slopes: a = 1 + 5*2^-52; b = (1 + 3*2^-53)/(1 + 2^-53) < 1 + 2^-51 < a.
    assert_eq!(cmp_slope_euclidean(&seg_a, &seg_b), Ordering::Greater);
    assert_eq!(cmp_slope_euclidean(&seg_b, &seg_a), Ordering::Less);

    // 4. Symmetric intersection_x_dd_f64 ceiling invariant for deep-exponent Euclidean pairs.
    let p_m500 = 2.0_f64.powi(-500);
    let p_m200 = 2.0_f64.powi(-200);
    let a = [p_m200, TORUS_MAX, 0.0, p_m200];
    let b = [p_m500, p_m200, TORUS_MAX, p_m500];
    let lo = a[0].min(a[2]).max(b[0].min(b[2]));
    let hi = a[0].max(a[2]).min(b[0].max(b[2]));
    let xi = intersection_x_dd_f64(a, b, lo, hi).unwrap();
    let pa1 = ShiftedPt::from_f64(a[0], a[1]);
    let pa2 = ShiftedPt::from_f64(a[2], a[3]);
    let pb1 = ShiftedPt::from_f64(b[0], b[1]);
    let pb2 = ShiftedPt::from_f64(b[2], b[3]);
    assert_ne!(
        exact_xi_cmp(pa1, pa2, pb1, pb2, xi),
        Some(Ordering::Greater)
    );
    if xi > lo {
        let prev = f64::from_bits(xi.to_bits() - 1);
        assert_eq!(
            exact_xi_cmp(pa1, pa2, pb1, pb2, prev),
            Some(Ordering::Greater)
        );
    }
}
