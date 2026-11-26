use std::cmp::Ordering;

use egraph_core::{CsrGraph, DeterministicRng, ManifoldKind, ManifoldLayout2d};

use super::super::torus_clip::TORUS_MAX;
use super::super::torus_exact::{
    ShiftedF64, ShiftedPt, cmp_slope_shifted, exact_intersection_in_torus_max_box,
    exact_intersection_in_unit_box, exact_xi_cmp, orient2d_shifted,
};
use super::super::{torus_crossing_edges, torus_crossing_number, torus_edge_segments};
use super::helpers::{assert_torus_matches_oracle, assert_torus_pair_intersects};

fn two_edge_graph() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap()
}

#[test]
fn test_clean_non_wrapping_near_one_lattice_pairs() {
    let edges = [(0, 1), (2, 3)];
    let g = two_edge_graph();

    // Pair A: 0.95 + 0.05 = 1 - 4.16e-17 < 1 vs (0.75, 0.25) -> (1 + 1e-12, -1e-12) (x + y > 1).
    let mut lay_a = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.95, 0.421_670_682_475_601_6, 0.75, 1e-12],
        vec![0.05, 0.569_849_243_140_097_3, 0.25, 1.0 - 1e-12],
    )
    .unwrap();
    lay_a.set_torus_extents(3.0, 0.5).unwrap();
    assert_eq!(torus_crossing_number(&g, &lay_a).unwrap(), 0);
    assert!(torus_crossing_edges(&g, &lay_a).unwrap().is_empty());
    assert_torus_matches_oracle("clean_near_one_pair_a", &edges, &lay_a);

    // Pair B: (0.75, 0.25) -> (1e-12, 1 - 1e-12) vs (0.95, 0.05) -> (0.8444, 0.6388).
    let mut lay_b = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.75, 1e-12, 0.95, 0.844_425_596_787_626_4],
        vec![0.25, 1.0 - 1e-12, 0.05, 0.638_846_931_970_323_7],
    )
    .unwrap();
    lay_b.set_torus_extents(3.0, 0.5).unwrap();
    assert_eq!(torus_crossing_number(&g, &lay_b).unwrap(), 0);
    assert!(torus_crossing_edges(&g, &lay_b).unwrap().is_empty());
    assert_torus_matches_oracle("clean_near_one_pair_b", &edges, &lay_b);
}

#[test]
fn test_near_endpoint_cancellation_clipping() {
    let edges = [(0, 1), (2, 3)];
    let g = two_edge_graph();
    let eps = f64::EPSILON;

    // Case 1: t0 ~ 1 - 7.26e-15 near x = 0.0 on unit square torus.
    let lay_unit = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![
            eps,
            0.969_408_602_573_636_7,
            1e-16,
            0.253_159_530_328_314_73,
        ],
        vec![0.0, 0.055_805_718_674_001_97, eps, 0.363_438_894_780_452_55],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &lay_unit).unwrap(), 1);
    let pairs_unit = assert_torus_matches_oracle("near_endpoint_unit", &edges, &lay_unit);
    assert_eq!(pairs_unit.len(), 1);
    assert_torus_pair_intersects("near_endpoint_unit", &pairs_unit[0]);
    let expected_y = 4.050_602_390_869_335e-16_f64;
    let clip_y = pairs_unit[0].seg_a[1].max(pairs_unit[0].seg_a[3]);
    assert!(
        (clip_y - expected_y).abs() <= 2.0 * expected_y * eps,
        "clipped y={clip_y} drifted from exact {expected_y}"
    );

    // Case 2: both t0 and t1 near 1.0 across Left and Bottom boundaries on (3.0, 0.5) torus.
    let mut lay_rect = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.5 * eps, 0.984_789_966_329_928_4, 1e-16, 0.5 * eps],
        vec![1.0 - 0.5 * eps, 0.153_445_883_127_257_27, 1e-12, 0.95],
    )
    .unwrap();
    lay_rect.set_torus_extents(3.0, 0.5).unwrap();
    assert_eq!(torus_crossing_number(&g, &lay_rect).unwrap(), 1);
    let pairs_rect = assert_torus_matches_oracle("near_endpoint_rect", &edges, &lay_rect);
    assert_eq!(pairs_rect.len(), 1);
    assert_torus_pair_intersects("near_endpoint_rect", &pairs_rect[0]);
}

#[test]
fn test_half_ulp_tilt_and_sub_ulp_wedge_repair() {
    let edges = [(0, 1), (2, 3)];
    let g = two_edge_graph();
    let half_ulp = 2.0_f64.powi(-54);
    let one_ulp = 2.0_f64.powi(-53);

    // 0.5 ULP boundary quantization tilt: exact y_sa(0) = 0.5 + 2^-55 rounds down to 0.5.
    let tilt = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.75, 0.25, 1e-17, 0.5],
        vec![0.75, 0.25 + half_ulp, 0.5, 0.5],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &tilt).unwrap(), 1);
    let tilt_pairs = assert_torus_matches_oracle("half_ulp_tilt", &edges, &tilt);
    assert_eq!(tilt_pairs.len(), 1);
    assert_torus_pair_intersects("half_ulp_tilt", &tilt_pairs[0]);

    // Sub-ULP wedge degeneracy: (-0.125, 0.625 + 2^-53) -> (0.25, 0.25) has y_sa(0) strictly
    // between 0.5 and 0.5 + 2^-53, while crossing sb at (0.0625, 0.4375 + 2^-54).
    let wedge = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.875, 0.25, 0.03125, 0.09375],
        vec![
            0.625 + one_ulp,
            0.25,
            0.46875 + half_ulp,
            0.40625 + half_ulp,
        ],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &wedge).unwrap(), 1);
    let wedge_pairs = assert_torus_matches_oracle("sub_ulp_wedge", &edges, &wedge);
    assert_eq!(wedge_pairs.len(), 1);
    assert_torus_pair_intersects("sub_ulp_wedge", &wedge_pairs[0]);
}

#[test]
fn test_steep_shifted_slope_collapsed_ties_and_deep_exponents() {
    let edges = [(0, 1), (2, 3)];
    let g = two_edge_graph();
    let eps = f64::EPSILON;

    // Steep shift = +1 edge where 1.0 + 1.5e-16 in f64 rounds to 1.0 + eps and would miss.
    let steep = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![1.0 - eps, 1.5e-16, TORUS_MAX - 1e-15, TORUS_MAX],
        vec![0.0, 0.49, 0.14, 0.14],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &steep).unwrap(), 1);
    let steep_pairs = assert_torus_matches_oracle("steep_shift_plus_one", &edges, &steep);
    assert_eq!(steep_pairs.len(), 1);
    assert_torus_pair_intersects("steep_shift_plus_one", &steep_pairs[0]);

    // Collapsed f64 slope tie where shift = +1 endpoints share identical f64 rounded slope.
    let slope_tie = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.5 + eps, 1.1e-16, 0.5 + eps, 0.5e-16],
        vec![1e-17, 0.49, 0.0, 0.49],
    )
    .unwrap();
    let cnt = torus_crossing_number(&g, &slope_tie).unwrap();
    let pairs = assert_torus_matches_oracle("collapsed_slope_tie", &edges, &slope_tie);
    assert_eq!(cnt, pairs.len());

    // Deep-exponent stack-limb exactness: (1 + 2^-200) vs (1 + 2^-300) cannot be resolved by 106-bit DD.
    let o = ShiftedPt::from_f64(0.0, 0.0);
    let a = ShiftedPt::new(
        ShiftedF64::new(2.0_f64.powi(-200), 1),
        ShiftedF64::new(0.0, 1),
    );
    let b = ShiftedPt::new(
        ShiftedF64::new(0.0, 1),
        ShiftedF64::new(2.0_f64.powi(-300), 1),
    );
    // (1 + 2^-200)*(1 + 2^-300) - 1*1 = 2^-200 + 2^-300 + 2^-500 > 0.
    assert_eq!(orient2d_shifted(o, a, b), Ordering::Greater);
    // Collinear-up-to-2^-300 test: (1 + 2^-200)*(1 + 2^-300) vs (1 + 2^-200 + 2^-300)*(1):
    let a2 = ShiftedPt::new(
        ShiftedF64::new(2.0_f64.powi(-200), 1),
        ShiftedF64::new(2.0_f64.powi(-200), 1),
    );
    let b2 = ShiftedPt::new(
        ShiftedF64::new(2.0_f64.powi(-300), 1),
        ShiftedF64::new(2.0_f64.powi(-300), 1),
    );
    assert_eq!(orient2d_shifted(o, a2, b2), Ordering::Equal);
    let b3 = ShiftedPt::new(
        ShiftedF64::new(2.0_f64.powi(-300), 1),
        ShiftedF64::new(2.0_f64.powi(-301), 1),
    );
    assert_eq!(orient2d_shifted(o, a2, b3), Ordering::Less);
    assert_eq!(cmp_slope_shifted(o, a, o, b), Ordering::Less);
    assert_eq!(
        exact_xi_cmp(
            ShiftedPt::from_f64(0.0, 0.0),
            a,
            ShiftedPt::from_f64(0.5, 0.0),
            ShiftedPt::from_f64(0.5, 1.0),
            0.5
        ),
        Some(Ordering::Equal)
    );
}

#[test]
fn test_torus_edge_segments_near_one_corner_accuracy() {
    let pieces = torus_edge_segments(0.9_f64, 1e-16, 1e-16, 0.9_f64).unwrap();
    assert_eq!(pieces.len(), 3);
    for p in &pieces {
        for &c in p {
            assert!(c.is_finite() && (0.0..1.0).contains(&c));
            assert_ne!(c.to_bits(), (-0.0_f64).to_bits());
        }
    }
    // At the x-cut (tx = 0.1 / (0.1 + 1e-16) = 1 - 1e-15, after ty = 1e-15),
    // 1 - tx = 1e-16 / (0.1 + 1e-16), so y(tx) = 0.9 + (1 - tx) * (0.1 + 1e-16) = 0.9 + 1e-16.
    let y_at_xcut = pieces[1][3];
    assert_eq!(pieces[1][3], pieces[2][1]);
    assert!(
        (y_at_xcut - (0.9 + 1e-16)).abs() <= 4.0 * f64::EPSILON,
        "unexpected corner cut y={y_at_xcut}, pieces={pieces:?}"
    );
}

#[test]
fn test_f32_boundary_parity_and_commutativity() {
    let eps32 = f32::EPSILON;
    let xs32 = vec![0.5 * eps32, 0.95_f32, 0.0_f32, 1.0_f32 - eps32, 0.25, 0.75];
    let ys32 = vec![1.0_f32 - eps32, 0.05_f32, 0.25_f32, 0.75_f32, 0.0, 0.9];
    let edges = vec![(0, 1), (2, 3), (4, 5), (0, 3), (1, 4)];
    let mut tor32 = ManifoldLayout2d::<f32>::from_xy(ManifoldKind::Torus2d, xs32, ys32).unwrap();
    tor32.set_torus_extents(3.0, 0.5).unwrap();
    let baseline = assert_torus_matches_oracle("f32_comm_base", &edges, &tor32);

    // Commutativity under endpoint swap (u, v) <-> (v, u) and edge list reversal.
    let flipped: Vec<(u32, u32)> = edges.iter().rev().map(|&(u, v)| (v, u)).collect();
    let rev_pairs = assert_torus_matches_oracle("f32_comm_rev", &flipped, &tor32);
    assert_eq!(baseline.len(), rev_pairs.len());
    for pair in &baseline {
        assert_torus_pair_intersects("f32_comm_base", pair);
    }
}

#[test]
fn test_exact_shifted_matrix_oracle_and_non_gap_intersection_invariant() {
    let near_one = f64::from_bits(1.0_f64.to_bits() - 1);
    let half_eps = 0.5 * f64::EPSILON;
    let pool = [
        0.0,
        1e-16,
        half_eps,
        f64::EPSILON,
        1e-12,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        1.0 - 1e-12,
        1.0 - f64::EPSILON,
        near_one,
    ];
    let aspects = [(1.0, 1.0), (3.0, 0.5), (0.5, 2.5)];
    for (a_idx, &(lx, ly)) in aspects.iter().enumerate() {
        for seed in 1..=60u64 {
            let mut rng = DeterministicRng::from_seed_u64(800_000 + (a_idx as u64) * 1000 + seed);
            let n = 14usize;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                if i < 10 {
                    xs.push(pool[rng.gen_range_usize(pool.len())]);
                    ys.push(pool[rng.gen_range_usize(pool.len())]);
                } else {
                    xs.push(rng.next_f64());
                    ys.push(rng.next_f64());
                }
            }
            let mut tor = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs, ys).unwrap();
            tor.set_torus_extents(lx, ly).unwrap();

            let mut edges = Vec::new();
            for _ in 0..24 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v && !edges.contains(&(u.min(v), u.max(v))) {
                    edges.push((u.min(v), u.max(v)));
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
            let label = format!("precision_matrix_a{a_idx}_s{seed}");
            let got = assert_torus_matches_oracle(&label, &edges, &tor);
            assert_eq!(torus_crossing_number(&g, &tor).unwrap(), got.len());

            // Verify every crossing whose exact shifted intersection lies in [0, TORUS_MAX]^2
            // yields intersecting `seg_a` and `seg_b` under both exact and f64 predicates.
            for c in &got {
                let is_non_gap = has_non_gap_copy(&tor, edges[c.edge_a], edges[c.edge_b]);
                if is_non_gap {
                    assert_torus_pair_intersects(&label, c);
                }
            }
        }
    }
}

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
