use std::f64::consts::FRAC_PI_2;

use egraph_core::{CsrGraph, DeterministicRng, ManifoldKind, ManifoldLayout2d};

use super::super::euclidean_geom::segments_geometrically_intersect;
use super::super::geometry::Seg;
use super::super::{
    torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_layout_edge_segments,
};
use super::helpers::{assert_torus_matches_oracle, assert_torus_pair_intersects};

fn assert_segments_valid(segs: &[[f64; 4]], label: &str) {
    for seg in segs {
        for &c in seg {
            assert!(
                c.is_finite() && (0.0..1.0).contains(&c),
                "{label}: segment coordinate out of [0, 1): {seg:?}"
            );
            assert_ne!(
                c.to_bits(),
                (-0.0_f64).to_bits(),
                "{label}: segment coordinate has -0.0: {seg:?}"
            );
        }
    }
}

#[test]
fn test_torus_boundary_corner_and_sub_ulp_reproducers() {
    let edges = [(0, 1), (2, 3)];
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();
    let near_one = f64::from_bits(1.0_f64.to_bits() - 1);

    // REPRO1: corner-wrapping edge vs unrelated y-wrapping edge must report 0 crossings.
    let repro1 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 0.5, 0.5],
        vec![0.9, 0.1, 0.95, 0.05],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro1).unwrap(), 0);
    assert_torus_matches_oracle("repro1", &edges, &repro1);
    assert_segments_valid(&torus_layout_edge_segments(&g, &repro1).unwrap(), "repro1");

    // REPRO2: corner-wrapping edge vs strictly interior edge must report 0 crossings.
    let repro2 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 0.4, 0.5],
        vec![0.9, 0.1, 0.85, 0.98],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro2).unwrap(), 0);
    assert_torus_matches_oracle("repro2", &edges, &repro2);
    assert_segments_valid(&torus_layout_edge_segments(&g, &repro2).unwrap(), "repro2");

    // Near-corner 1e-16 perturbation must also emit clean corner segments and 0 crossings.
    let near_corner = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.9, 0.5, 0.5],
        vec![0.1 + 1e-16, 0.9, 0.8, 0.2],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &near_corner).unwrap(), 0);
    assert_torus_matches_oracle("near_corner_tiny_pos", &edges, &near_corner);
    assert_segments_valid(
        &torus_layout_edge_segments(&g, &near_corner).unwrap(),
        "near_corner_tiny_pos",
    );

    // REPRO3: exact corner crossing at (0, 0) under (1, 1), (3.0, 0.5), and (0.5, 2.5) extents.
    let mut repro3 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 0.9, 0.1],
        vec![0.9, 0.1, 0.1, 0.9],
    )
    .unwrap();
    let pairs_sq = assert_torus_matches_oracle("repro3_sq", &edges, &repro3);
    assert_eq!(pairs_sq.len(), 1);
    assert!((pairs_sq[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);
    assert!(pairs_sq[0].cos_sq < 1e-12);

    for (lx, ly, label) in [
        (3.0_f64, 0.5_f64, "repro3_3_0.5"),
        (0.5_f64, 2.5_f64, "repro3_0.5_2.5"),
    ] {
        repro3.set_torus_extents(lx, ly).unwrap();
        assert_eq!(torus_crossing_number(&g, &repro3).unwrap(), 1);
        let pairs = assert_torus_matches_oracle(label, &edges, &repro3);
        assert_eq!(pairs.len(), 1);
        let dot = (0.2 * lx) * (0.2 * lx) - (0.2 * ly) * (0.2 * ly);
        let n1 = (0.2 * lx).hypot(0.2 * ly);
        let cos_val = (dot / (n1 * n1)).clamp(-1.0, 1.0);
        let exp_angle = cos_val.abs().acos().min(FRAC_PI_2);
        let exp_cos_sq = cos_val * cos_val;
        assert!((pairs[0].angle_rad.unwrap() - exp_angle).abs() < 1e-12);
        assert!((pairs[0].cos_sq - exp_cos_sq).abs() < 1e-12);
    }

    // REPRO4a / REPRO4b: sub-ULP near_one along x or y must not wrap 1.0 -> 0.0.
    let repro4a = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![near_one, near_one, 0.5, 0.5],
        vec![0.9, 0.1, 0.2, 0.8],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro4a).unwrap(), 0);
    assert_torus_matches_oracle("repro4a", &edges, &repro4a);
    assert_segments_valid(
        &torus_layout_edge_segments(&g, &repro4a).unwrap(),
        "repro4a",
    );

    let repro4b = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 0.2, 0.8],
        vec![near_one, near_one, 0.5, 0.5],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro4b).unwrap(), 0);
    assert_torus_matches_oracle("repro4b", &edges, &repro4b);
    assert_segments_valid(
        &torus_layout_edge_segments(&g, &repro4b).unwrap(),
        "repro4b",
    );

    // REPRO5: y0 = 0.0, y1_w = near_one must not suffer cy ~ 0 cancellation blowup.
    let repro5 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.0, 0.88, 0.88],
        vec![0.0, near_one, 0.9, 0.1],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro5).unwrap(), 0);
    assert_torus_matches_oracle("repro5", &edges, &repro5);
    assert_segments_valid(&torus_layout_edge_segments(&g, &repro5).unwrap(), "repro5");

    // REPRO6: boundary-touching stub at torus_max must preserve exact unwrapped crossing angle.
    let repro6 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.535, 0.496, 0.5, 0.544],
        vec![near_one, 0.1, near_one, near_one],
    )
    .unwrap();
    let pairs6 = assert_torus_matches_oracle("repro6_y01", &edges, &repro6);
    assert_eq!(pairs6.len(), 1);
    assert!((pairs6[0].angle_rad.unwrap() - 0.1_f64.atan2(0.039)).abs() < 1e-12);
    assert_segments_valid(&torus_layout_edge_segments(&g, &repro6).unwrap(), "repro6");

    let repro6_y05 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.535, 0.496, 0.5, 0.544],
        vec![near_one, 0.5, near_one, near_one],
    )
    .unwrap();
    let pairs6_y05 = assert_torus_matches_oracle("repro6_y05", &edges, &repro6_y05);
    assert_eq!(pairs6_y05.len(), 1);
    assert!((pairs6_y05[0].angle_rad.unwrap() - 1.492_953_935_857_574_5).abs() < 1e-12);

    // Grazing boundary variant: vertex on y = 0.0 wraps across y = 0 against on-boundary edge.
    let grazing = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.535, 0.496, 0.5, 0.544],
        vec![0.0, 0.9, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &grazing).unwrap(), 1);
    assert_torus_matches_oracle("grazing_y0", &edges, &grazing);
}

#[test]
fn test_torus_planar_4x4_grid_with_diagonals() {
    let rows = 4usize;
    let cols = 4usize;
    let n = rows * cols;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for r in 0..rows {
        for c in 0..cols {
            xs.push(c as f64 * 0.25);
            ys.push(r as f64 * 0.25);
        }
    }
    let layout = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs, ys).unwrap();

    for diag_dir in [1usize, cols - 1] {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                let right = (r * cols + (c + 1) % cols) as u32;
                let up = (((r + 1) % rows) * cols + c) as u32;
                let diag = (((r + 1) % rows) * cols + (c + diag_dir) % cols) as u32;
                edges.push((u, right));
                edges.push((u, up));
                edges.push((u, diag));
            }
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
        assert_eq!(
            torus_crossing_number(&g, &layout).unwrap(),
            0,
            "planar 4x4 torus triangulation (diag_dir={diag_dir}) must have 0 crossings"
        );
        assert!(torus_crossing_edges(&g, &layout).unwrap().is_empty());
        assert_torus_matches_oracle(&format!("torus_grid_4x4_diag_{diag_dir}"), &edges, &layout);
        assert_segments_valid(
            &torus_layout_edge_segments(&g, &layout).unwrap(),
            "torus_grid_4x4",
        );
    }
}

#[test]
fn test_torus_snapped_and_cornered_randomized_oracle() {
    let near_one = f64::from_bits(1.0_f64.to_bits() - 1);
    let pool = [
        0.0,
        1e-16,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        1.0 - f64::EPSILON,
        near_one,
    ];

    for (aspect_idx, (lx, ly)) in [(1.0, 1.0), (3.0, 0.5), (0.5, 2.5)].iter().enumerate() {
        for seed in 1..=30u64 {
            let mut rng =
                DeterministicRng::from_seed_u64(120_000 + (aspect_idx as u64) * 1000 + seed);
            let n = 16usize;
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
            tor.set_torus_extents(*lx, *ly).unwrap();

            let mut edges = Vec::new();
            for _ in 0..28 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v && !edges.contains(&(u.min(v), u.max(v))) {
                    edges.push((u.min(v), u.max(v)));
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
            let got = assert_torus_matches_oracle(
                &format!("snapped_oracle_a{aspect_idx}_s{seed}"),
                &edges,
                &tor,
            );
            assert_eq!(torus_crossing_number(&g, &tor).unwrap(), got.len());
            assert_eq!(
                torus_crossing_edges_from_edges(&edges, &tor).unwrap().len(),
                got.len()
            );
            assert_segments_valid(
                &torus_layout_edge_segments(&g, &tor).unwrap(),
                &format!("snapped_segs_a{aspect_idx}_s{seed}"),
            );
        }
    }
}

#[test]
fn test_torus_sub_ulp_unclipped_copy_and_disjoint_neighbor_regression() {
    let edges = [(0, 1), (2, 3)];
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();
    let eps = f64::EPSILON;
    let near_one = f64::from_bits(1.0_f64.to_bits() - 1);

    // Exact (0, 0)-centered copy intersects at (-0.5*eps, 0.0) while cx=-1 copy loses
    // 0.5*eps to 1.0 + 0.5*eps == 1.0 rounding; emitting all <= 4 lattice copies preserves it.
    let sub_ulp_eps = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.5 * eps, 1.0 - eps, 0.5 * eps, near_one],
        vec![eps, 1.0 - 0.5 * eps, 0.95, 0.0],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &sub_ulp_eps).unwrap(), 1);
    assert_eq!(torus_crossing_edges(&g, &sub_ulp_eps).unwrap().len(), 1);
    let pairs_eps = assert_torus_matches_oracle("sub_ulp_eps_crossing", &edges, &sub_ulp_eps);
    assert_eq!(pairs_eps.len(), 1);
    assert!(segments_geometrically_intersect(
        &Seg::from_endpoints(
            0,
            0,
            1,
            pairs_eps[0].seg_a[0],
            pairs_eps[0].seg_a[1],
            pairs_eps[0].seg_a[2],
            pairs_eps[0].seg_a[3],
        ),
        &Seg::from_endpoints(
            1,
            2,
            3,
            pairs_eps[0].seg_b[0],
            pairs_eps[0].seg_b[1],
            pairs_eps[0].seg_b[2],
            pairs_eps[0].seg_b[3],
        ),
    ));

    // Tiny non-0.5*eps coordinate variant where (0, 0)-centered copy crosses while cx=-1 misses.
    let sub_ulp_tiny_cross = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![1e-20, 1.0 - eps, 1e-20, near_one],
        vec![1.5 * eps, 1.0 - 1.5 * eps, 0.95, 0.0],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &sub_ulp_tiny_cross).unwrap(), 1);
    assert_eq!(
        torus_crossing_edges(&g, &sub_ulp_tiny_cross).unwrap().len(),
        1
    );
    assert_torus_matches_oracle("sub_ulp_tiny_crossing", &edges, &sub_ulp_tiny_cross);

    // Near-boundary non-crossing variant must remain 0 and match oracle without false positives.
    let sub_ulp_tiny_miss = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![1e-20, 1.0 - 1e-15, 1e-20, 1.0 - 5e-16],
        vec![1e-15, 1.0 - 1e-16, 0.95, 0.0],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &sub_ulp_tiny_miss).unwrap(), 0);
    assert_torus_matches_oracle("sub_ulp_tiny_non_crossing", &edges, &sub_ulp_tiny_miss);

    // Crossing at (1.0 - 5e-13, 0.5) must select the [0, 1)^2 cell with intersecting
    // clipped sub-segments rather than the disjoint neighbor cell at xi = -5e-13.
    let near_one_cell = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 1.0 - 1e-12, 0.0],
        vec![0.3, 0.7, 0.7, 0.3],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &near_one_cell).unwrap(), 1);
    let pairs_near_one =
        assert_torus_matches_oracle("near_one_intersecting_cell", &edges, &near_one_cell);
    assert_eq!(pairs_near_one.len(), 1);
    let c = &pairs_near_one[0];
    assert!(
        segments_geometrically_intersect(
            &Seg::from_endpoints(0, 0, 1, c.seg_a[0], c.seg_a[1], c.seg_a[2], c.seg_a[3]),
            &Seg::from_endpoints(1, 2, 3, c.seg_b[0], c.seg_b[1], c.seg_b[2], c.seg_b[3]),
        ),
        "clipped TorusCrossing segments must intersect: seg_a={:?}, seg_b={:?}",
        c.seg_a,
        c.seg_b
    );
}

#[test]
fn test_torus_clipping_boundary_snap_and_sterbenz_reproducers() {
    let edges = [(0, 1), (2, 3)];
    let g = CsrGraph::<f64>::from_edges(4, &edges, None, false, false).unwrap();

    // REPRO_R4_A_a0_s146: direct [0, TORUS_MAX]^2 clipping keeps endpoint-touching segments intersecting.
    let repro_a = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.95, 0.75, 0.95, 0.9],
        vec![0.9, 0.1, 1.0 - 1e-12, 0.95],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro_a).unwrap(), 1);
    let pairs_a = torus_crossing_edges(&g, &repro_a).unwrap();
    assert_eq!(pairs_a.len(), 1);
    assert_eq!(
        torus_crossing_edges_from_edges(&edges, &repro_a)
            .unwrap()
            .len(),
        1
    );
    assert_torus_pair_intersects("REPRO_R4_A_a0_s146", &pairs_a[0]);
    assert_torus_matches_oracle("REPRO_R4_A_a0_s146", &edges, &repro_a);

    // REPRO_R4_B_f64_y0_residual: snapping constrained y=0.0 eliminates 2.77e-17 residual.
    let repro_b_f64 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.5, 0.25, 0.0, 0.411_784_202_862_037_9],
        vec![0.0, 0.0, 0.25, 0.857_126_664_672_255_7],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro_b_f64).unwrap(), 1);
    let pairs_b_f64 = torus_crossing_edges(&g, &repro_b_f64).unwrap();
    assert_eq!(pairs_b_f64.len(), 1);
    assert_torus_pair_intersects("REPRO_R4_B_f64_y0_residual", &pairs_b_f64[0]);
    assert_torus_matches_oracle("REPRO_R4_B_f64_y0_residual", &edges, &repro_b_f64);

    // REPRO_R4_B_f32_y0_residual: snapping constrained y=0.0 eliminates 6.62e-24 f32 residual.
    let near_one_f32 = 1.0_f32 - 0.5 * f32::EPSILON;
    let repro_b_f32 = ManifoldLayout2d::<f32>::from_xy(
        ManifoldKind::Torus2d,
        vec![near_one_f32, 0.07138637, 0.95, 0.5 * f32::EPSILON],
        vec![0.5 * f32::EPSILON, 0.7705116, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro_b_f32).unwrap(), 1);
    let pairs_b_f32 = torus_crossing_edges(&g, &repro_b_f32).unwrap();
    assert_eq!(pairs_b_f32.len(), 1);
    assert_torus_pair_intersects("REPRO_R4_B_f32_y0_residual", &pairs_b_f32[0]);
    assert_torus_matches_oracle("REPRO_R4_B_f32_y0_residual", &edges, &repro_b_f32);

    // REPRO_R4_C_missed_dd_sub_ulp: Sterbenz shift for y>=0.9 avoids 1.0 + 1e-16 absorption.
    let eps = f64::EPSILON;
    let repro_c = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![eps, 1e-12, 0.0, 0.277_488_775_723_142_7],
        vec![1.0 - eps, 1.0 - eps, 1e-16, 0.616_063_961_723_204_7],
    )
    .unwrap();
    assert_eq!(torus_crossing_number(&g, &repro_c).unwrap(), 1);
    let pairs_c = torus_crossing_edges(&g, &repro_c).unwrap();
    assert_eq!(pairs_c.len(), 1);
    assert_eq!(
        torus_crossing_edges_from_edges(&edges, &repro_c)
            .unwrap()
            .len(),
        1
    );
    assert_torus_pair_intersects("REPRO_R4_C_missed_dd_sub_ulp", &pairs_c[0]);
}

#[test]
fn test_torus_non_sub_ulp_randomized_intersecting_segments() {
    let pool = [
        0.0,
        1e-6,
        0.07,
        0.19,
        0.31,
        0.47,
        0.61,
        0.73,
        0.89,
        1.0 - 3e-6,
    ];
    let aspects = [(1.0, 1.0), (3.0, 0.5), (0.5, 2.5), (2.0, 0.75)];
    for (aspect_idx, (lx, ly)) in aspects.iter().enumerate() {
        for seed in 1..=150u64 {
            let mut rng =
                DeterministicRng::from_seed_u64(420_000 + (aspect_idx as u64) * 1000 + seed);
            let n = 14usize;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                if i < 8 {
                    xs.push(pool[rng.gen_range_usize(pool.len())]);
                    ys.push(pool[rng.gen_range_usize(pool.len())]);
                } else {
                    xs.push(0.05 + 0.90 * rng.next_f64());
                    ys.push(0.05 + 0.90 * rng.next_f64());
                }
            }
            let mut tor = ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, xs, ys).unwrap();
            tor.set_torus_extents(*lx, *ly).unwrap();

            let mut edges = Vec::new();
            for _ in 0..24 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v && !edges.contains(&(u.min(v), u.max(v))) {
                    edges.push((u.min(v), u.max(v)));
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
            let label = format!("non_sub_ulp_a{aspect_idx}_s{seed}");
            let got = assert_torus_matches_oracle(&label, &edges, &tor);
            assert_eq!(torus_crossing_number(&g, &tor).unwrap(), got.len());
            assert_eq!(torus_crossing_edges(&g, &tor).unwrap().len(), got.len());
            for pair in &got {
                assert_torus_pair_intersects(&label, pair);
            }
        }
    }
}
