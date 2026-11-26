use std::f64::consts::FRAC_PI_2;

use egraph_core::{CsrGraph, DeterministicRng, ManifoldKind, ManifoldLayout2d};

use super::super::{
    torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_layout_edge_segments,
};
use super::helpers::assert_torus_matches_oracle;

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
