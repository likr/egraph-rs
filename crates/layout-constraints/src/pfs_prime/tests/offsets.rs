use egraph_core::{DeterministicRng, Layout2d, LayoutTransform, NodeExtents2d};

use super::super::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
use super::{
    assert_no_overlaps, assert_no_overlaps_all, count_orthogonal_inversions,
    count_orthogonal_inversions_scalar,
};

#[test]
fn test_pfs_prime_f32_large_offsets_zero_overlap_and_oo_nni() {
    let offsets = [0.0_f32, 100.0, 1000.0, 10000.0, -50000.0];
    let paddings = [0.0_f64, 0.25, 0.5, 1.0];

    for (offset_idx, &offset) in offsets.iter().enumerate() {
        let mut rng = DeterministicRng::from_seed_u64(9001 + offset_idx as u64);
        for trial in 0..30 {
            let n = 15 + rng.gen_range_usize(15);
            let use_equal_radii = trial % 3 == 0;
            let padding = paddings[trial % paddings.len()];
            let mut hw = Vec::with_capacity(n);
            let mut hh = Vec::with_capacity(n);
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);

            for i in 0..n {
                let r1 = (0.3 + 1.5 * rng.next_f64()) as f32;
                let r2 = if use_equal_radii {
                    r1
                } else {
                    (0.2 + 1.8 * rng.next_f64()) as f32
                };
                hw.push(r1);
                hh.push(r2);

                if trial % 7 == 0 && i < 3 {
                    // Include exact-coincident points alongside spread points.
                    xs.push(offset);
                    ys.push(offset);
                } else {
                    xs.push(offset + (rng.next_f64() * 6.0 - 3.0) as f32);
                    ys.push(offset + (rng.next_f64() * 6.0 - 3.0) as f32);
                }
            }

            let cfg = PfsPrimeConfig::new()
                .with_padding(padding)
                .with_seed(42 + trial as u64);
            let mut solver_a = if use_equal_radii {
                PfsPrimeOverlapRemoval::<f32>::from_radii(&hw, cfg.clone()).unwrap()
            } else {
                PfsPrimeOverlapRemoval::<f32>::new(
                    NodeExtents2d::new(&hw, &hh).unwrap(),
                    cfg.clone(),
                )
                .unwrap()
            };
            let mut solver_b = if use_equal_radii {
                PfsPrimeOverlapRemoval::<f32>::from_radii(&hw, cfg).unwrap()
            } else {
                PfsPrimeOverlapRemoval::<f32>::new(NodeExtents2d::new(&hw, &hh).unwrap(), cfg)
                    .unwrap()
            };

            let mut layout_a = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let mut layout_b = layout_a.clone();

            solver_a.apply(&mut layout_a).unwrap();
            solver_b.apply(&mut layout_b).unwrap();
            assert_eq!(solver_a.last_overlap_count(), 0);
            assert_eq!(solver_b.last_overlap_count(), 0);

            assert_no_overlaps_all(&layout_a, &hw, &hh, padding);
            assert_eq!(
                count_orthogonal_inversions_scalar(&xs, &ys, layout_a.x(), layout_a.y()),
                0,
                "oo_nni > 0 for f32 offset={offset} trial={trial}"
            );
            assert_eq!(
                layout_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                layout_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                layout_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                layout_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_pfs_prime_f64_large_offset_1e10_zero_overlap_and_oo_nni() {
    let offset = 1e10_f64;
    let paddings = [0.0_f64, 0.25, 0.75];
    let mut rng = DeterministicRng::from_seed_u64(77777);

    for trial in 0..30 {
        let n = 18 + rng.gen_range_usize(12);
        let use_equal_radii = trial % 2 == 0;
        let padding = paddings[trial % paddings.len()];
        let mut hw = Vec::with_capacity(n);
        let mut hh = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let r1 = 0.4 + 1.6 * rng.next_f64();
            let r2 = if use_equal_radii {
                r1
            } else {
                0.2 + 2.0 * rng.next_f64()
            };
            hw.push(r1);
            hh.push(r2);
            if trial % 6 == 0 && i < 2 {
                xs.push(offset);
                ys.push(offset);
            } else {
                xs.push(offset + rng.next_f64() * 8.0 - 4.0);
                ys.push(offset + rng.next_f64() * 8.0 - 4.0);
            }
        }

        let cfg = PfsPrimeConfig::new()
            .with_padding(padding)
            .with_seed(100 + trial as u64);
        let mut solver =
            PfsPrimeOverlapRemoval::<f64>::new(NodeExtents2d::new(&hw, &hh).unwrap(), cfg).unwrap();

        let mut layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        solver.apply(&mut layout).unwrap();
        assert_eq!(solver.last_overlap_count(), 0);
        assert_no_overlaps_all(&layout, &hw, &hh, padding);
        assert_eq!(
            count_orthogonal_inversions(&xs, &ys, layout.x(), layout.y()),
            0,
            "oo_nni > 0 for f64 offset=1e10 trial={trial}"
        );
    }
}

#[test]
fn test_pfs_prime_near_coincident_distinct_points_oo_nni_zero() {
    let xs = vec![0.0_f64, 5e-10, 1e-9, 1.5e-9];
    let ys = vec![0.0_f64, 5e-10, 1e-9, 1.5e-9];
    let hw = vec![0.8_f64, 0.6, 1.0, 0.5];
    let hh = vec![0.5_f64, 0.9, 0.4, 0.7];

    for &padding in &[0.0_f64, 0.25] {
        let cfg = PfsPrimeConfig::new().with_padding(padding).with_seed(42);
        let mut solver =
            PfsPrimeOverlapRemoval::<f64>::new(NodeExtents2d::new(&hw, &hh).unwrap(), cfg).unwrap();
        let mut layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        solver.apply(&mut layout).unwrap();
        assert_eq!(solver.last_overlap_count(), 0);
        assert_no_overlaps_all(&layout, &hw, &hh, padding);
        assert_eq!(
            count_orthogonal_inversions(&xs, &ys, layout.x(), layout.y()),
            0
        );
    }

    // Mixed exact-coincident pair alongside 5e-10-spaced distinct points.
    let xs_mixed = vec![0.0_f64, 0.0, 5e-10, 1e-9, 1.5e-9];
    let ys_mixed = vec![0.0_f64, 0.0, 5e-10, 1e-9, 1.5e-9];
    let radii_mixed = vec![0.75_f64; 5];
    let mut solver_mixed =
        PfsPrimeOverlapRemoval::<f64>::from_radii(&radii_mixed, PfsPrimeConfig::default()).unwrap();
    let mut layout_mixed = Layout2d::from_xy(xs_mixed.clone(), ys_mixed.clone()).unwrap();
    solver_mixed.apply(&mut layout_mixed).unwrap();
    assert_eq!(solver_mixed.last_overlap_count(), 0);
    assert_no_overlaps(&layout_mixed, &radii_mixed, &radii_mixed);
    assert_eq!(
        count_orthogonal_inversions(&xs_mixed, &ys_mixed, layout_mixed.x(), layout_mixed.y()),
        0
    );
}

#[test]
fn test_pfs_prime_repeat_apply_deterministic_f64_and_f32() {
    let xs64 = vec![0.0_f64, 0.0, 0.2, 0.5, 1.1];
    let ys64 = vec![0.0_f64, 0.0, -0.1, 0.3, 0.0];
    let hw64 = vec![0.9_f64, 0.7, 0.8, 0.6, 1.0];
    let hh64 = vec![0.5_f64, 0.8, 0.6, 0.7, 0.4];
    let cfg = PfsPrimeConfig::new().with_padding(0.25).with_seed(1234);
    let ext64 = NodeExtents2d::new(&hw64, &hh64).unwrap();
    let mut s64_a = PfsPrimeOverlapRemoval::<f64>::new(ext64, cfg.clone()).unwrap();
    let mut l64_a = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
    let mut l64_b = l64_a.clone();

    s64_a.apply(&mut l64_a).unwrap();
    s64_a.apply(&mut l64_b).unwrap();
    assert_eq!(
        l64_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l64_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l64_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l64_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let xs32 = vec![-1000.0_f32, -1000.0, -999.8, -999.5, -1000.3];
    let ys32 = vec![500.0_f32, 500.0, 500.2, 499.9, 500.1];
    let hw32 = vec![0.9_f32, 0.7, 0.8, 0.6, 1.0];
    let hh32 = vec![0.5_f32, 0.8, 0.6, 0.7, 0.4];
    let ext32 = NodeExtents2d::new(&hw32, &hh32).unwrap();
    let mut s32_a = PfsPrimeOverlapRemoval::<f32>::new(ext32, cfg.clone()).unwrap();
    let mut l32_a = Layout2d::from_xy(xs32.clone(), ys32.clone()).unwrap();
    let mut l32_b = l32_a.clone();

    s32_a.apply(&mut l32_a).unwrap();
    s32_a.apply(&mut l32_b).unwrap();
    assert_no_overlaps_all(&l32_a, &hw32, &hh32, cfg.padding);
    assert_eq!(
        count_orthogonal_inversions_scalar(&xs32, &ys32, l32_a.x(), l32_a.y()),
        0
    );
    assert_eq!(
        l32_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l32_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l32_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l32_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
