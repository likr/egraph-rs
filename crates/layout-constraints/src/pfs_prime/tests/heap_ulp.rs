use egraph_core::{Layout2d, LayoutTransform, NodeExtents2d};

use super::super::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
use super::{
    assert_no_overlaps, assert_no_overlaps_all, count_orthogonal_inversions,
    count_orthogonal_inversions_scalar,
};

#[test]
fn test_pfs_prime_disjoint_column_heap_order_and_cancellation_f64_f32() {
    let htops = [-5.0_f64, -3.0, 20.0, -2.0];
    let f64_cases = [(1e10_f64, 50usize), (1e8_f64, 20usize), (1e6_f64, 20usize)];

    for &(scale, trials) in &f64_cases {
        for k in 0..trials {
            let target_y = 0.05 + (k as f64) * 0.037;
            let mut xs = Vec::with_capacity(20);
            let mut ys = Vec::with_capacity(20);
            let mut hw = Vec::with_capacity(20);
            let mut hh = Vec::with_capacity(20);

            // Tiny-gap pair keeping min_pos_gap = 1e-12 so the cluster at scale is not jittered.
            xs.extend_from_slice(&[0.0, 1e-12]);
            ys.extend_from_slice(&[0.0, 1e-12]);
            hw.extend_from_slice(&[0.1, 0.1]);
            hh.extend_from_slice(&[0.1, 0.1]);

            // 4 disjoint-column helper pairs bracketing target_y to scramble active_y heap order.
            for (idx, &htop) in htops.iter().enumerate() {
                let hx = scale + 100.0 * ((idx + 1) as f64);
                xs.push(hx);
                ys.push(-scale - 10.0 - (idx as f64));
                hw.push(1.0);
                hh.push(scale + 10.0 + (idx as f64) + htop);

                xs.push(hx);
                ys.push(2.0 * scale + (idx as f64));
                hw.push(1.0);
                hh.push(1.0);
            }

            // Coincident cluster at (scale, -scale): u0, u1, and u2..u9 with sub-ULP hh.
            xs.push(scale);
            ys.push(-scale);
            hw.push(10.0);
            hh.push(scale + target_y);

            xs.push(scale);
            ys.push(-scale);
            hw.push(10.0);
            hh.push(1.0);

            for _ in 0..8 {
                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0);
                hh.push(1e-7);
            }

            let cfg = PfsPrimeConfig::new()
                .with_padding(0.0)
                .with_seed(42 + k as u64);
            let ext = NodeExtents2d::new(&hw, &hh).unwrap();
            let mut solver_step = PfsPrimeOverlapRemoval::<f64>::new(ext, cfg.clone()).unwrap();
            let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();

            solver_step.apply(&mut layout_step).unwrap();
            assert_eq!(solver_step.last_overlap_count(), 0);
            assert_no_overlaps(&layout_step, &hw, &hh);
            assert_eq!(
                count_orthogonal_inversions(&xs, &ys, layout_step.x(), layout_step.y()),
                0
            );

            // Transposed orientation exercising Phase 2 (X-sweep) heap-order + cancellation.
            let ext_t = NodeExtents2d::new(&hh, &hw).unwrap();
            let mut solver_t = PfsPrimeOverlapRemoval::<f64>::new(ext_t, cfg).unwrap();
            let mut layout_t = Layout2d::from_xy(ys.clone(), xs.clone()).unwrap();
            solver_t.apply(&mut layout_t).unwrap();
            assert_eq!(solver_t.last_overlap_count(), 0);
            assert_no_overlaps(&layout_t, &hh, &hw);
            assert_eq!(
                count_orthogonal_inversions(&ys, &xs, layout_t.x(), layout_t.y()),
                0
            );
        }
    }

    // f32 equivalent across scales where float32 ULP exceeds tiny cluster extents.
    let f32_cases = [(1e6_f32, 50usize), (1e5_f32, 20usize), (1e4_f32, 20usize)];
    for &(scale, trials) in &f32_cases {
        for k in 0..trials {
            let target_y = 0.05_f32 + (k as f32) * 0.037_f32;
            let mut xs = Vec::with_capacity(20);
            let mut ys = Vec::with_capacity(20);
            let mut hw = Vec::with_capacity(20);
            let mut hh = Vec::with_capacity(20);

            xs.extend_from_slice(&[0.0_f32, 1e-4_f32]);
            ys.extend_from_slice(&[0.0_f32, 1e-4_f32]);
            hw.extend_from_slice(&[0.1_f32, 0.1_f32]);
            hh.extend_from_slice(&[0.1_f32, 0.1_f32]);

            for (idx, &htop_f64) in htops.iter().enumerate() {
                let htop = htop_f64 as f32;
                let hx = scale + 100.0_f32 * ((idx + 1) as f32);
                xs.push(hx);
                ys.push(-scale - 10.0_f32 - (idx as f32));
                hw.push(1.0_f32);
                hh.push(scale + 10.0_f32 + (idx as f32) + htop);

                xs.push(hx);
                ys.push(2.0_f32 * scale + (idx as f32));
                hw.push(1.0_f32);
                hh.push(1.0_f32);
            }

            xs.push(scale);
            ys.push(-scale);
            hw.push(10.0_f32);
            hh.push(scale + target_y);

            xs.push(scale);
            ys.push(-scale);
            hw.push(10.0_f32);
            hh.push(1.0_f32);

            for _ in 0..8 {
                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0_f32);
                hh.push(1e-4_f32);
            }

            let cfg = PfsPrimeConfig::new()
                .with_padding(0.0)
                .with_seed(42 + k as u64);
            let ext = NodeExtents2d::new(&hw, &hh).unwrap();
            let mut solver_step = PfsPrimeOverlapRemoval::<f32>::new(ext, cfg.clone()).unwrap();
            let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            solver_step.apply(&mut layout_step).unwrap();
            assert_eq!(solver_step.last_overlap_count(), 0);
            assert_no_overlaps(&layout_step, &hw, &hh);
            assert_eq!(
                count_orthogonal_inversions_scalar(&xs, &ys, layout_step.x(), layout_step.y()),
                0
            );

            // Transposed f32 case exercising Phase 2.
            let ext_t = NodeExtents2d::new(&hh, &hw).unwrap();
            let mut solver_t = PfsPrimeOverlapRemoval::<f32>::new(ext_t, cfg).unwrap();
            let mut layout_t = Layout2d::from_xy(ys.clone(), xs.clone()).unwrap();
            solver_t.apply(&mut layout_t).unwrap();
            assert_eq!(solver_t.last_overlap_count(), 0);
            assert_no_overlaps(&layout_t, &hh, &hw);
            assert_eq!(
                count_orthogonal_inversions_scalar(&ys, &xs, layout_t.x(), layout_t.y()),
                0
            );
        }
    }
}

#[test]
fn test_pfs_prime_3node_pfsp_force_ulp_x_and_y() {
    let xs_base = vec![10000.0 - 1e-6, 10000.0, 10000.0 + 8e-10];
    let ys_base = vec![0.0, 0.0, -1e8];
    let hh_base = vec![1.0, 1.0, 1.0];

    for k in 0..50 {
        let hw_base = vec![40000.0 + (k as f64) * 13.7, 1.0, 476.0 + (k as f64) * 0.31];
        let cfg = PfsPrimeConfig::default();

        // X-axis scan-line orientation (50 trials).
        let ext_x = NodeExtents2d::new(&hw_base, &hh_base).unwrap();
        let mut sx = PfsPrimeOverlapRemoval::<f64>::new(ext_x, cfg.clone()).unwrap();
        let mut lx = Layout2d::from_xy(xs_base.clone(), ys_base.clone()).unwrap();

        sx.apply(&mut lx).unwrap();
        assert_eq!(sx.last_overlap_count(), 0);
        assert_no_overlaps_all(&lx, &hw_base, &hh_base, 0.0);
        assert_eq!(
            count_orthogonal_inversions(&xs_base, &ys_base, lx.x(), lx.y()),
            0,
            "3-node X oo_nni > 0 at k={k}"
        );

        // Y-axis transposed orientation (50 trials).
        let ext_y = NodeExtents2d::new(&hh_base, &hw_base).unwrap();
        let mut sy = PfsPrimeOverlapRemoval::<f64>::new(ext_y, cfg).unwrap();
        let mut ly = Layout2d::from_xy(ys_base.clone(), xs_base.clone()).unwrap();

        sy.apply(&mut ly).unwrap();
        assert_eq!(sy.last_overlap_count(), 0);
        assert_no_overlaps_all(&ly, &hh_base, &hw_base, 0.0);
        assert_eq!(
            count_orthogonal_inversions(&ys_base, &xs_base, ly.x(), ly.y()),
            0,
            "3-node Y oo_nni > 0 at k={k}"
        );
    }
}
