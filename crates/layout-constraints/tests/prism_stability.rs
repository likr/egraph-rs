use egraph_core::{DeterministicRng, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_constraints::constraints::project_rectangle_no_overlap_2d;
use egraph_layout_constraints::prism::{PrismConfig, PrismOverlapRemoval};
use egraph_spatial_trees::RectangleOverlapSweep;

fn seeded_layout_f64(n: usize, scale: f64, seed: u64) -> Layout2d<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push((rng.next_f64() * 2.0 - 1.0) * scale);
        y.push((rng.next_f64() * 2.0 - 1.0) * scale);
    }
    Layout2d::from_xy(x, y).unwrap()
}

fn naive_overlap_count_f32(layout: &Layout2d<f32>, hw: &[f32], hh: &[f32]) -> usize {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut count = 0usize;
    for i in 0..n {
        let ax_min = xs[i] - hw[i];
        let ax_max = xs[i] + hw[i];
        let ay_min = ys[i] - hh[i];
        let ay_max = ys[i] + hh[i];
        for j in (i + 1)..n {
            let bx_min = xs[j] - hw[j];
            let bx_max = xs[j] + hw[j];
            let by_min = ys[j] - hh[j];
            let by_max = ys[j] + hh[j];
            if ax_min < bx_max && bx_min < ax_max && ay_min < by_max && by_min < ay_max {
                count += 1;
            }
        }
    }
    count
}

fn naive_overlap_count_f64(layout: &Layout2d<f64>, hw: &[f64], hh: &[f64]) -> usize {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut count = 0usize;
    for i in 0..n {
        let ax_min = xs[i] - hw[i];
        let ax_max = xs[i] + hw[i];
        let ay_min = ys[i] - hh[i];
        let ay_max = ys[i] + hh[i];
        for j in (i + 1)..n {
            let bx_min = xs[j] - hw[j];
            let bx_max = xs[j] + hw[j];
            let by_min = ys[j] - hh[j];
            let by_max = ys[j] + hh[j];
            if ax_min < bx_max && bx_min < ax_max && ay_min < by_max && by_min < ay_max {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn test_prism_f32_large_coordinate_offsets_converge_zero_overlaps() {
    // Regression test: f32 layouts at coordinate offsets 5000.0 and 1_000_000.0
    // must converge with 0 overlaps across 20 random seeds (both Delaunay and StaticKdTree fallback).
    let n = 50usize;
    let hw = vec![0.5_f32; n];
    let hh = vec![0.5_f32; n];

    for &offset in &[5_000.0_f32, 1_000_000.0_f32] {
        for seed in 0..20u64 {
            let mut rng = DeterministicRng::from_seed_u64(5_000 + seed);
            let mut x = Vec::with_capacity(n);
            let mut y = Vec::with_capacity(n);
            for _ in 0..n {
                x.push(offset + (rng.next_f64() as f32 * 4.0 - 2.0));
                y.push(-offset + (rng.next_f64() as f32 * 4.0 - 2.0));
            }
            let mut layout = Layout2d::from_xy(x, y).unwrap();
            let mut solver =
                PrismOverlapRemoval::new(hw.clone(), hh.clone(), PrismConfig::default()).unwrap();
            let status = solver.run(&mut layout).unwrap();
            assert_eq!(
                status,
                SolverStatus::Converged,
                "PRISM f32 did not converge at offset={offset} seed={seed}: iters={} overlaps={}",
                solver.iteration(),
                solver.last_overlap_count()
            );
            assert_eq!(solver.last_overlap_count(), 0);
            assert_eq!(
                naive_overlap_count_f32(&layout, &hw, &hh),
                0,
                "naive O(N^2) overlap check failed at offset={offset} seed={seed}"
            );
            assert!(!RectangleOverlapSweep::has_layout_overlaps(&layout, &hw, &hh).unwrap());
        }
    }

    // Also verify project_rectangle_no_overlap_2d with non-power-of-two f32 extents at 1_000_000.0.
    let hw_nonpow = vec![0.3_f32; n];
    let hh_nonpow = vec![0.3_f32; n];
    for &offset in &[5_000.0_f32, 1_000_000.0_f32] {
        let mut rng = DeterministicRng::from_seed_u64(777);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(offset + (rng.next_f64() as f32 * 4.0 - 2.0));
            y.push(-offset + (rng.next_f64() as f32 * 4.0 - 2.0));
        }
        let mut layout = Layout2d::from_xy(x, y).unwrap();
        project_rectangle_no_overlap_2d(&mut layout, &hw_nonpow, &hh_nonpow).unwrap();
        assert_eq!(naive_overlap_count_f32(&layout, &hw_nonpow, &hh_nonpow), 0);
        assert!(
            !RectangleOverlapSweep::has_layout_overlaps(&layout, &hw_nonpow, &hh_nonpow).unwrap()
        );
    }
}

fn check_prism_case_f64(
    label: &str,
    init: Layout2d<f64>,
    hw: &[f64],
    hh: &[f64],
    cfg: PrismConfig,
) -> usize {
    let mut l_run = init.clone();
    let mut l_step = init;
    let mut s_run = PrismOverlapRemoval::new(hw.to_vec(), hh.to_vec(), cfg).unwrap();
    let mut s_step = PrismOverlapRemoval::new(hw.to_vec(), hh.to_vec(), cfg).unwrap();

    let st_run = s_run.run(&mut l_run).unwrap();
    assert_eq!(
        st_run,
        SolverStatus::Converged,
        "{label} (f64): did not converge (status={st_run:?}, iters={}, overlaps={})",
        s_run.iteration(),
        s_run.last_overlap_count()
    );
    assert_eq!(s_run.last_overlap_count(), 0, "{label} (f64)");
    assert_eq!(naive_overlap_count_f64(&l_run, hw, hh), 0, "{label} (f64)");
    assert!(
        !RectangleOverlapSweep::has_layout_overlaps(&l_run, hw, hh).unwrap(),
        "{label} (f64)"
    );

    let _ = s_step.run_steps(&mut l_step, 3).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status(), "{label} (f64)");
    assert_eq!(s_run.iteration(), s_step.iteration(), "{label} (f64)");
    assert_eq!(
        s_run.last_overlap_count(),
        s_step.last_overlap_count(),
        "{label} (f64)"
    );
    for i in 0..l_run.len() {
        assert_eq!(
            l_run.x()[i].to_bits(),
            l_step.x()[i].to_bits(),
            "{label} (f64) x[{i}] bit-exact mismatch"
        );
        assert_eq!(
            l_run.y()[i].to_bits(),
            l_step.y()[i].to_bits(),
            "{label} (f64) y[{i}] bit-exact mismatch"
        );
    }
    s_run.iteration()
}

fn check_prism_case_f32(
    label: &str,
    init: Layout2d<f32>,
    hw: &[f32],
    hh: &[f32],
    cfg: PrismConfig,
) -> usize {
    let mut l_run = init.clone();
    let mut l_step = init;
    let mut s_run = PrismOverlapRemoval::new(hw.to_vec(), hh.to_vec(), cfg).unwrap();
    let mut s_step = PrismOverlapRemoval::new(hw.to_vec(), hh.to_vec(), cfg).unwrap();

    let st_run = s_run.run(&mut l_run).unwrap();
    assert_eq!(
        st_run,
        SolverStatus::Converged,
        "{label} (f32): did not converge (status={st_run:?}, iters={}, overlaps={})",
        s_run.iteration(),
        s_run.last_overlap_count()
    );
    assert_eq!(s_run.last_overlap_count(), 0, "{label} (f32)");
    assert_eq!(naive_overlap_count_f32(&l_run, hw, hh), 0, "{label} (f32)");
    assert!(
        !RectangleOverlapSweep::has_layout_overlaps(&l_run, hw, hh).unwrap(),
        "{label} (f32)"
    );

    let _ = s_step.run_steps(&mut l_step, 3).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status(), "{label} (f32)");
    assert_eq!(s_run.iteration(), s_step.iteration(), "{label} (f32)");
    assert_eq!(
        s_run.last_overlap_count(),
        s_step.last_overlap_count(),
        "{label} (f32)"
    );
    for i in 0..l_run.len() {
        assert_eq!(
            l_run.x()[i].to_bits(),
            l_step.x()[i].to_bits(),
            "{label} (f32) x[{i}] bit-exact mismatch"
        );
        assert_eq!(
            l_run.y()[i].to_bits(),
            l_step.y()[i].to_bits(),
            "{label} (f32) y[{i}] bit-exact mismatch"
        );
    }
    s_run.iteration()
}

#[test]
fn test_prism_aspect_100_to_1_and_mixed_sizes_converge_bit_exact() {
    let default_cfg = PrismConfig::default();

    // 1. N=50 alternating 100:1 aspect ratio boxes ((5.0, 0.05) and (0.05, 5.0)),
    // including the offset=100, seed=4 limit-cycle reproducer (rng seed 5104).
    let n50 = 50usize;
    let hw50_64: Vec<f64> = (0..n50)
        .map(|i| if i % 2 == 0 { 5.0 } else { 0.05 })
        .collect();
    let hh50_64: Vec<f64> = (0..n50)
        .map(|i| if i % 2 == 0 { 0.05 } else { 5.0 })
        .collect();
    let hw50_32: Vec<f32> = hw50_64.iter().map(|&v| v as f32).collect();
    let hh50_32: Vec<f32> = hh50_64.iter().map(|&v| v as f32).collect();

    for &(offset, seed) in &[(100.0_f64, 4u64), (1000.0, 17), (50_000.0, 8)] {
        let mut rng = DeterministicRng::from_seed_u64(5_000 + offset as u64 + seed);
        let mut x64 = Vec::with_capacity(n50);
        let mut y64 = Vec::with_capacity(n50);
        for _ in 0..n50 {
            x64.push(offset + (rng.next_f64() * 8.0 - 4.0));
            y64.push(-offset + (rng.next_f64() * 8.0 - 4.0));
        }
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();

        let label = format!("aspect_100:1_n50_off={offset}_seed={seed}");
        check_prism_case_f64(
            &label,
            Layout2d::from_xy(x64.clone(), y64.clone()).unwrap(),
            &hw50_64,
            &hh50_64,
            default_cfg,
        );
        check_prism_case_f32(
            &label,
            Layout2d::from_xy(x32, y32).unwrap(),
            &hw50_32,
            &hh50_32,
            default_cfg,
        );

        // Also verify oscillation detection stops early well before max_iterations=200.
        if offset == 100.0 && seed == 4 {
            let iters = check_prism_case_f64(
                "aspect_100:1_n50_off=100_seed=4_max200",
                Layout2d::from_xy(x64, y64).unwrap(),
                &hw50_64,
                &hh50_64,
                PrismConfig {
                    max_iterations: 200,
                    ..default_cfg
                },
            );
            assert!(
                iters < 100,
                "expected early stall fallback convergence < 100 iters, got {iters}"
            );
        }
    }

    // 2. N=80 wide [-18, 18]^2 100:1 aspect ratio boxes across multiple seeds.
    let n80 = 80usize;
    let hw80_64: Vec<f64> = (0..n80)
        .map(|i| if i % 2 == 0 { 5.0 } else { 0.05 })
        .collect();
    let hh80_64: Vec<f64> = (0..n80)
        .map(|i| if i % 2 == 0 { 0.05 } else { 5.0 })
        .collect();
    let hw80_32: Vec<f32> = hw80_64.iter().map(|&v| v as f32).collect();
    let hh80_32: Vec<f32> = hh80_64.iter().map(|&v| v as f32).collect();

    for seed in 0..20u64 {
        let l64 = seeded_layout_f64(n80, 18.0, 9_000 + seed);
        let l32 = Layout2d::from_xy(
            l64.x().iter().map(|&v| v as f32).collect(),
            l64.y().iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let label = format!("aspect_100:1_n80_wide_seed={seed}");
        check_prism_case_f64(&label, l64, &hw80_64, &hh80_64, default_cfg);
        check_prism_case_f32(&label, l32, &hw80_32, &hh80_32, default_cfg);
    }

    // 3. Mixed-size boxes at offsets 1000, 50_000, 1_000_000 (plus 0) for both f64 and f32.
    let mut mixed_hw64 = Vec::with_capacity(n50);
    let mut mixed_hh64 = Vec::with_capacity(n50);
    for i in 0..n50 {
        let (w, h) = match i % 5 {
            0 => (3.5, 2.5),
            1 => (0.15, 0.15),
            2 => (2.0, 0.2),
            3 => (0.2, 2.0),
            _ => (0.6, 0.4),
        };
        mixed_hw64.push(w);
        mixed_hh64.push(h);
    }
    let mixed_hw32: Vec<f32> = mixed_hw64.iter().map(|&v| v as f32).collect();
    let mixed_hh32: Vec<f32> = mixed_hh64.iter().map(|&v| v as f32).collect();

    for &offset in &[0.0_f64, 1_000.0, 50_000.0, 1_000_000.0] {
        for seed in 0..20u64 {
            let mut rng = DeterministicRng::from_seed_u64(12_000 + offset as u64 + seed);
            let mut x64 = Vec::with_capacity(n50);
            let mut y64 = Vec::with_capacity(n50);
            for _ in 0..n50 {
                x64.push(offset + (rng.next_f64() * 8.0 - 4.0));
                y64.push(-offset + (rng.next_f64() * 8.0 - 4.0));
            }
            let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();
            let label = format!("mixed_sizes_off={offset}_seed={seed}");
            check_prism_case_f64(
                &label,
                Layout2d::from_xy(x64, y64).unwrap(),
                &mixed_hw64,
                &mixed_hh64,
                default_cfg,
            );
            check_prism_case_f32(
                &label,
                Layout2d::from_xy(x32, y32).unwrap(),
                &mixed_hw32,
                &mixed_hh32,
                default_cfg,
            );
        }
    }
}
