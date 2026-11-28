use std::time::Instant;

use egraph_core::{
    CoreError, DeterministicRng, IterativeSolver, Layout2d, NodeExtents2d, SolverStatus,
};
use egraph_layout_constraints::{PrismConfig, PrismOverlapRemoval, SparsePcgWorkspace};
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

fn seeded_layout_f32(n: usize, scale: f32, seed: u64) -> Layout2d<f32> {
    let l64 = seeded_layout_f64(n, scale as f64, seed);
    Layout2d::from_xy(
        l64.x().iter().map(|&v| v as f32).collect(),
        l64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap()
}

#[test]
fn test_sparse_pcg_workspace_basic_and_errors() {
    let mut pcg = SparsePcgWorkspace::new(3);
    // 2x2 SPD block + isolated singular row:
    // [3, 1, 0; 1, 2, 0; 0, 0, 0] x = [5, 5, 0] => x = [1, 2, x2]
    let indptr = [0u32, 2, 4, 5];
    let indices = [0u32, 1, 0, 1, 2];
    let values = [3.0, 1.0, 1.0, 2.0, 0.0];
    let b = [5.0, 5.0, 0.0];
    let mut x = [0.0, 0.0, 7.0];

    let iters = pcg
        .solve_csr(&indptr, &indices, &values, &b, &mut x, 1e-10, 50)
        .unwrap();
    assert!(iters > 0);
    assert!((x[0] - 1.0).abs() < 1e-8);
    assert!((x[1] - 2.0).abs() < 1e-8);
    assert_eq!(x[2], 7.0);

    // Zero residual exits immediately.
    assert_eq!(
        pcg.solve_csr(&indptr, &indices, &values, &b, &mut x, 1e-10, 50)
            .unwrap(),
        0
    );

    // Dimension mismatch and invalid tolerance guards.
    assert!(matches!(
        pcg.solve_csr(&[0u32, 1], &indices, &values, &b, &mut x, 1e-10, 50),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        pcg.solve_csr(&indptr, &indices, &values, &b, &mut x, -1.0, 50),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_prism_step_vs_run_bit_exact_f64_and_f32() {
    let cfg = PrismConfig::default();

    // f64 bit-exact check
    let hw64 = vec![0.6_f64; 24];
    let hh64 = vec![0.4_f64; 24];
    let mut l64_run = seeded_layout_f64(24, 1.0, 101);
    let mut l64_step = l64_run.clone();

    let ext64 = NodeExtents2d::new(&hw64, &hh64).unwrap();
    let mut s64_run = PrismOverlapRemoval::new(ext64, cfg.clone()).unwrap();
    let mut s64_step = PrismOverlapRemoval::new(ext64, cfg.clone()).unwrap();

    assert_eq!(s64_run.total_iterations(), Some(cfg.max_iterations));
    let st_run = s64_run.run(&mut l64_run).unwrap();
    assert_eq!(st_run, SolverStatus::Converged);
    assert_eq!(s64_run.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&l64_run, &hw64, &hh64).unwrap());

    assert_eq!(s64_step.status(), SolverStatus::Initialized);
    assert_eq!(
        s64_step.run_steps(&mut l64_step, 0).unwrap(),
        SolverStatus::Initialized
    );
    let _ = s64_step.run_steps(&mut l64_step, 2).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert_eq!(s64_run.status(), s64_step.status());
    assert_eq!(s64_run.iteration(), s64_step.iteration());
    assert_eq!(s64_run.last_overlap_count(), s64_step.last_overlap_count());
    for i in 0..24 {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    // f32 bit-exact check
    let radii32 = vec![0.5_f32; 20];
    let mut l32_run = seeded_layout_f32(20, 0.8, 202);
    let mut l32_step = l32_run.clone();
    let mut s32_run = PrismOverlapRemoval::from_radii(&radii32, cfg.clone()).unwrap();
    let mut s32_step = PrismOverlapRemoval::from_radii(&radii32, cfg).unwrap();

    let st32 = s32_run.run(&mut l32_run).unwrap();
    assert_eq!(st32, SolverStatus::Converged);
    assert_eq!(s32_run.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&l32_run, &radii32, &radii32).unwrap());

    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..20 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_prism_coincident_collinear_padding_and_errors() {
    // Coincident points at origin + padding.
    let cfg = PrismConfig::new().with_padding(0.2);
    let radii = vec![0.5_f64; 12];
    let mut coincident = Layout2d::zeros(12);
    let mut solver = PrismOverlapRemoval::from_radii(&radii, cfg).unwrap();
    let st = solver.run(&mut coincident).unwrap();
    assert_eq!(st, SolverStatus::Converged);
    assert_eq!(solver.last_overlap_count(), 0);
    // Effective half-extents include 0.5 * padding = 0.1 => 0.6.
    let eff = vec![0.6_f64; 12];
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&coincident, &eff, &eff).unwrap());

    // Collinear points overlapping along X.
    let mut collinear = Layout2d::from_xy(
        vec![0.0_f64, 0.2, 0.4, 0.6, 0.8],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let hw = vec![0.5_f64; 5];
    let hh = vec![0.5_f64; 5];
    let mut s_col = PrismOverlapRemoval::new(
        NodeExtents2d::new(&hw, &hh).unwrap(),
        PrismConfig::default(),
    )
    .unwrap();
    assert_eq!(s_col.run(&mut collinear).unwrap(), SolverStatus::Converged);
    assert_eq!(s_col.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&collinear, &hw, &hh).unwrap());

    // Already non-overlapping layout converges without altering coordinates.
    let clean = Layout2d::from_xy(vec![0.0_f64, 5.0], vec![0.0, 5.0]).unwrap();
    let mut clean_work = clean.clone();
    let mut s_clean = PrismOverlapRemoval::from_radii(&[0.5, 0.5], PrismConfig::default()).unwrap();
    assert_eq!(
        s_clean.run(&mut clean_work).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(clean, clean_work);

    // Invalid inputs and zero-step validation.
    let mut wrong_len = Layout2d::zeros(3);
    let mut s_v = PrismOverlapRemoval::from_radii(&[1.0, 1.0], PrismConfig::default()).unwrap();
    assert!(matches!(
        s_v.run_steps(&mut wrong_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_v.status(), SolverStatus::Initialized);
    assert!(matches!(
        NodeExtents2d::new(&[1.0], &[1.0, 2.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        PrismOverlapRemoval::from_radii(&[-0.1], PrismConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PrismOverlapRemoval::from_radii(&[1.0], PrismConfig::new().with_max_scale_factor(1.0)),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_prism_large_scale_1000_to_2000_zero_overlaps_fast() {
    for &n in &[1000usize, 1500usize, 2000usize] {
        let mut layout = seeded_layout_f64(n, (n as f64).sqrt() * 0.6, 777 + n as u64);
        let hw = vec![0.5_f64; n];
        let hh = vec![0.35_f64; n];
        let mut solver = PrismOverlapRemoval::new(
            NodeExtents2d::new(&hw, &hh).unwrap(),
            PrismConfig::new().with_max_iterations(50),
        )
        .unwrap();

        let t0 = Instant::now();
        let status = solver.run(&mut layout).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "PRISM n={n}: status={status:?} iters={} overlaps={} elapsed={elapsed_ms:.2} ms",
            solver.iteration(),
            solver.last_overlap_count()
        );

        assert_eq!(status, SolverStatus::Converged, "n={n} did not converge");
        assert_eq!(solver.last_overlap_count(), 0);
        assert!(!RectangleOverlapSweep::has_layout_overlaps(&layout, &hw, &hh).unwrap());
        let time_budget_ms = if cfg!(debug_assertions) {
            15_000.0
        } else {
            500.0
        };
        assert!(
            elapsed_ms < time_budget_ms,
            "n={n} took {elapsed_ms:.1} ms (budget {time_budget_ms:.1} ms)"
        );
    }
}
