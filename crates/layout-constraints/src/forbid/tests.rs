use super::*;
use egraph_core::{
    CoreError, DeterministicRng, IterativeSolver, Layout2d, NodeExtents2d, Scalar, SolverStatus,
};
use egraph_spatial_trees::RectangleOverlapSweep;

fn assert_no_overlaps<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S]) {
    let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, hw, hh).unwrap();
    assert!(
        overlaps.is_empty(),
        "expected 0 overlaps, found {}: {:?}",
        overlaps.len(),
        &overlaps[..overlaps.len().min(5)]
    );
}

#[test]
fn test_forbid_validation_and_degenerate_inputs() {
    assert_eq!(ForbidVariant::default(), ForbidVariant::SparseForbid);
    assert_eq!(ForbidConfig::default().variant, ForbidVariant::SparseForbid);
    assert!(matches!(
        ForbidConfig::new().with_scale_step(0.0).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ForbidConfig::new().with_scale_step(f64::NAN).validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ForbidConfig::new().with_padding(-1.0).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ForbidConfig::new()
            .with_overlap_weight_k(f64::INFINITY)
            .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        NodeExtents2d::new(&[1.0_f64], &[1.0, 2.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        NodeExtents2d::new(&[-0.5_f64], &[1.0]),
        Err(CoreError::InvalidInput(_))
    ));

    for n in [0usize, 1] {
        let radii = vec![1.0; n];
        let mut solver =
            ForbidOverlapRemoval::<f64>::from_radii(&radii, ForbidConfig::default()).unwrap();
        let mut layout = Layout2d::zeros(n);
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
        assert_eq!(solver.last_overlap_count(), 0);
        assert_eq!(solver.total_iterations(), Some(20));
    }

    let mut solver =
        ForbidOverlapRemoval::<f64>::from_radii(&[1.0, 1.0], ForbidConfig::default()).unwrap();
    let mut wrong = Layout2d::zeros(3);
    assert!(matches!(
        solver.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        solver.run_steps(&mut wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);
    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        solver.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);
}

#[test]
fn test_forbid_all_variants_zero_overlaps_f64_f32_and_step_eq_run() {
    let n = 30usize;
    let mut rng = DeterministicRng::from_seed_u64(9876);
    let mut xs64 = Vec::with_capacity(n);
    let mut ys64 = Vec::with_capacity(n);
    let mut hw64 = Vec::with_capacity(n);
    let mut hh64 = Vec::with_capacity(n);
    for i in 0..n {
        xs64.push((rng.next_f64() - 0.5) * 4.0);
        ys64.push((rng.next_f64() - 0.5) * 4.0);
        hw64.push(0.75 + (i % 4) as f64 * 0.35);
        hh64.push(0.65 + ((i + 1) % 3) as f64 * 0.3);
    }

    for variant in [
        ForbidVariant::Forbid,
        ForbidVariant::ForbidPrime,
        ForbidVariant::SparseForbid,
    ] {
        let cfg = ForbidConfig::new()
            .with_variant(variant)
            .with_max_scale_iters(20)
            .with_sgd_iterations(30)
            .with_sparse_k(10)
            .with_overlap_weight_k(1.0)
            .with_scale_step(0.1)
            .with_padding(0.2)
            .with_seed(42);
        let pad_half = 0.5 * cfg.padding;
        let eff_hw64: Vec<f64> = hw64.iter().map(|&w| w + pad_half).collect();
        let eff_hh64: Vec<f64> = hh64.iter().map(|&h| h + pad_half).collect();

        let mut l64_run = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
        let mut l64_step = l64_run.clone();
        let ext64 = NodeExtents2d::new(&hw64, &hh64).unwrap();
        let mut s64_run = ForbidOverlapRemoval::<f64>::new(ext64, cfg.clone()).unwrap();
        let mut s64_step = ForbidOverlapRemoval::<f64>::new(ext64, cfg.clone()).unwrap();

        assert_eq!(s64_run.run(&mut l64_run).unwrap(), SolverStatus::Converged);
        assert_eq!(s64_run.last_overlap_count(), 0);
        assert_no_overlaps(&l64_run, &eff_hw64, &eff_hh64);

        s64_step.run_steps(&mut l64_step, 1).unwrap();
        while !s64_step.status().is_finished() {
            s64_step.step(&mut l64_step).unwrap();
        }
        assert_eq!(s64_step.status(), SolverStatus::Converged);
        for i in 0..n {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        // Verify f32 precision at offset coordinates.
        let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32 + 250.0).collect();
        let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32 - 250.0).collect();
        let hw32: Vec<f32> = hw64.iter().map(|&v| v as f32).collect();
        let hh32: Vec<f32> = hh64.iter().map(|&v| v as f32).collect();
        let eff_hw32: Vec<f32> = hw32
            .iter()
            .map(|&v| (v as f64 + 0.5 * cfg.padding) as f32)
            .collect();
        let eff_hh32: Vec<f32> = hh32
            .iter()
            .map(|&v| (v as f64 + 0.5 * cfg.padding) as f32)
            .collect();

        let mut l32 = Layout2d::from_xy(xs32, ys32).unwrap();
        let ext32 = NodeExtents2d::new(&hw32, &hh32).unwrap();
        let mut s32 = ForbidOverlapRemoval::<f32>::new(ext32, cfg).unwrap();
        assert_eq!(s32.run(&mut l32).unwrap(), SolverStatus::Converged);
        assert_eq!(s32.last_overlap_count(), 0);
        assert_no_overlaps(&l32, &eff_hw32, &eff_hh32);
    }
}

#[test]
fn test_forbid_coincident_collinear_and_low_iter_fallback() {
    let n = 14usize;
    let hw = vec![1.0; n];
    let hh = vec![0.5; n];
    let ext = NodeExtents2d::new(&hw, &hh).unwrap();

    // 1. All coincident points at origin.
    let mut coincident = Layout2d::zeros(n);
    let mut s_coin = ForbidOverlapRemoval::new(ext, ForbidConfig::default()).unwrap();
    assert_eq!(
        s_coin.run(&mut coincident).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(s_coin.last_overlap_count(), 0);
    assert_no_overlaps(&coincident, &hw, &hh);

    // 2. Strictly collinear layout.
    let mut collinear =
        Layout2d::from_xy((0..n).map(|i| (i as f64) * 0.3).collect(), vec![0.0; n]).unwrap();
    let mut s_coll = ForbidOverlapRemoval::new(ext, ForbidConfig::default()).unwrap();
    assert_eq!(s_coll.run(&mut collinear).unwrap(), SolverStatus::Converged);
    assert_eq!(s_coll.last_overlap_count(), 0);
    assert_no_overlaps(&collinear, &hw, &hh);

    // 3. Tight max_scale_iters = 1 fallback guarantees 0 overlaps.
    let mut tight = Layout2d::zeros(n);
    let cfg_tight = ForbidConfig::new()
        .with_max_scale_iters(1)
        .with_sgd_iterations(5);
    let mut s_tight = ForbidOverlapRemoval::new(ext, cfg_tight).unwrap();
    assert_eq!(s_tight.run(&mut tight).unwrap(), SolverStatus::Converged);
    assert_eq!(s_tight.last_overlap_count(), 0);
    assert_no_overlaps(&tight, &hw, &hh);
}
