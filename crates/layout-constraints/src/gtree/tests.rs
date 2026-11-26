use super::*;
use egraph_core::{CoreError, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus};
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
fn test_gtree_validation_and_degenerate_inputs() {
    assert!(matches!(
        GTreeConfig {
            padding: -0.1,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        GTreeConfig {
            padding: f64::NAN,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        GTreeConfig {
            compact_t_max: Some(1.0),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        GTreeConfig {
            compact_t_max: Some(f64::INFINITY),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        GTreeOverlapRemoval::new(vec![1.0], vec![1.0, 2.0], GTreeConfig::default()),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        GTreeOverlapRemoval::new(vec![-1.0], vec![1.0], GTreeConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        GTreeOverlapRemoval::new(vec![f64::NAN], vec![1.0], GTreeConfig::default()),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Empty (N=0) and singleton (N=1).
    for n in [0usize, 1] {
        let mut solver =
            GTreeOverlapRemoval::<f64>::from_radii(vec![1.0; n], GTreeConfig::default()).unwrap();
        let mut layout = Layout2d::zeros(n);
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
        assert_eq!(solver.last_overlap_count(), 0);
    }

    // Layout mismatch and non-finite coordinates.
    let mut solver =
        GTreeOverlapRemoval::<f64>::from_radii(vec![1.0; 2], GTreeConfig::default()).unwrap();
    let mut wrong = Layout2d::zeros(3);
    assert!(matches!(
        solver.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        solver.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_gtree_zero_overlaps_f64_f32_and_step_eq_run() {
    let n = 48usize;
    let mut rng = DeterministicRng::from_seed_u64(12345);
    let mut xs64 = Vec::with_capacity(n);
    let mut ys64 = Vec::with_capacity(n);
    let mut hw64 = Vec::with_capacity(n);
    let mut hh64 = Vec::with_capacity(n);
    for i in 0..n {
        // Clustered overlapping layout with varied box aspect ratios.
        let cluster_x = if i % 2 == 0 { 0.0 } else { 20.0 };
        xs64.push(cluster_x + (rng.next_f64() - 0.5) * 2.0);
        ys64.push((rng.next_f64() - 0.5) * 2.0);
        hw64.push(0.8 + (i % 5) as f64 * 0.4);
        hh64.push(0.6 + ((i + 2) % 4) as f64 * 0.35);
    }

    let cfg = GTreeConfig {
        max_iterations: 20,
        padding: 0.2,
        compact_t_max: None,
        seed: 42,
    };
    let pad_half = 0.5 * cfg.padding;
    let eff_hw64: Vec<f64> = hw64.iter().map(|&w| w + pad_half).collect();
    let eff_hh64: Vec<f64> = hh64.iter().map(|&h| h + pad_half).collect();

    let mut l64_run = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
    let mut l64_step = l64_run.clone();
    let mut s64_run = GTreeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();
    let mut s64_step = GTreeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();

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

    // Compact mode & f32 precision.
    let cfg_compact = GTreeConfig {
        max_iterations: 10,
        padding: 0.1,
        compact_t_max: Some(1.5),
        seed: 77,
    };
    let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32 + 500.0).collect();
    let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32 - 500.0).collect();
    let hw32: Vec<f32> = hw64.iter().map(|&v| v as f32).collect();
    let hh32: Vec<f32> = hh64.iter().map(|&v| v as f32).collect();
    let eff_hw32: Vec<f32> = hw32
        .iter()
        .map(|&v| (v as f64 + 0.5 * cfg_compact.padding) as f32)
        .collect();
    let eff_hh32: Vec<f32> = hh32
        .iter()
        .map(|&v| (v as f64 + 0.5 * cfg_compact.padding) as f32)
        .collect();

    let mut l32_run = Layout2d::from_xy(xs32, ys32).unwrap();
    let mut l32_step = l32_run.clone();
    let mut s32_run =
        GTreeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), cfg_compact).unwrap();
    let mut s32_step = GTreeOverlapRemoval::<f32>::new(hw32, hh32, cfg_compact).unwrap();

    assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Converged);
    assert_eq!(s32_run.last_overlap_count(), 0);
    assert_no_overlaps(&l32_run, &eff_hw32, &eff_hh32);

    s32_step.run_steps(&mut l32_step, 2).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..n {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_gtree_coincident_collinear_and_zero_size_edge_cases() {
    // 1. All coincident points at origin.
    let n = 16usize;
    let hw = vec![1.0; n];
    let hh = vec![0.5; n];
    let mut coincident = Layout2d::zeros(n);
    let mut s_coin =
        GTreeOverlapRemoval::new(hw.clone(), hh.clone(), GTreeConfig::default()).unwrap();
    assert_eq!(
        s_coin.run(&mut coincident).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(s_coin.last_overlap_count(), 0);
    assert_no_overlaps(&coincident, &hw, &hh);

    // 2. Strictly collinear layout along y = 0.
    let mut collinear =
        Layout2d::from_xy((0..n).map(|i| (i as f64) * 0.25).collect(), vec![0.0; n]).unwrap();
    let mut s_coll =
        GTreeOverlapRemoval::new(hw.clone(), hh.clone(), GTreeConfig::default()).unwrap();
    assert_eq!(s_coll.run(&mut collinear).unwrap(), SolverStatus::Converged);
    assert_eq!(s_coll.last_overlap_count(), 0);
    assert_no_overlaps(&collinear, &hw, &hh);

    // 3. Zero-size nodes (no interior overlap possible).
    let zero_r = vec![0.0; n];
    let mut zero_layout = Layout2d::zeros(n);
    let mut s_zero =
        GTreeOverlapRemoval::from_radii(zero_r.clone(), GTreeConfig::default()).unwrap();
    assert_eq!(
        s_zero.run(&mut zero_layout).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(s_zero.last_overlap_count(), 0);
    assert_no_overlaps(&zero_layout, &zero_r, &zero_r);

    // 4. Two widely separated overlapping clusters (>12 nodes each, exercises bridging).
    let m = 32usize;
    let mut c_xs = Vec::with_capacity(m);
    let mut c_ys = Vec::with_capacity(m);
    for i in 0..m {
        let base_x = if i < 16 { -1000.0 } else { 1000.0 };
        c_xs.push(base_x + ((i % 4) as f64) * 0.2);
        c_ys.push(((i / 4 % 4) as f64) * 0.2);
    }
    let c_hw = vec![1.0; m];
    let c_hh = vec![1.0; m];
    let mut c_layout = Layout2d::from_xy(c_xs, c_ys).unwrap();
    let mut s_clust =
        GTreeOverlapRemoval::new(c_hw.clone(), c_hh.clone(), GTreeConfig::default()).unwrap();
    assert_eq!(s_clust.run(&mut c_layout).unwrap(), SolverStatus::Converged);
    assert_eq!(s_clust.last_overlap_count(), 0);
    assert_no_overlaps(&c_layout, &c_hw, &c_hh);
}
