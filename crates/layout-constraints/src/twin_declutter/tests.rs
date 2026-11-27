use super::*;
use egraph_core::{CoreError, IterativeSolver, Layout2d, SolverStatus};
use egraph_spatial_trees::StaticKdTree;

#[test]
fn test_twin_declutter_config_defaults_and_validation() {
    let cfg = TwinDeclutterConfig::default();
    assert!((cfg.eps_rel - 1e-6).abs() < 1e-15);
    assert!((cfg.r_frac - 0.35).abs() < 1e-15);
    assert!((cfg.r_cap_rel - 3e-3).abs() < 1e-15);
    assert!(
        (cfg.min_pair_rel - 5e-7).abs() < 1e-15,
        "min_pair_rel must be 5e-7"
    );
    assert_eq!(cfg.iterations, 4);
    cfg.validate().unwrap();

    assert!(matches!(
        TwinDeclutterConfig {
            min_pair_rel: -1.0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        TwinDeclutterConfig {
            eps_rel: f64::NAN,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_twin_declutter_empty_singleton_and_early_convergence() {
    let cfg = TwinDeclutterConfig::default();

    let mut empty = Layout2d::<f64>::zeros(0);
    assert_eq!(declutter_twins_2d(&mut empty, &cfg).unwrap(), 0);

    let mut single = Layout2d::from_xy(vec![3.5_f64], vec![-2.0_f64]).unwrap();
    let mut solver = TwinDeclutterer::new(cfg).unwrap();
    assert_eq!(solver.run(&mut single).unwrap(), SolverStatus::Converged);
    assert_eq!(solver.iteration(), 0);

    // Well-separated points: early convergence with 0 passes and zero displacement.
    let mut clean = Layout2d::from_xy(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, -1.0]).unwrap();
    let clean_orig = clean.clone();
    assert_eq!(declutter_twins_2d(&mut clean, &cfg).unwrap(), 0);
    assert_eq!(clean, clean_orig);
}

#[test]
fn test_twin_declutter_degenerate_zero_diag_f64_and_f32() {
    let cfg = TwinDeclutterConfig::default();

    let mut collapsed64 = Layout2d::from_xy(vec![5.0_f64; 6], vec![-3.0_f64; 6]).unwrap();
    let mut s64 = TwinDeclutterer::new(cfg).unwrap();
    assert_eq!(s64.run(&mut collapsed64).unwrap(), SolverStatus::Converged);
    assert_eq!(s64.iteration(), 1);
    collapsed64.validate_finite().unwrap();

    // Verify all nodes are distinct and bounded within unit radius around (5, -3).
    for i in 0..6 {
        let dx = collapsed64.x()[i] - 5.0;
        let dy = collapsed64.y()[i] + 3.0;
        assert!(dx.hypot(dy) <= 1.0 + 1e-12);
        for j in (i + 1)..6 {
            let dist = (collapsed64.x()[i] - collapsed64.x()[j])
                .hypot(collapsed64.y()[i] - collapsed64.y()[j]);
            assert!(dist > 0.1);
        }
    }

    let mut collapsed32 = Layout2d::from_xy(vec![-0.0_f32; 4], vec![0.0_f32; 4]).unwrap();
    assert_eq!(declutter_twins_2d(&mut collapsed32, &cfg).unwrap(), 1);
    collapsed32.validate_finite().unwrap();
}

#[test]
fn test_twin_declutter_exact_and_near_twins_multi_cluster_f64_and_f32() {
    let cfg = TwinDeclutterConfig::default();

    // Two distinct twin clusters separated by distance 10.0, plus one singleton anchor.
    // Cluster A: 4 exact twins at (0.0, 0.0) and 1 near twin at (1e-7, 0.0).
    // Cluster B: 3 exact twins at (10.0, 0.0).
    // Singleton C: (0.0, 10.0).
    let xs = vec![0.0_f64, 0.0, 0.0, 0.0, 1e-7, 10.0, 10.0, 10.0, 0.0];
    let ys = vec![0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
    let mut layout = Layout2d::from_xy(xs, ys).unwrap();
    let mut solver = TwinDeclutterer::new(cfg).unwrap();
    let status = solver.run(&mut layout).unwrap();
    assert_eq!(status, SolverStatus::Converged);
    assert!(solver.iteration() >= 1);
    assert!(solver.iteration() <= cfg.iterations);

    // Every pair must now exceed near-collision threshold `1e-7 * diag`.
    let min_x = layout.x().iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = layout.x().iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = layout.y().iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = layout.y().iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let diag = (max_x - min_x).hypot(max_y - min_y);
    let tree = StaticKdTree::<2, f64>::from_layout(&layout).unwrap();
    for i in 0..layout.len() {
        let (_, d2) = tree.nearest_excluding_self(i).unwrap().unwrap();
        assert!(
            d2.sqrt() > 1e-7 * diag,
            "node {i} still has near-collision dist={} threshold={}",
            d2.sqrt(),
            1e-7 * diag
        );
    }

    // f32 precision check at non-zero coordinate offset.
    let xs32 = vec![100.0_f32, 100.0, 100.0, 110.0, 110.0, 100.0];
    let ys32 = vec![-50.0_f32, -50.0, -50.0, -50.0, -50.0, -40.0];
    let mut layout32 = Layout2d::from_xy(xs32, ys32).unwrap();
    let passes = declutter_twins_2d(&mut layout32, &cfg).unwrap();
    assert!(passes >= 1);
    layout32.validate_finite().unwrap();

    let fx: Vec<f64> = layout32.x().iter().map(|&v| v as f64).collect();
    let fy: Vec<f64> = layout32.y().iter().map(|&v| v as f64).collect();
    let diag32 = (fx.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - fx.iter().copied().fold(f64::INFINITY, f64::min))
    .hypot(
        fy.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - fy.iter().copied().fold(f64::INFINITY, f64::min),
    );
    let pts32: Vec<[f64; 2]> = fx.iter().zip(fy.iter()).map(|(&x, &y)| [x, y]).collect();
    let tree32 = StaticKdTree::<2, f64>::build(&pts32).unwrap();
    for i in 0..layout32.len() {
        let (_, d2) = tree32.nearest_excluding_self(i).unwrap().unwrap();
        assert!(d2.sqrt() > 1e-7 * diag32);
    }
}

#[test]
fn test_twin_declutter_step_matches_run_and_respects_iterations() {
    let cfg = TwinDeclutterConfig {
        iterations: 2,
        ..Default::default()
    };
    let mut l_run = Layout2d::from_xy(vec![0.0_f64, 0.0, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    let mut l_step = l_run.clone();

    let mut s_run = TwinDeclutterer::new(cfg).unwrap();
    let mut s_step = TwinDeclutterer::new(cfg).unwrap();

    let st_run = s_run.run(&mut l_run).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(st_run, s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(l_run, l_step);
}

#[test]
fn test_twin_declutter_large_finite_coordinates_preserve_finiteness() {
    let cfg = TwinDeclutterConfig::default();

    // Coincident nodes at `1e308`: degenerate fallback must divide before summing.
    let mut coincident = Layout2d::from_xy(vec![1e308_f64; 4], vec![1e308_f64; 4]).unwrap();
    let mut solver_c = TwinDeclutterer::new(cfg).unwrap();
    assert_eq!(
        solver_c.run(&mut coincident).unwrap(),
        SolverStatus::Converged
    );
    assert!(coincident.is_all_finite());

    // Multi-cluster twins spanning `[-1e308, +1e308]`: bbox/diag and centroid averaging stay finite.
    let mut multi = Layout2d::from_xy(
        vec![-1e308_f64, -1e308_f64, 1e308_f64, 1e308_f64, 0.0],
        vec![-1e308_f64, -1e308_f64, 1e308_f64, 1e308_f64, 0.0],
    )
    .unwrap();
    let mut solver_m = TwinDeclutterer::new(cfg).unwrap();
    let status_m = solver_m.run(&mut multi).unwrap();
    assert!(matches!(
        status_m,
        SolverStatus::Converged | SolverStatus::Completed
    ));
    assert!(multi.is_all_finite());

    // f32 extreme coincident and multi-cluster coordinates stay finite as well.
    let mut coincident32 = Layout2d::from_xy(vec![3e38_f32; 3], vec![-3e38_f32; 3]).unwrap();
    assert_eq!(declutter_twins_2d(&mut coincident32, &cfg).unwrap(), 1);
    assert!(coincident32.is_all_finite());

    let mut multi32 = Layout2d::from_xy(
        vec![-3e38_f32, -3e38_f32, 3e38_f32, 3e38_f32],
        vec![0.0_f32, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let _ = declutter_twins_2d(&mut multi32, &cfg).unwrap();
    assert!(multi32.is_all_finite());
}
