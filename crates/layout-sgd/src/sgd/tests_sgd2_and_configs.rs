use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};

use super::super::sgd::{
    DrGraphConfig, GraphCneConfig, LargeVisConfig, OrtmannSparseStressConfig, Sgd2Config,
    SgdScheduleConfig, SgdScheduleKind, SgdSolver,
};
use super::test_support::{circle_layout, euclidean_apsp_stress, grid_graph};

#[test]
fn test_sgd2_and_squared_schedule_convergence_and_bit_exact() {
    let g = grid_graph(3, 4);

    // Verify SgdScheduleKind::Squared decay curve & eta_min clamping.
    let sq_sched = SgdScheduleConfig {
        t_max: 11,
        eps: 0.05,
        kind: SgdScheduleKind::Squared,
        step_scale: 1.0,
    };
    let mut sq_solver = SgdSolver::<f64>::full(&g, sq_sched, 42).unwrap();
    let mut etas = Vec::new();
    let mut sq_layout = circle_layout::<f64>(g.num_nodes(), 1.0);
    for _ in 0..11 {
        etas.push(sq_solver.current_eta());
        sq_solver.step(&mut sq_layout).unwrap();
    }
    assert_eq!(sq_solver.current_eta(), 0.0);
    assert!((etas[0] - sq_solver.eta_max).abs() < 1e-12);
    assert!((etas[10] - sq_solver.eta_min).abs() < 1e-12);
    for win in etas.windows(2) {
        assert!(win[0] >= win[1]);
    }

    // Test Sgd2 with small batch_size=7 so multiple mini-batches + tail flush are exercised.
    let cfg = Sgd2Config {
        t_max: 40,
        eps: 0.01,
        step_scale: 1.0,
        batch_size: 7,
        damping: 1e-4,
        seed: 42,
    };

    // Bit-exact step() == run() for f64 and f32.
    let mut s64_run = SgdSolver::<f64>::sgd2(&g, cfg).unwrap();
    let mut s64_step = SgdSolver::<f64>::sgd2(&g, cfg).unwrap();
    let mut l64_run = circle_layout::<f64>(g.num_nodes(), 0.25);
    let mut l64_step = l64_run.clone();
    let stress_before = euclidean_apsp_stress(&g, &l64_run);
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 11).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    for i in 0..g.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }
    let stress_after = euclidean_apsp_stress(&g, &l64_run);
    assert!(
        stress_after < stress_before * 0.2,
        "Sgd2 did not reduce stress: {stress_before} -> {stress_after}"
    );

    let mut s32_run = SgdSolver::<f32>::sgd2(&g, cfg).unwrap();
    let mut s32_step = SgdSolver::<f32>::sgd2(&g, cfg).unwrap();
    let mut l32_run = circle_layout::<f32>(g.num_nodes(), 0.25);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 13).unwrap();
    s32_step.run(&mut l32_step).unwrap();
    for i in 0..g.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // Large-n (> 256) sparse path uses bounded O(N) pairs instead of O(N^2) APSP.
    let big_grid = grid_graph(18, 18); // 324 nodes > 256
    let mut big_s2 = SgdSolver::<f64>::sgd2(&big_grid, Sgd2Config::default()).unwrap();
    assert!(big_s2.num_springs() < 324 * 20);
    let mut big_l = circle_layout::<f64>(big_grid.num_nodes(), 1.0);
    big_s2.run(&mut big_l).unwrap();
    assert!(big_l.is_all_finite());

    // Weighted triangle shortest-path check (0-1-2 = 3.0 vs 0-2 = 10.0).
    let wg = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.0, 2.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let mut w_s2 = SgdSolver::<f64>::sgd2(&wg, Sgd2Config::default()).unwrap();
    let mut w_l = circle_layout::<f64>(3, 2.0);
    w_s2.run(&mut w_l).unwrap();
    let d02 = (w_l.x()[0] - w_l.x()[2]).hypot(w_l.y()[0] - w_l.y()[2]);
    assert!((d02 - 3.0).abs() < 0.25, "expected dist ~3.0, got {d02}");

    // Validation & edge-case guards.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        SgdSolver::<f64>::sgd2(&empty, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SgdSolver::<f64>::sgd2(&single, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    for bad in [
        Sgd2Config { eps: 0.0, ..cfg },
        Sgd2Config {
            eps: f64::NAN,
            ..cfg
        },
        Sgd2Config {
            step_scale: -1.0,
            ..cfg
        },
        Sgd2Config {
            batch_size: 0,
            ..cfg
        },
        Sgd2Config {
            damping: 0.0,
            ..cfg
        },
    ] {
        assert!(SgdSolver::<f64>::sgd2(&g, bad).is_err());
    }
}

#[test]
fn test_sgd_variant_config_validations() {
    assert!(OrtmannSparseStressConfig::default().validate().is_ok());
    assert!(
        OrtmannSparseStressConfig {
            num_pivots: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            OrtmannSparseStressConfig {
                eps: bad,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            OrtmannSparseStressConfig {
                eta_max: Some(bad),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            OrtmannSparseStressConfig {
                eta_min: Some(bad),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    assert!(
        OrtmannSparseStressConfig {
            eta_min: Some(2.0),
            eta_max: Some(1.0),
            ..Default::default()
        }
        .validate()
        .is_err()
    );

    assert!(LargeVisConfig::default().validate().is_ok());
    assert!(
        LargeVisConfig {
            gamma: -1.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LargeVisConfig {
            gamma: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LargeVisConfig {
            initial_lr: 0.0,
            min_lr: 0.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LargeVisConfig {
            min_lr: -1e-4,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LargeVisConfig {
            initial_lr: 0.5,
            min_lr: 0.6,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LargeVisConfig {
            degree_exponent: -0.1,
            ..Default::default()
        }
        .validate()
        .is_err()
    );

    assert!(DrGraphConfig::default().validate().is_ok());
    assert!(
        DrGraphConfig {
            k_order: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DrGraphConfig {
            b: 0.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DrGraphConfig {
            b: f64::INFINITY,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DrGraphConfig {
            gamma_coarse: -0.01,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DrGraphConfig {
            gamma_fine: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    for bad_ratio in [0.0, -0.5, 1.01, f64::NAN] {
        assert!(
            DrGraphConfig {
                coarsen_stop_ratio: bad_ratio,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    assert!(
        DrGraphConfig {
            initial_lr: 0.0,
            min_lr: 0.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DrGraphConfig {
            initial_lr: 0.5,
            min_lr: 0.8,
            ..Default::default()
        }
        .validate()
        .is_err()
    );

    assert!(GraphCneConfig::default().validate().is_ok());
    assert!(
        GraphCneConfig {
            initial_lr: -1.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        GraphCneConfig {
            min_lr: -0.1,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        GraphCneConfig {
            initial_lr: 0.2,
            min_lr: 0.5,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        GraphCneConfig {
            repulsion_weight: -0.1,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}
