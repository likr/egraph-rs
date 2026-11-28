//! Opaque-box E2E tests for `SgdScheduleConfig::evaluate` parity and boundary contracts.

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d};
use egraph_layout_sgd::{
    Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
};

fn triangle_graph() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        Some(&[1.0, 2.0, 1.5, 3.0, 1.0]),
        false,
        false,
    )
    .unwrap()
}

const ALL_SCHEDULE_KINDS: [SgdScheduleKind; 6] = [
    SgdScheduleKind::Exponential,
    SgdScheduleKind::Linear,
    SgdScheduleKind::Quadratic,
    SgdScheduleKind::Reciprocal,
    SgdScheduleKind::Constant,
    SgdScheduleKind::Squared,
];

#[test]
fn schedule_evaluate_matches_solver_current_eta_across_all_kinds() {
    let g = triangle_graph();
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(15)
            .with_eps(0.02)
            .with_kind(kind)
            .with_step_scale(0.85);
        let mut solver = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 42).unwrap();
        let (emin, emax) = solver.eta_bounds();
        let mut layout = Layout2d::<f64>::init(4, 9, 1.0).unwrap();

        for step_idx in 0..15 {
            assert_eq!(solver.iteration(), step_idx);
            let eval_eta = sched.evaluate(step_idx, emin, emax);
            assert_eq!(
                eval_eta.to_bits(),
                solver.current_eta().to_bits(),
                "bit-exact mismatch for {kind:?} at step {step_idx}"
            );
            solver.step(&mut layout).unwrap();
        }
        assert_eq!(sched.evaluate(15, emin, emax), 0.0);
        assert_eq!(solver.current_eta(), 0.0);
    }
}

#[test]
fn schedule_evaluate_boundary_t0_and_final_step() {
    let (emin, emax, scale, iters) = (0.05, 4.0, 1.5, 25usize);
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(iters)
            .with_kind(kind)
            .with_step_scale(scale);
        let eta0 = sched.evaluate(0, emin, emax);
        let eta_last = sched.evaluate(iters - 1, emin, emax);
        if kind == SgdScheduleKind::Constant {
            assert!((eta0 - scale).abs() < 1e-12);
            assert!((eta_last - scale).abs() < 1e-12);
        } else {
            assert!((eta0 - scale * emax).abs() < 1e-12, "{kind:?}: eta0={eta0}");
            assert!(
                (eta_last - scale * emin).abs() < 1e-11,
                "{kind:?}: eta_last={eta_last}"
            );
        }
    }
}

#[test]
fn schedule_evaluate_monotonic_decay() {
    let (emin, emax, iters) = (0.01, 5.0, 50usize);
    for kind in [
        SgdScheduleKind::Exponential,
        SgdScheduleKind::Linear,
        SgdScheduleKind::Quadratic,
        SgdScheduleKind::Reciprocal,
        SgdScheduleKind::Squared,
    ] {
        let sched = SgdScheduleConfig::new(iters).with_kind(kind);
        let mut prev = sched.evaluate(0, emin, emax);
        for t in 1..iters {
            let cur = sched.evaluate(t, emin, emax);
            assert!(
                prev >= cur,
                "{kind:?} non-monotonic at t={t}: prev={prev} < cur={cur}"
            );
            prev = cur;
        }
    }
}

#[test]
fn schedule_evaluate_step_scale_linearity() {
    let (emin, emax) = (0.02, 2.5);
    for kind in ALL_SCHEDULE_KINDS {
        let base = SgdScheduleConfig::new(20)
            .with_kind(kind)
            .with_step_scale(1.0);
        let scaled = base.clone().with_step_scale(2.5);
        for t in [0usize, 1, 7, 19, 20] {
            let expected = 2.5 * base.evaluate(t, emin, emax);
            let actual = scaled.evaluate(t, emin, emax);
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn schedule_evaluate_matches_sgd2_squared_schedule() {
    let g = triangle_graph();
    let (iters, eps, step_scale) = (12usize, 0.03, 0.75);
    let ref_sgd =
        SgdSolver::<Layout2d<f64>>::full(&g, SgdScheduleConfig::new(iters).with_eps(eps), 42)
            .unwrap();
    let (emin, emax) = ref_sgd.eta_bounds();

    let mut sgd2 = Sgd2Solver::<Layout2d<f64>>::new(
        &g,
        Sgd2Config::new()
            .with_iterations(iters)
            .with_eps(eps)
            .with_step_scale(step_scale)
            .with_seed(42),
    )
    .unwrap();
    let sched = SgdScheduleConfig::new(iters)
        .with_kind(SgdScheduleKind::Squared)
        .with_eps(eps)
        .with_step_scale(step_scale);
    let mut layout = Layout2d::<f64>::init(4, 3, 1.0).unwrap();

    for t in 0..iters {
        assert!((sched.evaluate(t, emin, emax) - sgd2.current_eta()).abs() < 1e-12);
        sgd2.step(&mut layout).unwrap();
    }
    assert_eq!(sgd2.current_eta(), 0.0);
}

#[test]
fn schedule_evaluate_zero_iterations_returns_zero() {
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(0).with_kind(kind);
        assert_eq!(sched.evaluate(0, 0.01, 1.0), 0.0);
    }
}

#[test]
fn schedule_evaluate_past_end_returns_zero_for_all_kinds() {
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(10)
            .with_kind(kind)
            .with_step_scale(3.0);
        assert_eq!(sched.evaluate(10, 0.01, 2.0), 0.0);
        assert_eq!(sched.evaluate(100, 0.01, 2.0), 0.0);
        assert_eq!(sched.evaluate(usize::MAX, 0.01, 2.0), 0.0);
    }
}

#[test]
fn schedule_evaluate_single_iteration_avoids_zero_division() {
    let (emin, emax, scale) = (0.1, 3.0, 2.0);
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(1)
            .with_kind(kind)
            .with_step_scale(scale);
        let eta = sched.evaluate(0, emin, emax);
        assert!(eta.is_finite());
        let expected = if kind == SgdScheduleKind::Constant {
            scale
        } else {
            scale * emax
        };
        assert!((eta - expected).abs() < 1e-12, "{kind:?}: got {eta}");
        assert_eq!(sched.evaluate(1, emin, emax), 0.0);
    }
}

#[test]
fn schedule_evaluate_non_positive_eta_bounds() {
    let scale = 1.75;
    for kind in [SgdScheduleKind::Linear, SgdScheduleKind::Squared] {
        let sched = SgdScheduleConfig::new(5)
            .with_kind(kind)
            .with_step_scale(scale);
        assert_eq!(sched.evaluate(0, 0.01, 0.0), 0.0);
        assert_eq!(sched.evaluate(0, 0.01, -1.0), 0.0);
    }
    for kind in [
        SgdScheduleKind::Quadratic,
        SgdScheduleKind::Exponential,
        SgdScheduleKind::Reciprocal,
    ] {
        let sched = SgdScheduleConfig::new(5)
            .with_kind(kind)
            .with_step_scale(scale);
        assert_eq!(sched.evaluate(0, 0.0, 1.0), 0.0);
        assert_eq!(sched.evaluate(0, 0.01, 0.0), 0.0);
        assert_eq!(sched.evaluate(0, -0.5, 1.0), 0.0);
    }
    let constant = SgdScheduleConfig::new(5)
        .with_kind(SgdScheduleKind::Constant)
        .with_step_scale(scale);
    assert_eq!(constant.evaluate(0, 0.0, 0.0), scale);
}

#[test]
fn schedule_evaluate_equal_eta_bounds() {
    let (eta, scale, iters) = (2.25, 0.5, 10usize);
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(iters)
            .with_kind(kind)
            .with_step_scale(scale);
        let expected = if kind == SgdScheduleKind::Constant {
            scale
        } else {
            scale * eta
        };
        for t in 0..iters {
            assert!((sched.evaluate(t, eta, eta) - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn schedule_evaluate_inverted_and_non_finite_bounds_do_not_panic() {
    let (emin_gt, emax_lt, scale, iters) = (5.0, 1.0, 1.25, 6usize);
    for kind in ALL_SCHEDULE_KINDS {
        let sched = SgdScheduleConfig::new(iters)
            .with_kind(kind)
            .with_step_scale(scale);
        for t in 0..iters {
            let v = sched.evaluate(t, emin_gt, emax_lt);
            assert!(v.is_finite() && v >= 0.0, "{kind:?} t={t} got {v}");
        }
        // Negative infinity returns 0.0 for decay schedules and scale for Constant.
        let neg_inf_val = sched.evaluate(1, f64::NEG_INFINITY, 1.0);
        if kind == SgdScheduleKind::Constant {
            assert_eq!(neg_inf_val, scale);
        } else {
            assert_eq!(neg_inf_val, 0.0);
        }
        // NaN and +Inf bounds must execute cleanly without panicking.
        for (bad_min, bad_max) in [
            (f64::NAN, 1.0),
            (0.1, f64::NAN),
            (f64::INFINITY, 1.0),
            (0.1, f64::INFINITY),
        ] {
            let _ = sched.evaluate(0, bad_min, bad_max);
            let _ = sched.evaluate(2, bad_min, bad_max);
        }
    }
}

#[test]
fn schedule_invalid_configs_rejected_on_from_pair_buffer_and_reset() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1.0).unwrap();

    for bad_sched in [
        SgdScheduleConfig::new(5).with_eps(0.0),
        SgdScheduleConfig::new(5).with_eps(-0.1),
        SgdScheduleConfig::new(5).with_step_scale(0.0),
        SgdScheduleConfig::new(5).with_max_mu(Some(0.0)),
    ] {
        assert!(matches!(
            SgdSolver::<Layout2d<f64>>::from_pair_buffer(2, buf.clone(), bad_sched, 7),
            Err(CoreError::InvalidInput(_))
        ));
    }

    for nf_sched in [
        SgdScheduleConfig::new(5).with_eps(f64::NAN),
        SgdScheduleConfig::new(5).with_step_scale(f64::INFINITY),
        SgdScheduleConfig::new(5).with_max_mu(Some(f64::NAN)),
    ] {
        assert!(matches!(
            SgdSolver::<Layout2d<f64>>::from_pair_buffer(2, buf.clone(), nf_sched, 7),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    let valid_sched = SgdScheduleConfig::new(5).with_eps(0.05);
    let mut solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(2, buf, valid_sched.clone(), 7).unwrap();
    assert!(matches!(
        solver.reset_schedule(SgdScheduleConfig::new(5).with_eps(0.0), 11),
        Err(CoreError::InvalidInput(_))
    ));
    assert_eq!(solver.schedule_config(), &valid_sched);
}
