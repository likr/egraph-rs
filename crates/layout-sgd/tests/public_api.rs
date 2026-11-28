use egraph_core::{CsrGraph, IterativeSolver, Layout2d, Layout3d, SolverStatus};
use egraph_layout_sgd::{
    BarrierSymmetry, OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig,
    Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdPairKind, SgdPairRecord, SgdScheduleConfig,
    SgdScheduleKind, SgdSolver, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
    build_ortmann_sparse_problem,
};

#[test]
fn root_reexports_and_pair_buffer_contract_compile_and_run() {
    let _ = BarrierSymmetry::SymMax;
    let _ = OrtmannPivotStrategy::KMeansSp;
    let _ = OrtmannSparseStressConfig::default().with_num_pivots(16);
    let _ = Sgd2Config::default().with_iterations(10);
    let _ = SparseBarrierConfig::default().with_num_pivots(8);
    let _ = StreamingNegativeConfig::default().with_negatives_per_node(4);
    let _: Option<SparseBarrierStats> = None;

    let mut buffer = SgdPairBuffer::<f64>::with_capacity(3);
    assert!(buffer.is_empty());
    buffer.push_spring(0, 1, 1.0, 1.0).unwrap();
    buffer.push_spring(1, 2, 1.0, 4.0).unwrap();
    buffer.push_hinge(0, 2, 1.5, 0.5).unwrap();
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.num_springs(), 2);
    assert_eq!(buffer.num_hinges(), 1);
    buffer.condition_weights(4.0).unwrap();

    let sched = SgdScheduleConfig::new(10)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);
    let mut solver = SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buffer, sched, 42).unwrap();
    assert_eq!(solver.total_iterations(), Some(10));
    let mut layout = Layout2d::from_xy(vec![0.0, 0.2, 0.4], vec![0.0, 0.1, 0.0]).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert!(layout.is_all_finite());

    let tri =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (0, 2)], None, false, false).unwrap();
    let mut sgd_3d = SgdSolver::<Layout3d<f64>>::full(&tri, SgdScheduleConfig::new(8), 7).unwrap();
    let mut l3d = Layout3d::<f64>::init(3, 11, 1.0).unwrap();
    assert_eq!(sgd_3d.run(&mut l3d).unwrap(), SolverStatus::Completed);
    assert!(l3d.is_all_finite());

    let mut sgd2 =
        Sgd2Solver::<Layout2d<f64>>::new(&tri, Sgd2Config::new().with_iterations(8)).unwrap();
    let mut l2d = Layout2d::<f64>::init(3, 13, 1.0).unwrap();
    assert_eq!(sgd2.run(&mut l2d).unwrap(), SolverStatus::Completed);
    assert!(l2d.is_all_finite());
}

#[test]
fn pair_buffer_iter_schedule_evaluate_and_solver_accessors_smoke() {
    let mut buffer = SgdPairBuffer::<f64>::with_capacity(3);
    buffer.push_spring(0, 1, 1.0, 2.0).unwrap();
    buffer.push_hinge(1, 2, 1.5, 0.5).unwrap();
    buffer.push_spring(0, 2, 2.0, 0.25).unwrap();
    let recs: Vec<SgdPairRecord<f64>> = buffer.iter().collect();
    assert_eq!(recs.len(), 3);
    assert_eq!(recs[0].kind, SgdPairKind::Spring);
    assert_eq!(recs[1].kind, SgdPairKind::Hinge);
    assert_eq!(
        (recs[0].d_uv, recs[0].d_vu, recs[0].w_uv, recs[0].w_vu),
        (1.0, 1.0, 2.0, 2.0)
    );
    match recs[0].kind {
        SgdPairKind::Spring | SgdPairKind::Hinge => {}
        _ => unreachable!(),
    }

    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (0, 2)],
        Some(&[1.0, 2.0, 1.5, 3.0]),
        false,
        false,
    )
    .unwrap();
    for kind in [
        SgdScheduleKind::Exponential,
        SgdScheduleKind::Linear,
        SgdScheduleKind::Quadratic,
        SgdScheduleKind::Reciprocal,
        SgdScheduleKind::Constant,
        SgdScheduleKind::Squared,
    ] {
        let sched = SgdScheduleConfig::new(6)
            .with_eps(0.05)
            .with_kind(kind)
            .with_step_scale(0.9);
        let mut solver = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 42).unwrap();
        assert_eq!(solver.schedule_config(), &sched);
        let (emin, emax) = solver.eta_bounds();
        assert!(emin > 0.0 && emax >= emin);
        let mut layout = Layout2d::<f64>::init(4, 9, 1.0).unwrap();
        for step in 0..6 {
            assert_eq!(
                sched.evaluate(step, emin, emax).to_bits(),
                solver.current_eta().to_bits()
            );
            solver.step(&mut layout).unwrap();
        }
        assert_eq!(sched.evaluate(6, emin, emax), 0.0);
        assert_eq!(solver.current_eta(), 0.0);
    }

    let empty_sched = SgdScheduleConfig::new(0).with_kind(SgdScheduleKind::Exponential);
    assert_eq!(empty_sched.evaluate(0, 0.1, 1.0), 0.0);
    let const_sched = SgdScheduleConfig::new(4)
        .with_kind(SgdScheduleKind::Constant)
        .with_step_scale(2.25);
    assert_eq!(const_sched.evaluate(0, 0.0, 0.0), 2.25);
    let exp_sched = SgdScheduleConfig::new(4).with_kind(SgdScheduleKind::Exponential);
    assert_eq!(exp_sched.evaluate(0, 2.0, 1.0), 0.0);
    assert_eq!(exp_sched.evaluate(0, f64::NAN, 1.0), 0.0);

    let mut buf_check = SgdPairBuffer::<f32>::new();
    buf_check.push_spring(0, 1, 1.0, 1.0).unwrap();
    buf_check.push_hinge(1, 2, 1.5, 0.5).unwrap();
    let recs32: Vec<SgdPairRecord<f32>> = buf_check.iter().collect();
    let solver32 =
        SgdSolver::<Layout2d<f32>>::from_pair_buffer(3, buf_check, SgdScheduleConfig::new(4), 11)
            .unwrap();
    assert_eq!(solver32.iter_pairs().collect::<Vec<_>>(), recs32);

    let problem: OrtmannSparseProblem<f64> = build_ortmann_sparse_problem(
        &g,
        OrtmannSparseStressConfig::new()
            .with_num_pivots(2)
            .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
            .with_seed(7),
    )
    .unwrap();
    assert_eq!(problem.num_nodes, 4);
    assert!(!problem.two_sided.is_empty());
}
