use egraph_core::{CsrGraph, IterativeSolver, Layout2d, Layout3d, SolverStatus};
use egraph_layout_sgd::{
    BarrierSymmetry, OrtmannPivotStrategy, OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver,
    SgdPairBuffer, SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig,
    SparseBarrierStats, StreamingNegativeConfig,
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
