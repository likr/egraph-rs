use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_sgd::sgd::{PackedSymPair, SgdScheduleConfig, SgdScheduleKind, SgdSolver};
use egraph_layout_sgd::{
    BarrierSymmetry, CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats,
    CoarseDistanceBackend, CoarseEdgeLengthMode, DrGraphConfig, GraphCneConfig, GraphCneObjective,
    LargeVisConfig, OrtmannPivotStrategy, OrtmannSparseStressConfig, PacmapGraphConfig, Sgd2Config,
    SnapTFdpConfig, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
    coarse_backbone_init, crate_version,
};

#[test]
fn root_reexports_and_refiner_packed_contract_compile_and_run() {
    assert!(!crate_version().is_empty());

    // Touch all root-exported config/enum/result types to ensure external crate visibility.
    let _ = BarrierSymmetry::SymMax;
    let _ = CoarseDistanceBackend::Rhgh;
    let _ = CoarseEdgeLengthMode::ConductanceInverse;
    let _ = GraphCneObjective::InfoNce;
    let _ = OrtmannPivotStrategy::KMeansSp;
    let _ = DrGraphConfig::default();
    let _ = GraphCneConfig::default();
    let _ = LargeVisConfig::default();
    let _ = OrtmannSparseStressConfig::default();
    let _ = PacmapGraphConfig::default();
    let _ = Sgd2Config::default();
    let _ = SnapTFdpConfig::default();
    let _ = SparseBarrierConfig::default();
    let _ = StreamingNegativeConfig::default();
    let _: Option<SparseBarrierStats> = None;
    let _: Option<CoarseBackboneStats> = None;
    let _: Option<CoarseBackboneResult<f64>> = None;

    // Exercise coarse_backbone_init from the crate root.
    let tri =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (0, 2)], None, false, false).unwrap();
    let bb = coarse_backbone_init::<f64, f64>(&tri, &CoarseBackboneConfig::default()).unwrap();
    assert_eq!(bb.fine_layout.len(), 3);

    // Exercise the hidden cross-crate refiner contract on `egraph_layout_sgd::sgd`.
    let mut pairs = vec![
        PackedSymPair::new(0, 1, false, 1.0_f64, 1.0_f64),
        PackedSymPair::new(1, 2, false, 1.0_f64, 2.0_f64),
        PackedSymPair::new(0, 2, true, 1.5_f64, 0.5_f64),
    ];
    let w_before = pairs[0].w;
    assert_eq!(w_before, 1.0);
    pairs[0].w = 1.25;
    assert_eq!(pairs[0].w, 1.25);

    let sched = SgdScheduleConfig::new(10)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);
    let mut solver = SgdSolver::<f64>::from_packed_symmetric(3, pairs, 2, 1, sched, 42).unwrap();
    let mut layout = Layout2d::from_xy(vec![0.0, 0.2, 0.4], vec![0.0, 0.1, 0.0]).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert!(layout.is_all_finite());
}
