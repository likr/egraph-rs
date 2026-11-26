use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};

use super::super::*;
use super::{barbell_graph, grid_graph, seeded_layout};

#[test]
fn test_yifan_hu_ec_mivs_multilevel_and_step_bit_exact() {
    // 15x15 grid (225 nodes) must produce multilevel EC+MIVS hierarchy (depth > 1).
    let grid = grid_graph(15, 15);
    let cfg = YifanHuConfig {
        max_iterations_per_level: 40,
        min_coarsen_size: 20,
        ..Default::default()
    };
    let mut yh_run = YifanHuSolver::<f64>::new(&grid, cfg).unwrap();
    let mut yh_step = YifanHuSolver::<f64>::new(&grid, cfg).unwrap();
    assert!(
        yh_run.num_levels() >= 3,
        "expected EC+MIVS hierarchy >= 3 levels on 225-node grid, got {}",
        yh_run.num_levels()
    );
    assert_eq!(yh_run.current_level(), yh_run.num_levels() - 1);
    assert_eq!(yh_run.total_iterations(), yh_run.num_levels() * 40);

    let mut l_run = Layout2d::<f64>::zeros(225);
    let mut l_step = l_run.clone();
    let st_run = yh_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    assert_eq!(yh_run.current_level(), 0);
    assert!(l_run.is_all_finite());

    // Prolongated & refined grid must have positive spatial spread.
    let spread_x = l_run.x().iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - l_run.x().iter().copied().fold(f64::INFINITY, f64::min);
    assert!(spread_x > 1.0, "expected spread > 1.0, got {spread_x}");

    while !yh_step.status().is_finished() {
        yh_step.step(&mut l_step).unwrap();
    }
    assert_eq!(yh_run.iteration(), yh_step.iteration());
    assert_eq!(yh_run.status(), yh_step.status());
    for i in 0..225 {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact check via ContinuousFdp::yifan_hu
    let barbell = barbell_graph(18);
    let cfg_b = YifanHuConfig {
        max_iterations_per_level: 30,
        min_coarsen_size: 10,
        ..Default::default()
    };
    let mut l32_run = seeded_layout::<f32>(barbell.num_nodes(), 123);
    let mut l32_step = l32_run.clone();
    let mut c32_run = ContinuousFdp::<f32>::yifan_hu(&barbell, cfg_b).unwrap();
    let mut c32_step = ContinuousFdp::<f32>::yifan_hu(&barbell, cfg_b).unwrap();
    c32_run.run(&mut l32_run).unwrap();
    while !c32_step.status().is_finished() {
        c32_step.step(&mut l32_step).unwrap();
    }
    assert!(l32_run.is_all_finite());
    for i in 0..barbell.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_continuous_fdp_edge_cases_directed_weighted_and_errors() {
    // Empty, singleton, edgeless, and zero-iteration graphs.
    for n in [0usize, 1, 5] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l_fa2 = Layout2d::<f64>::zeros(n);
        let mut fa2 = ForceAtlas2Solver::new(&g, ForceAtlas2Config::default()).unwrap();
        assert!(fa2.run(&mut l_fa2).unwrap().is_finished());
        assert!(l_fa2.is_all_finite());

        let mut l_yh = Layout2d::<f64>::zeros(n);
        let mut yh = YifanHuSolver::new(&g, YifanHuConfig::default()).unwrap();
        assert!(yh.run(&mut l_yh).unwrap().is_finished());
        assert!(l_yh.is_all_finite());
    }

    // Zero-iteration configs complete immediately.
    let g4 = grid_graph(2, 2);
    let fa2_zero = ForceAtlas2Solver::<f64>::new(
        &g4,
        ForceAtlas2Config {
            iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(fa2_zero.status(), SolverStatus::Completed);
    assert_eq!(fa2_zero.total_iterations(), 0);

    let yh_zero = YifanHuSolver::<f64>::new(
        &g4,
        YifanHuConfig {
            max_iterations_per_level: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(yh_zero.status(), SolverStatus::Completed);
    assert_eq!(yh_zero.total_iterations(), 0);

    // Directed weighted graph canonicalization + negative weight rejection.
    let dir_g = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 4.0, 1.5, 3.0]),
        true,
        false,
    )
    .unwrap();
    let mut l_dir = seeded_layout::<f64>(4, 55);
    let mut s_dir = ContinuousFdp::force_atlas2(&dir_g, ForceAtlas2Config::default()).unwrap();
    assert!(s_dir.run(&mut l_dir).unwrap().is_finished());
    assert!(l_dir.is_all_finite());

    let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        ForceAtlas2Solver::<f64>::new(&neg_g, ForceAtlas2Config::default()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        YifanHuSolver::<f64>::new(&neg_g, YifanHuConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));

    // Dimension mismatch & non-finite layout errors.
    let mut fa2_err = ForceAtlas2Solver::<f64>::new(&g4, ForceAtlas2Config::default()).unwrap();
    let mut yh_err = YifanHuSolver::<f64>::new(&g4, YifanHuConfig::default()).unwrap();
    let mut wrong = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        fa2_err.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        yh_err.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0, 2.0], vec![0.0; 4]).unwrap();
    assert!(matches!(
        fa2_err.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        yh_err.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
