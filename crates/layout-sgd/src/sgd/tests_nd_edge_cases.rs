use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, SolverStatus};
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use super::super::sgd::{
    OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver, SgdScheduleConfig, SgdScheduleKind,
    SgdSolver, SparseBarrierConfig, StreamingNegativeConfig,
};
use super::test_support::grid_graph;
use super::tests_nd::{assert_step_eq_run_nd, build_mode_pair};

#[test]
fn test_nd_edge_cases_degenerate_zero_step_and_breakdown_guards() {
    let deg_graphs = [
        CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2)],
            Some(&[1.0, 2.0, 1.5]),
            true,
            false,
        )
        .unwrap(),
    ];
    let sched = SgdScheduleConfig::new(8);
    let emb = [0.0_f64; 18];
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(2).with_seed(42);
    for g in &deg_graphs {
        let gn = g.num_nodes();
        let zeros = LayoutNd::<f64>::zeros(gn, 3).unwrap();
        let (lms, _) = compute_regularized_dog_varadhan_landmarks(g, &dog_cfg).unwrap();
        let mut solvers: Vec<SgdSolver<LayoutNd<f64>>> = vec![
            SgdSolver::full(g, sched.clone(), 42).unwrap(),
            SgdSolver::random_pair(g, 2, sched.clone(), 42).unwrap(),
            SgdSolver::omega(g, &emb[..gn * 2], 2, 2, 1e-3, sched.clone(), 42).unwrap(),
            SgdSolver::streaming_negatives(
                g,
                StreamingNegativeConfig::default(),
                sched.clone(),
                42,
            )
            .unwrap(),
            SgdSolver::ortmann_sparse_stress(
                g,
                OrtmannSparseStressConfig::default().with_iterations(8),
            )
            .unwrap(),
        ];
        if gn > 0 {
            let sb_cfg = SparseBarrierConfig::default()
                .with_num_pivots(2)
                .with_schedule(sched.clone());
            solvers.push(
                SgdSolver::sparse_barrier_from_landmarks(g, &lms, None, &sb_cfg)
                    .unwrap()
                    .0,
            );
            solvers.push(
                SgdSolver::sparse_barrier_dog(g, &dog_cfg, &sb_cfg)
                    .unwrap()
                    .0,
            );
        }
        for mut s in solvers {
            let mut l = zeros.clone();
            assert_eq!(s.run(&mut l).unwrap(), SolverStatus::Completed);
            assert!(l.is_all_finite());
        }
        let mut s2 =
            Sgd2Solver::<LayoutNd<f64>>::new(g, Sgd2Config::new().with_iterations(8)).unwrap();
        let mut l2 = zeros;
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
        assert!(l2.is_all_finite());
    }

    // Verify dims=1 works bit-exactly across all 11 modes.
    let g_grid = grid_graph(3, 3);
    let ng = g_grid.num_nodes();
    let (lms, _) = compute_regularized_dog_varadhan_landmarks(&g_grid, &dog_cfg).unwrap();
    let init_1d = LayoutNd::<f64>::init(ng, 1, 55, 1.0).unwrap();
    for mode in 0..11 {
        let (s1, s2, name) = build_mode_pair::<LayoutNd<f64>>(mode, &g_grid, &lms, &emb[..ng * 2]);
        assert_step_eq_run_nd(s1, s2, &init_1d, 4, &format!("{name}_1d"));
    }

    // Error guards: dims=0, num_nodes mismatch, NaN/Inf, run_steps(_, 0), mid-run dims change.
    assert!(matches!(
        LayoutNd::<f64>::zeros(ng, 0),
        Err(CoreError::InvalidInput(_))
    ));
    let mut wrong_n = LayoutNd::<f64>::zeros(ng + 1, 3).unwrap();
    let mut s_guard = SgdSolver::<LayoutNd<f64>>::full(&g_grid, sched.clone(), 1).unwrap();
    assert!(matches!(
        s_guard.run_steps(&mut wrong_n, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_guard.status(), SolverStatus::Initialized);
    assert!(matches!(
        s_guard.step(&mut wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_guard.status(), SolverStatus::Initialized);

    let mut nan_l = LayoutNd::<f64>::from_flat(ng, 3, &vec![f64::NAN; ng * 3]).unwrap();
    assert!(matches!(
        s_guard.run_steps(&mut nan_l, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        s_guard.step(&mut nan_l),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_d3 = LayoutNd::<f64>::init(ng, 3, 1, 1.0).unwrap();
    let mut l_d4 = LayoutNd::<f64>::init(ng, 4, 1, 1.0).unwrap();
    for mode in 0..11 {
        let (mut s_3d, _, name) =
            build_mode_pair::<LayoutNd<f64>>(mode, &g_grid, &lms, &emb[..ng * 2]);
        s_3d.step(&mut l_d3).unwrap();
        assert!(
            matches!(
                s_3d.step(&mut l_d4),
                Err(CoreError::DimensionMismatch { .. })
            ),
            "{name}: expected 3D->4D rejection"
        );
        assert!(
            matches!(
                s_3d.run_steps(&mut l_d4, 0),
                Err(CoreError::DimensionMismatch { .. })
            ),
            "{name}: expected 3D->4D run_steps(_, 0) rejection"
        );
    }

    let mut s2_d3 =
        Sgd2Solver::<LayoutNd<f64>>::new(&g_grid, Sgd2Config::new().with_iterations(4)).unwrap();
    s2_d3.step(&mut l_d3).unwrap();
    assert!(matches!(
        s2_d3.run_steps(&mut l_d4, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Fatal numerical breakdown transitions solver to SolverStatus::Failed.
    let mut overflow_layout =
        Layout2d::from_xy(vec![-1.0_f32, 1.0_f32], vec![0.0_f32, 0.0_f32]).unwrap();
    let mut s_fail = SgdSolver::<Layout2d<f32>>::from_pairs(
        2,
        &[0],
        &[1],
        &[10.0_f32],
        &[10.0_f32],
        &[1.0_f32],
        &[1.0_f32],
        SgdScheduleConfig::new(4)
            .with_kind(SgdScheduleKind::Constant)
            .with_step_scale(1e39)
            .with_max_mu(Some(1e39)),
        42,
    )
    .unwrap();
    assert!(s_fail.step(&mut overflow_layout).is_err());
    assert_eq!(s_fail.status(), SolverStatus::Failed);
    assert!(s_fail.status().is_finished());
    assert!(s_fail.status().is_failed());
    assert_eq!(
        s_fail.step(&mut overflow_layout).unwrap(),
        SolverStatus::Failed
    );
}
