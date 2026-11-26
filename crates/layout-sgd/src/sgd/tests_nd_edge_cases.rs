use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, SolverStatus};
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use super::super::sgd::{
    DrGraphConfig, GraphCneConfig, LargeVisConfig, OrtmannSparseStressConfig, PacmapGraphConfig,
    Sgd2Config, SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SparseBarrierConfig,
    StreamingNegativeConfig,
};
use super::test_support::grid_graph;
use super::tests_nd::{assert_step_eq_run_nd, build_nd_mode_pair};

#[test]
fn test_nd_edge_cases_coincident_degenerate_and_error_guards() {
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
    let sched = SgdScheduleConfig {
        iterations: 8,
        ..Default::default()
    };
    let emb = [0.0_f64; 18];
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 2,
        seed: 42,
        ..Default::default()
    };
    for g in &deg_graphs {
        let gn = g.num_nodes();
        let zeros = LayoutNd::<f64>::zeros(gn, 3).unwrap();
        let (lms, _) = compute_regularized_dog_varadhan_landmarks(g, &dog_cfg).unwrap();
        let mut solvers: Vec<SgdSolver<f64>> = vec![
            SgdSolver::full(g, sched, 42).unwrap(),
            SgdSolver::random_pair(g, 2, sched, 42).unwrap(),
            SgdSolver::omega(g, &emb[..gn * 2], 2, 2, 1e-3, sched, 42).unwrap(),
            SgdSolver::streaming_negatives(g, StreamingNegativeConfig::default(), sched, 42)
                .unwrap(),
            SgdSolver::ortmann_sparse_stress(
                g,
                OrtmannSparseStressConfig {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::snap_t_fdp(
                g,
                SnapTFdpConfig {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::pacmap_graph(
                g,
                PacmapGraphConfig {
                    phase_iterations: [3, 3, 2],
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::sgd2(
                g,
                Sgd2Config {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::largevis(
                g,
                LargeVisConfig {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::drgraph(
                g,
                DrGraphConfig {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            SgdSolver::graph_cne(
                g,
                GraphCneConfig {
                    iterations: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
        ];
        if gn > 0 {
            let sb_cfg = SparseBarrierConfig {
                num_pivots: 2,
                schedule: sched,
                ..Default::default()
            };
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
            let mut l = zeros.copy();
            assert_eq!(s.run(&mut l).unwrap(), SolverStatus::Completed);
            assert!(l.is_all_finite());
        }
    }

    // Verify dims=1 works bit-exactly across all 17 modes.
    let g_grid = grid_graph(3, 3);
    let ng = g_grid.num_nodes();
    let (lms, _) = compute_regularized_dog_varadhan_landmarks(&g_grid, &dog_cfg).unwrap();
    let init_1d = LayoutNd::<f64>::init(ng, 1, 55, 1.0).unwrap();
    for mode in 0..17 {
        let (s1, s2, name) = build_nd_mode_pair::<f64>(mode, &g_grid, &lms, &emb[..ng * 2]);
        assert_step_eq_run_nd(s1, s2, &init_1d, 4, &format!("{name}_1d"));
    }

    // Coincident zero scatter breaks degeneracy on LargeVis / DRGraph / Graph-CNE in 3D.
    let mut lv_z = LayoutNd::<f64>::zeros(ng, 3).unwrap();
    SgdSolver::<f64>::largevis(
        &g_grid,
        LargeVisConfig {
            iterations: 8,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut lv_z)
    .unwrap();
    assert!(lv_z.coords().iter().any(|&v| v.abs() > 1e-6));

    // Error guards: dims=0, num_nodes mismatch, NaN/Inf, mid-run dims change & 2D/N-D switch.
    assert!(matches!(
        LayoutNd::<f64>::zeros(ng, 0),
        Err(CoreError::InvalidInput(_))
    ));
    let mut wrong_n = LayoutNd::<f64>::zeros(ng + 1, 3).unwrap();
    assert!(matches!(
        SgdSolver::<f64>::full(&g_grid, sched, 1)
            .unwrap()
            .step(&mut wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut nan_l = LayoutNd::<f64>::from_flat(ng, 3, vec![f64::NAN; ng * 3]).unwrap();
    assert!(matches!(
        SgdSolver::<f64>::full(&g_grid, sched, 1)
            .unwrap()
            .step(&mut nan_l),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut inf_l = LayoutNd::<f64>::from_flat(ng, 3, vec![f64::INFINITY; ng * 3]).unwrap();
    assert!(matches!(
        SgdSolver::<f64>::pacmap_graph(&g_grid, Default::default())
            .unwrap()
            .step(&mut inf_l),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_d3 = LayoutNd::<f64>::init(ng, 3, 1, 1.0).unwrap();
    let mut l_d4 = LayoutNd::<f64>::init(ng, 4, 1, 1.0).unwrap();
    let mut l_d2 = LayoutNd::<f64>::init(ng, 2, 1, 1.0).unwrap();
    let mut l_2d = Layout2d::<f64>::init(ng, 1, 1.0).unwrap();
    for mode in 0..17 {
        // 3D -> 4D and 3D -> 2D (Layout2d)
        let (mut s_3d, mut s_2d, name) =
            build_nd_mode_pair::<f64>(mode, &g_grid, &lms, &emb[..ng * 2]);
        s_3d.step(&mut l_d3).unwrap();
        assert!(
            matches!(
                s_3d.step(&mut l_d4),
                Err(CoreError::DimensionMismatch { .. })
            ),
            "{name}: expected 3D->4D rejection"
        );
        assert!(
            matches!(s_3d.step(&mut l_2d), Err(CoreError::InvalidInput(_))),
            "{name}: expected 3D->Layout2d rejection"
        );

        // Layout2d -> 3D and Layout2d -> Nd(2D)
        s_2d.step(&mut l_2d).unwrap();
        assert!(
            matches!(s_2d.step(&mut l_d3), Err(CoreError::InvalidInput(_))),
            "{name}: expected Layout2d->3D rejection"
        );
        assert!(
            matches!(s_2d.step(&mut l_d2), Err(CoreError::InvalidInput(_))),
            "{name}: expected Layout2d->Nd(2D) rejection"
        );
    }
}
