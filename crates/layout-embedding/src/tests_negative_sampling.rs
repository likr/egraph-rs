use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d,
    SolverStatus,
};

use crate::test_support::{
    ALL_MANIFOLDS, assert_manifold_step_eq_run, assert_riemannian_step_eq_run, barbell_graph,
    circle_layout, grid_graph, layout_2d_to_nd,
};
use crate::{
    DrGraphConfig, DrGraphSolver, GraphCneConfig, GraphCneObjective, GraphCneSolver,
    LargeVisConfig, LargeVisSolver,
};

#[test]
fn test_negative_sampling_solvers_parity_and_edge_cases() {
    let barbell = barbell_graph(6);
    let n = barbell.num_nodes();
    let grid = grid_graph(4, 4);
    let ng = grid.num_nodes();

    // 1. LargeVis (f64 & f32 bit-exact step() == run(), collapsed init, 3D & ND parity).
    let lv_cfg = LargeVisConfig::new()
        .with_iterations(30)
        .with_negative_samples(4)
        .with_gamma(5.0)
        .with_seed(42);
    let mut lv_run = LargeVisSolver::<Layout2d<f64>>::new(&barbell, lv_cfg.clone()).unwrap();
    let mut lv_step = LargeVisSolver::<Layout2d<f64>>::new(&barbell, lv_cfg.clone()).unwrap();
    let mut l_lv_run = Layout2d::<f64>::zeros(n);
    let mut l_lv_step = Layout2d::<f64>::zeros(n);
    lv_run.run(&mut l_lv_run).unwrap();
    lv_step.run_steps(&mut l_lv_step, 11).unwrap();
    while !lv_step.status().is_finished() {
        lv_step.step(&mut l_lv_step).unwrap();
    }
    assert!(l_lv_run.is_all_finite());
    for i in 0..n {
        assert_eq!(l_lv_run.x()[i].to_bits(), l_lv_step.x()[i].to_bits());
        assert_eq!(l_lv_run.y()[i].to_bits(), l_lv_step.y()[i].to_bits());
    }
    assert!((0..n).any(|i| l_lv_run.x()[i].abs() > 1e-6 || l_lv_run.y()[i].abs() > 1e-6));

    let lv32_init = circle_layout::<f32>(n, 0.5);
    let out_lv32 = assert_riemannian_step_eq_run(
        LargeVisSolver::<Layout2d<f32>>::new(&barbell, lv_cfg.clone()).unwrap(),
        LargeVisSolver::<Layout2d<f32>>::new(&barbell, lv_cfg.clone()).unwrap(),
        &lv32_init,
        7,
        "largevis_2d_f32",
    );
    let out_lv32_nd2 = assert_riemannian_step_eq_run(
        LargeVisSolver::<LayoutNd<f32>>::new(&barbell, lv_cfg.clone()).unwrap(),
        LargeVisSolver::<LayoutNd<f32>>::new(&barbell, lv_cfg.clone()).unwrap(),
        &layout_2d_to_nd(&lv32_init),
        7,
        "largevis_nd2_f32",
    );
    for i in 0..n {
        assert_eq!(
            out_lv32.x()[i].to_bits(),
            out_lv32_nd2.coords()[i * 2].to_bits()
        );
        assert_eq!(
            out_lv32.y()[i].to_bits(),
            out_lv32_nd2.coords()[i * 2 + 1].to_bits()
        );
    }

    let lv3_init = Layout3d::<f64>::init(n, 7, 1.0).unwrap();
    assert_riemannian_step_eq_run(
        LargeVisSolver::<Layout3d<f64>>::new(&barbell, lv_cfg.clone()).unwrap(),
        LargeVisSolver::<Layout3d<f64>>::new(&barbell, lv_cfg.clone()).unwrap(),
        &lv3_init,
        7,
        "largevis_3d",
    );

    // 2. DRGraph (k_order=1 and k_order=2, multilevel coarsening, 2D/3D/ND f64 & f32 bit-exact).
    for k_order in [1usize, 2usize] {
        let dr_cfg = DrGraphConfig::new()
            .with_k_order(k_order)
            .with_negative_samples(4)
            .with_b(2.0)
            .with_iterations(36)
            .with_coarsen_stop_ratio(0.85)
            .with_seed(42);
        let dr_run = DrGraphSolver::<Layout2d<f64>>::new(&grid, dr_cfg.clone()).unwrap();
        let dr_step = DrGraphSolver::<Layout2d<f64>>::new(&grid, dr_cfg.clone()).unwrap();
        assert!(dr_run.num_levels() >= 2);
        let l_dr_run = assert_riemannian_step_eq_run(
            dr_run,
            dr_step,
            &Layout2d::<f64>::zeros(ng),
            13,
            "drgraph_2d",
        );
        assert!((0..ng).any(|i| l_dr_run.x()[i].abs() > 1e-6 || l_dr_run.y()[i].abs() > 1e-6));

        let init_nd2 = layout_2d_to_nd(&circle_layout::<f64>(ng, 1.0));
        let out_2d = assert_riemannian_step_eq_run(
            DrGraphSolver::<Layout2d<f64>>::new(&grid, dr_cfg.clone()).unwrap(),
            DrGraphSolver::<Layout2d<f64>>::new(&grid, dr_cfg.clone()).unwrap(),
            &circle_layout::<f64>(ng, 1.0),
            11,
            "drgraph_2d_circle",
        );
        let out_nd2 = assert_riemannian_step_eq_run(
            DrGraphSolver::<LayoutNd<f64>>::new(&grid, dr_cfg.clone()).unwrap(),
            DrGraphSolver::<LayoutNd<f64>>::new(&grid, dr_cfg.clone()).unwrap(),
            &init_nd2,
            11,
            "drgraph_nd2_circle",
        );
        for i in 0..ng {
            assert_eq!(out_2d.x()[i].to_bits(), out_nd2.coords()[i * 2].to_bits());
            assert_eq!(
                out_2d.y()[i].to_bits(),
                out_nd2.coords()[i * 2 + 1].to_bits()
            );
        }
    }

    let dr32_cfg = DrGraphConfig::new()
        .with_k_order(1)
        .with_b(1.0)
        .with_iterations(24)
        .with_seed(7);
    assert_riemannian_step_eq_run(
        DrGraphSolver::<Layout2d<f32>>::new(&grid, dr32_cfg.clone()).unwrap(),
        DrGraphSolver::<Layout2d<f32>>::new(&grid, dr32_cfg.clone()).unwrap(),
        &circle_layout::<f32>(ng, 1.0),
        9,
        "drgraph_2d_f32",
    );
    assert_riemannian_step_eq_run(
        DrGraphSolver::<Layout3d<f64>>::new(&grid, dr32_cfg.clone()).unwrap(),
        DrGraphSolver::<Layout3d<f64>>::new(&grid, dr32_cfg.clone()).unwrap(),
        &Layout3d::<f64>::init(ng, 9, 1.0).unwrap(),
        9,
        "drgraph_3d",
    );
    assert_riemannian_step_eq_run(
        DrGraphSolver::<LayoutNd<f64>>::new(&grid, dr32_cfg.clone()).unwrap(),
        DrGraphSolver::<LayoutNd<f64>>::new(&grid, dr32_cfg.clone()).unwrap(),
        &LayoutNd::<f64>::init(ng, 5, 9, 1.0).unwrap(),
        9,
        "drgraph_5d",
    );

    // 3. Graph-CNE (InfoNce & NegTSne, row_normalize true/false, 2D/3D/ND f64 & f32 bit-exact).
    for objective in [GraphCneObjective::InfoNce, GraphCneObjective::NegTSne] {
        for row_normalize in [true, false] {
            let cne_cfg = GraphCneConfig::new()
                .with_objective(objective)
                .with_negative_samples(4)
                .with_iterations(30)
                .with_row_normalize(row_normalize)
                .with_seed(42);
            assert_riemannian_step_eq_run(
                GraphCneSolver::<Layout2d<f64>>::new(&barbell, cne_cfg.clone()).unwrap(),
                GraphCneSolver::<Layout2d<f64>>::new(&barbell, cne_cfg.clone()).unwrap(),
                &Layout2d::<f64>::zeros(n),
                10,
                "cne_2d_f64",
            );
            assert_riemannian_step_eq_run(
                GraphCneSolver::<Layout2d<f32>>::new(&barbell, cne_cfg.clone()).unwrap(),
                GraphCneSolver::<Layout2d<f32>>::new(&barbell, cne_cfg.clone()).unwrap(),
                &circle_layout::<f32>(n, 0.5),
                7,
                "cne_2d_f32",
            );
            assert_riemannian_step_eq_run(
                GraphCneSolver::<Layout3d<f64>>::new(&barbell, cne_cfg.clone()).unwrap(),
                GraphCneSolver::<Layout3d<f64>>::new(&barbell, cne_cfg.clone()).unwrap(),
                &Layout3d::<f64>::init(n, 5, 1.0).unwrap(),
                7,
                "cne_3d",
            );
            assert_riemannian_step_eq_run(
                GraphCneSolver::<LayoutNd<f32>>::new(&barbell, cne_cfg).unwrap(),
                GraphCneSolver::<LayoutNd<f32>>::new(
                    &barbell,
                    cne_cfg_clone(objective, row_normalize),
                )
                .unwrap(),
                &LayoutNd::<f32>::init(n, 4, 5, 1.0).unwrap(),
                7,
                "cne_4d_f32",
            );
        }
    }

    // 4. Non-collapsed inputs remain distinct from zero-init, and different seeds differ.
    let cne_s1 = GraphCneConfig::new().with_iterations(15).with_seed(101);
    let cne_s2 = cne_s1.clone().with_seed(202);
    let mut sc1 = GraphCneSolver::<Layout2d<f64>>::new(&barbell, cne_s1).unwrap();
    let mut sc2 = GraphCneSolver::<Layout2d<f64>>::new(&barbell, cne_s2).unwrap();
    let mut lc1 = circle_layout::<f64>(n, 1.0);
    let mut lc2 = lc1.clone();
    sc1.run(&mut lc1).unwrap();
    sc2.run(&mut lc2).unwrap();
    assert!((0..n).any(|i| lc1.x()[i].to_bits() != lc2.x()[i].to_bits()));

    // 5. All 4 manifolds bit-exact f64 & f32 across LargeVis, DRGraph, GraphCne.
    let lv_m_cfg = LargeVisConfig::new()
        .with_iterations(8)
        .with_negative_samples(2);
    let dr_m_cfg = DrGraphConfig::new()
        .with_k_order(2)
        .with_iterations(10)
        .with_negative_samples(2)
        .with_coarsen_stop_ratio(0.85);
    let cne_m_cfg = GraphCneConfig::new()
        .with_iterations(8)
        .with_negative_samples(2);
    for kind in ALL_MANIFOLDS {
        assert_manifold_step_eq_run(
            LargeVisSolver::<ManifoldLayout2d<f64>>::new(&grid, lv_m_cfg.clone()).unwrap(),
            LargeVisSolver::<ManifoldLayout2d<f64>>::new(&grid, lv_m_cfg.clone()).unwrap(),
            kind,
            ng,
            "largevis_manifold_f64",
        );
        assert_manifold_step_eq_run(
            DrGraphSolver::<ManifoldLayout2d<f64>>::new(&grid, dr_m_cfg.clone()).unwrap(),
            DrGraphSolver::<ManifoldLayout2d<f64>>::new(&grid, dr_m_cfg.clone()).unwrap(),
            kind,
            ng,
            "drgraph_manifold_f64",
        );
        assert_manifold_step_eq_run(
            GraphCneSolver::<ManifoldLayout2d<f64>>::new(&grid, cne_m_cfg.clone()).unwrap(),
            GraphCneSolver::<ManifoldLayout2d<f64>>::new(&grid, cne_m_cfg.clone()).unwrap(),
            kind,
            ng,
            "graph_cne_manifold_f64",
        );
        assert_manifold_step_eq_run(
            DrGraphSolver::<ManifoldLayout2d<f32>>::new(&grid, dr_m_cfg.clone()).unwrap(),
            DrGraphSolver::<ManifoldLayout2d<f32>>::new(&grid, dr_m_cfg.clone()).unwrap(),
            kind,
            ng,
            "drgraph_manifold_f32",
        );
    }

    // 6. run_steps(0) validation, ND dim-locking, and non-finite / dimension-mismatch guards.
    let mut bad_layout = Layout2d::from_xy(vec![f64::NAN; n], vec![0.0; n]).unwrap();
    let mut lv_guard = LargeVisSolver::<Layout2d<f64>>::new(&barbell, lv_cfg).unwrap();
    assert!(matches!(
        lv_guard.run_steps(&mut bad_layout, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(lv_guard.status(), SolverStatus::Initialized);
    assert!(lv_guard.step(&mut bad_layout).is_err());
    assert_eq!(lv_guard.status(), SolverStatus::Initialized);

    let mut wrong_len = Layout2d::<f64>::zeros(n + 1);
    assert!(
        DrGraphSolver::<Layout2d<f64>>::new(&barbell, DrGraphConfig::default())
            .unwrap()
            .step(&mut wrong_len)
            .is_err()
    );

    let mut nd_locked =
        GraphCneSolver::<LayoutNd<f64>>::new(&barbell, GraphCneConfig::new().with_iterations(4))
            .unwrap();
    let mut nd2 = LayoutNd::<f64>::init(n, 2, 1, 1.0).unwrap();
    let mut nd3 = LayoutNd::<f64>::init(n, 3, 1, 1.0).unwrap();
    assert_eq!(nd_locked.step(&mut nd2).unwrap(), SolverStatus::Running);
    assert!(matches!(
        nd_locked.run_steps(&mut nd3, 0),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));

    // 7. Degenerate graphs: N=0, N=1, edgeless, disconnected + isolated, collinear,
    //    weighted directed with reciprocal edges & self-loop.
    let dir_w = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (3, 4)],
        Some(&[1.0, 2.5, 1.5, 3.0, 4.0, 0.5]),
        true,
        true,
    )
    .unwrap();
    let deg_cases = vec![
        CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap(),
        dir_w,
    ];
    for g in deg_cases {
        let gn = g.num_nodes();
        let xs: Vec<f64> = (0..gn).map(|i| i as f64).collect();
        let ys = vec![0.0; gn];
        let init = Layout2d::from_xy(xs, ys).unwrap();

        let mut l1 = init.clone();
        LargeVisSolver::<Layout2d<f64>>::new(&g, LargeVisConfig::new().with_iterations(10))
            .unwrap()
            .run(&mut l1)
            .unwrap();
        assert!(l1.is_all_finite());

        let mut l2 = init.clone();
        DrGraphSolver::<Layout2d<f64>>::new(
            &g,
            DrGraphConfig::new().with_k_order(2).with_iterations(10),
        )
        .unwrap()
        .run(&mut l2)
        .unwrap();
        assert!(l2.is_all_finite());

        let mut l3 = init;
        GraphCneSolver::<Layout2d<f64>>::new(&g, GraphCneConfig::new().with_iterations(10))
            .unwrap()
            .run(&mut l3)
            .unwrap();
        assert!(l3.is_all_finite());
    }

    let mut inf_l = Layout2d::from_xy(vec![f64::INFINITY; n], vec![0.0; n]).unwrap();
    assert!(
        GraphCneSolver::<Layout2d<f64>>::new(&barbell, GraphCneConfig::default())
            .unwrap()
            .step(&mut inf_l)
            .is_err()
    );
}

fn cne_cfg_clone(objective: GraphCneObjective, row_normalize: bool) -> GraphCneConfig {
    GraphCneConfig::new()
        .with_objective(objective)
        .with_negative_samples(4)
        .with_iterations(30)
        .with_row_normalize(row_normalize)
        .with_seed(42)
}
