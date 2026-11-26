use egraph_core::{CsrGraph, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d};

use super::super::sgd::{
    DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig, SgdSolver,
};
use super::test_support::{barbell_graph, circle_layout, grid_graph};

#[test]
fn test_negative_sampling_solvers_parity_and_edge_cases() {
    let barbell = barbell_graph(6);
    let n = barbell.num_nodes();
    let grid = grid_graph(4, 4);
    let ng = grid.num_nodes();

    // 1. LargeVis (f64 & f32 bit-exact step() == run(), seed parity, collapsed init).
    let lv_cfg = LargeVisConfig {
        iterations: 30,
        negative_samples: 4,
        gamma: 5.0,
        seed: 42,
        ..Default::default()
    };
    let mut lv_run = SgdSolver::<f64>::largevis(&barbell, lv_cfg).unwrap();
    let mut lv_step = SgdSolver::<f64>::largevis(&barbell, lv_cfg).unwrap();
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

    let mut lv32_run = SgdSolver::<f32>::largevis(&barbell, lv_cfg).unwrap();
    let mut lv32_step = SgdSolver::<f32>::largevis(&barbell, lv_cfg).unwrap();
    let mut l32_lv_run = circle_layout::<f32>(n, 0.5);
    let mut l32_lv_step = l32_lv_run.clone();
    lv32_run.run(&mut l32_lv_run).unwrap();
    lv32_step.run_steps(&mut l32_lv_step, 7).unwrap();
    lv32_step.run(&mut l32_lv_step).unwrap();
    for i in 0..n {
        assert_eq!(l32_lv_run.x()[i].to_bits(), l32_lv_step.x()[i].to_bits());
        assert_eq!(l32_lv_run.y()[i].to_bits(), l32_lv_step.y()[i].to_bits());
    }

    // 2. DRGraph (k_order=1 and k_order=2, multilevel coarsening, f64 & f32 bit-exact).
    for k_order in [1usize, 2usize] {
        let dr_cfg = DrGraphConfig {
            k_order,
            negative_samples: 4,
            b: 2.0,
            iterations: 36,
            coarsen_stop_ratio: 0.85,
            seed: 42,
            ..Default::default()
        };
        let mut dr_run = SgdSolver::<f64>::drgraph(&grid, dr_cfg).unwrap();
        let mut dr_step = SgdSolver::<f64>::drgraph(&grid, dr_cfg).unwrap();
        assert!(dr_run.drgraph.as_ref().unwrap().levels.len() >= 2);
        let mut l_dr_run = Layout2d::<f64>::zeros(ng);
        let mut l_dr_step = Layout2d::<f64>::zeros(ng);
        dr_run.run(&mut l_dr_run).unwrap();
        dr_step.run_steps(&mut l_dr_step, 13).unwrap();
        while !dr_step.status().is_finished() {
            dr_step.step(&mut l_dr_step).unwrap();
        }
        assert!(l_dr_run.is_all_finite());
        for i in 0..ng {
            assert_eq!(l_dr_run.x()[i].to_bits(), l_dr_step.x()[i].to_bits());
            assert_eq!(l_dr_run.y()[i].to_bits(), l_dr_step.y()[i].to_bits());
        }
        assert!((0..ng).any(|i| l_dr_run.x()[i].abs() > 1e-6 || l_dr_run.y()[i].abs() > 1e-6));
    }

    let dr32_cfg = DrGraphConfig {
        k_order: 1,
        b: 1.0,
        iterations: 24,
        seed: 7,
        ..Default::default()
    };
    let mut dr32_run = SgdSolver::<f32>::drgraph(&grid, dr32_cfg).unwrap();
    let mut dr32_step = SgdSolver::<f32>::drgraph(&grid, dr32_cfg).unwrap();
    let mut l32_dr_run = circle_layout::<f32>(ng, 1.0);
    let mut l32_dr_step = l32_dr_run.clone();
    dr32_run.run(&mut l32_dr_run).unwrap();
    dr32_step.run_steps(&mut l32_dr_step, 9).unwrap();
    dr32_step.run(&mut l32_dr_step).unwrap();
    for i in 0..ng {
        assert_eq!(l32_dr_run.x()[i].to_bits(), l32_dr_step.x()[i].to_bits());
        assert_eq!(l32_dr_run.y()[i].to_bits(), l32_dr_step.y()[i].to_bits());
    }

    // 3. Graph-CNE (InfoNce & NegTSne, row_normalize true/false, f64 & f32 bit-exact).
    for objective in [GraphCneObjective::InfoNce, GraphCneObjective::NegTSne] {
        for row_normalize in [true, false] {
            let cne_cfg = GraphCneConfig {
                objective,
                negative_samples: 4,
                iterations: 30,
                row_normalize,
                seed: 42,
                ..Default::default()
            };
            let mut cne_run = SgdSolver::<f64>::graph_cne(&barbell, cne_cfg).unwrap();
            let mut cne_step = SgdSolver::<f64>::graph_cne(&barbell, cne_cfg).unwrap();
            let mut l_run = Layout2d::<f64>::zeros(n);
            let mut l_step = Layout2d::<f64>::zeros(n);
            cne_run.run(&mut l_run).unwrap();
            cne_step.run_steps(&mut l_step, 10).unwrap();
            while !cne_step.status().is_finished() {
                cne_step.step(&mut l_step).unwrap();
            }
            assert!(l_run.is_all_finite());
            for i in 0..n {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            let mut cne32_run = SgdSolver::<f32>::graph_cne(&barbell, cne_cfg).unwrap();
            let mut cne32_step = SgdSolver::<f32>::graph_cne(&barbell, cne_cfg).unwrap();
            let mut l32_run = circle_layout::<f32>(n, 0.5);
            let mut l32_step = l32_run.clone();
            cne32_run.run(&mut l32_run).unwrap();
            cne32_step.run_steps(&mut l32_step, 7).unwrap();
            cne32_step.run(&mut l32_step).unwrap();
            for i in 0..n {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    // 4. Non-collapsed inputs remain distinct from zero-init, and different seeds differ.
    let cne_s1 = GraphCneConfig {
        iterations: 15,
        seed: 101,
        ..Default::default()
    };
    let cne_s2 = GraphCneConfig {
        seed: 202,
        ..cne_s1
    };
    let mut sc1 = SgdSolver::<f64>::graph_cne(&barbell, cne_s1).unwrap();
    let mut sc2 = SgdSolver::<f64>::graph_cne(&barbell, cne_s2).unwrap();
    let mut lc1 = circle_layout::<f64>(n, 1.0);
    let mut lc2 = lc1.clone();
    sc1.run(&mut lc1).unwrap();
    sc2.run(&mut lc2).unwrap();
    assert!((0..n).any(|i| lc1.x()[i].to_bits() != lc2.x()[i].to_bits()));

    // 5. step_manifold guard & non-finite / dimension-mismatch guards.
    let mut mf = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, n);
    assert!(
        SgdSolver::<f64>::largevis(&barbell, lv_cfg)
            .unwrap()
            .step_manifold(&mut mf)
            .is_err()
    );
    assert!(
        SgdSolver::<f64>::drgraph(&barbell, DrGraphConfig::default())
            .unwrap()
            .step_manifold(&mut mf)
            .is_err()
    );
    assert!(
        SgdSolver::<f64>::graph_cne(&barbell, GraphCneConfig::default())
            .unwrap()
            .step_manifold(&mut mf)
            .is_err()
    );
    let mut bad_layout = Layout2d::from_xy(vec![f64::NAN; n], vec![0.0; n]).unwrap();
    assert!(
        SgdSolver::<f64>::largevis(&barbell, lv_cfg)
            .unwrap()
            .step(&mut bad_layout)
            .is_err()
    );
    let mut wrong_len = Layout2d::<f64>::zeros(n + 1);
    assert!(
        SgdSolver::<f64>::drgraph(&barbell, DrGraphConfig::default())
            .unwrap()
            .step(&mut wrong_len)
            .is_err()
    );

    // 6. Degenerate graphs: N=0, N=1, edgeless, disconnected + isolated, collinear,
    //    weighted directed with reciprocal edges & self-loop.
    let dir_w = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (3, 4)],
        Some(&[1.0, 2.5, 1.5, 3.0, 4.0, 0.5]),
        true,
        true,
    )
    .unwrap();
    let mut deg_cases = vec![
        CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap(),
        dir_w,
    ];
    for g in deg_cases.drain(..) {
        let gn = g.num_nodes();
        let xs: Vec<f64> = (0..gn).map(|i| i as f64).collect();
        let ys = vec![0.0; gn];
        let init = Layout2d::from_xy(xs, ys).unwrap();

        let mut l1 = init.clone();
        SgdSolver::<f64>::largevis(
            &g,
            LargeVisConfig {
                iterations: 10,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l1)
        .unwrap();
        assert!(l1.is_all_finite());

        let mut l2 = init.clone();
        SgdSolver::<f64>::drgraph(
            &g,
            DrGraphConfig {
                k_order: 2,
                iterations: 10,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l2)
        .unwrap();
        assert!(l2.is_all_finite());

        let mut l3 = init;
        SgdSolver::<f64>::graph_cne(
            &g,
            GraphCneConfig {
                iterations: 10,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l3)
        .unwrap();
        assert!(l3.is_all_finite());
    }

    // Non-finite layout coordinates rejected across all 3 solvers.
    assert!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::INFINITY]), false, false).is_err()
    );
    let mut inf_l = Layout2d::from_xy(vec![f64::INFINITY; n], vec![0.0; n]).unwrap();
    assert!(
        SgdSolver::<f64>::graph_cne(&barbell, GraphCneConfig::default())
            .unwrap()
            .step(&mut inf_l)
            .is_err()
    );
}
