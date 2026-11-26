mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::stress_kamada::{JsKamadaKawaiSolver, JsStressMajorizationSolver};
use egraph_wasm::stress_multilevel_baselines::JsMaxentStressSolver;
use util::fixtures::{
    assert_layouts_bit_exact, make_circle_layout, make_disconnected_graph, make_grid, make_opts,
    make_star_graph, make_two_cluster_graph,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn stress_majorization_and_kamada_kawai_solvers() {
    let small_grid = make_grid(3, 3);

    // 1. StressMajorizationSolver (fromGraph, fromDistanceMatrix, computeStress, lastStress, step == run)
    let sm_opts = make_opts(&[
        ("epsilon", JsValue::from_f64(1e-6)),
        ("maxIterations", JsValue::from_f64(30.0)),
    ]);
    let mut sm_run =
        JsStressMajorizationSolver::from_graph(&small_grid, &sm_opts, None, None, None, None)
            .expect("SM fromGraph");
    let mut sm_step =
        JsStressMajorizationSolver::from_graph(&small_grid, &sm_opts, None, None, None, None)
            .expect("SM fromGraph step");
    let mut sm_l_run = make_circle_layout(9, 0.2);
    let mut sm_l_step = sm_l_run.clone();
    let stress_before = sm_run.compute_stress(&sm_l_run).unwrap();
    assert!(sm_run.last_stress().is_none());
    sm_run.run(&mut sm_l_run).unwrap();
    let stress_after = sm_run.last_stress().unwrap();
    assert!(stress_after < stress_before * 0.5);
    assert!((stress_after - sm_run.compute_stress(&sm_l_run).unwrap()).abs() < 1e-5);

    sm_step.run_steps(&mut sm_l_step, 4.0).unwrap();
    while !sm_step.is_finished() {
        sm_step.step(&mut sm_l_step).unwrap();
    }
    assert_layouts_bit_exact(&sm_l_run, &sm_l_step);

    let sm_dm = JsStressMajorizationSolver::from_distance_matrix(
        2.0,
        &[0.0, 2.0, 2.0, 0.0],
        &JsValue::UNDEFINED,
        Some(15.0),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(sm_dm.num_nodes(), 2);

    // 2. KamadaKawaiSolver (fromGraph, fromDistanceMatrix, step == run)
    let kk_opts = make_opts(&[
        ("eps", JsValue::from_f64(1e-2)),
        ("maxInnerIters", JsValue::from_f64(50.0)),
        ("maxOuterIters", JsValue::from_f64(100.0)),
    ]);
    let mut kk_run = JsKamadaKawaiSolver::from_graph(&small_grid, &kk_opts, None, None).unwrap();
    let mut kk_step = JsKamadaKawaiSolver::from_graph(&small_grid, &kk_opts, None, None).unwrap();
    let mut kk_l_run = make_circle_layout(9, 1.5);
    let mut kk_l_step = kk_l_run.clone();
    kk_run.run(&mut kk_l_run).unwrap();
    kk_step.run_steps(&mut kk_l_step, 5.0).unwrap();
    while !kk_step.is_finished() {
        kk_step.step(&mut kk_l_step).unwrap();
    }
    assert_layouts_bit_exact(&kk_l_run, &kk_l_step);

    let mut kk_dm = JsKamadaKawaiSolver::from_distance_matrix(
        2.0,
        &[0.0, 3.0, 3.0, 0.0],
        &JsValue::UNDEFINED,
        None,
        Some(20.0),
    )
    .unwrap();
    let mut kk_l2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    kk_dm.run(&mut kk_l2).unwrap();
    let sep = (kk_l2.x(0.0).unwrap() - kk_l2.x(1.0).unwrap())
        .hypot(kk_l2.y(0.0).unwrap() - kk_l2.y(1.0).unwrap());
    assert!((sep - 3.0).abs() < 0.05);

    // Adversarial checks
    assert!(
        JsStressMajorizationSolver::from_distance_matrix(
            2.0,
            &[0.0, f64::NAN, 1.0, 0.0],
            &JsValue::UNDEFINED,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        JsKamadaKawaiSolver::from_distance_matrix(
            2.0,
            &[0.0, -1.0, 1.0, 0.0],
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn maxent_stress_step_eq_run_and_getters() {
    let (graph, _) = make_two_cluster_graph();
    let star = make_star_graph(7);
    let init = make_circle_layout(8, 1.5);

    // Auto-q rule: two-cluster has 0% deg-1 (q=0.0), star has >30% deg-1 (q=0.8)
    let auto_opts = make_opts(&[
        ("stages", JsValue::from_f64(2.0)),
        ("stepsPerStage", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let s_auto_clique = JsMaxentStressSolver::new(&graph, &auto_opts).unwrap();
    assert!((s_auto_clique.effective_q() - 0.0).abs() < 1e-12);
    let s_auto_star = JsMaxentStressSolver::new(&star, &auto_opts).unwrap();
    assert!((s_auto_star.effective_q() - 0.8).abs() < 1e-12);

    for use_bh in [true, false] {
        let opts = make_opts(&[
            ("q", JsValue::from_f64(0.4)),
            ("alphaInit", JsValue::from_f64(1.0)),
            ("alphaDecay", JsValue::from_f64(0.3)),
            ("alphaMin", JsValue::from_f64(0.008)),
            ("stages", JsValue::from_f64(2.0)),
            ("stepsPerStage", JsValue::from_f64(5.0)),
            ("tol", JsValue::from_f64(0.0)),
            ("cgTol", JsValue::from_f64(0.1)),
            ("cgMaxIters", JsValue::from_f64(10.0)),
            ("theta", JsValue::from_f64(1.2)),
            ("useBarnesHut", JsValue::from_bool(use_bh)),
            ("epsSq", JsValue::from_f64(1e-6)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsMaxentStressSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsMaxentStressSolver::new(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), 10);
        assert!((s_step.effective_q() - 0.4).abs() < 1e-12);
        let alpha_0 = s_step.current_alpha();
        assert!((alpha_0 - 1.0).abs() < 1e-12);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        let status = s_run.run(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(s_run.current_alpha() <= alpha_0);
        assert!(l_run.is_all_finite());
    }

    // Disconnected, N=0, and invalid option checks
    let disc = make_disconnected_graph();
    let mut s_disc = JsMaxentStressSolver::new(&disc, &auto_opts).unwrap();
    let mut l_disc = make_circle_layout(7, 1.0);
    s_disc.run(&mut l_disc).unwrap();
    assert!(l_disc.is_all_finite());

    let g0 = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let mut s0 = JsMaxentStressSolver::new(&g0, &JsValue::UNDEFINED).unwrap();
    let mut l0 = JsLayout2d::zeros(0.0).unwrap();
    s0.run(&mut l0).unwrap();

    let bad_q = make_opts(&[("q", JsValue::from_f64(-2.0))]);
    assert!(JsMaxentStressSolver::new(&graph, &bad_q).is_err());
}
