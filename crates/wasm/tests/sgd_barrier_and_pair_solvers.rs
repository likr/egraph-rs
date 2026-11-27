mod util;

use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::linalg_chebyshev::js_chebyshev_dog_jl_embedding;
use egraph_wasm::linalg_spectral::js_pivot_mds_init;
use egraph_wasm::sgd::JsSgdSolver;
use util::fixtures::{
    assert_layouts_bit_exact, make_circle_layout, make_grid, make_opts, make_two_cluster_graph,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn sgd_solver_all_constructors_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = js_pivot_mds_init(&grid, None, None, None, &JsValue::UNDEFINED).unwrap();

    // 1. sparseBarrierDog + stats() + step() == run() + in-place coordsPtr stability
    let dog_opts = make_opts(&[
        ("iterations", JsValue::from_f64(25.0)),
        ("hingeRepulsion", JsValue::from_bool(true)),
        ("barrierSym", JsValue::from_str("sym_avg")),
        ("kind", JsValue::from_str("exponential")),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut s_run = JsSgdSolver::sparse_barrier_dog(
        &grid, &dog_opts, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None,
    )
    .expect("sparseBarrierDog");
    let mut s_step = JsSgdSolver::sparse_barrier_dog(
        &grid, &dog_opts, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None,
    )
    .expect("sparseBarrierDog step");
    let st = s_run.stats().expect("barrier stats");
    assert!(st.n_pairs_1hop() > 0);
    assert!(st.n_pairs_2hop() > 0);
    assert!(st.n_pairs_hinge() > 0);
    assert_eq!(st.perplexity_source(), "adaptive_2hop");
    assert!(s_run.num_springs() > 0);
    assert!(s_run.num_hinges() > 0);
    assert!(s_run.current_eta() > 0.0);

    let mut l_run = init.clone();
    let mut l_step = init.clone();
    let ptr_before = l_step.coords_ptr();
    assert_eq!(s_run.run_2d(&mut l_run).unwrap(), "completed");
    assert_eq!(s_run.current_eta(), 0.0);
    assert_eq!(s_step.run_steps_2d(&mut l_step, 7.0).unwrap(), "running");
    while !s_step.is_finished() {
        s_step.step_2d(&mut l_step).unwrap();
    }
    assert_eq!(l_step.coords_ptr(), ptr_before);
    assert_layouts_bit_exact(&l_run, &l_step);

    // 2. sparseBarrierFromLandmarks
    let lm = js_compute_regularized_dog_varadhan_landmarks(
        &grid, None, None, None, None, None, None, None, None, None, None, None,
    )
    .unwrap();
    let s_lm = JsSgdSolver::sparse_barrier_from_landmarks(
        &grid,
        &lm,
        Some(vec![0, 5, 10, 15]),
        &JsValue::from_f64(16.0),
        None,
        None,
        Some("symMax".to_owned()),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(4.0),
        None,
        None,
        Some(10.0),
        None,
        Some("linear".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    assert!(s_lm.stats().is_some());
    assert_eq!(s_lm.stats().unwrap().perplexity_source(), "explicit");

    // 3. randomPair, omega, streamingNegatives, fromPairs, fromPairsWithHinges
    let s_rp = JsSgdSolver::random_pair(
        &grid,
        &JsValue::from_f64(6.0),
        &JsValue::from_f64(10.0),
        None,
        Some("quadratic".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    assert!(s_rp.stats().is_none());

    let emb =
        js_chebyshev_dog_jl_embedding(&grid, Some(8.0), None, None, None, None, None, None, None)
            .unwrap();
    let mut s_om = JsSgdSolver::omega(
        &grid,
        &emb,
        8.0,
        &JsValue::from_f64(5.0),
        None,
        &JsValue::from_f64(10.0),
        None,
        Some("reciprocal".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    let mut l_om = init.clone();
    assert_eq!(s_om.run_2d(&mut l_om).unwrap(), "completed");

    let mut s_sn = JsSgdSolver::streaming_negatives(
        &grid,
        &JsValue::from_f64(4.0),
        Some(2.5),
        Some(0.25),
        Some(true),
        Some(10.0),
        None,
        Some("constant".to_owned()),
        Some(0.5),
        Some(42.0),
    )
    .unwrap();
    let mut l_sn = init.clone();
    assert_eq!(s_sn.run_2d(&mut l_sn).unwrap(), "completed");

    let mut s_pairs = JsSgdSolver::from_pairs(
        2.0,
        &[0],
        &[1],
        &[1.0],
        Some(vec![1.0]),
        Some(vec![1.0]),
        Some(vec![1.0]),
        &JsValue::from_f64(5.0),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l2 = JsLayout2d::from_coords(&[0.0, 0.0, 3.0, 0.0]).unwrap();
    assert_eq!(s_pairs.run_2d(&mut l2).unwrap(), "completed");

    let mut s_hinges = JsSgdSolver::from_pairs_with_hinges(
        2.0,
        &[],
        &[],
        &[],
        None,
        None,
        None,
        &[0],
        &[1],
        &[2.0],
        Some(vec![2.0]),
        Some(vec![1.0]),
        Some(vec![1.0]),
        &JsValue::from_f64(5.0),
        None,
        Some("constant".to_owned()),
        Some(0.5),
        None,
    )
    .unwrap();
    let mut l_hinge = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    s_hinges.step_2d(&mut l_hinge).unwrap();
    assert!((l_hinge.x(1.0).unwrap() - l_hinge.x(0.0).unwrap()) > 0.5);

    // Invalid schedule / dimensions / NaN errors
    assert!(
        JsSgdSolver::full(
            &grid,
            &JsValue::from_f64(10.0),
            None,
            Some("bad_kind".to_owned()),
            None,
            None,
        )
        .is_err()
    );
    let mut bad_layout = JsLayout2d::zeros(3.0).unwrap();
    assert!(s_om.step_2d(&mut bad_layout).is_err());
    assert!(s_om.run_2d(&mut bad_layout).is_err());
}

#[wasm_bindgen_test]
fn ortmann_sparse_stress_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
    let init = make_circle_layout(8, 1.5);

    for strategy in ["kmeans_sp", "maxmin_random_sp", "maxmin_sp"] {
        let opts = make_opts(&[
            ("numPivots", JsValue::from_f64(4.0)),
            ("pivotStrategy", JsValue::from_str(strategy)),
            ("iterations", JsValue::from_f64(12.0)),
            ("eps", JsValue::from_f64(0.1)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsSgdSolver::ortmann_sparse_stress(&graph, &opts).unwrap();
        let mut s_run = JsSgdSolver::ortmann_sparse_stress(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), 12);
        assert!(s_step.num_springs() > 0);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step_2d(&mut l_step).unwrap();
        }
        let status = s_run.run_2d(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(l_run.is_all_finite());
    }

    let opts_eta = make_opts(&[
        ("numPivots", JsValue::from_f64(4.0)),
        ("iterations", JsValue::from_f64(6.0)),
        ("etaMax", JsValue::from_f64(0.8)),
        ("etaMin", JsValue::from_f64(0.05)),
        ("seed", JsValue::from_f64(7.0)),
    ]);
    let mut s_steps = JsSgdSolver::ortmann_sparse_stress(&graph, &opts_eta).unwrap();
    let mut l_steps = init;
    s_steps.run_steps_2d(&mut l_steps, 3.0).unwrap();
    assert_eq!(s_steps.iteration(), 3);
    s_steps.run_2d(&mut l_steps).unwrap();
    assert!(s_steps.is_finished());

    let bad_pivots = make_opts(&[("numPivots", JsValue::from_f64(0.0))]);
    assert!(JsSgdSolver::ortmann_sparse_stress(&graph, &bad_pivots).is_err());

    let bad_strat = make_opts(&[("pivotStrategy", JsValue::from_str("unknown"))]);
    assert!(JsSgdSolver::ortmann_sparse_stress(&graph, &bad_strat).is_err());
}

#[wasm_bindgen_test]
fn largevis_drgraph_graph_cne_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
    let init = make_circle_layout(8, 1.5);

    // LargeVis
    let lv_opts = make_opts(&[
        ("iterations", JsValue::from_f64(10.0)),
        ("negativeSamples", JsValue::from_f64(3.0)),
        ("gamma", JsValue::from_f64(7.0)),
        ("initialLr", JsValue::from_f64(1.0)),
        ("minLr", JsValue::from_f64(1e-4)),
        ("degreeExponent", JsValue::from_f64(0.75)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut lv_step = JsSgdSolver::largevis(&graph, &lv_opts).unwrap();
    let mut lv_run = JsSgdSolver::largevis(&graph, &lv_opts).unwrap();
    let mut l_lv_step = init.clone();
    let mut l_lv_run = init.clone();
    while !lv_step.is_finished() {
        lv_step.step_2d(&mut l_lv_step).unwrap();
    }
    lv_run.run_2d(&mut l_lv_run).unwrap();
    assert_layouts_bit_exact(&l_lv_step, &l_lv_run);
    assert!(l_lv_run.is_all_finite());

    // DRGraph
    let dr_opts = make_opts(&[
        ("kOrder", JsValue::from_f64(1.0)),
        ("negativeSamples", JsValue::from_f64(3.0)),
        ("b", JsValue::from_f64(2.0)),
        ("gammaCoarse", JsValue::from_f64(0.01)),
        ("gammaFine", JsValue::from_f64(0.1)),
        ("coarsenStopRatio", JsValue::from_f64(0.8)),
        ("iterations", JsValue::from_f64(12.0)),
        ("initialLr", JsValue::from_f64(1.0)),
        ("minLr", JsValue::from_f64(1e-4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut dr_step = JsSgdSolver::drgraph(&graph, &dr_opts).unwrap();
    let mut dr_run = JsSgdSolver::drgraph(&graph, &dr_opts).unwrap();
    let mut l_dr_step = init.clone();
    let mut l_dr_run = init.clone();
    while !dr_step.is_finished() {
        dr_step.step_2d(&mut l_dr_step).unwrap();
    }
    dr_run.run_2d(&mut l_dr_run).unwrap();
    assert_layouts_bit_exact(&l_dr_step, &l_dr_run);
    assert!(l_dr_run.is_all_finite());

    // Graph-CNE ("infonce" and "neg_tsne")
    for obj in ["infonce", "neg_tsne"] {
        let cne_opts = make_opts(&[
            ("objective", JsValue::from_str(obj)),
            ("negativeSamples", JsValue::from_f64(3.0)),
            ("iterations", JsValue::from_f64(10.0)),
            ("initialLr", JsValue::from_f64(1.0)),
            ("minLr", JsValue::from_f64(1e-4)),
            ("repulsionWeight", JsValue::from_f64(1.0)),
            ("rowNormalize", JsValue::from_bool(true)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut cne_step = JsSgdSolver::graph_cne(&graph, &cne_opts).unwrap();
        let mut cne_run = JsSgdSolver::graph_cne(&graph, &cne_opts).unwrap();
        let mut l_cne_step = init.clone();
        let mut l_cne_run = init.clone();
        while !cne_step.is_finished() {
            cne_step.step_2d(&mut l_cne_step).unwrap();
        }
        cne_run.run_2d(&mut l_cne_run).unwrap();
        assert_layouts_bit_exact(&l_cne_step, &l_cne_run);
        assert!(l_cne_run.is_all_finite());
    }

    // Invalid option errors
    assert!(
        JsSgdSolver::largevis(&graph, &make_opts(&[("gamma", JsValue::from_f64(-1.0))])).is_err()
    );
    assert!(
        JsSgdSolver::drgraph(&graph, &make_opts(&[("kOrder", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(
        JsSgdSolver::graph_cne(
            &graph,
            &make_opts(&[("objective", JsValue::from_str("bad"))]),
        )
        .is_err()
    );
}
