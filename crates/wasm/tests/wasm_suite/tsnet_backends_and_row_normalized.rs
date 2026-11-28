use crate::util;

use egraph_wasm::neighborhood_knn::{
    js_build_pure_2hop_knn, js_compute_knn_probabilities, js_compute_nodewise_2hop_perplexities,
    js_graph_row_normalized_probabilities,
};
use egraph_wasm::tsnet::JsTsNetSolver;
use util::fixtures::{assert_layouts_bit_exact, make_circle_layout, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn tsnet_pure_2hop_and_row_normalized_backends() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 0.5);
    let stages_4 = js_sys::Array::of3(
        &JsValue::from_f64(4.0),
        &JsValue::from_f64(4.0),
        &JsValue::from_f64(4.0),
    );
    let stages_2 = js_sys::Array::of3(
        &JsValue::from_f64(2.0),
        &JsValue::from_f64(2.0),
        &JsValue::from_f64(2.0),
    );

    for backend in ["exact", "barnesHut", "fitFft", "linearFft"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("stageIterations", stages_4.clone().into()),
            ("lambdaRStage3", JsValue::from_f64(0.5)),
            ("learningRate", JsValue::from_f64(25.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsTsNetSolver::pure_2hop(&grid, &opts).expect("tsnet pure2Hop");
        let mut s_step = JsTsNetSolver::pure_2hop(&grid, &opts).expect("tsnet pure2Hop step");

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.status(), "initialized");
        assert!(s_run.sampling_mode().is_none());
        assert_eq!(s_run.total_iterations(), Some(12));
        assert_eq!(s_run.run_2d(&mut l_run).unwrap(), "completed");
        assert!(s_run.is_finished());

        assert_eq!(s_step.run_steps_2d(&mut l_step, 5.0).unwrap(), "running");
        assert_eq!(s_step.iteration(), 5);
        assert_eq!(s_step.stage_index(), 1);
        while !s_step.is_finished() {
            s_step.step_2d(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_eq!(s_step.status(), "completed");
        assert_eq!(s_step.total_fft_2d_calls(), s_run.total_fft_2d_calls());
        assert_layouts_bit_exact(&l_run, &l_step);
        if matches!(backend, "fitFft" | "linearFft") {
            assert!(s_run.total_fft_2d_calls() > 0.0);
        } else {
            assert_eq!(s_run.total_fft_2d_calls(), 0.0);
        }
    }

    // fromKnn and fromProbabilities
    let pure = js_build_pure_2hop_knn(&grid).unwrap();
    let knn = pure.knn();
    let perps = js_compute_nodewise_2hop_perplexities(&grid, &JsValue::UNDEFINED).unwrap();
    let probs = js_compute_knn_probabilities(&knn, &perps, &JsValue::UNDEFINED).unwrap();

    let s_knn = JsTsNetSolver::from_knn(
        &knn,
        &make_opts(&[
            ("perplexities", perps.into()),
            ("backend", JsValue::from_str("fitFft")),
            ("stageIterations", stages_2.clone().into()),
            ("learningRate", JsValue::from_f64(20.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let s_probs = JsTsNetSolver::from_probabilities(
        &probs,
        &make_opts(&[
            ("backend", JsValue::from_str("fitFft")),
            ("stageIterations", stages_2.clone().into()),
            ("learningRate", JsValue::from_f64(20.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(s_knn.num_nodes(), 16);
    assert_eq!(s_probs.total_iterations(), Some(6));

    // Optional learningRate auto-scaling & Stage 3 LR cap (2 * N for N < 100) + fromGraphRowNormalized
    let rn_opts = make_opts(&[
        ("w2", JsValue::from_f64(0.5)),
        ("max2hopPerNode", JsValue::from_f64(8.0)),
        ("degreeBeta", JsValue::from_f64(0.35)),
        ("backend", JsValue::from_str("fitFft")),
        ("stageIterations", stages_2.clone().into()),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut rn_run =
        JsTsNetSolver::from_graph_row_normalized(&grid, &rn_opts).expect("fromGraphRowNormalized");
    assert_eq!(rn_run.sampling_mode().as_deref(), Some("twoStepRandomWalk"));
    let mut rn_step = JsTsNetSolver::from_graph_row_normalized(&grid, &rn_opts)
        .expect("fromGraphRowNormalized step");
    assert_eq!(
        rn_step.sampling_mode().as_deref(),
        Some("twoStepRandomWalk")
    );
    let mut lrn_run = init.clone();
    let mut lrn_step = init.clone();
    assert_eq!(rn_run.run_2d(&mut lrn_run).unwrap(), "completed");
    while !rn_step.is_finished() {
        rn_step.step_2d(&mut lrn_step).unwrap();
    }
    assert_layouts_bit_exact(&lrn_run, &lrn_step);
    assert!(rn_run.current_stage_lr() <= 32.0);

    // Explicit samplingMode override in fromGraphRowNormalized
    let rn_deg_opts = make_opts(&[
        ("w2", JsValue::from_f64(0.5)),
        (
            "samplingMode",
            JsValue::from_str("degreeOrderedHubSuppressed"),
        ),
        ("maxBridgeDegree", JsValue::from_f64(4.0)),
        ("backend", JsValue::from_str("exact")),
        ("stageIterations", stages_2.clone().into()),
    ]);
    let rn_deg = JsTsNetSolver::from_graph_row_normalized(&grid, &rn_deg_opts).unwrap();
    assert_eq!(
        rn_deg.sampling_mode().as_deref(),
        Some("degreeOrderedHubSuppressed")
    );

    // w2 = 0.0 parity with graphRowNormalizedProbabilities + fromProbabilities
    let probs_1hop = js_graph_row_normalized_probabilities(&grid).unwrap();
    let mut s_rn0 = JsTsNetSolver::from_graph_row_normalized(
        &grid,
        &make_opts(&[
            ("w2", JsValue::from_f64(0.0)),
            ("backend", JsValue::from_str("fitFft")),
            ("stageIterations", stages_2.clone().into()),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut s_p0 = JsTsNetSolver::from_probabilities(
        &probs_1hop,
        &make_opts(&[
            ("backend", JsValue::from_str("fitFft")),
            ("stageIterations", stages_2.into()),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut lp0 = init.clone();
    let mut lr0 = init;
    s_rn0.run_2d(&mut lr0).unwrap();
    s_p0.run_2d(&mut lp0).unwrap();
    assert_layouts_bit_exact(&lr0, &lp0);
    assert!(JsTsNetSolver::from_graph_row_normalized(&grid, &JsValue::from_f64(-0.1)).is_err());
}
