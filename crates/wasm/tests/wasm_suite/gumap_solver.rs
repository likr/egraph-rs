use crate::util;

use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::gumap::JsGuMapSolver;
use egraph_wasm::neighborhood_knn::{js_build_pure_2hop_knn, js_compute_fuzzy_simplicial_set};
use egraph_wasm::neighborhood_types::JsKnnGraph;
use util::fixtures::{assert_layouts_bit_exact, make_circle_layout, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn gumap_solver_constructors_symmetric_negative_and_errors() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    // 1. GuMapSolver (fromGraphDog, fromKnn, fromFuzzySet, step == run)
    let gm_opts = make_opts(&[
        ("iterations", JsValue::from_f64(20.0)),
        ("negativeSampleRate", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_run = JsGuMapSolver::from_graph_dog(&grid, &gm_opts).expect("GuMap.fromGraphDog");
    let mut g_step =
        JsGuMapSolver::from_graph_dog(&grid, &gm_opts).expect("GuMap.fromGraphDog step");
    assert!(g_run.num_edges() > 0);
    assert!(g_run.a() > 0.0 && g_run.b() > 0.0);

    let mut gl_run = init.clone();
    let mut gl_step = init.clone();
    let ptr_before = gl_step.coords_ptr();
    assert_eq!(g_run.run_2d(&mut gl_run).unwrap(), "completed");
    assert_eq!(g_step.run_steps_2d(&mut gl_step, 5.0).unwrap(), "running");
    while !g_step.is_finished() {
        g_step.step_2d(&mut gl_step).unwrap();
    }
    assert_eq!(gl_step.coords_ptr(), ptr_before);
    assert_layouts_bit_exact(&gl_run, &gl_step);

    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let fuzzy = js_compute_fuzzy_simplicial_set(&knn, None, None).unwrap();
    let iter_opts = make_opts(&[("iterations", JsValue::from_f64(5.0))]);
    let g_knn = JsGuMapSolver::from_knn(&knn, &iter_opts).unwrap();
    let g_fuz = JsGuMapSolver::from_fuzzy_set(&fuzzy, &iter_opts).unwrap();
    assert_eq!(g_knn.num_edges(), g_fuz.num_edges());

    // GuMapSolver.fromLandmarks
    let lm_barbell = js_compute_regularized_dog_varadhan_landmarks(
        &grid,
        &make_opts(&[
            ("degree", JsValue::from_f64(12.0)),
            ("numLandmarks", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let g_lm_opts = make_opts(&[
        ("max2hopPerNode", JsValue::from_f64(32.0)),
        ("iterations", JsValue::from_f64(10.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_lm_run =
        JsGuMapSolver::from_landmarks(&grid, &lm_barbell, &g_lm_opts).expect("GuMap.fromLandmarks");
    let mut g_lm_step = JsGuMapSolver::from_landmarks(&grid, &lm_barbell, &g_lm_opts)
        .expect("GuMap.fromLandmarks step");
    let mut glm_l1 = init.clone();
    let mut glm_l2 = init.clone();
    assert_eq!(g_lm_run.run_2d(&mut glm_l1).unwrap(), "completed");
    while !g_lm_step.is_finished() {
        g_lm_step.step_2d(&mut glm_l2).unwrap();
    }
    assert_layouts_bit_exact(&glm_l1, &glm_l2);

    // symmetricNegative=false diverges from true while keeping step == run across fromGraphDog & fromLandmarks
    let gm_asym_opts = make_opts(&[
        ("iterations", JsValue::from_f64(20.0)),
        ("negativeSampleRate", JsValue::from_f64(4.0)),
        ("symmetricNegative", JsValue::from_bool(false)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_asym_r = JsGuMapSolver::from_graph_dog(&grid, &gm_asym_opts).unwrap();
    let mut g_asym_s = JsGuMapSolver::from_graph_dog(&grid, &gm_asym_opts).unwrap();
    let mut gl_asym_r = init.clone();
    let mut gl_asym_s = init.clone();
    assert_eq!(g_asym_r.run_2d(&mut gl_asym_r).unwrap(), "completed");
    while !g_asym_s.is_finished() {
        g_asym_s.step_2d(&mut gl_asym_s).unwrap();
    }
    assert_layouts_bit_exact(&gl_asym_r, &gl_asym_s);
    let max_dog_diff = gl_run
        .copy_coords()
        .iter()
        .zip(gl_asym_r.copy_coords().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_dog_diff > 1e-3,
        "expected symmetricNegative=false to diverge on fromGraphDog"
    );

    let g_lm_asym_opts = make_opts(&[
        ("max2hopPerNode", JsValue::from_f64(32.0)),
        ("iterations", JsValue::from_f64(10.0)),
        ("symmetricNegative", JsValue::from_bool(false)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_lm_asym = JsGuMapSolver::from_landmarks(&grid, &lm_barbell, &g_lm_asym_opts).unwrap();
    let mut glm_asym_l = init;
    assert_eq!(g_lm_asym.run_2d(&mut glm_asym_l).unwrap(), "completed");
    let max_lm_diff = glm_l1
        .copy_coords()
        .iter()
        .zip(glm_asym_l.copy_coords().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_lm_diff > 1e-3,
        "expected symmetricNegative=false to diverge on fromLandmarks"
    );
    assert!(
        JsGuMapSolver::from_landmarks(
            &grid,
            &lm_barbell,
            &make_opts(&[("symmetricNegative", JsValue::from_str("nope"))]),
        )
        .is_err()
    );
    let lm_small = js_compute_regularized_dog_varadhan_landmarks(
        &make_grid(2, 2),
        &make_opts(&[
            ("numLandmarks", JsValue::from_f64(2.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(JsGuMapSolver::from_landmarks(&grid, &lm_small, &JsValue::UNDEFINED).is_err());
    assert!(
        JsGuMapSolver::from_knn(
            &knn,
            &make_opts(&[
                ("iterations", JsValue::from_f64(5.0)),
                ("learningRate", JsValue::from_f64(0.0)),
            ]),
        )
        .is_err()
    );
    assert!(JsKnnGraph::from_csr(2.0, &[0, 0, 0], &[], &[]).is_ok());
}
