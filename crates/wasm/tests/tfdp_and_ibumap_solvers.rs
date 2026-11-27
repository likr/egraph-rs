mod util;

use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::ib_umap::JsIbUmapSolver;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::neighborhood_knn::{js_build_pure_2hop_knn, js_compute_fuzzy_simplicial_set};
use egraph_wasm::t_fdp::JsTFdpSolver;
use util::fixtures::{assert_layouts_bit_exact, make_circle_layout, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn t_fdp_solver_all_backends_from_knn_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    for backend in ["ibfft", "barnes_hut", "exact"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("iterations", JsValue::from_f64(12.0)),
            ("learningRate", JsValue::from_f64(0.5)),
            ("momentum", JsValue::from_f64(0.75)),
            ("rhoStart", JsValue::from_f64(1.2)),
            ("rhoEnd", JsValue::from_f64(0.2)),
            ("rhoPower", JsValue::from_f64(1.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsTFdpSolver::new(&grid, &opts).expect("TFdpSolver::new");
        let mut s_step = JsTFdpSolver::new(&grid, &opts).expect("TFdpSolver::new step");
        assert_eq!(s_run.status(), "initialized");
        assert_eq!(s_run.num_nodes(), 16);
        assert_eq!(s_run.total_iterations(), 12);
        assert!(s_run.num_edges() > 0);
        assert!((s_run.current_rho() - 1.2).abs() < 1e-12);

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
        assert!(s_run.is_finished());
        assert!((s_run.current_rho() - 0.2).abs() < 1e-12);
        assert!(l_run.is_all_finite());

        assert_eq!(s_step.run_steps(&mut l_step, 5.0).unwrap(), "running");
        assert_eq!(s_step.iteration(), 5);
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_eq!(s_step.status(), "completed");
        assert_layouts_bit_exact(&l_run, &l_step);
        if backend == "ibfft" {
            assert!(s_run.total_fft_2d_calls() > 0.0);
        } else {
            assert_eq!(s_run.total_fft_2d_calls(), 0.0);
        }
    }

    // TFdpSolver.fromKnn with raw CSR arrays (both omitted weights and explicit Float64Array)
    let indptr = [0u32, 1, 3, 4];
    let indices = [1u32, 0, 2, 1];
    let knn_opts = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("iterations", JsValue::from_f64(8.0)),
    ]);
    let indptr_js: JsValue = js_sys::Uint32Array::from(&indptr[..]).into();
    let indices_js: JsValue = js_sys::Uint32Array::from(&indices[..]).into();
    let mut k_default =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &JsValue::UNDEFINED, &knn_opts).unwrap();
    let mut k_opts_short =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &knn_opts, &JsValue::UNDEFINED).unwrap();
    let weights_arr = js_sys::Float64Array::from(&[0.5_f64, 0.5, 1.0, 1.0][..]);
    let mut k_weighted =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &weights_arr, &knn_opts).unwrap();
    let mut l3_a = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
    let mut l3_b = l3_a.clone();
    let mut l3_c = l3_a.clone();
    assert_eq!(k_default.run(&mut l3_a).unwrap(), "completed");
    assert_eq!(k_opts_short.run(&mut l3_b).unwrap(), "completed");
    assert_layouts_bit_exact(&l3_a, &l3_b);
    assert_eq!(k_weighted.run(&mut l3_c).unwrap(), "completed");
    assert!(l3_c.is_all_finite());

    // Adversarial checks
    assert!(
        JsTFdpSolver::new(
            &grid,
            &make_opts(&[("backend", JsValue::from_str("unknown"))])
        )
        .is_err()
    );
    assert!(JsTFdpSolver::new(&grid, &JsValue::from_str("bad")).is_err());
    assert!(JsTFdpSolver::new(&grid, &make_opts(&[("alpha", JsValue::from_f64(0.0))])).is_err());
    let empty_u32: JsValue = js_sys::Uint32Array::new_with_length(0).into();
    assert!(
        JsTFdpSolver::from_knn(
            &empty_u32,
            &empty_u32,
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED
        )
        .is_err()
    );
    let mut bad_layout = JsLayout2d::zeros(2.0).unwrap();
    assert!(k_default.step(&mut bad_layout).is_err());
}

#[wasm_bindgen_test]
fn ib_umap_solver_all_constructors_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    for backend in ["ibfft", "barnes_hut", "exact"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("max2hopPerNode", JsValue::from_f64(16.0)),
            ("iterations", JsValue::from_f64(10.0)),
            ("cK", JsValue::from_f64(4.0)),
            ("tauScale", JsValue::from_f64(4.0)),
            ("hubDamping", JsValue::from_bool(true)),
            ("degree", JsValue::from_f64(6.0)),
            ("numLandmarks", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsIbUmapSolver::new(&grid, &opts).expect("IbUmapSolver::new");
        let mut s_step = JsIbUmapSolver::new(&grid, &opts).expect("IbUmapSolver::new step");
        assert_eq!(s_run.num_nodes(), 16);
        assert_eq!(s_run.total_iterations(), 10);
        assert_eq!(s_run.current_rho(), 1.0);
        assert!(s_run.a() > 0.0 && s_run.b() > 0.0);
        assert!(s_run.num_edges() > 0);

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
        assert!(l_run.is_all_finite());
        assert_eq!(s_step.run_steps(&mut l_step, 4.0).unwrap(), "running");
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_layouts_bit_exact(&l_run, &l_step);
    }

    // fromKnn and fromFuzzySet (with and without explicit degrees)
    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let fuzzy = js_compute_fuzzy_simplicial_set(&knn, None, None).unwrap();
    let opts = make_opts(&[
        ("backend", JsValue::from_str("barnes_hut")),
        ("iterations", JsValue::from_f64(6.0)),
    ]);
    let mut s_knn = JsIbUmapSolver::from_knn(&knn, &opts).unwrap();
    let mut s_fuz_a = JsIbUmapSolver::from_fuzzy_set(&fuzzy, &opts, &JsValue::UNDEFINED).unwrap();
    let degs = js_sys::Uint32Array::from(&vec![3u32; 16][..]);
    let mut s_fuz_b = JsIbUmapSolver::from_fuzzy_set(&fuzzy, &degs, &opts).unwrap();

    let mut l_a = init.clone();
    let mut l_b = init.clone();
    let mut l_c = init.clone();
    assert_eq!(s_knn.run(&mut l_a).unwrap(), "completed");
    assert_eq!(s_fuz_a.run(&mut l_b).unwrap(), "completed");
    assert_eq!(s_fuz_b.run(&mut l_c).unwrap(), "completed");
    assert_layouts_bit_exact(&l_a, &l_b);
    assert!(l_c.is_all_finite());

    // fromLandmarks (both options-object and positional max2hopPerNode)
    let lm = js_compute_regularized_dog_varadhan_landmarks(
        &grid,
        Some(6.0),
        Some(8.0),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let ib_lm_opts = make_opts(&[
        ("max2hopPerNode", JsValue::from_f64(32.0)),
        ("backend", JsValue::from_str("exact")),
        ("iterations", JsValue::from_f64(6.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut s_lm_a =
        JsIbUmapSolver::from_landmarks(&grid, &lm, &ib_lm_opts, &JsValue::UNDEFINED).unwrap();
    let mut s_lm_b = JsIbUmapSolver::from_landmarks(
        &grid,
        &lm,
        &JsValue::from_f64(32.0),
        &make_opts(&[
            ("backend", JsValue::from_str("exact")),
            ("iterations", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_lm_a = init.clone();
    let mut l_lm_b = init;
    assert_eq!(s_lm_a.run(&mut l_lm_a).unwrap(), "completed");
    while !s_lm_b.is_finished() {
        s_lm_b.step(&mut l_lm_b).unwrap();
    }
    assert_layouts_bit_exact(&l_lm_a, &l_lm_b);

    let lm_bad = js_compute_regularized_dog_varadhan_landmarks(
        &make_grid(2, 2),
        None,
        Some(2.0),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        JsIbUmapSolver::from_landmarks(&grid, &lm_bad, &JsValue::UNDEFINED, &JsValue::UNDEFINED)
            .is_err()
    );

    // Adversarial checks
    assert!(JsIbUmapSolver::new(&grid, &make_opts(&[("cK", JsValue::from_f64(-1.0))])).is_err());
    assert!(JsIbUmapSolver::from_fuzzy_set(&fuzzy, &opts, &opts).is_err());
}
