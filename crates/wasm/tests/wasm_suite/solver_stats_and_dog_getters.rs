use crate::util;

use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::gumap::JsGuMapSolver;
use egraph_wasm::ib_umap::JsIbUmapSolver;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::linalg_spectral::js_classical_mds_2d;
use egraph_wasm::neighborhood_knn::js_build_pure_2hop_knn;
use egraph_wasm::sgd::JsSgdSolver;
use egraph_wasm::t_fdp::JsTFdpSolver;
use egraph_wasm::tsnet::JsTsNetSolver;
use util::fixtures::{make_grid, make_opts, make_path_graph};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn solvers_fft_stats_dog_and_ab_getters() {
    let grid = make_grid(4, 4);
    let init = js_classical_mds_2d(&grid).unwrap();

    // TsNetSolver.fromGraphPartialBfs + lastFftStats + short FFT options
    let ts_opts = make_opts(&[
        ("k", JsValue::from_f64(8.0)),
        ("uniformPerplexity", JsValue::from_f64(6.0)),
        ("backend", JsValue::from_str("fitFft")),
        (
            "stageIterations",
            js_sys::Array::of3(
                &JsValue::from_f64(2.0),
                &JsValue::from_f64(2.0),
                &JsValue::from_f64(2.0),
            )
            .into(),
        ),
        ("intervals", JsValue::from_f64(1.0)),
        ("interpolationOrder", JsValue::from_f64(1.0)),
        ("minGridDim", JsValue::from_f64(16.0)),
        ("maxGridDim", JsValue::from_f64(64.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut ts = JsTsNetSolver::from_graph_partial_bfs(&grid, &ts_opts).unwrap();
    let mut l_ts = init.clone();
    assert!(ts.last_fft_stats().is_none());
    ts.run_2d(&mut l_ts).unwrap();
    let fft_stats = ts.last_fft_stats().expect("lastFftStats after FFT run");
    assert!(fft_stats.total_fft_2d_calls() > 0.0);
    assert!(fft_stats.particle_forward_ffts() > 0.0);

    // TFdpSolver.fromKnn accepting a KnnGraph instance directly
    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let tfdp_opts = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("iterations", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut tfdp = JsTFdpSolver::from_knn(
        &JsValue::from(knn.clone()),
        &tfdp_opts,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    let mut l_tfdp = init.clone();
    tfdp.run_2d(&mut l_tfdp).unwrap();
    assert!(l_tfdp.is_all_finite());

    // SgdSolver.dogStats + optional fromPairs args
    let mut sgd = JsSgdSolver::sparse_barrier_dog(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let d_stats = sgd.dog_stats().expect("dogStats from sparseBarrierDog");
    assert!(d_stats.degree() > 0);
    assert!(d_stats.t2() > d_stats.t1() && d_stats.t1() > 0.0);
    let mut l_sgd = init;
    sgd.run_2d(&mut l_sgd).unwrap();

    let mut pairs_opt = JsSgdSolver::from_pairs(
        2.0,
        &[0],
        &[1],
        &[1.5],
        &make_opts(&[("iterations", JsValue::from_f64(4.0))]),
    )
    .unwrap();
    let mut l2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    pairs_opt.run_2d(&mut l2).unwrap();
    assert!(l2.is_all_finite());

    // GuMapSolver.fromGraphDog minDegree/jlDim + fromFuzzySet sigmaIters/sigmaTolerance
    let path = make_path_graph(6);
    let gm_pos = JsGuMapSolver::from_graph_dog(
        &path,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("minDegree", JsValue::from_f64(2.0)),
            ("jlDim", JsValue::from_f64(8.0)),
        ]),
    )
    .unwrap();
    assert!(gm_pos.num_edges() > 0);
    let gm_ab = gm_pos.ab();
    assert_eq!(gm_ab.len(), 2);
    assert!((gm_ab[0] - gm_pos.a()).abs() < 1e-12);
    assert!((gm_ab[1] - gm_pos.b()).abs() < 1e-12);

    let ib = JsIbUmapSolver::new(
        &grid,
        &make_opts(&[
            ("backend", JsValue::from_str("exact")),
            ("iterations", JsValue::from_f64(3.0)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("jlDim", JsValue::from_f64(8.0)),
        ]),
    )
    .unwrap();
    let ib_ab = ib.ab();
    assert_eq!(ib_ab.len(), 2);
    assert!((ib_ab[0] - ib.a()).abs() < 1e-12);
    assert!((ib_ab[1] - ib.b()).abs() < 1e-12);

    // Default degree in computeRegularizedDogVaradhanLandmarks matches adaptive degree >= 40
    let lm_def = js_compute_regularized_dog_varadhan_landmarks(&grid, &JsValue::UNDEFINED).unwrap();
    assert_eq!(lm_def.num_nodes(), 16);
}
