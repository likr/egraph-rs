use crate::util;

use egraph_wasm::edge_polylines::JsEdgePolylines;
use egraph_wasm::fdeb::JsFdebSolver;
use egraph_wasm::force_classic::JsLinLogSolver;
use egraph_wasm::force_fr::JsFruchtermanReingoldSolver;
use egraph_wasm::force_multilevel::{JsForceAtlas2Solver, JsYifanHuSolver};
use egraph_wasm::gumap::JsGuMapSolver;
use egraph_wasm::layout_buffers::JsLayoutNd;
use egraph_wasm::layout3d::JsLayout3d;
use egraph_wasm::manifold::JsManifoldLayout2d;
use egraph_wasm::manifold_geodesic::{
    js_lorentz_distance, js_lorentz_einstein_midpoint, js_lorentz_exp_map, js_lorentz_log_map,
    js_lorentz_to_poincare_point, js_poincare_einstein_midpoint, js_torus_spectral_phase_init,
};
use egraph_wasm::mds_generators::{JsClassicalMds, JsHde, JsKorenSpectral, JsPivotMds, JsRdMds};
use egraph_wasm::metrics_3d::{js_np2_3d, js_sns_3d};
use egraph_wasm::metrics_eval_dispatch::{
    js_evaluate_metrics_3d, js_evaluate_metrics_manifold, js_evaluate_metrics_nd,
};
use egraph_wasm::metrics_nd_manifold::{
    js_layout_degeneracy_manifold, js_layout_degeneracy_nd, js_np2_manifold, js_np2_nd,
    js_sns_manifold, js_sns_nd,
};
use egraph_wasm::metrics_non_euclidean_crossings::{
    js_crossing_edges_hyperbolic, js_crossing_edges_spherical, js_crossing_edges_torus_manifold,
    js_hyperbolic_crossing_number, js_spherical_crossing_number, js_torus_crossing_number_manifold,
    js_torus_edge_segments, js_torus_layout_edge_segments_manifold,
};
use egraph_wasm::sgd::JsSgdSolver;
use egraph_wasm::stress_kamada::JsStressMajorizationSolver;
use egraph_wasm::stress_kamada_kawai::JsKamadaKawaiSolver;
use egraph_wasm::stress_multilevel_baselines::JsMaxentStressSolver;
use egraph_wasm::tsnet::JsTsNetSolver;
use util::fixtures::{make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn nd_and_3d_solvers_generators_and_metrics() {
    let grid = make_grid(4, 4);
    let init_opts = make_opts(&[("seed", JsValue::from_f64(42.0))]);
    let init_nd = JsLayoutNd::init(16.0, 3.0, &init_opts).unwrap();
    assert_eq!(init_nd.num_nodes(), 16);
    assert_eq!(init_nd.dims(), 3);
    assert_eq!(init_nd.coords_len(), 48);

    // Layout3d + 5 MDS LayoutGenerators targeting layout3d
    let init_3d = JsLayout3d::init(16.0, &init_opts).unwrap();
    assert_eq!(init_3d.num_nodes(), 16);
    assert_eq!(init_3d.coords_len(), 48);
    let opts_3d = make_opts(&[
        ("target", JsValue::from_str("layout3d")),
        ("iterations", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut sgd_3d = JsSgdSolver::full(&grid, &opts_3d).unwrap();
    let mut l3d = init_3d.clone();
    assert_eq!(sgd_3d.run_3d(&mut l3d).unwrap(), "completed");
    assert!(js_sns_3d(&grid, &l3d).unwrap().is_finite());
    assert!(js_np2_3d(&grid, &l3d).unwrap().is_finite());
    let rep_3d = js_evaluate_metrics_3d(
        &grid,
        &l3d,
        &js_sys::Array::of2(&"sns".into(), &"np2".into()),
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(rep_3d.scale_normalized_stress().is_some());

    for g_val in [
        JsClassicalMds::new(&opts_3d)
            .unwrap()
            .generate(&grid)
            .unwrap(),
        JsPivotMds::new(&opts_3d).unwrap().generate(&grid).unwrap(),
        JsHde::new(&opts_3d).unwrap().generate(&grid).unwrap(),
        JsRdMds::new(&opts_3d).unwrap().generate(&grid).unwrap(),
        JsKorenSpectral::new(&opts_3d)
            .unwrap()
            .generate(&grid)
            .unwrap(),
    ] {
        assert!(!g_val.is_undefined());
    }

    // Exercise all 11 N-D cooperative solvers on LayoutNd with target="layoutNd"
    let mut sgd_nd = JsSgdSolver::full(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("iterations", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l = init_nd.clone();
    assert_eq!(sgd_nd.run_nd(&mut l).unwrap(), "completed");
    assert!(l.is_all_finite());

    let mut ts_nd = JsTsNetSolver::pure_2hop(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("backend", JsValue::from_str("exact")),
            (
                "stageIterations",
                js_sys::Array::of3(
                    &JsValue::from_f64(2.0),
                    &JsValue::from_f64(2.0),
                    &JsValue::from_f64(2.0),
                )
                .into(),
            ),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_ts = init_nd.clone();
    assert_eq!(ts_nd.run_nd(&mut l_ts).unwrap(), "completed");
    assert!(l_ts.is_all_finite());

    let mut gm_nd = JsGuMapSolver::from_graph_dog(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_gm = init_nd.clone();
    assert_eq!(gm_nd.run_nd(&mut l_gm).unwrap(), "completed");

    let mut sm_nd = JsStressMajorizationSolver::from_graph(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("maxIterations", JsValue::from_f64(4.0)),
        ]),
    )
    .unwrap();
    let mut l_sm = init_nd.clone();
    let sm_st = sm_nd.run_nd(&mut l_sm).unwrap();
    assert!(sm_st == "completed" || sm_st == "converged");

    let mut kk_nd = JsKamadaKawaiSolver::from_graph(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("maxIterations", JsValue::from_f64(4.0)),
        ]),
    )
    .unwrap();
    let mut l_kk = init_nd.clone();
    let kk_st = kk_nd.run_nd(&mut l_kk).unwrap();
    assert!(kk_st == "completed" || kk_st == "converged");

    let mut fa2_nd = JsForceAtlas2Solver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_fa2 = init_nd.clone();
    assert_eq!(fa2_nd.run_nd(&mut l_fa2).unwrap(), "completed");

    let mut yh_nd = JsYifanHuSolver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("maxIterationsPerLevel", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_yh = init_nd.clone();
    let yh_st = yh_nd.run_nd(&mut l_yh).unwrap();
    assert!(yh_st == "completed" || yh_st == "converged");

    let mut ll_nd = JsLinLogSolver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_ll = init_nd.clone();
    let ll_st = ll_nd.run_nd(&mut l_ll).unwrap();
    assert!(ll_st == "completed" || ll_st == "converged");

    let mut fr_nd = JsFruchtermanReingoldSolver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("iterations", JsValue::from_f64(5.0)),
            ("backend", JsValue::from_str("exact")),
        ]),
    )
    .unwrap();
    let mut l_fr = init_nd.clone();
    assert_eq!(fr_nd.run_nd(&mut l_fr).unwrap(), "completed");

    let mut mx_nd = JsMaxentStressSolver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("layoutNd")),
            ("stages", JsValue::from_f64(2.0)),
            ("stepsPerStage", JsValue::from_f64(3.0)),
        ]),
    )
    .unwrap();
    let mut l_mx = init_nd.clone();
    assert_eq!(mx_nd.run_nd(&mut l_mx).unwrap(), "completed");

    let mut poly_nd = JsEdgePolylines::straight_line_from_graph_nd(&grid, &init_nd).unwrap();
    let mut fdeb_nd = JsFdebSolver::new(
        &poly_nd,
        &make_opts(&[
            ("cycles", JsValue::from_f64(2.0)),
            ("stepsPerCycle", JsValue::from_f64(4.0)),
        ]),
    )
    .unwrap();
    assert_eq!(fdeb_nd.run(&mut poly_nd).unwrap(), "completed");
    assert_eq!(poly_nd.copy_points().len() % 3, 0);

    // N-D metrics & evaluateMetricsNd
    assert!(js_sns_nd(&grid, &l).unwrap().is_finite());
    assert!(js_np2_nd(&grid, &l).unwrap().is_finite());
    let deg_nd = js_layout_degeneracy_nd(&grid, &l).unwrap();
    assert_eq!(deg_nd.num_nodes(), 16);
    let targets = js_sys::Array::of2(&"sns".into(), &"np2".into());
    let rep_nd = js_evaluate_metrics_nd(&grid, &l, &targets, &JsValue::UNDEFINED).unwrap();
    assert!(rep_nd.scale_normalized_stress().is_some());
    assert!(rep_nd.neighborhood_preservation_2hop().is_some());
}

#[wasm_bindgen_test]
fn manifold_solvers_lorentz_crossings_and_kind_lock() {
    let grid = make_grid(4, 4);

    // 1. HyperbolicLorentz, shadow coords sync, geodesic helpers & toPoincareLayout
    let mut lorentz = JsManifoldLayout2d::hyperbolic_lorentz(16.0).unwrap();
    assert_eq!(lorentz.kind(), "hyperbolicLorentz");
    assert_eq!(lorentz.coords_len(), 32);
    let ptr = lorentz.coords_ptr();
    lorentz.set_point(0.0, 0.25, -0.15).unwrap();
    assert_eq!(lorentz.coords_ptr(), ptr);
    lorentz.set_scale(1.5).unwrap();
    assert!((lorentz.scale() - 1.5).abs() < 1e-12);
    lorentz.rescale_geodesic(1.2).unwrap();
    let poincare = lorentz.to_poincare_layout().unwrap();
    assert_eq!(poincare.kind(), "hyperbolicPoincare");
    assert!(poincare.is_all_finite());

    assert!(js_lorentz_distance(0.0, 0.0, 0.3, 0.4) > 0.0);
    let p_pt = js_lorentz_to_poincare_point(0.3, 0.4);
    assert!(p_pt[0].hypot(p_pt[1]) < 1.0);
    let log_v = js_lorentz_log_map(0.0, 0.0, 0.3, 0.4);
    let exp_p = js_lorentz_exp_map(0.0, 0.0, log_v[0], log_v[1]);
    assert!((exp_p[0] - 0.3).abs() < 1e-6);
    assert!(js_lorentz_einstein_midpoint(&[0.0, 0.2], &[0.0, -0.2]).is_ok());
    assert!(js_poincare_einstein_midpoint(&[0.0, 0.2], &[0.0, -0.2]).is_ok());

    // 2. Torus spectral phase init, auto pan, segments, crossings
    let mut torus = js_torus_spectral_phase_init(&grid, &JsValue::from_f64(42.0)).unwrap();
    assert_eq!(torus.kind(), "torus");
    torus.set_torus_extents(2.0, 1.0).unwrap();
    assert!((torus.aspect() - 2.0).abs() < 1e-12);
    let panned = torus.torus_auto_pan(&grid).unwrap();
    assert!(panned.is_all_finite());
    let segs = js_torus_edge_segments(0.9, 0.1, 0.1, 0.9).unwrap();
    assert!(segs.len() >= 8);
    let layout_segs = js_torus_layout_edge_segments_manifold(&grid, &panned).unwrap();
    assert_eq!(layout_segs.len() % 4, 0);
    let _tc = js_torus_crossing_number_manifold(&grid, &panned).unwrap();
    let _te = js_crossing_edges_torus_manifold(&grid, &panned).unwrap();

    // 3. Spherical & Hyperbolic crossings + metrics
    let sph_opts = make_opts(&[
        ("seed", JsValue::from_f64(42.0)),
        ("scale", JsValue::from_f64(0.5)),
    ]);
    let mut sph = JsManifoldLayout2d::init("spherical", 16.0, &sph_opts).unwrap();
    let _sc = js_spherical_crossing_number(&grid, &sph).unwrap();
    let _se = js_crossing_edges_spherical(&grid, &sph).unwrap();
    let _hc = js_hyperbolic_crossing_number(&grid, &poincare).unwrap();
    let _he = js_crossing_edges_hyperbolic(&grid, &poincare).unwrap();

    // 4. All 6 Manifold solvers with target="manifold2d"
    let hyp_opts = make_opts(&[
        ("seed", JsValue::from_f64(42.0)),
        ("scale", JsValue::from_f64(0.3)),
    ]);
    let base_m = JsManifoldLayout2d::init("hyperbolic", 16.0, &hyp_opts).unwrap();
    let mut m_sgd = base_m.copy();
    let mut sgd = JsSgdSolver::full(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("iterations", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(sgd.run_manifold(&mut m_sgd).unwrap(), "completed");

    // ManifoldKind lock: passing Spherical layout after locking to HyperbolicPoincare fails with InvalidInput
    let lock_err = sgd.step_manifold(&mut sph).unwrap_err();
    assert_eq!(
        js_sys::Reflect::get(&lock_err, &"code".into())
            .unwrap()
            .as_string(),
        Some("InvalidInput".to_owned())
    );

    let mut m_ts = base_m.copy();
    let mut ts = JsTsNetSolver::pure_2hop(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("backend", JsValue::from_str("exact")),
            (
                "stageIterations",
                js_sys::Array::of3(
                    &JsValue::from_f64(2.0),
                    &JsValue::from_f64(2.0),
                    &JsValue::from_f64(2.0),
                )
                .into(),
            ),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(ts.run_manifold(&mut m_ts).unwrap(), "completed");

    let mut m_gm = base_m.copy();
    let mut gm = JsGuMapSolver::from_graph_dog(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("iterations", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(gm.run_manifold(&mut m_gm).unwrap(), "completed");

    let mut m_sm = base_m.copy();
    let mut sm = JsStressMajorizationSolver::from_graph(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("maxIterations", JsValue::from_f64(3.0)),
        ]),
    )
    .unwrap();
    let _ = sm.run_manifold(&mut m_sm).unwrap();

    let mut m_kk = base_m.copy();
    let mut kk = JsKamadaKawaiSolver::from_graph(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("maxIterations", JsValue::from_f64(3.0)),
        ]),
    )
    .unwrap();
    let _ = kk.run_manifold(&mut m_kk).unwrap();

    let mut m_fr = base_m.copy();
    let mut fr = JsFruchtermanReingoldSolver::new(
        &grid,
        &make_opts(&[
            ("target", JsValue::from_str("manifold2d")),
            ("iterations", JsValue::from_f64(3.0)),
            ("backend", JsValue::from_str("exact")),
        ]),
    )
    .unwrap();
    assert_eq!(fr.run_manifold(&mut m_fr).unwrap(), "completed");

    assert!(js_sns_manifold(&grid, &m_sgd).unwrap().is_finite());
    assert!(js_np2_manifold(&grid, &m_sgd).unwrap().is_finite());
    assert_eq!(
        js_layout_degeneracy_manifold(&grid, &m_sgd)
            .unwrap()
            .num_nodes(),
        16
    );
    let rep_m = js_evaluate_metrics_manifold(
        &grid,
        &m_sgd,
        &js_sys::Array::of2(&"sns".into(), &"geodesicDistortion".into()),
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(rep_m.geodesic_distortion().is_some());
}
