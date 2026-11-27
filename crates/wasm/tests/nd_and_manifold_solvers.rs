mod util;

use egraph_wasm::fdeb::JsFdebSolver;
use egraph_wasm::force_classic::{JsFruchtermanReingoldSolver, JsLinLogSolver};
use egraph_wasm::force_multilevel::{JsForceAtlas2Solver, JsYifanHuSolver};
use egraph_wasm::gumap::JsGuMapSolver;
use egraph_wasm::layout_buffers::JsLayoutNd;
use egraph_wasm::manifold::JsManifoldLayout2d;
use egraph_wasm::manifold_geodesic::{
    js_lorentz_distance, js_lorentz_einstein_midpoint, js_lorentz_exp_map, js_lorentz_log_map,
    js_lorentz_to_poincare_point, js_poincare_einstein_midpoint, js_torus_spectral_phase_init,
};
use egraph_wasm::metrics_eval_dispatch::{js_evaluate_metrics_manifold, js_evaluate_metrics_nd};
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
fn nd_solvers_and_metrics_evaluation() {
    let grid = make_grid(4, 4);
    let init3d = JsLayoutNd::init(16.0, 3.0, Some(42.0), Some(1.0)).unwrap();
    assert_eq!(init3d.num_nodes(), 16);
    assert_eq!(init3d.dims(), 3);
    assert_eq!(init3d.coords_len(), 48);

    // Exercise all 11 N-D cooperative solvers on LayoutNd
    let mut sgd_nd = JsSgdSolver::full(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l = init3d.clone();
    assert_eq!(sgd_nd.run_nd(&mut l).unwrap(), "completed");
    assert!(l.is_all_finite());

    let mut ts_nd = JsTsNetSolver::pure_2hop(
        &grid,
        &make_opts(&[
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
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    let mut l_ts = init3d.clone();
    assert_eq!(ts_nd.run_nd(&mut l_ts).unwrap(), "completed");
    assert!(l_ts.is_all_finite());

    let mut gm_nd = JsGuMapSolver::from_graph_dog(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
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
    let mut l_gm = init3d.clone();
    assert_eq!(gm_nd.run_nd(&mut l_gm).unwrap(), "completed");

    let mut sm_nd = JsStressMajorizationSolver::from_graph(
        &grid,
        &make_opts(&[("maxIterations", JsValue::from_f64(4.0))]),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l_sm = init3d.clone();
    let sm_st = sm_nd.run_nd(&mut l_sm).unwrap();
    assert!(sm_st == "completed" || sm_st == "converged");

    let mut kk_nd = JsKamadaKawaiSolver::from_graph(
        &grid,
        &make_opts(&[("maxIterations", JsValue::from_f64(4.0))]),
        None,
        None,
    )
    .unwrap();
    let mut l_kk = init3d.clone();
    let kk_st = kk_nd.run_nd(&mut l_kk).unwrap();
    assert!(kk_st == "completed" || kk_st == "converged");

    let mut fa2_nd = JsForceAtlas2Solver::new(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_fa2 = init3d.clone();
    assert_eq!(fa2_nd.run_nd(&mut l_fa2).unwrap(), "completed");

    let mut yh_nd = JsYifanHuSolver::new(
        &grid,
        &make_opts(&[
            ("maxIterationsPerLevel", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_yh = init3d.clone();
    let yh_st = yh_nd.run_nd(&mut l_yh).unwrap();
    assert!(yh_st == "completed" || yh_st == "converged");

    let mut ll_nd = JsLinLogSolver::new(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_ll = init3d.clone();
    let ll_st = ll_nd.run_nd(&mut l_ll).unwrap();
    assert!(ll_st == "completed" || ll_st == "converged");

    let mut fr_nd = JsFruchtermanReingoldSolver::new(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(5.0)),
            ("backend", JsValue::from_str("exact")),
        ]),
    )
    .unwrap();
    let mut l_fr = init3d.clone();
    assert_eq!(fr_nd.run_nd(&mut l_fr).unwrap(), "completed");

    let mut mx_nd = JsMaxentStressSolver::new(
        &grid,
        &make_opts(&[
            ("stages", JsValue::from_f64(2.0)),
            ("stepsPerStage", JsValue::from_f64(3.0)),
        ]),
    )
    .unwrap();
    let mut l_mx = init3d.clone();
    assert_eq!(mx_nd.run_nd(&mut l_mx).unwrap(), "completed");

    let mut fdeb_nd = JsFdebSolver::new(
        &grid,
        &make_opts(&[
            ("cycles", JsValue::from_f64(2.0)),
            ("stepsPerCycle", JsValue::from_f64(4.0)),
        ]),
    )
    .unwrap();
    let mut l_fdeb = init3d.clone();
    assert_eq!(fdeb_nd.run_nd(&mut l_fdeb).unwrap(), "completed");
    assert_eq!(fdeb_nd.points().len() % 3, 0);

    // N-D metrics & evaluateMetricsNd
    assert!(js_sns_nd(&grid, &l).unwrap().is_finite());
    assert!(js_np2_nd(&grid, &l).unwrap().is_finite());
    let deg_nd = js_layout_degeneracy_nd(&grid, &l).unwrap();
    assert_eq!(deg_nd.num_nodes(), 16);
    let targets = js_sys::Array::of2(&"sns".into(), &"np2".into());
    let rep_nd = js_evaluate_metrics_nd(
        &grid,
        &l,
        &targets,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(rep_nd.scale_normalized_stress().is_some());
    assert!(rep_nd.neighborhood_preservation_2hop().is_some());
}

#[wasm_bindgen_test]
fn manifold_solvers_lorentz_crossings_and_shadow_sync() {
    let grid = make_grid(4, 4);

    // 1. HyperbolicLorentz, shadow coords sync, geodesic helpers & toPoincareLayout
    let mut lorentz = JsManifoldLayout2d::hyperbolic_lorentz(16.0).unwrap();
    assert_eq!(lorentz.kind(), "hyperbolic_lorentz");
    assert_eq!(lorentz.coords_len(), 32);
    let ptr = lorentz.coords_ptr();
    lorentz.set_point(0.0, 0.25, -0.15).unwrap();
    assert_eq!(lorentz.coords_ptr(), ptr);
    lorentz.set_scale(1.5).unwrap();
    assert!((lorentz.scale() - 1.5).abs() < 1e-12);
    lorentz.rescale_geodesic(1.2).unwrap();
    let poincare = lorentz.to_poincare_layout().unwrap();
    assert_eq!(poincare.kind(), "hyperbolic");
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
    let mut torus = js_torus_spectral_phase_init(&grid, Some(42.0)).unwrap();
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
    let sph = JsManifoldLayout2d::init("spherical", 16.0, Some(42.0), Some(0.5)).unwrap();
    let _sc = js_spherical_crossing_number(&grid, &sph).unwrap();
    let _se = js_crossing_edges_spherical(&grid, &sph).unwrap();
    let _hc = js_hyperbolic_crossing_number(&grid, &poincare).unwrap();
    let _he = js_crossing_edges_hyperbolic(&grid, &poincare).unwrap();

    // 4. All 6 Manifold solvers
    let base_m = JsManifoldLayout2d::init("hyperbolic", 16.0, Some(42.0), Some(0.3)).unwrap();
    let mut m_sgd = base_m.copy();
    let mut sgd = JsSgdSolver::full(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(sgd.run_manifold(&mut m_sgd).unwrap(), "completed");

    let mut m_ts = base_m.copy();
    let mut ts = JsTsNetSolver::pure_2hop(
        &grid,
        &make_opts(&[
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
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    assert_eq!(ts.run_manifold(&mut m_ts).unwrap(), "completed");

    let mut m_gm = base_m.copy();
    let mut gm = JsGuMapSolver::from_graph_dog(
        &grid,
        &make_opts(&[
            ("iterations", JsValue::from_f64(4.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
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
    assert_eq!(gm.run_manifold(&mut m_gm).unwrap(), "completed");

    let mut m_sm = base_m.copy();
    let mut sm = JsStressMajorizationSolver::from_graph(
        &grid,
        &make_opts(&[("maxIterations", JsValue::from_f64(3.0))]),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let _ = sm.run_manifold(&mut m_sm).unwrap();

    let mut m_kk = base_m.copy();
    let mut kk = JsKamadaKawaiSolver::from_graph(
        &grid,
        &make_opts(&[("maxIterations", JsValue::from_f64(3.0))]),
        None,
        None,
    )
    .unwrap();
    let _ = kk.run_manifold(&mut m_kk).unwrap();

    let mut m_fr = base_m.copy();
    let mut fr = JsFruchtermanReingoldSolver::new(
        &grid,
        &make_opts(&[
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
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(rep_m.geodesic_distortion().is_some());
}
