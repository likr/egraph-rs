mod util;

use egraph_wasm::constraints_layering::js_transitive_reduction;
use egraph_wasm::constraints_twin_declutter::{JsTwinDeclutterer, js_declutter_twins_2d};
use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::datasets_catalog::{
    js_dataset_catalog, js_dataset_info, js_generate_hyperbolic_tree, js_generate_layered_dag,
    js_generate_sbm_graph, js_generate_spherical_grid, js_generate_spherical_icosphere,
    js_generate_torus_grid, js_generate_weighted_graph, js_list_datasets,
};
use egraph_wasm::layout_buffers::{JsLayout2d, JsLayoutNd};
use egraph_wasm::linalg_chebyshev_kernel::JsChebyshevDogKernel;
use egraph_wasm::linalg_dijkstra::{
    JsDijkstraWorkspace, js_all_pairs_dijkstra, js_dijkstra_single_source,
};
use egraph_wasm::metrics_rta::js_rta_from_landmarks_2d;
use egraph_wasm::sgd::JsSgdSolver;
use egraph_wasm::spatial_trees::{JsArenaOctree3d, JsDynamicKdTree};
use egraph_wasm::tsnet::JsTsNetSolver;
use egraph_wasm::tsnet_stages::JsTsNetStage;
use util::fixtures::{make_circle_layout, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn spatial_trees_octree_and_kdtree() {
    let pts3 = [
        0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
    ];
    let nd = JsLayoutNd::from_coords(4.0, 3.0, &pts3).unwrap();

    let mut oct = JsArenaOctree3d::new();
    oct.build_nd(&nd).unwrap();
    assert_eq!(oct.point_count(), 4);
    assert!(oct.node_count() >= 1);
    assert_eq!(
        oct.compute_coulomb_repulsion(Some(0.5), None)
            .unwrap()
            .len(),
        12
    );
    assert_eq!(
        oct.compute_maxent_repulsion(Some(0.5), Some(1.0), None)
            .unwrap()
            .len(),
        12
    );
    assert_eq!(oct.compute_kl_repulsion(Some(0.5)).unwrap().len(), 12);
    assert_eq!(
        oct.compute_entropy_repulsion(Some(0.5), Some(0.05))
            .unwrap()
            .len(),
        12
    );
    oct.build_weighted_nd(&nd, &[1.0, 2.0, 1.0, 2.0]).unwrap();
    oct.build_xyz(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0], None)
        .unwrap();
    assert_eq!(oct.point_count(), 2);
    oct.clear();
    assert_eq!(oct.point_count(), 0);

    let kd = JsDynamicKdTree::from_layout_nd(&nd).unwrap();
    assert_eq!(kd.dims(), 3);
    assert_eq!(kd.num_nodes(), 4);
    assert_eq!(kd.point(1.0).unwrap(), vec![1.0, 0.0, 0.0]);
    assert_eq!(kd.k_nearest(&[0.0, 0.0, 0.0], 2.0).unwrap().length(), 2);
    assert_eq!(kd.k_nearest_excluding_self(0.0, 2.0).unwrap().length(), 2);
    assert!(kd.nearest_excluding_self(0.0).unwrap().is_object());
    assert_eq!(kd.within_radius(&[0.0, 0.0, 0.0], 1.1).unwrap().length(), 3);
    assert_eq!(
        kd.within_radius_sq(&[0.0, 0.0, 0.0], 1.1).unwrap().length(),
        3
    );
}

#[wasm_bindgen_test]
fn datasets_catalog_and_synthetic_generators() {
    let names = js_list_datasets();
    assert!(!names.is_empty());
    let catalog = js_dataset_catalog();
    assert_eq!(catalog.length() as usize, names.len());
    let info = js_dataset_info(&names[0]).expect("first dataset");
    assert_eq!(info.name(), names[0]);
    assert!(info.num_nodes() > 0);
    assert!(info.to_record().unwrap().is_object());

    assert_eq!(
        js_generate_weighted_graph(10.0, Some(3.0), Some(0.5), Some(2.0), Some(7.0))
            .unwrap()
            .num_nodes(),
        10
    );
    let dag = js_generate_layered_dag(3.0, 4.0, Some(0.4), Some(7.0)).unwrap();
    assert!(dag.is_directed());
    let red = js_transitive_reduction(&dag).unwrap();
    assert!(red.num_edges() <= dag.num_edges());
    assert!(
        js_generate_sbm_graph(&[4, 4], Some(0.7), Some(0.1), Some(7.0))
            .unwrap()
            .is_object()
    );
    assert_eq!(
        js_generate_torus_grid(3.0, 4.0, Some(true))
            .unwrap()
            .num_nodes(),
        12
    );
    assert!(
        js_generate_spherical_icosphere(Some(1.0))
            .unwrap()
            .num_nodes()
            >= 12
    );
    assert!(js_generate_spherical_grid(4.0, 6.0).unwrap().num_nodes() > 0);
    assert!(js_generate_hyperbolic_tree(2.0, 2.0).unwrap().num_nodes() >= 7);
}

#[wasm_bindgen_test]
fn dijkstra_chebyshev_kernel_declutter_rta_and_stages() {
    let g = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2],
        Some(vec![1.5, 2.5]),
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let mut ws = JsDijkstraWorkspace::new(Some(4.0)).unwrap();
    ws.ensure_capacity(6.0).unwrap();
    let d0 = ws.dijkstra_from(&g, 0.0).unwrap();
    assert!((d0[2] - 4.0).abs() < 1e-9);
    assert!(d0[3].is_infinite());
    assert_eq!(js_dijkstra_single_source(&g, 0.0).unwrap(), d0);
    assert_eq!(js_all_pairs_dijkstra(&g).unwrap().len(), 16);

    let grid = make_grid(4, 4);
    let kern =
        JsChebyshevDogKernel::new(&grid, &make_opts(&[("jlDim", JsValue::from_f64(8.0))])).unwrap();
    assert_eq!(kern.num_nodes(), 16);
    assert!(kern.terms_psd().length() > 0);

    // Twin decluttering
    let mut dup = JsLayout2d::from_coords(&[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0]).unwrap();
    let mut declutter =
        JsTwinDeclutterer::new(&make_opts(&[("iterations", JsValue::from_f64(3.0))])).unwrap();
    let step_st = declutter.step(&mut dup).unwrap();
    assert!(step_st == "running" || step_st == "converged");
    assert!(declutter.last_moved_count() > 0);
    let dec_st = declutter.run(&mut dup).unwrap();
    assert!(dec_st == "converged" || dec_st == "completed");
    let mut dup2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0]).unwrap();
    assert!(js_declutter_twins_2d(&mut dup2, &JsValue::UNDEFINED).unwrap() > 0);

    // RTA from landmarks with Float64Array / Float32Array / Uint32Array (including Infinity)
    let lm_ids = js_sys::Uint32Array::from(&[0u32, 3][..]);
    let f64_dists: JsValue = js_sys::Float64Array::from(
        &[0.0, 1.0, f64::INFINITY, 2.0, 2.0, 1.0, f64::INFINITY, 0.0][..],
    )
    .into();
    let f32_dists: JsValue = js_sys::Float32Array::from(
        &[
            0.0_f32,
            1.0,
            f32::INFINITY,
            2.0,
            2.0,
            1.0,
            f32::INFINITY,
            0.0,
        ][..],
    )
    .into();
    let r64 = js_rta_from_landmarks_2d(
        &dup,
        &lm_ids.clone().into(),
        &f64_dists,
        Some(64.0),
        Some(42.0),
    )
    .unwrap();
    let r32 =
        js_rta_from_landmarks_2d(&dup, &lm_ids.into(), &f32_dists, Some(64.0), Some(42.0)).unwrap();
    assert!(r64.rta().is_finite() && r32.rta().is_finite());

    // SgdSolver packed symmetric, resetSchedule, runScheduleWindow, maxMu
    let mut sgd = JsSgdSolver::from_packed_symmetric(
        4.0,
        &[0, 1, 2],
        &[1, 2, 3],
        &[1.0, 1.0, 1.0],
        None,
        None,
        &make_opts(&[
            ("iterations", JsValue::from_f64(6.0)),
            ("maxMu", JsValue::from_f64(0.5)),
        ]),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(sgd.max_mu(), Some(0.5));
    let mut l4 = make_circle_layout(4, 1.0);
    assert_eq!(
        sgd.run_schedule_window_2d(&mut l4, 0.0, 3.0).unwrap(),
        "running"
    );
    sgd.reset_schedule(&JsValue::from_f64(4.0), None, None, None, Some(42.0))
        .unwrap();
    assert_eq!(sgd.run_2d(&mut l4).unwrap(), "completed");

    // TsNetBackend::Auto + TsNetStage
    let ts_auto = JsTsNetSolver::pure_2hop(
        &grid,
        &make_opts(&[
            ("backend", JsValue::from_str("auto")),
            ("learningRateDivisor", JsValue::from_f64(2.0)),
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
    assert!(!ts_auto.resolved_backend().is_empty());
    let stage = JsTsNetStage::new(3.0, 2.0, 1.0, 0.1, 0.0).unwrap();
    assert_eq!(stage.iterations(), 3);
}
