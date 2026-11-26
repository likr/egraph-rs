mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::metrics_cluster_rta::{
    js_landmark_stratified_rta, js_landmark_stratified_rta_detail,
    js_landmark_stratified_rta_from_landmarks,
};
use egraph_wasm::metrics_crossings::{
    js_angular_resolution, js_aspect_ratio, js_crosslessness, js_edge_angle, js_layout_degeneracy,
    js_minimum_angle, js_minimum_angle_summary, js_node_resolution, js_sampled_crossing_number,
    js_sampled_crosslessness, js_sweep_crossing_angle, js_sweep_crossing_number,
    js_torus_crossing_number,
};
use egraph_wasm::metrics_eval::js_evaluate_metrics_with_options;
use egraph_wasm::metrics_functions::{
    js_adjusted_patch_nig, js_adjusted_patch_nig_multiscale, js_dgg_preservation,
    js_drng_preservation, js_gabriel_graph_property, js_ideal_edge_lengths,
    js_neighborhood_preservation_1hop, js_neighborhood_preservation_2hop, js_prism_distortion,
    js_sampled_np1, js_sampled_np2, js_sampled_scale_normalized_stress, js_scale_normalized_stress,
};
use util::fixtures::{make_grid_4x4_with_layout, u32_typed_js};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn stress_and_crossings_and_torus() {
    // 4-cycle with crossing diagonals (K4 without one edge or K4 full).
    let k4 = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3],
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("K4");
    let sq = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).expect("square");

    let sns_1 = js_scale_normalized_stress(&k4, &sq).expect("sns 1x");
    let sq_scaled =
        JsLayout2d::from_coords(&[0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0]).expect("scaled");
    let sns_50 = js_scale_normalized_stress(&k4, &sq_scaled).expect("sns 50x");
    assert!((sns_1 - sns_50).abs() < 1e-6);

    let sampled_sns =
        js_sampled_scale_normalized_stress(&k4, &sq, Some(4.0), Some(42.0)).expect("sampled sns");
    assert!((sampled_sns - sns_1).abs() < 1e-6);

    assert_eq!(js_sweep_crossing_number(&k4, &sq).expect("crossings"), 1);
    assert!(
        js_sweep_crossing_angle(&k4, &sq)
            .expect("crossing angle")
            .abs()
            < 1e-6
    );
    assert_eq!(
        js_sampled_crossing_number(&k4, &sq, Some(100.0), Some(42.0)).expect("sampled crossing"),
        1.0
    );

    // Torus wrap crossing on [0, 1)^2.
    let two_edges =
        JsCsrGraph::from_edge_list(4.0, &[0, 1, 2, 3], None, &JsValue::UNDEFINED, None).unwrap();
    let torus_layout = JsLayout2d::from_coords(&[0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1]).unwrap();
    assert!(js_torus_crossing_number(&two_edges, &torus_layout).expect("torus") >= 1);
}

#[wasm_bindgen_test]
fn neighborhood_proximity_geometry_and_prism() {
    let (grid, layout) = make_grid_4x4_with_layout();
    assert!((js_neighborhood_preservation_1hop(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert!((js_neighborhood_preservation_2hop(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(
        js_sampled_np1(&grid, &layout, Some(16.0), Some(42.0)).unwrap(),
        js_neighborhood_preservation_1hop(&grid, &layout).unwrap()
    );
    assert_eq!(
        js_sampled_np2(&grid, &layout, Some(16.0), Some(42.0)).unwrap(),
        js_neighborhood_preservation_2hop(&grid, &layout).unwrap()
    );
    let snp1_a = js_sampled_np1(&grid, &layout, Some(6.0), Some(7.0)).unwrap();
    let snp1_b = js_sampled_np1(&grid, &layout, Some(6.0), Some(7.0)).unwrap();
    assert_eq!(snp1_a, snp1_b);
    assert!((0.0..=1.0).contains(&snp1_a));
    assert!((js_drng_preservation(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert!((js_dgg_preservation(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(js_gabriel_graph_property(&grid, &layout).unwrap(), 0.0);
    assert!((js_aspect_ratio(&layout).unwrap() - 1.0).abs() < 1e-6);
    assert!(js_node_resolution(&layout).unwrap() > 0.0);
    assert!(js_angular_resolution(&grid, &layout).unwrap() > 0.0);
    assert!(js_ideal_edge_lengths(&grid, &layout).unwrap().abs() < 1e-6);

    // edgeAngle vector checks
    let right = js_edge_angle(1.0, 0.0, 0.0, 1.0).expect("90 deg");
    assert!((right - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert!(js_edge_angle(0.0, 0.0, 1.0, 0.0).is_none());
    assert!(js_edge_angle(f64::NAN, 0.0, 1.0, 0.0).is_none());

    // PRISM similarity invariance
    let mut scaled_coords = Vec::with_capacity(32);
    for i in 0..16 {
        let x = layout.x(i as f64).unwrap();
        let y = layout.y(i as f64).unwrap();
        scaled_coords.push(2.5 * x + 10.0);
        scaled_coords.push(2.5 * y - 3.0);
    }
    let adjusted = JsLayout2d::from_coords(&scaled_coords).unwrap();
    let dist = js_prism_distortion(&layout, &adjusted).expect("prism distortion");
    assert!(dist.sigma_dist() < 1e-5);
    assert!(dist.sigma_disp() < 1e-5);
}

#[wasm_bindgen_test]
fn crosslessness_minimum_angle_degeneracy_nig_and_rta() {
    let k4 = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3],
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .unwrap();
    let sq = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    let expected_cl = 1.0 - (1.0_f64 / 3.0).sqrt();
    assert!((js_crosslessness(&k4, &sq).unwrap() - expected_cl).abs() < 1e-12);
    assert!(
        (js_sampled_crosslessness(&k4, &sq, Some(100.0), Some(42.0)).unwrap() - expected_cl).abs()
            < 1e-12
    );

    let ma_sum = js_minimum_angle_summary(&k4, &sq).unwrap();
    assert!((ma_sum.a1() - js_minimum_angle(&k4, &sq).unwrap()).abs() < 1e-12);
    let ma_rec = ma_sum.to_record().unwrap();
    assert!(
        js_sys::Reflect::get(&ma_rec, &"a2".into())
            .unwrap()
            .as_f64()
            .is_some()
    );

    let deg = js_layout_degeneracy(&k4, &sq).unwrap();
    assert_eq!(deg.num_nodes(), 4);
    assert_eq!(deg.num_edges(), 6);
    assert_eq!(deg.duplicate_nodes_f32(), 0);
    assert_eq!(deg.zero_length_edges(), 0);
    assert!(!deg.is_bbox_degenerate());
    let deg_rec = deg.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&deg_rec, &"duplicateNodesF32".into())
            .unwrap()
            .as_f64(),
        Some(0.0)
    );

    let nig = js_adjusted_patch_nig(
        &sq,
        &[0, 0, 1, 1],
        Some(16.0),
        Some(2048.0),
        None,
        Some(42.0),
    )
    .unwrap();
    let nig_ms = js_adjusted_patch_nig_multiscale(
        &sq,
        &[0, 0, 1, 1],
        Some(vec![16, 32]),
        Some(2048.0),
        None,
        Some(42.0),
    )
    .unwrap();
    assert!((nig_ms[0] - nig).abs() < 1e-12);

    let (grid, grid_layout) = make_grid_4x4_with_layout();
    let rta = js_landmark_stratified_rta(
        &grid,
        &grid_layout,
        Some(4.0),
        Some(10000.0),
        Some(2.0),
        Some(42.0),
    )
    .unwrap();
    let rta_det = js_landmark_stratified_rta_detail(
        &grid,
        &grid_layout,
        Some(4.0),
        Some(10000.0),
        Some(2.0),
        Some(42.0),
    )
    .unwrap();
    assert!((rta_det.rta() - rta).abs() < 1e-12);
    assert!(rta_det.is_exact());
    let rta_rec = rta_det.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&rta_rec, &"isExact".into())
            .unwrap()
            .as_bool(),
        Some(true)
    );
    let rta_tbl = js_landmark_stratified_rta_from_landmarks(
        &sq,
        &[0, 2],
        &[0, 1, 2, 1, 2, 1, 0, 1],
        Some(1000.0),
        Some(42.0),
    )
    .unwrap();
    assert!(rta_tbl.is_exact());

    let targets = js_sys::Array::new();
    for t in [
        "crosslessness",
        "sampledCrosslessness",
        "minimumAngle",
        "minimumAngleA1",
        "layoutDegeneracy",
        "adjustedPatchNig",
        "landmarkStratifiedRta",
    ] {
        targets.push(&t.into());
    }
    let k4_labels_js = u32_typed_js(&[0, 0, 1, 1]);
    let rep = js_evaluate_metrics_with_options(
        &k4,
        &sq,
        &targets,
        None,
        Some(100.0),
        Some(42.0),
        None,
        &k4_labels_js,
        Some(4.0),
        Some(1000.0),
        Some(1.0),
        Some(16.0),
        Some(2048.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!((rep.crosslessness().unwrap() - expected_cl).abs() < 1e-12);
    assert!(rep.minimum_angle().is_some());
    assert!(rep.minimum_angle_a1().is_some());
    assert_eq!(rep.layout_degeneracy(), Some(0.0));
    assert!(rep.adjusted_patch_nig().is_some());
    assert!(rep.landmark_stratified_rta().is_some());
    assert!(rep.cluster_area_faithfulness().is_none());

    let need_labels = js_sys::Array::of1(&"adjustedPatchNig".into());
    assert!(
        js_evaluate_metrics_with_options(
            &k4,
            &sq,
            &need_labels,
            None,
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
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn crosslessness_directed_reciprocal_regression() {
    // 1. Single reciprocal pair: c_max == 0 -> crosslessness == 1.0.
    let g_pair = JsCsrGraph::from_edge_list(
        2.0,
        &[0, 1, 1, 0],
        None,
        &JsValue::from_bool(true),
        Some(true),
    )
    .unwrap();
    let l_pair = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(js_crosslessness(&g_pair, &l_pair).unwrap(), 1.0);
    assert_eq!(
        js_sampled_crosslessness(&g_pair, &l_pair, Some(32.0), Some(42.0)).unwrap(),
        1.0
    );

    // 2. Reciprocal 3-leaf star: adj_pairs > total_pairs, yet c_max == 0 -> crosslessness == 1.0.
    let g_star = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0],
        None,
        &JsValue::from_bool(true),
        Some(true),
    )
    .unwrap();
    let l_star = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap();
    assert_eq!(js_crosslessness(&g_star, &l_star).unwrap(), 1.0);
    assert_eq!(
        js_sampled_crosslessness(&g_star, &l_star, Some(64.0), Some(42.0)).unwrap(),
        1.0
    );

    // 3. 4-node 6-edge directed graph: adj_pairs (16) > total_pairs (15), c_max = 1, c = 1 -> 0.0.
    let g_dir4 = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 0, 3, 1, 3, 3, 0, 3, 1, 3, 2],
        None,
        &JsValue::from_bool(true),
        Some(true),
    )
    .unwrap();
    let l_dir4 = JsLayout2d::from_coords(&[0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0]).unwrap();
    assert_eq!(js_sweep_crossing_number(&g_dir4, &l_dir4).unwrap(), 1);
    assert_eq!(js_crosslessness(&g_dir4, &l_dir4).unwrap(), 0.0);
    assert_eq!(
        js_sampled_crosslessness(&g_dir4, &l_dir4, Some(100.0), Some(42.0)).unwrap(),
        0.0
    );
}
