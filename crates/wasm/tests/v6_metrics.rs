#[cfg(feature = "delaunay")]
use egraph_wasm::v6::js_delaunay_triangulation;
use egraph_wasm::v6::{
    JsCsrGraph, JsLayout2d, js_angular_resolution_v6, js_aspect_ratio_v6, js_dgg_preservation,
    js_drng_preservation, js_edge_angle, js_evaluate_metrics, js_gabriel_graph_property_v6,
    js_ideal_edge_lengths_v6, js_neighborhood_preservation_1hop,
    js_neighborhood_preservation_2hop_v6, js_node_resolution_v6, js_prism_distortion,
    js_sampled_crossing_number, js_sampled_np1, js_sampled_np2, js_sampled_scale_normalized_stress,
    js_scale_normalized_stress, js_sweep_crossing_angle, js_sweep_crossing_number,
    js_torus_crossing_number,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_grid_4x4() -> (JsCsrGraph, JsLayout2d) {
    let mut edges = Vec::new();
    let mut coords = Vec::with_capacity(32);
    for r in 0..4u32 {
        for c in 0..4u32 {
            let id = r * 4 + c;
            coords.push(c as f32);
            coords.push(r as f32);
            if c + 1 < 4 {
                edges.push(id);
                edges.push(id + 1);
            }
            if r + 1 < 4 {
                edges.push(id);
                edges.push(id + 4);
            }
        }
    }
    let graph = JsCsrGraph::from_edge_list(16.0, &edges, None, &JsValue::from_bool(false), None)
        .expect("grid CSR");
    let layout = JsLayout2d::from_coords(&coords).expect("grid layout");
    (graph, layout)
}

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
    let (grid, layout) = make_grid_4x4();
    assert!((js_neighborhood_preservation_1hop(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert!((js_neighborhood_preservation_2hop_v6(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(
        js_sampled_np1(&grid, &layout, Some(16.0), Some(42.0)).unwrap(),
        js_neighborhood_preservation_1hop(&grid, &layout).unwrap()
    );
    assert_eq!(
        js_sampled_np2(&grid, &layout, Some(16.0), Some(42.0)).unwrap(),
        js_neighborhood_preservation_2hop_v6(&grid, &layout).unwrap()
    );
    let snp1_a = js_sampled_np1(&grid, &layout, Some(6.0), Some(7.0)).unwrap();
    let snp1_b = js_sampled_np1(&grid, &layout, Some(6.0), Some(7.0)).unwrap();
    assert_eq!(snp1_a, snp1_b);
    assert!((0.0..=1.0).contains(&snp1_a));
    assert!((js_drng_preservation(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert!((js_dgg_preservation(&grid, &layout).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(js_gabriel_graph_property_v6(&grid, &layout).unwrap(), 0.0);
    assert!((js_aspect_ratio_v6(&layout).unwrap() - 1.0).abs() < 1e-6);
    assert!(js_node_resolution_v6(&layout).unwrap() > 0.0);
    assert!(js_angular_resolution_v6(&grid, &layout).unwrap() > 0.0);
    assert!(js_ideal_edge_lengths_v6(&grid, &layout).unwrap().abs() < 1e-6);

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
fn selective_evaluate_metrics_and_adversarial_errors() {
    let (grid, layout) = make_grid_4x4();
    let targets = js_sys::Array::of5(
        &"sns".into(),
        &"np2".into(),
        &"sampledNp1".into(),
        &"sampled_np2".into(),
        &"aspectRatio".into(),
    );
    let report = js_evaluate_metrics(
        &grid,
        &layout,
        &targets,
        Some(8.0),
        Some(64.0),
        Some(42.0),
        None,
    )
    .expect("selective report");

    assert!(report.scale_normalized_stress().is_some());
    assert!(report.neighborhood_preservation_2hop().is_some());
    assert!(report.sampled_np1().is_some());
    assert!(report.sampled_np2().is_some());
    assert!(report.aspect_ratio().is_some());
    assert!(report.crossing_number().is_none());
    assert!(report.crossing_angle().is_none());
    assert!(report.sampled_crossing_number().is_none());
    assert!(report.ideal_edge_lengths().is_none());

    let record = report.to_record().expect("toRecord");
    let sns_val = js_sys::Reflect::get(&record, &"scaleNormalizedStress".into())
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((sns_val - report.scale_normalized_stress().unwrap()).abs() < 1e-12);
    assert_eq!(
        js_sys::Reflect::get(&record, &"sampledNp1".into())
            .unwrap()
            .as_f64(),
        report.sampled_np1()
    );
    assert_eq!(
        js_sys::Reflect::get(&record, &"sampledNp2".into())
            .unwrap()
            .as_f64(),
        report.sampled_np2()
    );
    assert!(
        js_sys::Reflect::get(&record, &"crossingNumber".into())
            .unwrap()
            .is_undefined()
    );

    // Adversarial inputs return Err(JsError) without panicking.
    let short_layout = JsLayout2d::zeros(3.0).unwrap();
    assert!(js_scale_normalized_stress(&grid, &short_layout).is_err());
    assert!(js_sweep_crossing_number(&grid, &short_layout).is_err());
    assert!(js_neighborhood_preservation_1hop(&grid, &short_layout).is_err());
    assert!(js_prism_distortion(&layout, &short_layout).is_err());

    let mut nan_layout = JsLayout2d::zeros(16.0).unwrap();
    nan_layout.set_x(0.0, f32::NAN).unwrap();
    assert!(js_scale_normalized_stress(&grid, &nan_layout).is_err());
    assert!(js_node_resolution_v6(&nan_layout).is_err());
    assert!(js_aspect_ratio_v6(&nan_layout).is_err());

    // Seed and sample/pivot bounds
    assert!(js_sampled_scale_normalized_stress(&grid, &layout, Some(0.0), Some(42.0)).is_err());
    assert!(js_sampled_scale_normalized_stress(&grid, &layout, Some(4.0), Some(-1.0)).is_err());
    assert!(js_sampled_scale_normalized_stress(&grid, &layout, Some(4.0), Some(1.5)).is_err());
    assert!(
        js_sampled_scale_normalized_stress(&grid, &layout, Some(4.0), Some(9007199254740992.0))
            .is_err()
    );
    assert!(js_sampled_crossing_number(&grid, &layout, Some(0.0), Some(42.0)).is_err());
    assert!(js_sampled_np1(&grid, &layout, Some(0.0), Some(42.0)).is_err());
    assert!(js_sampled_np2(&grid, &layout, Some(0.0), Some(42.0)).is_err());
    let empty_g = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let empty_l = JsLayout2d::zeros(0.0).unwrap();
    assert_eq!(js_sampled_np1(&empty_g, &empty_l, None, None).unwrap(), 0.0);
    assert_eq!(js_sampled_np2(&empty_g, &empty_l, None, None).unwrap(), 0.0);

    // Bad targets
    assert!(
        js_evaluate_metrics(
            &grid,
            &layout,
            &JsValue::from_str("sns"),
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    let bad_target = js_sys::Array::of1(&"nonExistentMetric".into());
    assert!(js_evaluate_metrics(&grid, &layout, &bad_target, None, None, None, None).is_err());
    let non_str_target = js_sys::Array::of1(&JsValue::from_f64(123.0));
    assert!(js_evaluate_metrics(&grid, &layout, &non_str_target, None, None, None, None).is_err());
}

#[cfg(feature = "delaunay")]
#[wasm_bindgen_test]
fn delaunay_triangulation_square_collinear_and_errors() {
    let square = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let dt = js_delaunay_triangulation(&square).expect("square delaunay");
    assert_eq!(dt.num_nodes(), 4);
    assert_eq!(dt.edges().len(), 10);
    assert_eq!(dt.triangles().len(), 6);

    let collinear = JsLayout2d::from_coords(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0]).unwrap();
    let dt_col = js_delaunay_triangulation(&collinear).expect("collinear delaunay");
    assert_eq!(dt_col.edges(), vec![0, 1, 1, 2]);
    assert!(dt_col.triangles().is_empty());

    let mut nan_layout = JsLayout2d::zeros(3.0).unwrap();
    nan_layout.set_y(1.0, f32::INFINITY).unwrap();
    assert!(js_delaunay_triangulation(&nan_layout).is_err());
}
