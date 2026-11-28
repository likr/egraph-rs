mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::metrics_crossings::{
    js_aspect_ratio, js_crossing_edges, js_crossing_summary, js_node_resolution,
    js_sampled_crossing_number, js_sweep_crossing_number,
};
use egraph_wasm::metrics_eval_dispatch::js_evaluate_metrics_2d;
use egraph_wasm::metrics_functions::{
    js_gabriel_graph_edges, js_ideal_edge_lengths_with_unit, js_prism_distortion,
    js_prism_distortion_with_edges, js_relative_neighborhood_edges,
};
use egraph_wasm::metrics_geometry::{
    js_all_pairs_bfs, js_bfs_single_source, js_neighborhood_preservation_1hop, js_sampled_np1,
    js_sampled_np2, js_sampled_scale_normalized_stress, js_scale_normalized_stress,
    js_scale_normalized_stress_from_distances,
};
use egraph_wasm::metrics_non_euclidean_crossings::js_crossing_edges_torus_2d;
use egraph_wasm::spatial_dataset::{
    js_delaunay_to_csr, js_delaunay_triangulation, js_parse_csr_csv, js_rectangle_overlaps,
};
use util::fixtures::{make_grid, make_grid_4x4_with_layout, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn selective_evaluate_metrics_and_adversarial_errors() {
    let (grid, layout) = make_grid_4x4_with_layout();
    let targets = js_sys::Array::of5(
        &"sns".into(),
        &"np2".into(),
        &"sampledNp1".into(),
        &"sampledNp2".into(),
        &"aspectRatio".into(),
    );
    let report = js_evaluate_metrics_2d(
        &grid,
        &layout,
        &targets,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(8.0)),
            ("numSamples", JsValue::from_f64(64.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
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

    // Adversarial inputs return Err(JsValue) with .code without panicking.
    let short_layout = JsLayout2d::zeros(3.0).unwrap();
    let dim_err = js_scale_normalized_stress(&grid, &short_layout).unwrap_err();
    assert_eq!(
        js_sys::Reflect::get(&dim_err, &"code".into())
            .unwrap()
            .as_string(),
        Some("DimensionMismatch".to_owned())
    );
    assert!(js_sweep_crossing_number(&grid, &short_layout).is_err());
    assert!(js_neighborhood_preservation_1hop(&grid, &short_layout).is_err());
    assert!(js_prism_distortion(&layout, &short_layout).is_err());

    let mut nan_layout = JsLayout2d::zeros(16.0).unwrap();
    nan_layout.inner_mut().as_mut_slice()[0] = f32::NAN;
    let nan_err = js_scale_normalized_stress(&grid, &nan_layout).unwrap_err();
    assert_eq!(
        js_sys::Reflect::get(&nan_err, &"code".into())
            .unwrap()
            .as_string(),
        Some("NonFiniteValue".to_owned())
    );
    assert!(js_node_resolution(&nan_layout).is_err());
    assert!(js_aspect_ratio(&nan_layout).is_err());

    // Seed and sample/pivot bounds
    assert!(
        js_sampled_scale_normalized_stress(
            &grid,
            &layout,
            &make_opts(&[
                ("numPivots", JsValue::from_f64(0.0)),
                ("seed", JsValue::from_f64(42.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_scale_normalized_stress(
            &grid,
            &layout,
            &make_opts(&[
                ("numPivots", JsValue::from_f64(4.0)),
                ("seed", JsValue::from_f64(-1.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_scale_normalized_stress(
            &grid,
            &layout,
            &make_opts(&[
                ("numPivots", JsValue::from_f64(4.0)),
                ("seed", JsValue::from_f64(1.5)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_scale_normalized_stress(
            &grid,
            &layout,
            &make_opts(&[
                ("numPivots", JsValue::from_f64(4.0)),
                ("seed", JsValue::from_f64(9007199254740992.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_crossing_number(
            &grid,
            &layout,
            &make_opts(&[
                ("numSamples", JsValue::from_f64(0.0)),
                ("seed", JsValue::from_f64(42.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_np1(
            &grid,
            &layout,
            &make_opts(&[
                ("numSamples", JsValue::from_f64(0.0)),
                ("seed", JsValue::from_f64(42.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_sampled_np2(
            &grid,
            &layout,
            &make_opts(&[
                ("numSamples", JsValue::from_f64(0.0)),
                ("seed", JsValue::from_f64(42.0)),
            ]),
        )
        .is_err()
    );
    let empty_g = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let empty_l = JsLayout2d::zeros(0.0).unwrap();
    assert_eq!(
        js_sampled_np1(&empty_g, &empty_l, &JsValue::UNDEFINED).unwrap(),
        0.0
    );
    assert_eq!(
        js_sampled_np2(&empty_g, &empty_l, &JsValue::UNDEFINED).unwrap(),
        0.0
    );

    // Bad targets
    assert!(
        js_evaluate_metrics_2d(
            &grid,
            &layout,
            &JsValue::from_str("sns"),
            &JsValue::UNDEFINED
        )
        .is_err()
    );
    let bad_target = js_sys::Array::of1(&"nonExistentMetric".into());
    assert!(js_evaluate_metrics_2d(&grid, &layout, &bad_target, &JsValue::UNDEFINED).is_err());
    let non_str_target = js_sys::Array::of1(&JsValue::from_f64(123.0));
    assert!(js_evaluate_metrics_2d(&grid, &layout, &non_str_target, &JsValue::UNDEFINED).is_err());
}

#[wasm_bindgen_test]
fn dataset_spatial_crossings_and_bfs_metrics() {
    let parsed = js_parse_csr_csv("3,2\n0,1\n1,2\n", Some(false), Some(true)).unwrap();
    assert_eq!(parsed.num_nodes(), 3);
    assert_eq!(parsed.num_edges(), 2);

    let parsed_w =
        js_parse_csr_csv("3,2\n0,1,2.5\n1, 2, 1.25\n", Some(false), Some(false)).unwrap();
    assert!(parsed_w.is_weighted());
    assert_eq!(parsed_w.weights().unwrap(), vec![2.5_f32, 2.5, 1.25, 1.25]);
    assert!(js_parse_csr_csv("3,2\n0,1,2.0\n1,2\n", Some(false), Some(false)).is_err());

    let rects = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0, 5.0, 0.0]).unwrap();
    let overlaps = js_rectangle_overlaps(&rects, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]).unwrap();
    assert_eq!(overlaps, vec![0, 1]);

    // X-shaped graph for crossings
    let cross_graph = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 3, 1, 2],
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let cross_layout = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let pairs = js_crossing_edges(&cross_graph, &cross_layout).unwrap();
    assert_eq!(pairs.len(), 1);
    let summary = js_crossing_summary(&cross_graph, &cross_layout).unwrap();
    assert_eq!(summary.count(), 1);
    assert!(summary.min_angle_rad().unwrap_or(0.0) > 0.0);
    assert!(summary.angle_cos_sq_sum() >= 0.0);

    let torus_pairs = js_crossing_edges_torus_2d(&cross_graph, &cross_layout).unwrap();
    assert!(!torus_pairs.is_empty());

    let grid = make_grid(3, 3);
    let mut grid_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            grid_coords.push(c as f32);
            grid_coords.push(r as f32);
        }
    }
    let grid_layout = JsLayout2d::from_coords(&grid_coords).unwrap();
    let iel_unit = js_ideal_edge_lengths_with_unit(&grid, &grid_layout, 1.0).unwrap();
    assert!(iel_unit.is_finite());

    let targets = js_sys::Array::of2(&"sns".into(), &"idealEdgeLengths".into());
    let report = js_evaluate_metrics_2d(
        &grid,
        &grid_layout,
        &targets,
        &make_opts(&[("idealEdgeLength", JsValue::from_f64(1.0))]),
    )
    .unwrap();
    assert!(report.scale_normalized_stress().is_some());
    assert!(report.ideal_edge_lengths().is_some());

    let gg_edges = js_gabriel_graph_edges(&grid_layout).unwrap();
    let rng_edges = js_relative_neighborhood_edges(&grid_layout).unwrap();
    assert!(!gg_edges.is_empty() && !rng_edges.is_empty());

    let apsp = js_all_pairs_bfs(&grid, Some(1.0)).unwrap();
    assert_eq!(apsp.len(), 81);
    let sssp = js_bfs_single_source(&grid, 0.0, Some(2.0)).unwrap();
    assert_eq!(sssp.len(), 9);
    assert!((sssp[8] - 8.0).abs() < 1e-9);
    let sns = js_scale_normalized_stress_from_distances(&grid_layout, &apsp, None).unwrap();
    assert!(sns.is_finite() && sns < 0.25);

    let mut adjusted = grid_layout.clone();
    adjusted.set_x(0.0, 0.05).unwrap();
    let p_dist = js_prism_distortion_with_edges(&grid_layout, &adjusted, &gg_edges).unwrap();
    assert!(p_dist.sigma_dist().is_finite() && p_dist.sigma_disp().is_finite());

    let del_csr = js_delaunay_to_csr(&grid_layout).unwrap();
    assert_eq!(del_csr.num_nodes(), 9);
    assert!(del_csr.num_edges() >= 12);
}

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
    nan_layout.inner_mut().as_mut_slice()[3] = f32::INFINITY;
    assert!(js_delaunay_triangulation(&nan_layout).is_err());
}
