#[cfg(feature = "delaunay")]
use egraph_wasm::v6::js_delaunay_triangulation;
use egraph_wasm::v6::{
    JsClusterAreaReport, JsCsrGraph, JsLayout2d, js_adjusted_patch_nig,
    js_adjusted_patch_nig_multiscale, js_angular_resolution_v6, js_aspect_ratio_v6,
    js_cluster_area_faithfulness, js_cluster_area_faithfulness_detail, js_crosslessness,
    js_dgg_preservation, js_drng_preservation, js_edge_angle, js_evaluate_metrics,
    js_evaluate_metrics_with_options, js_gabriel_graph_property_v6, js_ideal_edge_lengths_v6,
    js_landmark_stratified_rta, js_landmark_stratified_rta_detail,
    js_landmark_stratified_rta_from_landmarks, js_layout_degeneracy, js_minimum_angle,
    js_minimum_angle_summary, js_neighborhood_preservation_1hop,
    js_neighborhood_preservation_2hop_v6, js_node_resolution_v6, js_prism_distortion,
    js_sampled_crossing_number, js_sampled_crosslessness, js_sampled_np1, js_sampled_np2,
    js_sampled_scale_normalized_stress, js_scale_normalized_stress, js_sweep_crossing_angle,
    js_sweep_crossing_number, js_torus_crossing_number,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn u32_js(labels: &[u32]) -> JsValue {
    js_sys::Uint32Array::from(labels).into()
}

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
        None,
        None,
        None,
        None,
        None,
        None,
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    let bad_target = js_sys::Array::of1(&"nonExistentMetric".into());
    assert!(
        js_evaluate_metrics(
            &grid,
            &layout,
            &bad_target,
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
        .is_err()
    );
    let non_str_target = js_sys::Array::of1(&JsValue::from_f64(123.0));
    assert!(
        js_evaluate_metrics(
            &grid,
            &layout,
            &non_str_target,
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
        .is_err()
    );
}

#[wasm_bindgen_test]
fn w3_five_new_metrics_and_dispatcher_parity() {
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

    let (grid, grid_layout) = make_grid_4x4();
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
    let k4_labels_js = u32_js(&[0, 0, 1, 1]);
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
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn cluster_area_faithfulness_detail_invariance_and_adversarial() {
    let mut coords = Vec::with_capacity(100);
    let mut labels = Vec::with_capacity(50);
    let mut mixed_labels = Vec::with_capacity(50);
    for c in 0..5u32 {
        for r in 0..5u32 {
            coords.push(c as f32);
            coords.push(r as f32);
            labels.push(0);
            mixed_labels.push((c + r) % 2);
        }
    }
    for c in 0..5u32 {
        for r in 0..5u32 {
            coords.push((c + 10) as f32);
            coords.push(r as f32);
            labels.push(1);
            mixed_labels.push((c + r) % 2);
        }
    }
    let sep_layout = JsLayout2d::from_coords(&coords).unwrap();
    let labels_js = u32_js(&labels);
    let mixed_labels_js = u32_js(&mixed_labels);
    let caf_sep =
        js_cluster_area_faithfulness(&sep_layout, &labels_js, None, None, None, None, None, None)
            .unwrap();
    let det: JsClusterAreaReport = js_cluster_area_faithfulness_detail(
        &sep_layout,
        &labels_js,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!((caf_sep - det.caf()).abs() < 1e-12);
    assert!(
        caf_sep > 0.5,
        "separated clusters should score high: {caf_sep}"
    );
    assert!((0.0..=1.0).contains(&det.purity_adj()));
    assert!((0.0..=1.0).contains(&det.prop_vis()));
    assert!((0.0..=1.0).contains(&det.res_density()));
    assert!((0.0..=1.0).contains(&det.res_deconv()));
    assert!((0.0..=1.0).contains(&det.res_combined()));
    assert!((0.0..=1.0).contains(&det.spatial_coherence()));
    assert!((0.0..=1.0).contains(&det.in_frac()));
    assert_eq!(det.num_clusters(), 2);
    assert_eq!(det.num_nodes(), 50);

    assert_eq!(det.cluster_ids(), vec![0, 1]);
    assert_eq!(det.counts(), vec![25, 25]);
    assert_eq!(det.p().len(), 2);
    assert_eq!(det.q().len(), 2);
    assert_eq!(det.purity().len(), 2);
    assert_eq!(det.resolution().len(), 2);
    assert_eq!(det.res_density_per_cluster().len(), 2);
    assert_eq!(det.res_deconv_per_cluster().len(), 2);
    assert_eq!(det.caf_per_cluster().len(), 2);
    assert_eq!(det.coherence().len(), 2);
    assert!((det.p().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((det.q().iter().sum::<f64>() - 1.0).abs() < 1e-12);

    let rec = det.to_record().unwrap();
    assert!(
        (js_sys::Reflect::get(&rec, &"caf".into())
            .unwrap()
            .as_f64()
            .unwrap()
            - caf_sep)
            .abs()
            < 1e-12
    );
    assert_eq!(
        js_sys::Reflect::get(&rec, &"numClusters".into())
            .unwrap()
            .as_f64(),
        Some(2.0)
    );
    assert_eq!(
        js_sys::Reflect::get(&rec, &"numNodes".into())
            .unwrap()
            .as_f64(),
        Some(50.0)
    );
    for key in [
        "purityAdj",
        "propVis",
        "resDensity",
        "resDeconv",
        "resCombined",
        "spatialCoherence",
        "inFrac",
        "clusterIds",
        "counts",
        "p",
        "q",
        "purity",
        "resolution",
        "resDensityPerCluster",
        "resDeconvPerCluster",
        "cafPerCluster",
        "coherence",
    ] {
        assert!(
            !js_sys::Reflect::get(&rec, &key.into())
                .unwrap()
                .is_undefined(),
            "missing key {key} in ClusterAreaReport.toRecord()"
        );
    }

    let caf_mixed = js_cluster_area_faithfulness(
        &sep_layout,
        &mixed_labels_js,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(caf_mixed < caf_sep);

    // Exact grid-isometry invariance: scale + shift, 90-degree rotation, reflection.
    let (coord_pairs, _) = coords.as_chunks::<2>();
    let scaled_shifted: Vec<f32> = coord_pairs
        .iter()
        .flat_map(|p| [p[0] * 4.0 + 12.0, p[1] * 4.0 - 7.0])
        .collect();
    let rot90: Vec<f32> = coord_pairs.iter().flat_map(|p| [-p[1], p[0]]).collect();
    let refl: Vec<f32> = coord_pairs.iter().flat_map(|p| [p[0], -p[1]]).collect();
    for variant in [scaled_shifted, rot90, refl] {
        let lv = JsLayout2d::from_coords(&variant).unwrap();
        let caf_v =
            js_cluster_area_faithfulness(&lv, &labels_js, None, None, None, None, None, None)
                .unwrap();
        assert!((caf_v - caf_sep).abs() < 1e-12);
    }

    // Imbalanced clusters: stevensGamma and boundaryR95 = +inf.
    let mut imb_coords = Vec::with_capacity(48);
    let mut imb_labels = Vec::with_capacity(24);
    for i in 0..24u32 {
        let x = if i < 20 {
            -2.0 + 0.5 * (i as f32 / 19.0)
        } else {
            1.5 + 0.5 * ((i - 20) as f32 / 3.0)
        };
        let y = (i as f32 * 0.7).sin() * 0.2;
        imb_coords.push(x);
        imb_coords.push(y);
        imb_labels.push(if i < 20 { 0 } else { 1 });
    }
    let imb_layout = JsLayout2d::from_coords(&imb_coords).unwrap();
    let imb_labels_js = u32_js(&imb_labels);
    let det_g05 = js_cluster_area_faithfulness_detail(
        &imb_layout,
        &imb_labels_js,
        None,
        None,
        None,
        None,
        None,
        Some(0.5),
    )
    .unwrap();
    let det_g20 = js_cluster_area_faithfulness_detail(
        &imb_layout,
        &imb_labels_js,
        None,
        None,
        None,
        None,
        None,
        Some(2.0),
    )
    .unwrap();
    assert!((det_g05.p().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((det_g20.p().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(det_g05.p()[0] < det_g20.p()[0]);

    let det_inf = js_cluster_area_faithfulness_detail(
        &imb_layout,
        &imb_labels_js,
        None,
        None,
        None,
        None,
        Some(f64::INFINITY),
        None,
    )
    .unwrap();
    assert!((det_inf.in_frac() - 1.0).abs() < 1e-12);
    assert!((0.0..=1.0).contains(&det_inf.caf()));

    // Plain JS Array of integers matches Uint32Array, including u32::MAX.
    let l5 = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 3.0]).unwrap();
    let l5_u32_vec = [100u32, 5, 100, u32::MAX, 5];
    let l5_u32_js = u32_js(&l5_u32_vec);
    let l5_arr = js_sys::Array::new();
    for v in l5_u32_vec {
        l5_arr.push(&JsValue::from_f64(v as f64));
    }
    assert_eq!(
        js_cluster_area_faithfulness(&l5, &l5_u32_js, None, None, None, None, None, None).unwrap(),
        js_cluster_area_faithfulness(&l5, &l5_arr, None, None, None, None, None, None).unwrap()
    );

    // evaluateMetrics integration + lazy validation.
    let path_edges: Vec<u32> = (0..23u32).flat_map(|i| [i, i + 1]).collect();
    let imb_graph =
        JsCsrGraph::from_edge_list(24.0, &path_edges, None, &JsValue::from_bool(false), None)
            .unwrap();
    let expected_custom = js_cluster_area_faithfulness(
        &imb_layout,
        &imb_labels_js,
        Some(64.0),
        Some(0.9),
        Some(8.0),
        Some(0.2),
        Some(2.5),
        Some(0.7),
    )
    .unwrap();
    for alias in [
        "clusterAreaFaithfulness",
        "cluster_area_faithfulness",
        "gcaf",
    ] {
        let t_arr = js_sys::Array::of1(&alias.into());
        let rep = js_evaluate_metrics_with_options(
            &imb_graph,
            &imb_layout,
            &t_arr,
            None,
            None,
            None,
            None,
            &imb_labels_js,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(64.0),
            Some(0.9),
            Some(8.0),
            Some(0.2),
            Some(2.5),
            Some(0.7),
        )
        .unwrap();
        let val = rep.cluster_area_faithfulness().unwrap();
        assert!((val - expected_custom).abs() < 1e-12);
        let rep_rec = rep.to_record().unwrap();
        let rec_val = js_sys::Reflect::get(&rep_rec, &"clusterAreaFaithfulness".into())
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((rec_val - expected_custom).abs() < 1e-12);
    }

    // Lazy gMax validation: gMax=4 succeeds when G-CAF is not requested, fails when requested.
    let ar_only = js_sys::Array::of1(&"aspectRatio".into());
    assert!(
        js_evaluate_metrics_with_options(
            &imb_graph,
            &imb_layout,
            &ar_only,
            None,
            None,
            None,
            None,
            &JsValue::NULL,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(4.0),
            None,
            None,
            None,
            None,
            None,
        )
        .is_ok()
    );
    let gcaf_only = js_sys::Array::of1(&"gcaf".into());
    assert!(
        js_evaluate_metrics_with_options(
            &imb_graph,
            &imb_layout,
            &gcaf_only,
            None,
            None,
            None,
            None,
            &imb_labels_js,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(4.0),
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        js_evaluate_metrics_with_options(
            &imb_graph,
            &imb_layout,
            &gcaf_only,
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
        )
        .is_err()
    );

    // Degenerate boundaries: N=0, N=1, K=1, collapsed coordinates.
    let l0 = JsLayout2d::zeros(0.0).unwrap();
    let d0 =
        js_cluster_area_faithfulness_detail(&l0, &u32_js(&[]), None, None, None, None, None, None)
            .unwrap();
    assert_eq!(d0.caf(), 0.0);
    assert_eq!(d0.num_clusters(), 0);
    assert_eq!(d0.num_nodes(), 0);

    let l1 = JsLayout2d::from_coords(&[1.0, 2.0]).unwrap();
    let d1 =
        js_cluster_area_faithfulness_detail(&l1, &u32_js(&[5]), None, None, None, None, None, None)
            .unwrap();
    assert_eq!(d1.caf(), 0.0);
    assert_eq!(d1.num_clusters(), 1);
    assert_eq!(d1.num_nodes(), 1);

    let l4 = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let dk1 = js_cluster_area_faithfulness_detail(
        &l4,
        &u32_js(&[3, 3, 3, 3]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(dk1.caf(), 0.0);
    assert_eq!(dk1.num_clusters(), 1);
    assert_eq!(dk1.num_nodes(), 4);

    let collapsed = JsLayout2d::from_coords(&[2.0, -1.0].repeat(6)).unwrap();
    assert_eq!(
        js_cluster_area_faithfulness(
            &collapsed,
            &u32_js(&[0, 0, 0, 1, 1, 1]),
            None,
            None,
            None,
            None,
            None,
            None
        )
        .unwrap(),
        0.0
    );

    // Adversarial error checks (must return Err, never panic).
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &u32_js(&imb_labels[..23]),
            None,
            None,
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    // Invalid JS communityLabels must reject with Err rather than ToUint32 coercion.
    let make_bad_arr = |first: JsValue| -> JsValue {
        let arr = js_sys::Array::new();
        arr.push(&first);
        for &v in &imb_labels[1..] {
            arr.push(&JsValue::from_f64(v as f64));
        }
        arr.into()
    };
    let array_like_obj = js_sys::Object::new();
    js_sys::Reflect::set(&array_like_obj, &"length".into(), &24.0.into()).unwrap();
    let bad_label_inputs: Vec<JsValue> = vec![
        make_bad_arr(JsValue::from_f64(-1.0)),
        make_bad_arr(JsValue::from_f64(1.5)),
        make_bad_arr(JsValue::from_f64(f64::NAN)),
        make_bad_arr(JsValue::from_f64(f64::INFINITY)),
        make_bad_arr(JsValue::from_f64(4_294_967_296.0)),
        make_bad_arr(JsValue::from_str("bad")),
        make_bad_arr(JsValue::NULL),
        make_bad_arr(JsValue::UNDEFINED),
        array_like_obj.into(),
        JsValue::from_str("000000000000000000001111"),
    ];
    for bad in &bad_label_inputs {
        assert!(
            js_cluster_area_faithfulness(&imb_layout, bad, None, None, None, None, None, None)
                .is_err()
        );
        assert!(
            js_cluster_area_faithfulness_detail(
                &imb_layout,
                bad,
                None,
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            js_evaluate_metrics_with_options(
                &imb_graph,
                &imb_layout,
                &gcaf_only,
                None,
                None,
                None,
                None,
                bad,
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
            .is_err()
        );
    }
    for nullish in [JsValue::NULL, JsValue::UNDEFINED] {
        assert!(
            js_cluster_area_faithfulness(&imb_layout, &nullish, None, None, None, None, None, None)
                .is_err()
        );
        assert!(
            js_cluster_area_faithfulness_detail(
                &imb_layout,
                &nullish,
                None,
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            js_evaluate_metrics_with_options(
                &imb_graph,
                &imb_layout,
                &gcaf_only,
                None,
                None,
                None,
                None,
                &nullish,
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
            .is_err()
        );
    }

    let mut nan_coords = imb_coords.clone();
    nan_coords[0] = f32::NAN;
    let nan_layout = JsLayout2d::from_coords(&nan_coords).unwrap();
    assert!(
        js_cluster_area_faithfulness(
            &nan_layout,
            &imb_labels_js,
            None,
            None,
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    for bad_g in [11.0, 1.5, -1.0] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                Some(bad_g),
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
    }
    for bad_sigma in [0.0, f64::NAN, 500.0] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                None,
                Some(bad_sigma),
                None,
                None,
                None,
                None
            )
            .is_err()
        );
    }
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &imb_labels_js,
            None,
            None,
            Some(0.0),
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &imb_labels_js,
            None,
            None,
            None,
            Some(-1.0),
            None,
            None
        )
        .is_err()
    );
    for bad_r95 in [0.0, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                None,
                None,
                None,
                None,
                Some(bad_r95),
                None
            )
            .is_err()
        );
    }
    for bad_gamma in [0.0, f64::INFINITY] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                None,
                None,
                None,
                None,
                None,
                Some(bad_gamma)
            )
            .is_err()
        );
    }
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
