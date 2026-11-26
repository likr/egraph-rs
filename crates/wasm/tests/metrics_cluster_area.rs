mod util;

use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::metrics_cluster_rta::{
    JsClusterAreaReport, js_cluster_area_faithfulness, js_cluster_area_faithfulness_detail,
};
use egraph_wasm::metrics_eval::js_evaluate_metrics_with_options;
use util::fixtures::{make_imbalanced_cluster_fixture, u32_typed_js};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn cluster_area_faithfulness_detail_and_isometry() {
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
    let labels_js = u32_typed_js(&labels);
    let mixed_labels_js = u32_typed_js(&mixed_labels);
    let caf_sep = js_cluster_area_faithfulness(
        &sep_layout,
        &labels_js,
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
    let det: JsClusterAreaReport = js_cluster_area_faithfulness_detail(
        &sep_layout,
        &labels_js,
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
    assert_eq!(det.effective_boundary_r95(), 2.0);
    assert_eq!(det.g_eff(), 12);
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
        "effectiveBoundaryR95",
        "gEff",
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
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
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
        let caf_v = js_cluster_area_faithfulness(
            &lv,
            &labels_js,
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
        assert!((caf_v - caf_sep).abs() < 1e-12);
    }
}

#[wasm_bindgen_test]
fn cluster_area_faithfulness_stevens_gamma_and_eval_integration() {
    // Imbalanced clusters: stevensGamma and boundaryR95 = +inf.
    let (imb_graph, imb_layout, _imb_coords, _imb_labels, imb_labels_js) =
        make_imbalanced_cluster_fixture();
    let det_g05 = js_cluster_area_faithfulness_detail(
        &imb_layout,
        &imb_labels_js,
        None,
        None,
        None,
        None,
        None,
        Some(0.5),
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
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
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
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
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!((det_inf.in_frac() - 1.0).abs() < 1e-12);
    assert!((0.0..=1.0).contains(&det_inf.caf()));

    // Plain JS Array of integers matches Uint32Array, including u32::MAX.
    let l5 = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 3.0]).unwrap();
    let l5_u32_vec = [100u32, 5, 100, u32::MAX, 5];
    let l5_u32_js = u32_typed_js(&l5_u32_vec);
    let l5_arr = js_sys::Array::new();
    for v in l5_u32_vec {
        l5_arr.push(&JsValue::from_f64(v as f64));
    }
    assert_eq!(
        js_cluster_area_faithfulness(
            &l5,
            &l5_u32_js,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED
        )
        .unwrap(),
        js_cluster_area_faithfulness(
            &l5,
            &l5_arr,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED
        )
        .unwrap()
    );

    // evaluateMetrics integration + lazy validation.
    let expected_custom = js_cluster_area_faithfulness(
        &imb_layout,
        &imb_labels_js,
        Some(64.0),
        Some(0.9),
        Some(8.0),
        Some(0.2),
        Some(2.5),
        Some(0.7),
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
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
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED,
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
}
