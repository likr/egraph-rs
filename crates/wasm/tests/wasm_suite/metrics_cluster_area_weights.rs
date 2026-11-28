use crate::util;

use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::metrics_cluster_rta::{
    js_cluster_area_faithfulness, js_cluster_area_faithfulness_detail,
};
use egraph_wasm::metrics_eval_dispatch::js_evaluate_metrics_2d;
use util::fixtures::{make_imbalanced_cluster_fixture, make_opts, u32_typed_js};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn cluster_area_faithfulness_weights_and_degenerate_boundaries() {
    let (imb_graph, imb_layout, _imb_coords, imb_labels, imb_labels_js) =
        make_imbalanced_cluster_fixture();

    // Lazy gMax validation: gMax=4 succeeds when G-CAF is not requested, fails when requested.
    let ar_only = js_sys::Array::of1(&"aspectRatio".into());
    assert!(
        js_evaluate_metrics_2d(
            &imb_graph,
            &imb_layout,
            &ar_only,
            &make_opts(&[("gMax", JsValue::from_f64(4.0))]),
        )
        .is_ok()
    );
    let gcaf_only = js_sys::Array::of1(&"gcaf".into());
    assert!(
        js_evaluate_metrics_2d(
            &imb_graph,
            &imb_layout,
            &gcaf_only,
            &make_opts(&[
                ("labels", imb_labels_js.clone()),
                ("gMax", JsValue::from_f64(4.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_evaluate_metrics_2d(&imb_graph, &imb_layout, &gcaf_only, &JsValue::UNDEFINED).is_err()
    );

    // Degenerate boundaries: N=0, N=1, K=1, collapsed coordinates.
    let l0 = JsLayout2d::zeros(0.0).unwrap();
    let d0 =
        js_cluster_area_faithfulness_detail(&l0, &u32_typed_js(&[]), &JsValue::UNDEFINED).unwrap();
    assert_eq!(d0.caf(), 0.0);
    assert_eq!(d0.effective_boundary_r95(), 0.0);
    assert_eq!(d0.g_eff(), 0);
    assert_eq!(d0.num_clusters(), 0);
    assert_eq!(d0.num_nodes(), 0);

    let l1 = JsLayout2d::from_coords(&[1.0, 2.0]).unwrap();
    let d1 =
        js_cluster_area_faithfulness_detail(&l1, &u32_typed_js(&[5]), &JsValue::UNDEFINED).unwrap();
    assert_eq!(d1.caf(), 0.0);
    assert_eq!(d1.effective_boundary_r95(), 0.0);
    assert_eq!(d1.g_eff(), 0);
    assert_eq!(d1.num_clusters(), 1);
    assert_eq!(d1.num_nodes(), 1);

    let l4 = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let dk1 =
        js_cluster_area_faithfulness_detail(&l4, &u32_typed_js(&[3, 3, 3, 3]), &JsValue::UNDEFINED)
            .unwrap();
    assert_eq!(dk1.caf(), 0.0);
    assert_eq!(dk1.effective_boundary_r95(), 0.0);
    assert_eq!(dk1.g_eff(), 0);
    assert_eq!(dk1.num_clusters(), 1);
    assert_eq!(dk1.num_nodes(), 4);

    let collapsed = JsLayout2d::from_coords(&[2.0, -1.0].repeat(6)).unwrap();
    assert_eq!(
        js_cluster_area_faithfulness(
            &collapsed,
            &u32_typed_js(&[0, 0, 0, 1, 1, 1]),
            &JsValue::UNDEFINED,
        )
        .unwrap(),
        0.0
    );

    // nodeWeights & clusterWeights positive parity + adversarial non-panic errors.
    let nw_vec: Vec<f64> = imb_labels
        .iter()
        .map(|&l| if l == 0 { 1.5 } else { 3.0 })
        .collect();
    let nw_js: JsValue = js_sys::Float64Array::from(nw_vec.as_slice()).into();
    let cw_js: JsValue = js_sys::Float64Array::from([30.0_f64, 12.0].as_slice()).into();
    let caf_nw = js_cluster_area_faithfulness(
        &imb_layout,
        &imb_labels_js,
        &make_opts(&[("nodeWeights", nw_js.clone())]),
    )
    .unwrap();
    let caf_cw = js_cluster_area_faithfulness(
        &imb_layout,
        &imb_labels_js,
        &make_opts(&[("clusterWeights", cw_js.clone())]),
    )
    .unwrap();
    assert!((caf_nw - caf_cw).abs() < 1e-12);
    let eval_cw = js_evaluate_metrics_2d(
        &imb_graph,
        &imb_layout,
        &gcaf_only,
        &make_opts(&[
            ("labels", imb_labels_js.clone()),
            ("clusterWeights", cw_js.clone()),
        ]),
    )
    .unwrap();
    assert!((eval_cw.cluster_area_faithfulness().unwrap() - caf_cw).abs() < 1e-12);

    // Adversarial weight errors must return Err, never panic.
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &imb_labels_js,
            &make_opts(&[("nodeWeights", nw_js), ("clusterWeights", cw_js)]),
        )
        .is_err()
    );
    let bad_cw_neg: JsValue = js_sys::Float64Array::from([-1.0_f64, 2.0].as_slice()).into();
    let bad_cw_nan: JsValue = js_sys::Float64Array::from([f64::NAN, 2.0].as_slice()).into();
    let bad_cw_len: JsValue = js_sys::Float64Array::from([1.0_f64].as_slice()).into();
    let bad_cw_str: JsValue = JsValue::from_str("12");
    for bad_w in [&bad_cw_neg, &bad_cw_nan, &bad_cw_len, &bad_cw_str] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                &make_opts(&[("clusterWeights", bad_w.clone())]),
            )
            .is_err()
        );
    }
}
