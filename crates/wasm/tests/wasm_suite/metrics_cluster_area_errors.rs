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
fn cluster_area_faithfulness_adversarial_label_and_param_errors() {
    let (imb_graph, imb_layout, imb_coords, imb_labels, imb_labels_js) =
        make_imbalanced_cluster_fixture();
    let gcaf_only = js_sys::Array::of1(&"gcaf".into());

    // Adversarial error checks (must return Err, never panic).
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &u32_typed_js(&imb_labels[..23]),
            &JsValue::UNDEFINED,
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
        assert!(js_cluster_area_faithfulness(&imb_layout, bad, &JsValue::UNDEFINED).is_err());
        assert!(
            js_cluster_area_faithfulness_detail(&imb_layout, bad, &JsValue::UNDEFINED).is_err()
        );
        assert!(
            js_evaluate_metrics_2d(
                &imb_graph,
                &imb_layout,
                &gcaf_only,
                &make_opts(&[("labels", bad.clone())]),
            )
            .is_err()
        );
    }
    for nullish in [JsValue::NULL, JsValue::UNDEFINED] {
        assert!(js_cluster_area_faithfulness(&imb_layout, &nullish, &JsValue::UNDEFINED).is_err());
        assert!(
            js_cluster_area_faithfulness_detail(&imb_layout, &nullish, &JsValue::UNDEFINED)
                .is_err()
        );
        assert!(
            js_evaluate_metrics_2d(
                &imb_graph,
                &imb_layout,
                &gcaf_only,
                &make_opts(&[("labels", nullish)]),
            )
            .is_err()
        );
    }

    let mut nan_coords = imb_coords.clone();
    nan_coords[0] = f32::NAN;
    assert!(JsLayout2d::from_coords(&nan_coords).is_err());

    let mut nan_layout = JsLayout2d::from_coords(&imb_coords).unwrap();
    nan_layout.inner_mut().as_mut_slice()[0] = f32::NAN;
    assert!(
        js_cluster_area_faithfulness(&nan_layout, &imb_labels_js, &JsValue::UNDEFINED).is_err()
    );
    for bad_g in [11.0, 1.5, -1.0] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                &make_opts(&[("gMax", JsValue::from_f64(bad_g))]),
            )
            .is_err()
        );
    }
    for bad_sigma in [0.0, f64::NAN, 500.0] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                &make_opts(&[("sigmaCells", JsValue::from_f64(bad_sigma))]),
            )
            .is_err()
        );
    }
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &imb_labels_js,
            &make_opts(&[("kappa", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );
    assert!(
        js_cluster_area_faithfulness(
            &imb_layout,
            &imb_labels_js,
            &make_opts(&[("tauFrac", JsValue::from_f64(-1.0))]),
        )
        .is_err()
    );
    for bad_r95 in [0.0, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                &make_opts(&[("boundaryR95", JsValue::from_f64(bad_r95))]),
            )
            .is_err()
        );
    }
    for bad_gamma in [0.0, f64::INFINITY] {
        assert!(
            js_cluster_area_faithfulness(
                &imb_layout,
                &imb_labels_js,
                &make_opts(&[("stevensGamma", JsValue::from_f64(bad_gamma))]),
            )
            .is_err()
        );
    }
}
