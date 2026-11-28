mod util;

use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::metrics_cluster_baselines::{
    js_calinski_harabasz_score, js_class_density_measure, js_cq_hac, js_cq_kmeans,
    js_davies_bouldin_score, js_distance_consistency, js_distribution_consistency,
    js_neighborhood_hit, js_nor_metrics, js_nor_metrics_without_graph,
    js_pairwise_distance_overlap_error, js_partition_agreement, js_sepme_gab, js_sepme_gong,
    js_sepme_knng, js_sepme_rng, js_silhouette_score, js_sprawlter_nn, js_sprawlter_nn_radii,
};
use egraph_wasm::metrics_eval_dispatch::js_evaluate_metrics_2d;
use egraph_wasm::overlap_tree_forbid::JsGTreeOverlapRemoval;
use util::fixtures::{
    make_opts, make_overlapping_layout, make_separated_cluster_layout, make_two_cluster_graph,
    u32_array_js,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn cluster_baseline_metrics_and_evaluate_targets() {
    let (graph, labels) = make_two_cluster_graph();
    let labels_js = u32_array_js(&labels);
    let sep = make_separated_cluster_layout();
    let overlap = make_overlapping_layout();

    let pa = js_partition_agreement(&labels_js, &labels_js).unwrap();
    assert!((pa.ari() - 1.0).abs() < 1e-12);
    assert!((pa.ami() - 1.0).abs() < 1e-12);
    assert!((pa.v_measure() - 1.0).abs() < 1e-12);
    assert!((pa.fmi() - 1.0).abs() < 1e-12);
    let pa_rec = pa.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&pa_rec, &JsValue::from_str("vMeasure"))
            .unwrap()
            .as_f64(),
        Some(pa.v_measure())
    );

    let cqk = js_cq_kmeans(
        &sep,
        &labels_js,
        &make_opts(&[
            ("nInit", JsValue::from_f64(5.0)),
            ("maxIterations", JsValue::from_f64(50.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!((cqk.ari() - 1.0).abs() < 1e-12);
    let cqh = js_cq_hac(&sep, &labels_js).unwrap();
    assert!((cqh.homogeneity() - 1.0).abs() < 1e-12);

    assert!((js_distance_consistency(&sep, &labels_js).unwrap() - 1.0).abs() < 1e-12);
    assert!(
        js_distribution_consistency(
            &sep,
            &labels_js,
            &make_opts(&[("gridSize", JsValue::from_f64(16.0))]),
        )
        .unwrap()
            > 0.9
    );
    let cd = js_class_density_measure(
        &sep,
        &labels_js,
        &make_opts(&[("gridSize", JsValue::from_f64(32.0))]),
    )
    .unwrap();
    assert!(cd.cdm() > 0.8 && cd.hdm() > 0.8);
    let cd_rec = cd.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&cd_rec, &JsValue::from_str("cdm"))
            .unwrap()
            .as_f64(),
        Some(cd.cdm())
    );

    assert!(
        (js_sepme_knng(
            &sep,
            &labels_js,
            &make_opts(&[("k", JsValue::from_f64(2.0))])
        )
        .unwrap()
            - 1.0)
            .abs()
            < 1e-12
    );
    let rng_score = js_sepme_rng(&sep, &labels_js).unwrap();
    let gab_score = js_sepme_gab(&sep, &labels_js).unwrap();
    let gong_score = js_sepme_gong(
        &sep,
        &labels_js,
        &make_opts(&[("gamma", JsValue::from_f64(0.35))]),
    )
    .unwrap();
    assert!(rng_score > 0.5 && rng_score <= 1.0);
    assert!(gab_score > 0.5 && gab_score <= 1.0);
    assert!(gong_score > 0.5 && gong_score <= 1.0);
    assert!(
        (js_neighborhood_hit(
            &sep,
            &labels_js,
            &make_opts(&[("k", JsValue::from_f64(3.0))])
        )
        .unwrap()
            - 1.0)
            .abs()
            < 1e-12
    );
    assert!(js_silhouette_score(&sep, &labels_js).unwrap() > 0.8);
    assert!(js_calinski_harabasz_score(&sep, &labels_js).unwrap() > 50.0);
    assert!(js_davies_bouldin_score(&sep, &labels_js).unwrap() < 0.3);
    assert!(
        js_pairwise_distance_overlap_error(
            &sep,
            &labels_js,
            &make_opts(&[("numBins", JsValue::from_f64(32.0))]),
        )
        .unwrap()
            < 0.1
    );

    let hw = vec![0.4_f32; 8];
    let hh = vec![0.4_f32; 8];
    let sp1 = js_sprawlter_nn(&overlap, &hw, &hh).unwrap();
    let sp2 = js_sprawlter_nn_radii(&overlap, &hw).unwrap();
    assert!((sp1 - sp2).abs() < 1e-12);
    assert!(sp1 > 0.0);

    let nor = js_nor_metrics(&graph, &overlap, &sep, &hw, &hh).unwrap();
    assert!(nor.oo_nni().is_finite() && nor.el_rsd().is_finite());
    let nor_no_g = js_nor_metrics_without_graph(&overlap, &sep, &hw, &hh).unwrap();
    assert_eq!(nor_no_g.el_rsd(), 0.0);
    let nor_rec = nor.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&nor_rec, &JsValue::from_str("ooNni"))
            .unwrap()
            .as_f64(),
        Some(nor.oo_nni())
    );

    // evaluateMetrics with all 18 cluster & overlap baseline targets in canonical camelCase
    let targets = js_sys::Array::new();
    for t in [
        "dsc",
        "dc",
        "cdm",
        "hdm",
        "sepmeKnng2",
        "sepmeRng",
        "sepmeGab",
        "sepmeGong",
        "neighborhoodHit",
        "silhouette",
        "calinskiHarabasz",
        "daviesBouldin",
        "cqKmeansAri",
        "cqKmeansAmi",
        "cqHacHom",
        "cqHacAmi",
        "pairwiseDistanceOverlapError",
        "sprawlterNn",
    ] {
        targets.push(&JsValue::from_str(t));
    }
    let eval_opts = make_opts(&[
        ("labels", labels_js.clone()),
        ("sepmeKnngK", JsValue::from_f64(2.0)),
        ("neighborhoodHitK", JsValue::from_f64(3.0)),
        ("dcGridSize", JsValue::from_f64(16.0)),
        ("cdmGridSize", JsValue::from_f64(32.0)),
        ("numBins", JsValue::from_f64(32.0)),
        ("gongGamma", JsValue::from_f64(0.35)),
        ("kmeansNInit", JsValue::from_f64(3.0)),
        ("kmeansMaxIterations", JsValue::from_f64(50.0)),
        ("sprawlterNodeRadius", JsValue::from_f64(0.5)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let rep = js_evaluate_metrics_2d(&graph, &sep, &targets, &eval_opts).unwrap();
    assert!(rep.distance_consistency().is_some());
    assert!(rep.distribution_consistency().is_some());
    assert!(rep.class_density_measure().is_some());
    assert!(rep.histogram_density_measure().is_some());
    assert!(rep.sepme_knng().is_some());
    assert!(rep.sepme_rng().is_some());
    assert!(rep.sepme_gab().is_some());
    assert!(rep.sepme_gong().is_some());
    assert!(rep.neighborhood_hit().is_some());
    assert!(rep.silhouette_score().is_some());
    assert!(rep.calinski_harabasz_score().is_some());
    assert!(rep.davies_bouldin_score().is_some());
    assert!(rep.cq_kmeans_ari().is_some());
    assert!(rep.cq_kmeans_ami().is_some());
    assert!(rep.cq_hac_hom().is_some());
    assert!(rep.cq_hac_ami().is_some());
    assert!(rep.pairwise_distance_overlap_error().is_some());
    assert!(rep.sprawlter_nn().is_some());

    let rep_rec = rep.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&rep_rec, &JsValue::from_str("sepmeKnng"))
            .unwrap()
            .as_f64(),
        rep.sepme_knng()
    );

    // Nested clusterBaselines options support check
    let cb_opts = make_opts(&[("sepmeKnngK", JsValue::from_f64(3.0))]);
    let knng_target = js_sys::Array::of1(&JsValue::from_str("sepmeKnng2"));
    let rep_nested = js_evaluate_metrics_2d(
        &graph,
        &sep,
        &knng_target,
        &make_opts(&[
            ("labels", labels_js.clone()),
            ("seed", JsValue::from_f64(42.0)),
            ("clusterBaselines", cb_opts.into()),
        ]),
    )
    .unwrap();
    assert!(
        (rep_nested.sepme_knng().unwrap()
            - js_sepme_knng(
                &sep,
                &labels_js,
                &make_opts(&[("k", JsValue::from_f64(3.0))])
            )
            .unwrap())
        .abs()
            < 1e-12
    );

    // Lazy validation & error path: `sns` without `labels` succeeds; `dsc` without `labels` fails.
    let sns_only = js_sys::Array::of1(&JsValue::from_str("sns"));
    let sns_rep = js_evaluate_metrics_2d(&graph, &sep, &sns_only, &JsValue::UNDEFINED).unwrap();
    assert!(sns_rep.scale_normalized_stress().is_some());

    let dsc_only = js_sys::Array::of1(&JsValue::from_str("dsc"));
    assert!(js_evaluate_metrics_2d(&graph, &sep, &dsc_only, &JsValue::UNDEFINED).is_err());

    // Degenerate N=0 / N=1 non-panic checks
    let l0 = JsLayout2d::from_coords(&[]).unwrap();
    let empty_labels = u32_array_js(&[]);
    assert_eq!(js_distance_consistency(&l0, &empty_labels).unwrap(), 0.0);
    assert_eq!(js_silhouette_score(&l0, &empty_labels).unwrap(), 0.0);
    let mut gt0 = JsGTreeOverlapRemoval::new(&[], &[], &JsValue::UNDEFINED).unwrap();
    let st0 = gt0.run(&mut l0.clone()).unwrap();
    assert!(matches!(st0.as_str(), "converged" | "completed"));
}
