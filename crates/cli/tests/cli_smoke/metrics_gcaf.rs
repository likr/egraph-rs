use std::fs;

use crate::common::{run_egraph, temp_dir};

#[test]
fn cli_metrics_cluster_area_faithfulness_and_formats() {
    let dir = temp_dir("gcaf");
    let layout_path = dir.join("karate_layout.json");
    let cluster_path = dir.join("karate_cluster.json");
    let labels_txt_path = dir.join("karate_labels.txt");

    run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-pure2hop",
        "--init",
        "pivot-mds",
        "--seed",
        "42",
        "--output",
        layout_path.to_str().expect("utf8 path"),
    ]);

    run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "louvain",
        "--seed",
        "42",
        "--output",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let cluster_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&cluster_path).expect("read cluster json"))
            .expect("parse cluster json");
    let labels = cluster_json["labels"]
        .as_array()
        .expect("cluster labels array");
    assert_eq!(labels.len(), 34);

    // 1. JSON {"labels": [...]} from `egraph cluster` + `cluster-area-faithfulness` target.
    let out_json = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "sns,cluster-area-faithfulness",
        "--labels",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let metrics_from_json: serde_json::Value =
        serde_json::from_str(&out_json).expect("parse metrics json");
    let caf_json = metrics_from_json["cluster_area_faithfulness"]
        .as_f64()
        .expect("cluster_area_faithfulness float");
    assert!((0.0..=1.0).contains(&caf_json));
    assert!(
        metrics_from_json["scale_normalized_stress"]
            .as_f64()
            .is_some()
    );

    // 2. Plain-text/CSV labels + `gcaf` alias must match JSON result.
    let txt_body = format!(
        "# karate community labels\n{}\n",
        labels
            .iter()
            .map(|v| v.as_u64().expect("u64 label").to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    fs::write(&labels_txt_path, txt_body).expect("write text labels");
    let out_txt = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "gcaf",
        "--labels",
        labels_txt_path.to_str().expect("utf8 path"),
    ]);
    let metrics_from_txt: serde_json::Value =
        serde_json::from_str(&out_txt).expect("parse txt metrics json");
    let caf_txt = metrics_from_txt["cluster_area_faithfulness"]
        .as_f64()
        .expect("gcaf float");
    assert!((caf_txt - caf_json).abs() < 1e-12);

    // 3. `--targets all` remains backward-compatible without `--labels`.
    let out_all = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "all",
    ]);
    let metrics_all: serde_json::Value =
        serde_json::from_str(&out_all).expect("parse all metrics json");
    assert!(metrics_all["cluster_area_faithfulness"].is_null());
    assert!(metrics_all["adjusted_patch_nig"].is_null());
    assert!(metrics_all["gcaf_detail"].is_null());
    assert!(metrics_all["scale_normalized_stress"].as_f64().is_some());
    assert!(
        metrics_all["landmark_stratified_rta"].as_f64().is_some(),
        "expected all standard metrics including landmark_stratified_rta"
    );

    // 4. `--targets all` WITH `--labels` auto-includes `gcaf` and `adjusted-patch-nig`
    // and populates G-CAF summary fields + effective_boundary_r95 + g_eff (with gcaf_detail=null).
    let out_all_comm = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "all",
        "--labels",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let metrics_all_comm: serde_json::Value =
        serde_json::from_str(&out_all_comm).expect("parse all+comm metrics json");
    let caf_all_comm = metrics_all_comm["cluster_area_faithfulness"]
        .as_f64()
        .expect("non-null cluster_area_faithfulness for --targets all --labels");
    assert!((caf_all_comm - caf_json).abs() < 1e-12);
    assert!(
        metrics_all_comm["adjusted_patch_nig"].as_f64().is_some(),
        "expected non-null adjusted_patch_nig for --targets all --labels"
    );
    assert!(metrics_all_comm["gcaf_purity_adj"].as_f64().is_some());
    assert!(metrics_all_comm["gcaf_prop_vis"].as_f64().is_some());
    assert!(metrics_all_comm["gcaf_res_density"].as_f64().is_some());
    assert!(metrics_all_comm["gcaf_res_deconv"].as_f64().is_some());
    assert!(metrics_all_comm["gcaf_res_combined"].as_f64().is_some());
    assert!(
        metrics_all_comm["gcaf_spatial_coherence"]
            .as_f64()
            .is_some()
    );
    assert!(metrics_all_comm["gcaf_in_frac"].as_f64().is_some());
    assert!(metrics_all_comm["gcaf_num_clusters"].as_u64().is_some());
    assert_eq!(
        metrics_all_comm["effective_boundary_r95"].as_f64(),
        Some(2.0)
    );
    assert_eq!(metrics_all_comm["g_eff"].as_u64(), Some(12));
    assert!(metrics_all_comm["gcaf_detail"].is_null());

    // 5. Explicit `--targets adjusted-patch-nig` with `--labels`.
    let out_patch_nig = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "adjusted-patch-nig",
        "--labels",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let metrics_patch_nig: serde_json::Value =
        serde_json::from_str(&out_patch_nig).expect("parse adjusted-patch-nig metrics json");
    assert!(metrics_patch_nig["adjusted_patch_nig"].as_f64().is_some());
    assert!(
        metrics_patch_nig["cluster_area_faithfulness"].is_null(),
        "expected cluster_area_faithfulness to stay null when only adjusted-patch-nig requested"
    );

    // 6. 6 `--gcaf-*` flags + `--gcaf-detail` populate per-cluster arrays matching num_clusters.
    let out_gcaf_custom = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "gcaf",
        "--labels",
        cluster_path.to_str().expect("utf8 path"),
        "--gcaf-g-max",
        "64",
        "--gcaf-sigma-cells",
        "0.9",
        "--gcaf-kappa",
        "8.0",
        "--gcaf-tau-frac",
        "0.2",
        "--gcaf-boundary-r95",
        "2.5",
        "--gcaf-stevens-gamma",
        "0.7",
        "--gcaf-detail",
    ]);
    let metrics_gcaf_custom: serde_json::Value =
        serde_json::from_str(&out_gcaf_custom).expect("parse custom gcaf detail json");
    assert_eq!(
        metrics_gcaf_custom["effective_boundary_r95"].as_f64(),
        Some(2.5)
    );
    let k_clusters = metrics_gcaf_custom["gcaf_num_clusters"]
        .as_u64()
        .expect("gcaf_num_clusters") as usize;
    assert!(k_clusters >= 2);
    let detail_obj = metrics_gcaf_custom["gcaf_detail"]
        .as_object()
        .expect("non-null gcaf_detail when --gcaf-detail passed");
    for arr_key in [
        "cluster_ids",
        "counts",
        "p",
        "q",
        "purity",
        "resolution",
        "res_density",
        "res_deconv",
        "caf_per_cluster",
        "coherence",
    ] {
        assert_eq!(
            detail_obj[arr_key]
                .as_array()
                .unwrap_or_else(|| panic!("expected array for gcaf_detail.{arr_key}"))
                .len(),
            k_clusters
        );
    }

    // 7. `--targets all,gcaf` and `--targets gcaf,all` with `--labels`
    // must emit both non-null `cluster_area_faithfulness` and all standard metrics.
    for combined_targets in ["all,gcaf", "gcaf,all"] {
        let out_combined = run_egraph(&[
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            combined_targets,
            "--labels",
            cluster_path.to_str().expect("utf8 path"),
        ]);
        let metrics_combined: serde_json::Value =
            serde_json::from_str(&out_combined).expect("parse combined metrics json");
        let caf_combined = metrics_combined["cluster_area_faithfulness"]
            .as_f64()
            .unwrap_or_else(|| {
                panic!(
                    "expected non-null cluster_area_faithfulness for --targets {combined_targets}"
                )
            });
        assert!(
            (caf_combined - caf_json).abs() < 1e-12,
            "combined {combined_targets} gcaf ({caf_combined}) != standalone ({caf_json})"
        );
        assert!(
            metrics_combined["scale_normalized_stress"]
                .as_f64()
                .is_some()
        );
        assert!(metrics_combined["crossing_number"].as_u64().is_some());
        assert!(
            metrics_combined["neighborhood_preservation_2hop"]
                .as_f64()
                .is_some()
        );
        assert!(
            metrics_combined["landmark_stratified_rta"]
                .as_f64()
                .is_some()
        );
    }

    // 8. JSON schemas ({"community_labels": [...]}, bare [...]) and empty --targets "".
    let shifted_labels: Vec<u64> = labels
        .iter()
        .map(|v| 100_000 + v.as_u64().expect("u64 label"))
        .collect();
    let comm_obj_path = dir.join("karate_comm_labels.json");
    fs::write(
        &comm_obj_path,
        serde_json::json!({ "community_labels": shifted_labels }).to_string(),
    )
    .expect("write community_labels json");
    let out_comm_obj = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "gcaf",
        "--labels",
        comm_obj_path.to_str().expect("utf8 path"),
    ]);
    let metrics_comm_obj: serde_json::Value =
        serde_json::from_str(&out_comm_obj).expect("parse community_labels metrics json");
    let caf_comm_obj = metrics_comm_obj["cluster_area_faithfulness"]
        .as_f64()
        .expect("cluster_area_faithfulness float");
    assert!((caf_comm_obj - caf_json).abs() < 1e-12);

    let bare_arr_path = dir.join("karate_bare_labels.json");
    fs::write(
        &bare_arr_path,
        serde_json::to_string(&shifted_labels).expect("serialize bare array"),
    )
    .expect("write bare json array");
    let out_bare_arr = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "gcaf",
        "--labels",
        bare_arr_path.to_str().expect("utf8 path"),
    ]);
    let metrics_bare_arr: serde_json::Value =
        serde_json::from_str(&out_bare_arr).expect("parse bare array metrics json");
    let caf_bare_arr = metrics_bare_arr["cluster_area_faithfulness"]
        .as_f64()
        .expect("cluster_area_faithfulness float");
    assert!((caf_bare_arr - caf_json).abs() < 1e-12);

    let out_empty_targets = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "",
    ]);
    let metrics_empty: serde_json::Value =
        serde_json::from_str(&out_empty_targets).expect("parse empty targets json");
    assert!(metrics_empty["scale_normalized_stress"].is_null());
    assert!(metrics_empty["cluster_area_faithfulness"].is_null());

    let _ = fs::remove_dir_all(&dir);
}
