use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("egraph_cli_smoke_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn run_egraph(args: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_egraph");
    let output = Command::new(bin)
        .args(args)
        .output()
        .expect("execute egraph binary");
    assert!(
        output.status.success(),
        "egraph {:?} failed (status={:?}): stderr={}",
        args,
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("valid utf8 stdout")
}

#[test]
fn cli_layout_metrics_cluster_bench_embedded_dataset() {
    let dir = temp_dir("karate");
    let layout_path = dir.join("karate_layout.json");
    let metrics_path = dir.join("karate_metrics.json");
    let cluster_path = dir.join("karate_cluster.json");
    let bench_path = dir.join("karate_bench.json");

    run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-pure2hop",
        "--init",
        "pivot-mds",
        "--refine",
        "--prism",
        "--seed",
        "42",
        "--output",
        layout_path.to_str().expect("utf8 path"),
    ]);
    let layout_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&layout_path).expect("read layout"))
            .expect("parse layout json");
    assert_eq!(layout_json["num_nodes"], 34);
    assert_eq!(layout_json["num_edges"], 78);
    assert_eq!(layout_json["refined"], true);
    assert_eq!(layout_json["prism_applied"], true);
    assert_eq!(layout_json["prism_overlaps_remaining"], 0);
    assert_eq!(
        layout_json["coords"]
            .as_array()
            .expect("coords array")
            .len(),
        34
    );

    run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "sns,sampled-sns,crossings,np1,np2,sampled-np1,sampled-np2,ideal-edge-lengths",
        "--num-pivots",
        "8",
        "--num-samples",
        "256",
        "--ideal-edge-length",
        "1.0",
        "--seed",
        "42",
        "--output",
        metrics_path.to_str().expect("utf8 path"),
    ]);
    let metrics_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&metrics_path).expect("read metrics"))
            .expect("parse metrics json");
    assert!(metrics_json["scale_normalized_stress"].as_f64().is_some());
    assert!(
        metrics_json["sampled_scale_normalized_stress"]
            .as_f64()
            .is_some()
    );
    assert!(metrics_json["crossing_number"].as_u64().is_some());
    assert!(
        metrics_json["neighborhood_preservation_2hop"]
            .as_f64()
            .is_some()
    );
    assert!(metrics_json["sampled_np1"].as_f64().is_some());
    assert!(metrics_json["sampled_np2"].as_f64().is_some());

    for (method, extra_k) in [
        ("louvain", false),
        ("spectral", true),
        ("label-prop", false),
        ("kmeans", true),
    ] {
        let mut args = vec![
            "cluster",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
            "--output",
            cluster_path.to_str().expect("utf8 path"),
        ];
        if extra_k {
            args.extend(["--k", "3"]);
        }
        run_egraph(&args);
        let cluster_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&cluster_path).expect("read cluster"))
                .expect("parse cluster json");
        assert_eq!(
            cluster_json["labels"]
                .as_array()
                .expect("labels array")
                .len(),
            34
        );
        assert!(cluster_json["num_clusters"].as_u64().expect("num_clusters") >= 1);
    }

    run_egraph(&[
        "bench",
        "--dataset",
        "karate",
        "--method",
        "sgd-barrier-dog",
        "--init",
        "koren",
        "--seed",
        "42",
        "--output",
        bench_path.to_str().expect("utf8 path"),
    ]);
    let bench_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&bench_path).expect("read bench"))
            .expect("parse bench json");
    assert_eq!(bench_json["num_nodes"], 34);
    assert!(bench_json["total_ms"].as_f64().expect("total_ms") >= 0.0);
    assert!(
        bench_json["neighborhood_preservation_2hop"]
            .as_f64()
            .is_some()
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_reads_json_mtx_and_csv_inputs() {
    let dir = temp_dir("formats");

    // 1. JSON edge-list
    let json_path = dir.join("cycle.json");
    fs::write(
        &json_path,
        r#"{"num_nodes": 5, "edges": [[0,1],[1,2],[2,3],[3,4],[4,0]]}"#,
    )
    .expect("write json");
    let json_out = run_egraph(&[
        "layout",
        "--input",
        json_path.to_str().expect("utf8 path"),
        "--method",
        "sgd-full",
        "--init",
        "random",
        "--seed",
        "7",
    ]);
    let parsed_json: serde_json::Value =
        serde_json::from_str(&json_out).expect("parse stdout json");
    assert_eq!(parsed_json["num_nodes"], 5);
    assert_eq!(parsed_json["num_edges"], 5);

    // 2. JSON CSR
    let csr_path = dir.join("csr.json");
    fs::write(
        &csr_path,
        r#"{"num_nodes": 4, "indptr": [0, 2, 4, 6, 8], "indices": [1, 3, 0, 2, 1, 3, 0, 2]}"#,
    )
    .expect("write csr json");
    let csr_out = run_egraph(&[
        "layout",
        "--input",
        csr_path.to_str().expect("utf8 path"),
        "--method",
        "stress",
    ]);
    let parsed_csr: serde_json::Value = serde_json::from_str(&csr_out).expect("parse csr out");
    assert_eq!(parsed_csr["num_nodes"], 4);

    // 3. Matrix Market (.mtx)
    let mtx_path = dir.join("square.mtx");
    fs::write(
        &mtx_path,
        "%%MatrixMarket matrix coordinate pattern symmetric\n% test square\n4 4 4\n1 2\n2 3\n3 4\n4 1\n",
    )
    .expect("write mtx");
    let mtx_out = run_egraph(&[
        "layout",
        "--input",
        mtx_path.to_str().expect("utf8 path"),
        "--method",
        "kk",
    ]);
    let parsed_mtx: serde_json::Value = serde_json::from_str(&mtx_out).expect("parse mtx out");
    assert_eq!(parsed_mtx["num_nodes"], 4);
    assert_eq!(parsed_mtx["num_edges"], 4);

    // 4. CSR CSV (.csv)
    let csv_path = dir.join("path.csv");
    fs::write(&csv_path, "4,3\n0,1\n1,2\n2,3\n").expect("write csv");
    let csv_out = run_egraph(&[
        "cluster",
        "--input",
        csv_path.to_str().expect("utf8 path"),
        "--method",
        "louvain",
    ]);
    let parsed_csv: serde_json::Value = serde_json::from_str(&csv_out).expect("parse csv out");
    assert_eq!(parsed_csv["num_nodes"], 4);
    assert_eq!(parsed_csv["num_edges"], 3);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_supports_all_layout_methods_and_datasets() {
    for method in [
        "tsnet-pure2hop",
        "sgd-barrier-dog",
        "sgd-full",
        "sgd-snap",
        "sgd-pacmap",
        "gumap",
        "tfdp",
        "ibumap",
        "fa2",
        "yifanhu",
        "stress",
        "kk",
    ] {
        let out = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse layout output");
        assert_eq!(val["num_nodes"], 34, "method {method} num_nodes");
    }

    for dataset in [
        "1138_bus",
        "3_elt",
        "dwt_1005",
        "dwt_2680",
        "poli",
        "qh882",
        "uspower_grid",
        "karate",
        "lesmis",
    ] {
        let out = run_egraph(&[
            "cluster",
            "--dataset",
            dataset,
            "--method",
            "label-prop",
            "--seed",
            "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse dataset output");
        assert!(val["num_nodes"].as_u64().expect("num_nodes") > 0);
    }
}

#[test]
fn cli_rejects_invalid_prism_radius_fast() {
    let bin = env!("CARGO_BIN_EXE_egraph");
    for bad_radius in ["0.0", "-0.1", "NaN", "inf"] {
        let t0 = std::time::Instant::now();
        let output = Command::new(bin)
            .args([
                "layout",
                "--dataset",
                "karate",
                "--method",
                "tsnet-pure2hop",
                "--prism",
                "--prism-radius",
                bad_radius,
            ])
            .output()
            .expect("execute egraph binary");
        assert!(
            !output.status.success(),
            "expected non-zero exit for --prism-radius {bad_radius}"
        );
        assert!(
            t0.elapsed().as_secs_f64() < 5.0,
            "expected fast rejection for --prism-radius {bad_radius}"
        );
    }
}

#[test]
fn cli_metrics_cluster_area_faithfulness_and_error_guards() {
    let dir = temp_dir("gcaf");
    let layout_path = dir.join("karate_layout.json");
    let cluster_path = dir.join("karate_cluster.json");
    let labels_txt_path = dir.join("karate_labels.txt");
    let labels_bad_len_path = dir.join("karate_labels_short.txt");

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
        "--community-labels",
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
        "--community-labels",
        labels_txt_path.to_str().expect("utf8 path"),
    ]);
    let metrics_from_txt: serde_json::Value =
        serde_json::from_str(&out_txt).expect("parse txt metrics json");
    let caf_txt = metrics_from_txt["cluster_area_faithfulness"]
        .as_f64()
        .expect("gcaf float");
    assert!((caf_txt - caf_json).abs() < 1e-12);

    // 3. `--targets all` remains backward-compatible without `--community-labels`.
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

    // 3a. G5: `--targets all` WITH `--community-labels` auto-includes `gcaf` and `adjusted-patch-nig`
    // and populates G-CAF summary fields + effective_boundary_r95 + g_eff (with gcaf_detail=null).
    let out_all_comm = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "all",
        "--community-labels",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let metrics_all_comm: serde_json::Value =
        serde_json::from_str(&out_all_comm).expect("parse all+comm metrics json");
    let caf_all_comm = metrics_all_comm["cluster_area_faithfulness"]
        .as_f64()
        .expect("non-null cluster_area_faithfulness for --targets all --community-labels");
    assert!((caf_all_comm - caf_json).abs() < 1e-12);
    assert!(
        metrics_all_comm["adjusted_patch_nig"].as_f64().is_some(),
        "expected non-null adjusted_patch_nig for --targets all --community-labels"
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

    // G5: explicit `--targets adjusted-patch-nig` with and without `--community-labels`.
    let out_patch_nig = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "adjusted-patch-nig",
        "--community-labels",
        cluster_path.to_str().expect("utf8 path"),
    ]);
    let metrics_patch_nig: serde_json::Value =
        serde_json::from_str(&out_patch_nig).expect("parse adjusted-patch-nig metrics json");
    assert!(metrics_patch_nig["adjusted_patch_nig"].as_f64().is_some());
    assert!(
        metrics_patch_nig["cluster_area_faithfulness"].is_null(),
        "expected cluster_area_faithfulness to stay null when only adjusted-patch-nig requested"
    );

    // G4: 6 `--gcaf-*` flags + `--gcaf-detail` populate per-cluster arrays matching num_clusters.
    let out_gcaf_custom = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "gcaf",
        "--community-labels",
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

    // 3b. `--targets all,gcaf` and `--targets gcaf,all` with `--community-labels`
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
            "--community-labels",
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

    // 4. Missing `--community-labels` when requesting `cluster-area-faithfulness` errors cleanly.
    let bin = env!("CARGO_BIN_EXE_egraph");
    let missing_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "cluster-area-faithfulness",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!missing_out.status.success());
    let missing_stderr = String::from_utf8_lossy(&missing_out.stderr);
    assert!(
        missing_stderr.contains("community_labels"),
        "expected stderr to mention community_labels, got: {missing_stderr}"
    );

    let missing_patch_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "adjusted-patch-nig",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!missing_patch_out.status.success());

    let bad_gcaf_flag_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "gcaf",
            "--community-labels",
            cluster_path.to_str().expect("utf8 path"),
            "--gcaf-g-max",
            "4",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!bad_gcaf_flag_out.status.success());

    // 5. Length mismatch in `--community-labels` errors cleanly.
    fs::write(&labels_bad_len_path, "0 1 2 3\n").expect("write short labels");
    let mismatch_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "gcaf",
            "--community-labels",
            labels_bad_len_path.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!mismatch_out.status.success());

    // 6. Trailing invalid target after `all` must be rejected cleanly with non-zero exit.
    let invalid_after_all_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "all,invalid_target",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(
        !invalid_after_all_out.status.success(),
        "expected non-zero exit for --targets all,invalid_target"
    );
    let invalid_stderr = String::from_utf8_lossy(&invalid_after_all_out.stderr);
    assert!(
        invalid_stderr.contains("unknown metric target"),
        "expected stderr to mention unknown metric target, got: {invalid_stderr}"
    );

    // 7. Gap 7: JSON schemas ({"community_labels": [...]}, bare [...]), empty --targets "",
    // and malformed label / eager --num-pivots 0 error guards.
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
        "--community-labels",
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
        "--community-labels",
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

    let bad_label_cases: &[(&str, &str, &str, &str)] = &[
        (
            "bad_labels_not_array.json",
            r#"{"labels": "not_array"}"#,
            "'labels' field",
            "non-array labels",
        ),
        (
            "bad_comm_not_array.json",
            r#"{"community_labels": 123}"#,
            "'community_labels' field",
            "non-array community_labels",
        ),
        (
            "bad_other_schema.json",
            r#"{"other": []}"#,
            "unrecognized community labels JSON schema",
            "unrecognized JSON schema",
        ),
        (
            "bad_neg_elem.json",
            "[0, -1, 1, 1]",
            "non-negative integer",
            "negative JSON integer",
        ),
        (
            "bad_u32_overflow.json",
            "[0, 4294967296, 1, 1]",
            "exceeds u32::MAX",
            "u32 overflow in JSON",
        ),
        (
            "bad_plain_overflow.txt",
            "4294967296\n",
            "invalid community label",
            "u32 overflow in plain text",
        ),
        (
            "bad_malformed_prefix.txt",
            "[0, 0, 1",
            "EOF",
            "malformed JSON prefix",
        ),
    ];
    for &(fname, contents, expected_stderr, ctx) in bad_label_cases {
        let p = dir.join(fname);
        fs::write(&p, contents).expect("write bad label fixture");
        let out = Command::new(bin)
            .args([
                "metrics",
                "--dataset",
                "karate",
                "--layout",
                layout_path.to_str().expect("utf8 path"),
                "--targets",
                "gcaf",
                "--community-labels",
                p.to_str().expect("utf8 path"),
            ])
            .output()
            .expect("execute egraph binary");
        assert!(!out.status.success(), "expected non-zero exit for {ctx}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(expected_stderr),
            "expected stderr containing {expected_stderr:?} for {ctx}, got: {stderr}"
        );
    }

    let bad_pivots_out = Command::new(bin)
        .args([
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--targets",
            "gcaf",
            "--community-labels",
            cluster_path.to_str().expect("utf8 path"),
            "--num-pivots",
            "0",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(
        !bad_pivots_out.status.success(),
        "expected non-zero exit for --num-pivots 0 --targets gcaf"
    );
    let bad_pivots_stderr = String::from_utf8_lossy(&bad_pivots_out.stderr);
    assert!(
        bad_pivots_stderr.contains("sns_num_pivots"),
        "expected stderr mentioning sns_num_pivots, got: {bad_pivots_stderr}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_cluster_leiden_mod_and_cpm() {
    let bin = env!("CARGO_BIN_EXE_egraph");

    let out_mod_1 = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "leiden-mod",
        "--resolution",
        "1.0",
        "--randomness",
        "0.01",
        "--seed",
        "42",
    ]);
    let mod_1: serde_json::Value = serde_json::from_str(&out_mod_1).expect("parse leiden-mod");
    assert_eq!(mod_1["method"], "leiden-mod");
    assert_eq!(mod_1["labels"].as_array().expect("labels").len(), 34);
    assert!(mod_1["num_clusters"].as_u64().expect("num_clusters") >= 2);
    let q_mod = mod_1["quality"].as_f64().expect("quality f64");
    let m_mod = mod_1["modularity"].as_f64().expect("modularity f64");
    assert!(q_mod.is_finite() && m_mod.is_finite());
    assert!((q_mod - m_mod).abs() < 1e-12);
    assert!(mod_1["inertia"].is_null());

    // Determinism under identical seed.
    let out_mod_2 = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "leiden-mod",
        "--resolution",
        "1.0",
        "--randomness",
        "0.01",
        "--seed",
        "42",
    ]);
    let mod_2: serde_json::Value = serde_json::from_str(&out_mod_2).expect("parse leiden-mod 2");
    assert_eq!(mod_1["labels"], mod_2["labels"]);
    assert!((mod_2["quality"].as_f64().expect("quality2") - q_mod).abs() < 1e-12);

    // CPM objective.
    let out_cpm = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "leiden-cpm",
        "--resolution",
        "0.1",
        "--randomness",
        "0.01",
        "--seed",
        "42",
    ]);
    let cpm: serde_json::Value = serde_json::from_str(&out_cpm).expect("parse leiden-cpm");
    assert_eq!(cpm["method"], "leiden-cpm");
    assert_eq!(cpm["labels"].as_array().expect("labels").len(), 34);
    assert!(cpm["num_clusters"].as_u64().expect("num_clusters") >= 1);
    assert!(cpm["quality"].as_f64().expect("cpm quality").is_finite());
    assert!(
        cpm["modularity"]
            .as_f64()
            .expect("cpm modularity")
            .is_finite()
    );

    // Existing methods keep quality = null.
    let out_louvain = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "louvain",
        "--seed",
        "42",
    ]);
    let louvain_json: serde_json::Value =
        serde_json::from_str(&out_louvain).expect("parse louvain");
    assert!(louvain_json["quality"].is_null());
    assert!(louvain_json["modularity"].as_f64().is_some());

    // Error guards for invalid resolution / randomness.
    for args in [
        vec![
            "cluster",
            "--dataset",
            "karate",
            "--method",
            "leiden-mod",
            "--resolution",
            "-0.1",
        ],
        vec![
            "cluster",
            "--dataset",
            "karate",
            "--method",
            "leiden-cpm",
            "--randomness",
            "0.0",
        ],
    ] {
        let out = Command::new(bin)
            .args(&args)
            .output()
            .expect("execute egraph binary");
        assert!(
            !out.status.success(),
            "expected non-zero exit for {:?}",
            args
        );
    }
}

#[test]
fn cli_layout_w1_baselines_cluster_refine_and_bench() {
    let dir = temp_dir("w1_layout");
    let labels_path = dir.join("karate_leiden.json");
    let bad_labels_path = dir.join("bad_labels.txt");

    run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "leiden-mod",
        "--seed",
        "42",
        "--output",
        labels_path.to_str().expect("utf8 path"),
    ]);

    for method in ["linlog", "fr", "fa2-overlap", "backbone", "gib"] {
        let out = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse layout json");
        assert_eq!(val["num_nodes"], 34, "method {method}");
        assert_eq!(val["method"], method);
        assert_eq!(val["cluster_refine"], "none");
        assert_eq!(val["overlap_removal"], "none");
        let xs = val["x"].as_array().expect("x array");
        let ys = val["y"].as_array().expect("y array");
        assert_eq!(xs.len(), 34);
        assert_eq!(ys.len(), 34);
        assert!(xs.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)));
        assert!(ys.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)));
    }

    // Explicit --community-labels with gib.
    let gib_explicit = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "gib",
        "--community-labels",
        labels_path.to_str().expect("utf8 path"),
        "--seed",
        "42",
    ]);
    let gib_json: serde_json::Value =
        serde_json::from_str(&gib_explicit).expect("parse gib explicit");
    assert_eq!(gib_json["num_nodes"], 34);
    assert_eq!(gib_json["method"], "gib");

    // --cluster-refine kmeans and hac (both explicit labels and auto-Leiden fallback).
    for refine_mode in ["kmeans", "hac"] {
        let with_labels = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--cluster-refine",
            refine_mode,
            "--community-labels",
            labels_path.to_str().expect("utf8 path"),
            "--seed",
            "42",
        ]);
        let v1: serde_json::Value =
            serde_json::from_str(&with_labels).expect("parse cluster-refine with labels");
        assert_eq!(v1["num_nodes"], 34);
        assert_eq!(v1["cluster_refine"], refine_mode);

        let auto_labels = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--cluster-refine",
            refine_mode,
            "--seed",
            "42",
        ]);
        let v2: serde_json::Value =
            serde_json::from_str(&auto_labels).expect("parse cluster-refine auto");
        assert_eq!(v2["num_nodes"], 34);
        assert_eq!(v2["cluster_refine"], refine_mode);
        assert_eq!(v1["coords"], v2["coords"]);
    }

    // Bench shares expanded LayoutMethod cleanly.
    for bench_method in ["linlog", "gib"] {
        let bench_out = run_egraph(&[
            "bench",
            "--dataset",
            "karate",
            "--method",
            bench_method,
            "--seed",
            "42",
        ]);
        let bench_json: serde_json::Value =
            serde_json::from_str(&bench_out).expect("parse bench json");
        assert_eq!(bench_json["num_nodes"], 34);
        assert_eq!(bench_json["method"], bench_method);
        assert!(bench_json["total_ms"].as_f64().expect("total_ms") >= 0.0);
    }

    // Error guard: mismatched --community-labels length fails cleanly in layout.
    fs::write(&bad_labels_path, "0 1 2\n").expect("write bad labels");
    let bin = env!("CARGO_BIN_EXE_egraph");
    let bad_out = Command::new(bin)
        .args([
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--community-labels",
            bad_labels_path.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!bad_out.status.success());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_layout_overlap_removal_variants_and_prism_compat() {
    let bin = env!("CARGO_BIN_EXE_egraph");

    for variant in [
        "none",
        "prism",
        "gtree",
        "forbid",
        "forbid-prime",
        "sparse-forbid",
    ] {
        let out = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--overlap-removal",
            variant,
            "--prism-radius",
            "0.05",
            "--seed",
            "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse overlap json");
        assert_eq!(val["overlap_removal"], variant);
        if variant == "none" {
            assert_eq!(val["prism_applied"], false);
            assert!(val["prism_overlaps_remaining"].is_null());
            assert!(val["overlaps_remaining"].is_null());
        } else {
            assert_eq!(
                val["overlaps_remaining"], 0,
                "expected 0 overlaps remaining for variant {variant}"
            );
            if variant == "prism" {
                assert_eq!(val["prism_applied"], true);
                assert_eq!(val["prism_overlaps_remaining"], 0);
            } else {
                assert_eq!(val["prism_applied"], false);
                assert!(val["prism_overlaps_remaining"].is_null());
            }
        }
    }

    // Backward-compatible `--prism` flag.
    let prism_compat = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-pure2hop",
        "--prism",
        "--prism-radius",
        "0.05",
        "--seed",
        "42",
    ]);
    let prism_val: serde_json::Value =
        serde_json::from_str(&prism_compat).expect("parse prism compat");
    assert_eq!(prism_val["prism_applied"], true);
    assert_eq!(prism_val["overlap_removal"], "prism");
    assert_eq!(prism_val["prism_overlaps_remaining"], 0);
    assert_eq!(prism_val["overlaps_remaining"], 0);

    // Conflict guard between `--prism` and `--overlap-removal`.
    let conflict_out = Command::new(bin)
        .args([
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--prism",
            "--overlap-removal",
            "gtree",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!conflict_out.status.success());

    // Invalid radius guard across new overlap-removal variants (must fail fast).
    for variant in ["gtree", "forbid"] {
        for bad_radius in ["0.0", "-0.1", "NaN", "inf"] {
            let t0 = std::time::Instant::now();
            let out = Command::new(bin)
                .args([
                    "layout",
                    "--dataset",
                    "karate",
                    "--method",
                    "stress",
                    "--overlap-removal",
                    variant,
                    "--prism-radius",
                    bad_radius,
                ])
                .output()
                .expect("execute egraph binary");
            assert!(
                !out.status.success(),
                "expected non-zero exit for --overlap-removal {variant} --prism-radius {bad_radius}"
            );
            assert!(
                t0.elapsed().as_secs_f64() < 5.0,
                "expected fast rejection for --overlap-removal {variant} --prism-radius {bad_radius}"
            );
        }
    }
}

#[test]
fn cli_w1_degenerate_graphs_n0_n1_and_disconnected() {
    let dir = temp_dir("degenerate_w1");
    let n0_path = dir.join("n0.json");
    let n1_path = dir.join("n1.json");
    let disc_path = dir.join("disc.json");
    fs::write(&n0_path, r#"{"num_nodes": 0, "edges": []}"#).expect("write n0");
    fs::write(&n1_path, r#"{"num_nodes": 1, "edges": []}"#).expect("write n1");
    fs::write(
        &disc_path,
        r#"{"num_nodes": 6, "edges": [[0,1],[1,2],[2,0],[3,4]]}"#,
    )
    .expect("write disc");

    for (path, expected_n) in [(&n0_path, 0usize), (&n1_path, 1usize), (&disc_path, 6usize)] {
        let p = path.to_str().expect("utf8 path");
        for c_method in ["leiden-mod", "leiden-cpm"] {
            let out = run_egraph(&[
                "cluster", "--input", p, "--method", c_method, "--seed", "42",
            ]);
            let val: serde_json::Value = serde_json::from_str(&out).expect("parse cluster");
            assert_eq!(val["num_nodes"].as_u64(), Some(expected_n as u64));
            assert_eq!(val["labels"].as_array().expect("labels").len(), expected_n);
        }

        for l_method in ["linlog", "fr", "fa2-overlap", "backbone", "gib"] {
            let out = run_egraph(&[
                "layout",
                "--input",
                p,
                "--method",
                l_method,
                "--cluster-refine",
                "kmeans",
                "--overlap-removal",
                "gtree",
                "--seed",
                "42",
            ]);
            let val: serde_json::Value = serde_json::from_str(&out).expect("parse layout");
            assert_eq!(val["num_nodes"].as_u64(), Some(expected_n as u64));
            assert_eq!(val["overlaps_remaining"], 0);
            let xs = val["x"].as_array().expect("x");
            let ys = val["y"].as_array().expect("y");
            assert_eq!(xs.len(), expected_n);
            assert_eq!(ys.len(), expected_n);
            assert!(xs.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)));
            assert!(ys.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)));
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_metrics_eval_baselines_18_targets() {
    let dir = temp_dir("eval_baselines");
    let layout_path = dir.join("karate_stress.json");
    let cluster_path = dir.join("karate_leiden.json");

    run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
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
        "leiden-mod",
        "--seed",
        "42",
        "--output",
        cluster_path.to_str().expect("utf8 path"),
    ]);

    let all_18_targets = "dsc,dc,cdm,hdm,sepme-knng,sepme-rng,sepme-gab,sepme-gong,neighborhood-hit,silhouette,calinski-harabasz,davies-bouldin,cq-kmeans-ari,cq-kmeans-ami,cq-hac-hom,cq-hac-ami,pairwise-distance-overlap-error,sprawlter-nn";
    let out_18 = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--community-labels",
        cluster_path.to_str().expect("utf8 path"),
        "--targets",
        all_18_targets,
        "--seed",
        "42",
    ]);
    let m18: serde_json::Value = serde_json::from_str(&out_18).expect("parse 18 metrics");
    let baseline_keys = [
        "dsc",
        "dc",
        "cdm",
        "hdm",
        "sepme_knng",
        "sepme_rng",
        "sepme_gab",
        "sepme_gong",
        "neighborhood_hit",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "cq_kmeans_ari",
        "cq_kmeans_ami",
        "cq_hac_hom",
        "cq_hac_ami",
        "pairwise_distance_overlap_error",
        "sprawlter_nn",
    ];
    for key in baseline_keys {
        let val = m18[key]
            .as_f64()
            .unwrap_or_else(|| panic!("expected finite float for {key}, got {:?}", m18[key]));
        assert!(val.is_finite(), "{key} not finite: {val}");
    }
    for unit_key in [
        "dsc",
        "dc",
        "cdm",
        "hdm",
        "sepme_knng",
        "sepme_rng",
        "sepme_gab",
        "sepme_gong",
        "neighborhood_hit",
        "cq_hac_hom",
        "pairwise_distance_overlap_error",
    ] {
        let v = m18[unit_key].as_f64().expect("unit metric");
        assert!((0.0..=1.0).contains(&v), "{unit_key} out of [0,1]: {v}");
    }
    for signed_unit_key in ["silhouette", "cq_kmeans_ari", "cq_kmeans_ami", "cq_hac_ami"] {
        let v = m18[signed_unit_key].as_f64().expect("signed unit metric");
        assert!(
            (-1.0..=1.0).contains(&v),
            "{signed_unit_key} out of [-1,1]: {v}"
        );
    }
    for nonneg_key in ["calinski_harabasz", "davies_bouldin", "sprawlter_nn"] {
        let v = m18[nonneg_key].as_f64().expect("nonneg metric");
        assert!(v >= 0.0, "{nonneg_key} < 0: {v}");
    }

    // sepme-knng / sepme_knng / sepme-knng2 / sepme_knng2 alias parity.
    let expected_sepme_knng = m18["sepme_knng"].as_f64().expect("sepme_knng");
    for alias in ["sepme-knng", "sepme_knng", "sepme-knng2", "sepme_knng2"] {
        let out_alias = run_egraph(&[
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            layout_path.to_str().expect("utf8 path"),
            "--community-labels",
            cluster_path.to_str().expect("utf8 path"),
            "--targets",
            alias,
        ]);
        let alias_json: serde_json::Value =
            serde_json::from_str(&out_alias).expect("parse alias json");
        let v = alias_json["sepme_knng"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing sepme_knng for alias {alias}"));
        assert!(
            (v - expected_sepme_knng).abs() < 1e-12,
            "alias {alias} mismatch: {v} vs {expected_sepme_knng}"
        );
    }

    // Label-free sprawlter-nn succeeds without --community-labels and omits unrequested baseline keys.
    let out_sprawlter = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "sprawlter-nn",
    ]);
    let sp_obj =
        serde_json::from_str::<serde_json::Value>(&out_sprawlter).expect("parse sprawlter json");
    let sp_val = sp_obj["sprawlter_nn"].as_f64().expect("sprawlter_nn");
    assert!(sp_val.is_finite() && sp_val >= 1.0);
    assert!(sp_obj.as_object().expect("obj").get("dsc").is_none());

    // Regression: `--targets all` without `--community-labels` omits all 18 eval-baselines keys.
    let out_all = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        layout_path.to_str().expect("utf8 path"),
        "--targets",
        "all",
    ]);
    let all_val: serde_json::Value = serde_json::from_str(&out_all).expect("parse all");
    let all_obj = all_val.as_object().expect("obj");
    assert!(all_val["cluster_area_faithfulness"].is_null());
    assert!(all_val["adjusted_patch_nig"].is_null());
    assert!(all_val["scale_normalized_stress"].as_f64().is_some());
    for key in baseline_keys {
        assert!(
            all_obj.get(key).is_none(),
            "expected {key} to be omitted from `--targets all`"
        );
    }

    // Missing `--community-labels` error guard for label-dependent W1-6 targets.
    let bin = env!("CARGO_BIN_EXE_egraph");
    for target in ["dsc", "sepme-knng2", "cq-hac-ami"] {
        let missing_out = Command::new(bin)
            .args([
                "metrics",
                "--dataset",
                "karate",
                "--layout",
                layout_path.to_str().expect("utf8 path"),
                "--targets",
                target,
            ])
            .output()
            .expect("execute egraph binary");
        assert!(
            !missing_out.status.success(),
            "expected non-zero exit for --targets {target} without --community-labels"
        );
        let stderr = String::from_utf8_lossy(&missing_out.stderr);
        assert!(
            stderr.contains("community_labels"),
            "expected stderr mentioning community_labels for {target}, got: {stderr}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

const W2_METHODS: [&str; 10] = [
    "hde",
    "sgd-ortmann",
    "maxent",
    "largevis",
    "drgraph",
    "graph-cne",
    "grip",
    "gem",
    "sel",
    "sfc",
];

fn assert_finite_layout_json(val: &serde_json::Value, expected_n: usize, ctx: &str) {
    assert_eq!(
        val["num_nodes"].as_u64(),
        Some(expected_n as u64),
        "{ctx}: num_nodes"
    );
    let xs = val["x"].as_array().expect("x array");
    let ys = val["y"].as_array().expect("y array");
    let coords = val["coords"].as_array().expect("coords array");
    assert_eq!(xs.len(), expected_n, "{ctx}: x len");
    assert_eq!(ys.len(), expected_n, "{ctx}: y len");
    assert_eq!(coords.len(), expected_n, "{ctx}: coords len");
    for i in 0..expected_n {
        let xi = xs[i].as_f64().expect("x f64");
        let yi = ys[i].as_f64().expect("y f64");
        assert!(xi.is_finite(), "{ctx}: non-finite x[{i}]");
        assert!(yi.is_finite(), "{ctx}: non-finite y[{i}]");
    }
}

fn assert_no_pairwise_overlap(xs: &[f64], ys: &[f64], radius: f64, ctx: &str) {
    let min_sep = 2.0 * radius - 1e-12;
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            let dx = (xs[i] - xs[j]).abs();
            let dy = (ys[i] - ys[j]).abs();
            assert!(
                dx >= min_sep || dy >= min_sep,
                "{ctx}: pair ({i}, {j}) overlaps at radius={radius}: dx={dx}, dy={dy}"
            );
        }
    }
}

#[test]
fn cli_layout_w2_baselines_and_bench() {
    for method in W2_METHODS {
        let out = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse layout json");
        assert_eq!(val["method"], method);
        assert_eq!(val["overlap_removal"], "none");
        assert_finite_layout_json(&val, 34, method);
        let xs: Vec<f64> = val["x"]
            .as_array()
            .expect("x")
            .iter()
            .map(|v| v.as_f64().expect("f64"))
            .collect();
        let ys: Vec<f64> = val["y"]
            .as_array()
            .expect("y")
            .iter()
            .map(|v| v.as_f64().expect("f64"))
            .collect();
        let span_x = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - xs.iter().copied().fold(f64::INFINITY, f64::min);
        let span_y = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - ys.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            span_x > 0.0 && span_y > 0.0,
            "{method}: expected non-zero 2D coordinate span, got span_x={span_x}, span_y={span_y}"
        );
    }

    for bench_method in ["hde", "sgd-ortmann", "maxent", "drgraph", "grip", "sfc"] {
        let bench_out = run_egraph(&[
            "bench",
            "--dataset",
            "karate",
            "--method",
            bench_method,
            "--seed",
            "42",
        ]);
        let bench_json: serde_json::Value =
            serde_json::from_str(&bench_out).expect("parse bench json");
        assert_eq!(bench_json["num_nodes"], 34);
        assert_eq!(bench_json["method"], bench_method);
        assert!(
            bench_json["total_ms"].as_f64().expect("total_ms") >= 0.0,
            "{bench_method}: total_ms"
        );
        let np2 = bench_json["neighborhood_preservation_2hop"]
            .as_f64()
            .expect("np2");
        assert!(np2.is_finite(), "{bench_method}: finite np2");
    }
}

#[test]
fn cli_w2_determinism_with_seed() {
    for method in W2_METHODS {
        let out1 = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let out2 = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let v1: serde_json::Value = serde_json::from_str(&out1).expect("parse 1");
        let v2: serde_json::Value = serde_json::from_str(&out2).expect("parse 2");
        assert_eq!(
            v1["coords"], v2["coords"],
            "{method}: expected bit-exact deterministic coordinates for seed 42"
        );
    }

    // `pfs-prime` overlap removal determinism.
    let pfs1 = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--overlap-removal",
        "pfs-prime",
        "--prism-radius",
        "0.05",
        "--seed",
        "42",
    ]);
    let pfs2 = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--overlap-removal",
        "pfs-prime",
        "--prism-radius",
        "0.05",
        "--seed",
        "42",
    ]);
    let p1: serde_json::Value = serde_json::from_str(&pfs1).expect("parse pfs1");
    let p2: serde_json::Value = serde_json::from_str(&pfs2).expect("parse pfs2");
    assert_eq!(
        p1["coords"], p2["coords"],
        "pfs-prime: expected bit-exact deterministic coordinates for seed 42"
    );

    // Different seeds must produce distinct layouts on randomized/stochastic W2 methods.
    for method in ["sgd-ortmann", "largevis", "drgraph", "graph-cne", "gem"] {
        let out_a = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "42",
        ]);
        let out_b = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--seed",
            "99",
        ]);
        let va: serde_json::Value = serde_json::from_str(&out_a).expect("parse seed 42");
        let vb: serde_json::Value = serde_json::from_str(&out_b).expect("parse seed 99");
        assert_ne!(
            va["coords"], vb["coords"],
            "{method}: expected distinct layouts for seed 42 vs 99"
        );
    }

    // HDE replaces the initial layout directly, so `--init random` vs `--init pivot-mds` with same `--seed` must match.
    let hde_pmds = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "hde",
        "--init",
        "pivot-mds",
        "--seed",
        "42",
    ]);
    let hde_rand = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "hde",
        "--init",
        "random",
        "--seed",
        "42",
    ]);
    let vp: serde_json::Value = serde_json::from_str(&hde_pmds).expect("parse hde pmds");
    let vr: serde_json::Value = serde_json::from_str(&hde_rand).expect("parse hde rand");
    assert_eq!(
        vp["coords"], vr["coords"],
        "hde output must depend on seed and ignore initial placement"
    );
}

#[test]
fn cli_layout_pfs_prime_overlap_removal_and_guards() {
    let bin = env!("CARGO_BIN_EXE_egraph");

    let out = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--overlap-removal",
        "pfs-prime",
        "--prism-radius",
        "0.05",
        "--seed",
        "42",
    ]);
    let val: serde_json::Value = serde_json::from_str(&out).expect("parse pfs-prime json");
    assert_eq!(val["overlap_removal"], "pfs-prime");
    assert_eq!(val["overlaps_remaining"], 0);
    assert_eq!(val["prism_applied"], false);
    assert!(val["prism_overlaps_remaining"].is_null());
    assert_finite_layout_json(&val, 34, "pfs-prime stress");
    let xs: Vec<f64> = val["x"]
        .as_array()
        .expect("x")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let ys: Vec<f64> = val["y"]
        .as_array()
        .expect("y")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    assert_no_pairwise_overlap(&xs, &ys, 0.05, "pfs-prime stress");

    // `--overlap-radius` alias + large-scale `--method gem` (`edge_length = 128.0`).
    let gem_out = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "gem",
        "--overlap-removal",
        "pfs-prime",
        "--overlap-radius",
        "0.1",
        "--seed",
        "42",
    ]);
    let gem_val: serde_json::Value = serde_json::from_str(&gem_out).expect("parse pfs-prime gem");
    assert_eq!(gem_val["overlap_removal"], "pfs-prime");
    assert_eq!(gem_val["overlaps_remaining"], 0);
    assert_eq!(gem_val["prism_applied"], false);
    assert_finite_layout_json(&gem_val, 34, "pfs-prime gem");
    let gem_xs: Vec<f64> = gem_val["x"]
        .as_array()
        .expect("x")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let gem_ys: Vec<f64> = gem_val["y"]
        .as_array()
        .expect("y")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    assert_no_pairwise_overlap(&gem_xs, &gem_ys, 0.1, "pfs-prime gem");

    // Conflict guard between `--prism` and `--overlap-removal pfs-prime`.
    let conflict_out = Command::new(bin)
        .args([
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--prism",
            "--overlap-removal",
            "pfs-prime",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(!conflict_out.status.success());

    // Invalid radius guard for `pfs-prime` (must fail fast).
    for bad_radius in ["0.0", "-0.1", "NaN", "inf"] {
        let t0 = std::time::Instant::now();
        let bad_out = Command::new(bin)
            .args([
                "layout",
                "--dataset",
                "karate",
                "--method",
                "stress",
                "--overlap-removal",
                "pfs-prime",
                "--prism-radius",
                bad_radius,
            ])
            .output()
            .expect("execute egraph binary");
        assert!(
            !bad_out.status.success(),
            "expected non-zero exit for --overlap-removal pfs-prime --prism-radius {bad_radius}"
        );
        assert!(
            t0.elapsed().as_secs_f64() < 5.0,
            "expected fast rejection for --overlap-removal pfs-prime --prism-radius {bad_radius}"
        );
    }
}

#[test]
fn cli_w2_degenerate_and_edge_case_graphs() {
    let dir = temp_dir("degenerate_w2");
    let n0_path = dir.join("n0.json");
    let n1_path = dir.join("n1.json");
    let n2_path = dir.join("n2.json");
    let edgeless_path = dir.join("edgeless.json");
    let disc_path = dir.join("disc.json");
    let zero_w_path = dir.join("zero_w.json");

    fs::write(&n0_path, r#"{"num_nodes": 0, "edges": []}"#).expect("write n0");
    fs::write(&n1_path, r#"{"num_nodes": 1, "edges": []}"#).expect("write n1");
    fs::write(&n2_path, r#"{"num_nodes": 2, "edges": [[0, 1]]}"#).expect("write n2");
    fs::write(&edgeless_path, r#"{"num_nodes": 6, "edges": []}"#).expect("write edgeless");
    fs::write(
        &disc_path,
        r#"{"num_nodes": 6, "edges": [[0, 1], [1, 2], [2, 0], [3, 4]]}"#,
    )
    .expect("write disc");
    fs::write(
        &zero_w_path,
        r#"{"num_nodes": 3, "edges": [[0, 1], [1, 2]], "weights": [1.0, 0.0]}"#,
    )
    .expect("write zero_w");

    let fixtures = [
        ("n0", &n0_path, 0usize),
        ("n1", &n1_path, 1usize),
        ("n2", &n2_path, 2usize),
        ("edgeless", &edgeless_path, 6usize),
        ("disc", &disc_path, 6usize),
    ];

    for (label, path, expected_n) in fixtures {
        let p = path.to_str().expect("utf8 path");
        for method in W2_METHODS {
            let out = run_egraph(&[
                "layout",
                "--input",
                p,
                "--method",
                method,
                "--overlap-removal",
                "pfs-prime",
                "--prism-radius",
                "0.05",
                "--seed",
                "42",
            ]);
            let val: serde_json::Value = serde_json::from_str(&out).expect("parse layout");
            let ctx = format!("{label}/{method}");
            assert_eq!(val["overlaps_remaining"], 0, "{ctx}: overlaps_remaining");
            assert_finite_layout_json(&val, expected_n, &ctx);
        }

        // SFC must guarantee zero co-location even without overlap removal for N >= 2.
        if expected_n >= 2 {
            let sfc_out = run_egraph(&["layout", "--input", p, "--method", "sfc", "--seed", "42"]);
            let sfc_val: serde_json::Value =
                serde_json::from_str(&sfc_out).expect("parse sfc layout");
            assert_finite_layout_json(&sfc_val, expected_n, &format!("{label}/sfc-raw"));
            let xs: Vec<f64> = sfc_val["x"]
                .as_array()
                .expect("x")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect();
            let ys: Vec<f64> = sfc_val["y"]
                .as_array()
                .expect("y")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect();
            for i in 0..expected_n {
                for j in (i + 1)..expected_n {
                    let dist_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
                    assert!(
                        dist_sq > 1e-12,
                        "{label}/sfc-raw: co-located vertices {i} and {j}"
                    );
                }
            }
        }
    }

    // Zero-weight edge rejection for `hde` vs clean handling across remaining W2 methods.
    let bin = env!("CARGO_BIN_EXE_egraph");
    let zero_p = zero_w_path.to_str().expect("utf8 path");
    let hde_zero_out = Command::new(bin)
        .args([
            "layout", "--input", zero_p, "--method", "hde", "--seed", "42",
        ])
        .output()
        .expect("execute egraph binary");
    assert!(
        !hde_zero_out.status.success(),
        "expected hde to reject zero-weight edge"
    );
    let hde_stderr = String::from_utf8_lossy(&hde_zero_out.stderr);
    assert!(
        hde_stderr.contains("positive"),
        "expected stderr mentioning positive weights for hde, got: {hde_stderr}"
    );

    for method in W2_METHODS.into_iter().filter(|m| *m != "hde") {
        let out = run_egraph(&[
            "layout", "--input", zero_p, "--method", method, "--seed", "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse zero_w layout");
        assert_finite_layout_json(&val, 3, &format!("zero_w/{method}"));
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_help_includes_w2_methods_and_clarified_docstrings() {
    let layout_help = run_egraph(&["layout", "--help"]);
    for method in W2_METHODS {
        assert!(
            layout_help.contains(method),
            "layout --help missing method {method}"
        );
    }
    assert!(
        layout_help.contains("pfs-prime"),
        "layout --help missing pfs-prime"
    );
    assert!(
        layout_help.contains("linlog_mode"),
        "layout --help missing linlog_mode distinction"
    );
    assert!(
        layout_help.contains("muelder2008_space_filling") && layout_help.contains("Hilbert"),
        "layout --help missing SFC canonical Hilbert/Morton & MISSING_PDF note"
    );
    assert!(
        layout_help.contains("sgd2"),
        "layout --help missing sgd2 single-criteria SGD clarification"
    );

    let metrics_help = run_egraph(&["metrics", "--help"]);
    assert!(
        metrics_help.contains("drng")
            && metrics_help.contains("dgg")
            && metrics_help.contains("S_RNG")
            && metrics_help.contains("S_GG")
            && metrics_help.contains("dRNG")
            && metrics_help.contains("dGG"),
        "metrics --help missing Eades 2017 S_RNG/S_GG vs Hong 2022 dRNG/dGG clarification"
    );

    let bench_help = run_egraph(&["bench", "--help"]);
    for method in W2_METHODS {
        assert!(
            bench_help.contains(method),
            "bench --help missing method {method}"
        );
    }
}
