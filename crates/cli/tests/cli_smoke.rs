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
    assert!(metrics_all["scale_normalized_stress"].as_f64().is_some());
    assert!(
        metrics_all["landmark_stratified_rta"].as_f64().is_some(),
        "expected all standard metrics including landmark_stratified_rta"
    );

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
