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
        "sns,sampled-sns,crossings,np1,np2,ideal-edge-lengths",
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
