use std::fs;
use std::process::Command;

use crate::common::{run_egraph, temp_dir};

#[test]
fn cli_layout_community_baselines_cluster_refine_and_bench() {
    let dir = temp_dir("community_layout");
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
fn cli_community_baselines_degenerate_graphs() {
    let dir = temp_dir("degenerate_community");
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
