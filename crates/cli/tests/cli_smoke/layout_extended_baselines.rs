use std::fs;
use std::process::Command;

use crate::common::{EXTENDED_BASELINE_METHODS, assert_finite_layout_json, run_egraph, temp_dir};

#[test]
fn cli_layout_extended_baselines_and_bench() {
    for method in EXTENDED_BASELINE_METHODS {
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
fn cli_extended_baselines_determinism_with_seed() {
    for method in EXTENDED_BASELINE_METHODS {
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

    // Different seeds must produce distinct layouts on randomized/stochastic baseline methods.
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
fn cli_extended_baselines_degenerate_and_edge_case_graphs() {
    let dir = temp_dir("degenerate_extended");
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
        for method in EXTENDED_BASELINE_METHODS {
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

    // Zero-weight edge rejection for `hde` vs clean handling across remaining baseline methods.
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

    for method in EXTENDED_BASELINE_METHODS
        .into_iter()
        .filter(|m| *m != "hde")
    {
        let out = run_egraph(&[
            "layout", "--input", zero_p, "--method", method, "--seed", "42",
        ]);
        let val: serde_json::Value = serde_json::from_str(&out).expect("parse zero_w layout");
        assert_finite_layout_json(&val, 3, &format!("zero_w/{method}"));
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_help_includes_extended_methods_and_clarified_docstrings() {
    let layout_help = run_egraph(&["layout", "--help"]);
    for method in EXTENDED_BASELINE_METHODS {
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
    for method in EXTENDED_BASELINE_METHODS {
        assert!(
            bench_help.contains(method),
            "bench --help missing method {method}"
        );
    }
}
