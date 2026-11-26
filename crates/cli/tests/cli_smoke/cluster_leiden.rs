use std::process::Command;

use crate::common::run_egraph;

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
