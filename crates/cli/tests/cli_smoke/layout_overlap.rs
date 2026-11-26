use std::process::Command;

use crate::common::{assert_finite_layout_json, assert_no_pairwise_overlap, run_egraph};

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

    // Invalid radius guard across overlap-removal variants (must fail fast).
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
