use std::fs;
use std::process::Command;

use crate::common::{run_egraph, temp_dir};

fn parse_json(raw: &str, ctx: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("{ctx}: invalid json: {e}"))
}

#[test]
fn cli_nor_prism_cluster_diagnostics_and_guards() {
    let dir = temp_dir("nor_prism_guards");
    let orig_path = dir.join("orig.json");
    let removed_path = dir.join("removed.json");
    let hw_path = dir.join("half_widths.txt");
    let hh_path = dir.join("half_heights.txt");

    run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--seed",
        "42",
        "--output",
        orig_path.to_str().expect("utf8 path"),
    ]);
    run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--overlap-removal",
        "prism",
        "--overlap-radius",
        "0.05",
        "--seed",
        "42",
        "--output",
        removed_path.to_str().expect("utf8 path"),
    ]);

    let radii_txt = vec!["0.05"; 34].join(" ");
    fs::write(&hw_path, &radii_txt).expect("write half widths");
    fs::write(&hh_path, &radii_txt).expect("write half heights");

    let m_out = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        removed_path.to_str().expect("utf8 path"),
        "--original-layout",
        orig_path.to_str().expect("utf8 path"),
        "--overlap-radius",
        "0.05",
        "--sprawlter-half-widths",
        hw_path.to_str().expect("utf8 path"),
        "--sprawlter-half-heights",
        hh_path.to_str().expect("utf8 path"),
        "--targets",
        "prism-distortion,nor-metrics,sprawlter-nn",
    ]);
    let m_val = parse_json(&m_out, "nor/prism metrics");
    for key in [
        "prism_sigma_dist",
        "prism_sigma_disp",
        "nor_oo_nni",
        "nor_sp_ch_a",
        "nor_gs_bb_iar",
        "nor_nm_dm_imse",
        "nor_el_rsd",
        "sprawlter_nn",
    ] {
        assert!(m_val[key].as_f64().is_some_and(f64::is_finite), "{key}");
    }

    let lp_out = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "label-prop",
        "--seed",
        "42",
    ]);
    let lp_val = parse_json(&lp_out, "lp");
    assert!(lp_val["iterations"].as_u64().is_some());
    assert!(lp_val["converged"].as_bool().is_some());

    let sp_out = run_egraph(&[
        "cluster",
        "--dataset",
        "karate",
        "--method",
        "spectral",
        "--k",
        "3",
        "--seed",
        "42",
    ]);
    let sp_val = parse_json(&sp_out, "spectral");
    assert!(sp_val["eigen_iterations"].as_u64().is_some());

    let bin = env!("CARGO_BIN_EXE_egraph");
    for bad_args in [
        vec![
            "layout",
            "--dataset",
            "karate",
            "--dims",
            "3",
            "--manifold",
            "torus",
        ],
        vec!["layout", "--dataset", "karate", "--torus-auto-pan"],
        vec![
            "layout",
            "--dataset",
            "karate",
            "--manifold-rescale-final",
            "2.0",
        ],
        vec![
            "layout",
            "--dataset",
            "karate",
            "--method",
            "stress",
            "--schedule-windows",
            "2",
        ],
        vec![
            "layout",
            "--dataset",
            "karate",
            "--method",
            "ibumap",
            "--dims",
            "3",
        ],
        vec![
            "layout",
            "--dataset",
            "karate",
            "--method",
            "tfdp",
            "--manifold",
            "torus",
        ],
        vec![
            "layout",
            "--dataset",
            "karate",
            "--method",
            "fa2",
            "--manifold",
            "spherical",
        ],
    ] {
        let res = Command::new(bin)
            .args(&bad_args)
            .output()
            .expect("run egraph");
        assert!(!res.status.success(), "expected failure for {:?}", bad_args);
    }

    let _ = fs::remove_dir_all(&dir);
}
