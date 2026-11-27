use std::fs;
use std::process::Command;

use crate::common::{run_egraph, temp_dir};

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
        "--labels",
        cluster_path.to_str().expect("utf8 path"),
        "--targets",
        all_18_targets,
        "--seed",
        "42",
    ]);
    let m18: serde_json::Value = serde_json::from_str(&out_18).expect("parse 18 metrics");
    let baseline_keys = [
        "distance_consistency",
        "distribution_consistency",
        "class_density_measure",
        "histogram_density_measure",
        "sepme_knng",
        "sepme_rng",
        "sepme_gab",
        "sepme_gong",
        "neighborhood_hit",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
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
        "distance_consistency",
        "distribution_consistency",
        "class_density_measure",
        "histogram_density_measure",
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
    for signed_unit_key in [
        "silhouette_score",
        "cq_kmeans_ari",
        "cq_kmeans_ami",
        "cq_hac_ami",
    ] {
        let v = m18[signed_unit_key].as_f64().expect("signed unit metric");
        assert!(
            (-1.0..=1.0).contains(&v),
            "{signed_unit_key} out of [-1,1]: {v}"
        );
    }
    for nonneg_key in [
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "sprawlter_nn",
    ] {
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
            "--labels",
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

    // Label-free sprawlter-nn succeeds without --labels and omits unrequested baseline keys.
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
    assert!(
        sp_obj
            .as_object()
            .expect("obj")
            .get("distance_consistency")
            .is_none()
    );

    // Regression: `--targets all` without `--labels` omits all 18 eval-baselines keys.
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

    // Missing `--labels` error guard for label-dependent cluster baseline targets.
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
            "expected non-zero exit for --targets {target} without --labels"
        );
        let stderr = String::from_utf8_lossy(&missing_out.stderr);
        assert!(
            stderr.contains("labels"),
            "expected stderr mentioning labels for {target}, got: {stderr}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}
