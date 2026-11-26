use std::fs;
use std::process::Command;

use crate::common::{run_egraph, temp_dir};

#[test]
fn cli_metrics_cluster_area_faithfulness_error_guards() {
    let dir = temp_dir("gcaf_guards");
    let layout_path = dir.join("karate_layout.json");
    let cluster_path = dir.join("karate_cluster.json");
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

    let bin = env!("CARGO_BIN_EXE_egraph");

    // 1. Missing `--community-labels` when requesting `cluster-area-faithfulness` errors cleanly.
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

    // 2. Length mismatch in `--community-labels` errors cleanly.
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

    // 3. Trailing invalid target after `all` must be rejected cleanly with non-zero exit.
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

    // 4. Malformed label files and `--num-pivots 0` error guards.
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
