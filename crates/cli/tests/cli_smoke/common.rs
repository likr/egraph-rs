use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub(crate) const EXTENDED_BASELINE_METHODS: [&str; 10] = [
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

pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("egraph_cli_smoke_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

pub(crate) fn run_egraph(args: &[&str]) -> String {
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

pub(crate) fn assert_finite_layout_json(val: &serde_json::Value, expected_n: usize, ctx: &str) {
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

pub(crate) fn assert_no_pairwise_overlap(xs: &[f64], ys: &[f64], radius: f64, ctx: &str) {
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
