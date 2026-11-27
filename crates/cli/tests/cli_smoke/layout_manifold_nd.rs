use std::fs;

use crate::common::{assert_finite_layout_json, run_egraph, temp_dir};

fn parse_json(raw: &str, ctx: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("{ctx}: invalid json: {e}"))
}

#[test]
fn cli_tsnet_rn_and_twin_declutter() {
    let dir = temp_dir("tsnet_rn_twin_declutter");

    let rn1 = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--seed",
        "42",
    ]);
    let rn2 = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--seed",
        "42",
    ]);
    let v1 = parse_json(&rn1, "tsnet-rn 1");
    let v2 = parse_json(&rn2, "tsnet-rn 2");
    assert_eq!(v1["method"], "tsnet-rn");
    assert_eq!(v1["dims"], 2);
    assert_eq!(v1["coords"], v2["coords"]);
    assert_finite_layout_json(&v1, 34, "tsnet-rn");

    let bench_rn = run_egraph(&[
        "bench",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--seed",
        "42",
    ]);
    let bench_val = parse_json(&bench_rn, "bench tsnet-rn");
    assert_eq!(bench_val["method"], "tsnet-rn");
    assert!(
        bench_val["neighborhood_preservation_2hop"]
            .as_f64()
            .is_some()
    );

    let twin_layout = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--overlap-removal",
        "twin-declutter",
        "--seed",
        "42",
    ]);
    let twin_val = parse_json(&twin_layout, "twin-declutter");
    assert_eq!(twin_val["overlap_removal"], "twin-declutter");
    assert_eq!(twin_val["prism_applied"], false);
    assert_finite_layout_json(&twin_val, 34, "twin-declutter");

    let twin_path = dir.join("twin_layout.json");
    fs::write(&twin_path, &twin_layout).expect("write twin layout");
    let deg_out = run_egraph(&[
        "metrics",
        "--dataset",
        "karate",
        "--layout",
        twin_path.to_str().expect("utf8 path"),
        "--targets",
        "layout-degeneracy,crossing-edges",
    ]);
    let deg_val = parse_json(&deg_out, "deg out");
    assert_eq!(deg_val["degeneracy_excess_duplicate_ratio"], 0.0);
    assert!(
        deg_val["degeneracy_near_collision_ratio"]
            .as_f64()
            .is_some()
    );
    assert!(deg_val["crossing_edges_count"].as_u64().is_some());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_nd_layouts_and_metrics() {
    let dir = temp_dir("nd_layouts_metrics");

    for method in [
        "tsnet-rn",
        "tsnet-pure2hop",
        "sgd-full",
        "stress",
        "fr",
        "hde",
    ] {
        let out = run_egraph(&[
            "layout",
            "--dataset",
            "karate",
            "--method",
            method,
            "--dims",
            "3",
            "--seed",
            "42",
        ]);
        let val = parse_json(&out, method);
        assert_eq!(val["dims"], 3, "{method}: dims");
        assert!(val.get("manifold").is_none(), "{method}: no manifold");
        let coords_nd = val["coords_nd"].as_array().expect("coords_nd array");
        assert_eq!(coords_nd.len(), 34, "{method}: coords_nd len");
        for row in coords_nd {
            let pts = row.as_array().expect("row array");
            assert_eq!(pts.len(), 3, "{method}: 3D point len");
            assert!(pts.iter().all(|c| c.as_f64().is_some_and(f64::is_finite)));
        }

        if method == "sgd-full" {
            let nd_path = dir.join("karate_3d.json");
            fs::write(&nd_path, &out).expect("write 3d layout");
            let m_out = run_egraph(&[
                "metrics",
                "--dataset",
                "karate",
                "--layout",
                nd_path.to_str().expect("utf8 path"),
                "--targets",
                "sns,np1,np2,layout-degeneracy",
            ]);
            let m_val = parse_json(&m_out, "3d metrics");
            assert!(m_val["scale_normalized_stress"].as_f64().is_some());
            assert!(m_val["neighborhood_preservation_1hop"].as_f64().is_some());
            assert!(m_val["neighborhood_preservation_2hop"].as_f64().is_some());
            assert!(m_val["layout_degeneracy"].as_f64().is_some());
            assert!(
                m_val["degeneracy_excess_duplicate_ratio"]
                    .as_f64()
                    .is_some()
            );
            assert!(m_val["degeneracy_near_collision_ratio"].as_f64().is_some());
        }
    }

    let bench_3d = run_egraph(&[
        "bench",
        "--dataset",
        "karate",
        "--method",
        "stress",
        "--dims",
        "3",
        "--seed",
        "42",
    ]);
    let b3 = parse_json(&bench_3d, "3d bench");
    assert_eq!(b3["dims"], 3);
    assert!(b3["crossing_number"].is_null());
    assert!(b3["scale_normalized_stress"].as_f64().is_some());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_manifold_layouts_schedule_windows_and_metrics() {
    let dir = temp_dir("manifold_layouts_metrics");

    for manifold in ["hyperbolic", "hyperbolic-lorentz", "spherical", "torus"] {
        let mut args = vec![
            "layout",
            "--dataset",
            "karate",
            "--method",
            "sgd-full",
            "--manifold",
            manifold,
            "--manifold-scale",
            "1.25",
            "--schedule-windows",
            "2",
            "--schedule-window-iters",
            "15",
            "--manifold-rescale-final",
            "2.0",
            "--seed",
            "42",
        ];
        if manifold == "torus" {
            args.extend(["--torus-aspect", "1.5", "--torus-auto-pan"]);
        }
        let out = run_egraph(&args);
        let val = parse_json(&out, manifold);
        assert_eq!(val["manifold"], manifold);
        assert_eq!(val["manifold_scale"], 2.0);
        assert_finite_layout_json(&val, 34, manifold);

        let xs = val["x"].as_array().expect("x");
        let ys = val["y"].as_array().expect("y");
        if manifold == "hyperbolic" {
            for (xv, yv) in xs.iter().zip(ys.iter()) {
                let r = xv.as_f64().unwrap().hypot(yv.as_f64().unwrap());
                assert!(r < 1.0, "poincare point out of disk: r={r}");
            }
        } else if manifold == "torus" {
            assert_eq!(val["torus_aspect"], 1.5);
            for (xv, yv) in xs.iter().zip(ys.iter()) {
                let x = xv.as_f64().unwrap();
                let y = yv.as_f64().unwrap();
                assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
            }
        }

        let m_path = dir.join(format!("karate_{manifold}.json"));
        fs::write(&m_path, &out).expect("write manifold layout");

        let targets = if manifold == "torus" {
            "sns,geodesic-distortion,torus-crossing-number,crossing-edges-torus,np1,np2,layout-degeneracy"
        } else {
            "sns,geodesic-distortion,np1,np2,layout-degeneracy"
        };
        let m_out = run_egraph(&[
            "metrics",
            "--dataset",
            "karate",
            "--layout",
            m_path.to_str().expect("utf8 path"),
            "--targets",
            targets,
        ]);
        let m_val = parse_json(&m_out, manifold);
        assert!(m_val["scale_normalized_stress"].as_f64().is_some());
        assert!(m_val["geodesic_distortion"].as_f64().is_some());
        assert!(m_val["neighborhood_preservation_2hop"].as_f64().is_some());
        if manifold == "torus" {
            assert!(m_val["torus_crossing_number"].as_u64().is_some());
            assert!(m_val["crossing_edges_torus_count"].as_u64().is_some());
        }
    }

    let torus_rn = run_egraph(&[
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--init",
        "koren",
        "--manifold",
        "torus",
        "--torus-auto-pan",
        "--seed",
        "42",
    ]);
    let tr_val = parse_json(&torus_rn, "torus tsnet-rn");
    assert_eq!(tr_val["manifold"], "torus");
    assert_finite_layout_json(&tr_val, 34, "torus tsnet-rn");

    let _ = fs::remove_dir_all(&dir);
}
