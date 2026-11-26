use std::f64::consts::PI;

use egraph_core::Layout2d;

use super::super::mec::exact_mec;
use super::super::*;
use super::{assert_detail_unit_interval, assert_report_close};

#[test]
fn mec_100k_circle_timing() {
    let n = 100_000usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let theta = (2.0 * PI * (i as f64)) / (n as f64);
        pts.push((theta.cos(), theta.sin()));
    }
    let t0 = std::time::Instant::now();
    let (c, r) = exact_mec(&pts);
    let elapsed = t0.elapsed();
    eprintln!(
        "mec_100k_circle_timing: {:?} (center={:?}, r={})",
        elapsed, c, r
    );
    assert!(c.0.hypot(c.1) < 1e-12, "center={:?}", c);
    assert!((r - 1.0).abs() < 1e-12, "r={}", r);
    // Release normally runs around 10-25ms locally; allow generous headroom for shared CI
    // runners while still failing immediately on an O(h^3) regression (minutes/hours).
    #[cfg(not(debug_assertions))]
    assert!(elapsed.as_secs_f64() < 0.5, "elapsed={:?}", elapsed);
    #[cfg(debug_assertions)]
    assert!(elapsed.as_secs_f64() < 2.0, "elapsed={:?}", elapsed);
}

#[test]
fn test_small_n_mec_quantile_tie_stability() {
    // 1. Synthetic equidistant Stage-1/Stage-2 MEC tie (N = 30 and N = 41) with ±1e-15 jitter
    // on vertex 0 across both boundary_r95 = 2.0 (default) and boundary_r95 = 1.0.
    for &n in &[30usize, 41usize] {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for i in 0..3 {
            let theta = 0.37 + 2.0 * PI * (i as f64) / 3.0;
            xs.push(theta.cos());
            ys.push(theta.sin());
            labels.push((i % 3) as u32);
        }
        let inner = n - 3;
        for j in 0..inner {
            let r = 0.65 * (((j + 1) as f64) / (inner as f64)).sqrt();
            let theta = 2.399_963_229_728_653 * ((j + 1) as f64);
            xs.push(r * theta.cos());
            ys.push(r * theta.sin());
            labels.push(((j + 3) % 3) as u32);
        }

        let base_l = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let mut xs_p = xs.clone();
        let mut ys_p = ys.clone();
        xs_p[0] *= 1.0 + 1e-15;
        ys_p[0] *= 1.0 + 1e-15;
        let plus_l = Layout2d::from_xy(xs_p, ys_p).unwrap();

        let mut xs_m = xs;
        let mut ys_m = ys;
        xs_m[0] *= 1.0 - 1e-15;
        ys_m[0] *= 1.0 - 1e-15;
        let minus_l = Layout2d::from_xy(xs_m, ys_m).unwrap();

        for &b_r95 in &[2.0_f64, 1.0_f64] {
            let cfg = ClusterAreaConfig {
                boundary_r95: b_r95,
                ..ClusterAreaConfig::default()
            };
            let r_base = cluster_area_faithfulness_detail(&base_l, &labels, &cfg).unwrap();
            let r_plus = cluster_area_faithfulness(&plus_l, &labels, &cfg).unwrap();
            let r_minus = cluster_area_faithfulness(&minus_l, &labels, &cfg).unwrap();
            assert_detail_unit_interval(&r_base, &format!("synthetic_tie n={n} b={b_r95}"));
            assert_report_close(
                &r_base.report,
                &r_plus,
                1e-12,
                &format!("synthetic_tie +1e-15 n={n} b={b_r95}"),
            );
            assert_report_close(
                &r_base.report,
                &r_minus,
                1e-12,
                &format!("synthetic_tie -1e-15 n={n} b={b_r95}"),
            );
        }
    }

    // 2. Exact f32 N <= 40 cross-target parity regression fixtures (n30_k3, n39_k5, n40_b1, n6_b1).
    let n30_k3_coords: [f32; 60] = [
        -0.28154773,
        2.70013,
        1.5668566,
        0.07951676,
        -1.750706,
        3.2445662,
        -0.72710097,
        -1.1585857,
        -0.90493894,
        -0.016284715,
        0.64667726,
        0.38064554,
        -0.6412784,
        1.3978007,
        -0.4455091,
        -0.490506,
        2.5104165,
        1.2259845,
        0.4358604,
        -1.8759084,
        -2.0040073,
        1.1233397,
        1.072034,
        0.9364058,
        0.14748514,
        -0.58216316,
        -1.3126808,
        2.8923485,
        1.1164329,
        -2.4579682,
        -0.78497416,
        -4.539269,
        -1.1620724,
        -0.71214044,
        -0.56531346,
        -0.73464775,
        -2.099709,
        -0.093_136_556_446_552_28_f64 as f32,
        1.1614314,
        0.35444275,
        0.46526924,
        1.7718444,
        -0.44344476,
        1.4617308,
        -1.6758195,
        1.5955342,
        -3.2307613,
        -0.6647125,
        0.012851243,
        1.9697237,
        0.26684767,
        3.094563,
        -0.8386496,
        0.16661248,
        2.7859595,
        -2.0786989,
        -3.6580622,
        2.894068,
        4.2667637,
        0.91116905,
    ];
    let n30_k3_labels: [u32; 30] = [
        2, 1, 2, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
    ];

    let n39_k5_coords: [f32; 78] = [
        2.317929,
        -2.377_370_119_094_848_6_f64 as f32,
        -2.5181954,
        2.293665,
        1.5150898,
        -0.16263618,
        -0.541605,
        1.4072235,
        -2.6598365,
        1.2367346,
        0.23080106,
        2.890641,
        0.14955002,
        -2.7411644,
        2.9515316,
        0.5958319,
        -2.2528713,
        -1.2812335,
        -4.208866,
        -2.0573194,
        1.2402142,
        -0.64551497,
        -3.7162738,
        -1.5085694,
        0.5517861,
        0.43332744,
        -2.0856888,
        0.82401204,
        1.1661124,
        0.5285953,
        -1.7103713,
        0.91390324,
        -1.1898159,
        -2.2106411,
        2.106105,
        1.6581815,
        1.8313656,
        -5.0268283,
        0.159858,
        2.3602962,
        1.1456916,
        -0.6920323,
        -1.9657927,
        -1.7649826,
        -2.4530983,
        -2.5909653,
        0.39769387,
        0.16761969,
        -3.082718,
        -0.32256436,
        -3.2988117,
        0.354599,
        -2.2738812,
        0.9519124,
        -0.38830006,
        -1.5800346,
        -0.73172736,
        2.5496125,
        0.18937981,
        2.016829,
        -0.643776,
        -1.1126575,
        -1.6523632,
        -1.4371283,
        1.399753,
        -1.3744355,
        0.34608787,
        -0.20996541,
        0.8923608,
        -0.337803,
        0.48425642,
        -1.9588091,
        2.5424654,
        0.02001159,
        1.4568391,
        -3.30631,
        -2.2863953,
        -0.16748564,
    ];
    let n39_k5_labels: [u32; 39] = [
        3, 0, 0, 1, 4, 3, 0, 3, 1, 3, 3, 2, 2, 0, 0, 3, 4, 2, 0, 3, 0, 1, 1, 4, 1, 1, 3, 1, 0, 3,
        1, 2, 2, 3, 2, 3, 3, 2, 1,
    ];

    let n40_b1_coords: [f32; 80] = [
        -2.9233508,
        1.3258305,
        -1.2700397,
        0.055691067,
        -3.5743446,
        0.08754364,
        2.8365555,
        -1.2860936,
        0.67369485,
        2.1501174,
        0.027947975,
        -0.5190643,
        1.8806356,
        0.38272673,
        0.59430355,
        -5.774634,
        1.3010856,
        0.15634698,
        -0.19979255,
        2.8649082,
        -0.4877356,
        -0.35270914,
        1.160403,
        1.5458733,
        -3.2697394,
        -1.7934492,
        -1.9938316,
        0.7835421,
        2.9356518,
        -1.3010819,
        2.1025224,
        3.2613504,
        -3.110684,
        -3.0776134,
        4.413053,
        1.5375748,
        1.0798331,
        -0.56084615,
        -0.2496351,
        0.99739313,
        2.0195343,
        -1.7061172,
        -3.5770214,
        3.366582,
        1.310306,
        -1.3890607,
        0.76046354,
        0.39164633,
        1.9436023,
        1.0128682,
        -0.8362288,
        3.489198,
        -2.9951606,
        2.4799,
        -0.43555063,
        1.4823762,
        3.8454504,
        -0.4614333,
        -4.027053,
        -0.3833692,
        -2.1162024,
        2.3836763,
        2.3869655,
        -1.7980483,
        1.800023,
        1.4273919,
        -2.7043612,
        -2.191859,
        1.9361794,
        1.299877,
        -0.2770219,
        1.5509106,
        1.5726527,
        -3.354655,
        -0.31485745,
        2.2452877,
        1.2533443,
        -2.2307775,
        -0.2523287,
        0.99455553,
    ];
    let n40_b1_labels: [u32; 40] = [
        3, 1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 0, 2, 3, 2, 1, 0, 1, 2, 1, 0, 3, 1, 1, 1, 0, 0, 2, 1, 3,
        0, 3, 2, 2, 0, 0, 1, 1, 0, 1,
    ];

    let n6_b1_coords: [f32; 12] = [
        0.09900733,
        3.4892879,
        -2.2565875,
        1.0310328,
        -0.25905266,
        0.7601389,
        0.6031524,
        1.156712,
        4.331787,
        2.1050458,
        -2.2524312,
        3.5577226,
    ];
    let n6_b1_labels: [u32; 6] = [3, 0, 5, 1, 3, 5];

    let cfg_n30_custom = ClusterAreaConfig {
        g_max: 12,
        sigma_cells: 0.15,
        boundary_r95: 0.8,
        ..ClusterAreaConfig::default()
    };
    let cfg_n39_custom = ClusterAreaConfig {
        g_max: 32,
        sigma_cells: 1.8,
        boundary_r95: 5.0,
        ..ClusterAreaConfig::default()
    };
    let cfg_b1 = ClusterAreaConfig {
        boundary_r95: 1.0,
        ..ClusterAreaConfig::default()
    };

    let cases: &[(&str, &[f32], &[u32], ClusterAreaConfig)] = &[
        (
            "n30_k3_default",
            &n30_k3_coords,
            &n30_k3_labels,
            ClusterAreaConfig::default(),
        ),
        (
            "n30_k3_custom",
            &n30_k3_coords,
            &n30_k3_labels,
            cfg_n30_custom,
        ),
        (
            "n39_k5_default",
            &n39_k5_coords,
            &n39_k5_labels,
            ClusterAreaConfig::default(),
        ),
        (
            "n39_k5_custom",
            &n39_k5_coords,
            &n39_k5_labels,
            cfg_n39_custom,
        ),
        ("n40_b1", &n40_b1_coords, &n40_b1_labels, cfg_b1),
        ("n6_b1", &n6_b1_coords, &n6_b1_labels, cfg_b1),
    ];

    for &(name, xy, labs, ref cfg) in cases {
        let xs32: Vec<f32> = xy.iter().step_by(2).copied().collect();
        let ys32: Vec<f32> = xy.iter().skip(1).step_by(2).copied().collect();
        let l32 = Layout2d::from_xy(xs32.clone(), ys32.clone()).unwrap();
        let det = cluster_area_faithfulness_detail(&l32, labs, cfg).unwrap();
        let rep = cluster_area_faithfulness(&l32, labs, cfg).unwrap();
        assert_eq!(rep, det.report, "{name}: report != detail.report");
        assert_detail_unit_interval(&det, name);

        let xs64: Vec<f64> = xs32.iter().map(|&v| v as f64).collect();
        let ys64: Vec<f64> = ys32.iter().map(|&v| v as f64).collect();
        let cx = xs64.iter().sum::<f64>() / (xs64.len() as f64);
        let cy = ys64.iter().sum::<f64>() / (ys64.len() as f64);
        let max_idx = (0..xs64.len())
            .max_by(|&a, &b| {
                (xs64[a] - cx)
                    .hypot(ys64[a] - cy)
                    .total_cmp(&(xs64[b] - cx).hypot(ys64[b] - cy))
            })
            .unwrap();

        for &(tag, eps) in &[("+1e-15", 1.0 + 1e-15), ("-1e-15", 1.0 - 1e-15)] {
            let mut px = xs64.clone();
            let mut py = ys64.clone();
            px[max_idx] = cx + (px[max_idx] - cx) * eps;
            py[max_idx] = cy + (py[max_idx] - cy) * eps;
            let pl = Layout2d::from_xy(px, py).unwrap();
            let prep = cluster_area_faithfulness(&pl, labs, cfg).unwrap();
            assert_report_close(&rep, &prep, 1e-12, &format!("{name} {tag}"));
        }
    }
}
