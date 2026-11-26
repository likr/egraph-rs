use egraph_core::{DeterministicRng, Layout2d};

use super::super::*;
use super::{
    assert_detail_unit_interval, assert_report_close, make_ideal_10_disks, sample_disk_points,
};

#[test]
fn test_pair_purity_cancellation_regression() {
    // Regression fixtures from 500-case N in [6, 45] differential stress testing where
    // sub-cell sigma_cells = 0.15 produces a dominant Gaussian weight + tiny tail weight
    // (ratio ~1e13), causing catastrophic subtraction cancellation in `wk^2 - sum(w^2)`.
    let idx445_coords: [f32; 18] = [
        -0.93399155,
        -0.08828876,
        0.23824371,
        -1.6165369,
        0.3195233,
        -1.3285851,
        -0.2986154,
        1.6585639,
        -1.2817469,
        -0.23555677,
        0.9802638,
        -0.2675741,
        2.068217,
        0.88675106,
        0.44698393,
        1.5724736,
        0.40123367,
        -1.8822963,
    ];
    let idx445_labels: [u32; 9] = [1, 4, 4, 0, 3, 3, 3, 3, 4];
    let idx445_cfg = ClusterAreaConfig {
        g_max: 12,
        sigma_cells: 0.15,
        boundary_r95: 1.0,
        stevens_gamma: 0.3,
        ..ClusterAreaConfig::default()
    };

    let idx451_coords: [f32; 60] = [
        -0.2518732,
        0.07975522,
        1.3726488,
        0.8500951,
        -1.3431914,
        0.062369324,
        0.39769107,
        1.9115974,
        -0.9139043,
        -0.88805175,
        1.6514385,
        -0.22949933,
        -1.1391747,
        -0.951801,
        0.2050648,
        -1.6683543,
        -0.2511857,
        -0.68052495,
        0.300962,
        0.107651636,
        -0.111619756,
        -0.9943411,
        0.18235113,
        -0.6725447,
        -0.5753704,
        1.084559,
        -0.15376493,
        0.6685573,
        -0.7996044,
        -0.7258471,
        1.8715934,
        -0.4190787,
        -0.7483076,
        0.6918077,
        0.79846996,
        1.8965863,
        0.39656964,
        -0.7518983,
        -1.6543025,
        -0.009605116,
        -1.5369639,
        -0.49072692,
        -1.4884776,
        0.6713425,
        0.050311044,
        0.057186887,
        -0.53254545,
        -0.8697066,
        1.6423913,
        -1.789239,
        1.8272529,
        0.60150194,
        0.21556842,
        -0.25793338,
        1.4036045,
        0.28627896,
        -0.39399222,
        0.44464102,
        0.3066869,
        -0.12467769,
    ];
    let idx451_labels: [u32; 30] = [
        0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
    ];
    let idx451_cfg = ClusterAreaConfig {
        g_max: 32,
        sigma_cells: 0.15,
        boundary_r95: f64::INFINITY,
        stevens_gamma: 0.3,
        ..ClusterAreaConfig::default()
    };

    // idx=445: Cluster 4 (index 3) shares zero grid cells with any other cluster -> exact 1.0.
    let xs445: Vec<f32> = idx445_coords.iter().step_by(2).copied().collect();
    let ys445: Vec<f32> = idx445_coords.iter().skip(1).step_by(2).copied().collect();
    let l445 = Layout2d::from_xy(xs445.clone(), ys445.clone()).unwrap();
    let det445 = cluster_area_faithfulness_detail(&l445, &idx445_labels, &idx445_cfg).unwrap();
    assert_detail_unit_interval(&det445, "idx445_pure_tail");
    assert_eq!(det445.cluster_ids, vec![0, 1, 3, 4]);
    assert!(
        (det445.purity[3] - 1.0).abs() == 0.0,
        "idx445 cluster 4 must be exact 1.0, got {}",
        det445.purity[3]
    );

    let xs451: Vec<f32> = idx451_coords.iter().step_by(2).copied().collect();
    let ys451: Vec<f32> = idx451_coords.iter().skip(1).step_by(2).copied().collect();
    let l451 = Layout2d::from_xy(xs451.clone(), ys451.clone()).unwrap();
    let det451 = cluster_area_faithfulness_detail(&l451, &idx451_labels, &idx451_cfg).unwrap();
    assert_detail_unit_interval(&det451, "idx451_sparse_tail");

    // s500_292: sparse tail density phi_all[c] = 1 - exp(-x) cancellation regression (K=5 and K=6).
    let s500_292_coords: [f32; 74] = [
        -0.8589298,
        -0.8472999,
        -0.01037515,
        -1.7285855,
        0.00178981,
        2.9769006,
        -0.9761431,
        1.2507176,
        0.7966066,
        -0.43772125,
        -1.909174,
        0.88139206,
        -1.493889,
        0.055759937,
        -0.3533779,
        -1.4706259,
        0.6511914,
        -0.35015807,
        0.9030685,
        -1.5927824,
        -0.3389033,
        -1.0626435,
        -0.2578625,
        0.23115428,
        -1.4865725,
        0.38411444,
        0.55941355,
        -0.8532565,
        -1.2615016,
        -0.53420925,
        -0.9444083,
        0.5791151,
        -0.6351455,
        -0.9675821,
        -0.5154885,
        0.91004246,
        0.9587222,
        0.26757297,
        -0.87733287,
        -1.1966555,
        0.7086302,
        1.4324192,
        -1.5779736,
        -0.4205665,
        -0.9108668,
        0.9537005,
        0.76539826,
        0.99292314,
        -1.4883577,
        1.0644866,
        -0.7887485,
        0.22597103,
        -0.87951875,
        -0.9363758,
        0.8706658,
        0.7357092,
        0.95346594,
        -1.8786527,
        0.2655913,
        1.7928542,
        -0.10050357,
        -0.90063393,
        1.3498366,
        0.59151834,
        1.1290692,
        -0.5772221,
        1.0819533,
        -0.86151373,
        0.39588717,
        0.60193646,
        0.4208206,
        -1.8643262,
        0.8065901,
        0.9162831,
    ];
    let s500_292_k5_labels: [u32; 37] = [
        2, 4, 2, 4, 3, 0, 4, 1, 4, 1, 1, 0, 1, 0, 0, 3, 2, 0, 1, 3, 1, 3, 3, 0, 4, 4, 0, 1, 2, 4,
        4, 0, 3, 3, 4, 3, 4,
    ];
    let s500_292_k6_labels: [u32; 37] = [
        3, 5, 3, 4, 4, 0, 5, 1, 5, 1, 2, 0, 1, 0, 0, 4, 2, 0, 2, 3, 1, 4, 4, 1, 5, 5, 0, 2, 3, 5,
        5, 0, 4, 4, 5, 4, 5,
    ];
    let s500_292_cfg = ClusterAreaConfig {
        g_max: 12,
        sigma_cells: 0.15,
        boundary_r95: f64::INFINITY,
        stevens_gamma: 2.5,
        ..ClusterAreaConfig::default()
    };
    let xs292: Vec<f32> = s500_292_coords.iter().step_by(2).copied().collect();
    let ys292: Vec<f32> = s500_292_coords.iter().skip(1).step_by(2).copied().collect();
    let l292 = Layout2d::from_xy(xs292.clone(), ys292.clone()).unwrap();
    let rep292_k5 = cluster_area_faithfulness(&l292, &s500_292_k5_labels, &s500_292_cfg).unwrap();
    let det292_k5 =
        cluster_area_faithfulness_detail(&l292, &s500_292_k5_labels, &s500_292_cfg).unwrap();
    assert_eq!(
        rep292_k5, det292_k5.report,
        "s500_292_k5: report != detail.report"
    );
    assert_detail_unit_interval(&det292_k5, "s500_292_k5");

    let rep292_k6 = cluster_area_faithfulness(&l292, &s500_292_k6_labels, &s500_292_cfg).unwrap();
    let det292_k6 =
        cluster_area_faithfulness_detail(&l292, &s500_292_k6_labels, &s500_292_cfg).unwrap();
    assert_eq!(
        rep292_k6, det292_k6.report,
        "s500_292_k6: report != detail.report"
    );
    assert_detail_unit_interval(&det292_k6, "s500_292_k6");

    // Verify 1-ULP perturbation stability (< 1e-12) across all scalars and per-cluster purities.
    for (name, xs32, ys32, labs, cfg, base_det) in [
        (
            "idx445_pure_tail",
            &xs445,
            &ys445,
            &idx445_labels[..],
            &idx445_cfg,
            &det445,
        ),
        (
            "idx451_sparse_tail",
            &xs451,
            &ys451,
            &idx451_labels[..],
            &idx451_cfg,
            &det451,
        ),
        (
            "s500_292_k5",
            &xs292,
            &ys292,
            &s500_292_k5_labels[..],
            &s500_292_cfg,
            &det292_k5,
        ),
        (
            "s500_292_k6",
            &xs292,
            &ys292,
            &s500_292_k6_labels[..],
            &s500_292_cfg,
            &det292_k6,
        ),
    ] {
        let xs64_base: Vec<f64> = xs32.iter().map(|&v| v as f64).collect();
        let ys64_base: Vec<f64> = ys32.iter().map(|&v| v as f64).collect();
        for &(tag, eps) in &[("+1ulp", 1.0 + 2.22e-16), ("-1ulp", 1.0 - 2.22e-16)] {
            let mut px = xs64_base.clone();
            let mut py = ys64_base.clone();
            for i in 0..px.len() {
                px[i] *= eps;
                py[i] *= eps;
            }
            let pl = Layout2d::from_xy(px, py).unwrap();
            let pdet = cluster_area_faithfulness_detail(&pl, labs, cfg).unwrap();
            assert_detail_unit_interval(&pdet, &format!("{name} {tag}"));
            assert_report_close(
                &base_det.report,
                &pdet.report,
                1e-12,
                &format!("{name} {tag}"),
            );
            for (k, (&pa, &pb)) in base_det.purity.iter().zip(pdet.purity.iter()).enumerate() {
                assert!(
                    (pa - pb).abs() < 1e-12,
                    "{name} {tag}: purity[{k}] diff={} >= 1e-12 (a={pa}, b={pb})",
                    (pa - pb).abs()
                );
            }
        }
    }
}

#[test]
fn test_phi_all_tiny_wt_tau_no_nan() {
    // Verify that tiny Gaussian tail weights (wt < 1e-162) combined with tiny tau (tau <= 1e-162)
    // never trigger 0.0 * inf = NaN in phi_all[c], and all outputs remain finite in [0.0, 1.0].
    let xs = vec![0.0_f64, 0.2, 0.4, 2.0, 2.2, 2.4];
    let ys = vec![0.0_f64, 0.1, -0.1, 0.0, 0.1, -0.1];
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let layout = Layout2d::from_xy(xs, ys).unwrap();
    for &tau_frac in &[1e-200_f64, 1e-320_f64] {
        let cfg = ClusterAreaConfig {
            g_max: 24,
            sigma_cells: 0.04,
            tau_frac,
            ..ClusterAreaConfig::default()
        };
        let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
        let det = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
        assert_eq!(
            rep, det.report,
            "tiny tau_frac={tau_frac}: report != detail.report"
        );
        assert_detail_unit_interval(&det, &format!("tiny_wt_tau_{tau_frac:e}"));
    }
}

#[test]
fn test_unit_interval_clamping_adversarial() {
    let cfg = ClusterAreaConfig::default();

    // 1. K = N singletons (N = 100) triggers ULP overshoot on sum(p * e_k), sum(p * d_k) without clamping.
    let mut xs_s = Vec::with_capacity(100);
    let mut ys_s = Vec::with_capacity(100);
    let labs_s: Vec<u32> = (0..100).collect();
    for i in 0..100usize {
        xs_s.push((i % 10) as f64 * 2.0);
        ys_s.push((i / 10) as f64 * 2.0);
    }
    let l_s = Layout2d::from_xy(xs_s, ys_s).unwrap();
    let det_s = cluster_area_faithfulness_detail(&l_s, &labs_s, &cfg).unwrap();
    assert_detail_unit_interval(&det_s, "K=N singletons (N=100)");

    // Also test a_tot <= 1e-15 early-return path with K = N singletons.
    let det_s_empty = cluster_area_faithfulness_detail(
        &l_s,
        &labs_s,
        &ClusterAreaConfig {
            boundary_r95: 1e-6,
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(det_s_empty.report.in_frac, 0.0);
    assert_detail_unit_interval(&det_s_empty, "K=N singletons a_tot<=1e-15");

    // 2. Extreme scales 1e-50 / 1e50 and translation 1e8.
    let (base_layout, base_labels) = make_ideal_10_disks(500, 3.0, 10.0);
    let bx = base_layout.x();
    let by = base_layout.y();
    for &scale in &[1e-50, 1e50] {
        let l_sc = Layout2d::from_xy(
            bx.iter().map(|&v| v * scale).collect(),
            by.iter().map(|&v| v * scale).collect(),
        )
        .unwrap();
        let det = cluster_area_faithfulness_detail(&l_sc, &base_labels, &cfg).unwrap();
        assert_detail_unit_interval(&det, &format!("scale={scale:e}"));
    }
    let l_tr = Layout2d::from_xy(
        bx.iter().map(|&v| v + 1e8).collect(),
        by.iter().map(|&v| v + 1e8).collect(),
    )
    .unwrap();
    let det_tr = cluster_area_faithfulness_detail(&l_tr, &base_labels, &cfg).unwrap();
    assert_detail_unit_interval(&det_tr, "trans=1e8");

    // 3. Asymmetric aspect ratio 1e6 : 1 (horizontal and vertical).
    let mut rng = DeterministicRng::from_seed_u64(808);
    let mut ax = Vec::with_capacity(400);
    let mut ay = Vec::with_capacity(400);
    let mut alabs = Vec::with_capacity(400);
    for k in 0..4usize {
        let (dx, dy) = sample_disk_points(&mut rng, (k as f64) * 1e6, 0.0, 1.0, 100);
        ax.extend(dx);
        ay.extend(dy);
        alabs.extend(std::iter::repeat_n(k as u32, 100));
    }
    let l_ah = Layout2d::from_xy(ax.clone(), ay.clone()).unwrap();
    let det_ah = cluster_area_faithfulness_detail(&l_ah, &alabs, &cfg).unwrap();
    assert_detail_unit_interval(&det_ah, "aspect=1e6:1 horizontal");
    let l_av = Layout2d::from_xy(ay, ax).unwrap();
    let det_av = cluster_area_faithfulness_detail(&l_av, &alabs, &cfg).unwrap();
    assert_detail_unit_interval(&det_av, "aspect=1e6:1 vertical");

    // 4. Stevens gamma = 3.0.
    let det_g3 = cluster_area_faithfulness_detail(
        &base_layout,
        &base_labels,
        &ClusterAreaConfig {
            stevens_gamma: 3.0,
            ..cfg
        },
    )
    .unwrap();
    assert_detail_unit_interval(&det_g3, "stevens_gamma=3.0");
}
