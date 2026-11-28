use std::f64::consts::PI;

use egraph_core::{CoreError, DeterministicRng, Layout2d};

use super::super::*;
use super::{
    assert_detail_unit_interval, assert_report_close, make_ideal_10_disks, sample_disk_points,
};

#[test]
fn test_split_4islands_coherence() {
    let cfg = ClusterAreaConfig::default();
    let mut rng = DeterministicRng::from_seed_u64(2026);
    let mut xs = Vec::with_capacity(4000);
    let mut ys = Vec::with_capacity(4000);
    let mut labels = Vec::with_capacity(4000);

    // Cluster 0 split into 4 equal distant islands; Cluster 1 split into 4 equal distant islands.
    let c0_centers = [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)];
    let c1_centers = [(-10.0, 0.0), (10.0, 0.0), (0.0, -10.0), (0.0, 10.0)];
    let (tmpl_x, tmpl_y) = sample_disk_points(&mut rng, 0.0, 0.0, 2.0, 500);
    for &(cx, cy) in &c0_centers {
        xs.extend(tmpl_x.iter().map(|&x| cx + x));
        ys.extend(tmpl_y.iter().map(|&y| cy + y));
        labels.extend(std::iter::repeat_n(0u32, 500));
    }
    for &(cx, cy) in &c1_centers {
        xs.extend(tmpl_x.iter().map(|&x| cx + x));
        ys.extend(tmpl_y.iter().map(|&y| cy + y));
        labels.extend(std::iter::repeat_n(1u32, 500));
    }

    let layout = Layout2d::from_xy(xs, ys).unwrap();
    let detail = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
    assert!(detail.report.caf > 0.90, "caf={}", detail.report.caf);
    assert!(detail.report.purity_adj > 0.99);
    assert!(
        (detail.coherence[0] - 0.25).abs() < 0.005,
        "c0={}",
        detail.coherence[0]
    );
    assert!(
        (detail.coherence[1] - 0.25).abs() < 0.005,
        "c1={}",
        detail.coherence[1]
    );
    assert!(
        (detail.report.spatial_coherence - 0.25).abs() < 0.005,
        "coherence={}",
        detail.report.spatial_coherence
    );

    // Mixed: C0 = 4 islands (2000 pts), C1 = 1 disk of 4x area / equal visual weight (2000 pts)
    let mut xs_mix = Vec::with_capacity(4000);
    let mut ys_mix = Vec::with_capacity(4000);
    let mut labs_mix = Vec::with_capacity(4000);
    for &(cx, cy) in &c0_centers {
        xs_mix.extend(tmpl_x.iter().map(|&x| cx + x));
        ys_mix.extend(tmpl_y.iter().map(|&y| cy + y));
        labs_mix.extend(std::iter::repeat_n(0u32, 500));
    }
    // Radius 4.0 gives 4 * pi * 2.0^2 = pi * 4.0^2 equal area to the 4 islands combined.
    let (dx, dy) = sample_disk_points(&mut rng, 0.0, 0.0, 4.0, 2000);
    xs_mix.extend(dx);
    ys_mix.extend(dy);
    labs_mix.extend(std::iter::repeat_n(1u32, 2000));
    let mix_layout = Layout2d::from_xy(xs_mix, ys_mix).unwrap();
    let mix_detail = cluster_area_faithfulness_detail(&mix_layout, &labs_mix, &cfg).unwrap();
    assert!((mix_detail.coherence[0] - 0.25).abs() < 0.005);
    assert!((mix_detail.coherence[1] - 1.0).abs() < 1e-6);
    let expected_mix_coh =
        mix_detail.q[0] * mix_detail.coherence[0] + mix_detail.q[1] * mix_detail.coherence[1];
    assert!((mix_detail.report.spatial_coherence - expected_mix_coh).abs() < 1e-12);
    assert!(
        (mix_detail.report.spatial_coherence - 0.625).abs() < 0.05,
        "mix spatial_coherence={}",
        mix_detail.report.spatial_coherence
    );
}

#[test]
fn test_f2_coherence_shapes_saturation_and_barriers() {
    let cfg = ClusterAreaConfig::default();
    let mut rng = DeterministicRng::from_seed_u64(2027);

    // 1. Single continuous disk + C-shape + L-shape -> coherence == 1.0.
    let (dx, dy) = sample_disk_points(&mut rng, -8.0, 0.0, 2.0, 600);
    let mut xs = dx;
    let mut ys = dy;
    let mut labs = vec![0u32; 600];

    // C-shape arc (cluster 1) around (0, 0)
    for i in 0..400usize {
        let theta = 0.25 * PI + 1.5 * PI * (i as f64) / 399.0;
        for &dr in &[-0.25, 0.0, 0.25] {
            xs.push((2.5 + dr) * theta.cos());
            ys.push((2.5 + dr) * theta.sin());
            labs.push(1);
        }
    }
    // L-shape (cluster 2) near (8, 0)
    for i in 0..100usize {
        let t = (i as f64) / 99.0 * 3.0;
        xs.push(7.0 + t);
        ys.push(-1.5);
        labs.push(2);
        xs.push(7.0);
        ys.push(-1.5 + t);
        labs.push(2);
    }
    let l_shapes = Layout2d::from_xy(xs, ys).unwrap();
    let det_shapes = cluster_area_faithfulness_detail(&l_shapes, &labs, &cfg).unwrap();
    assert!((det_shapes.coherence[0] - 1.0).abs() < 1e-12);
    assert!((det_shapes.coherence[1] - 1.0).abs() < 1e-12);
    assert!((det_shapes.coherence[2] - 1.0).abs() < 1e-12);

    // 2. Simulated g_eff = g_max saturation where small connected cluster has max_wk < tau -> C_coh == 1.0.
    let (sx0, sy0) = sample_disk_points(&mut rng, -5.0, 0.0, 1.0, 5000);
    let mut sxs = sx0;
    let mut sys = sy0;
    let mut slabs = vec![0u32; 5000];
    for i in 0..5usize {
        sxs.push(5.0 + (i as f64) * 0.02);
        sys.push(0.0);
        slabs.push(1);
    }
    let l_sat = Layout2d::from_xy(sxs, sys).unwrap();
    let det_sat = cluster_area_faithfulness_detail(
        &l_sat,
        &slabs,
        &ClusterAreaConfig::default()
            .with_g_max(12)
            .with_tau_frac(5.0)
            .with_boundary_r95(f64::INFINITY),
    )
    .unwrap();
    assert!((det_sat.report.in_frac - 1.0).abs() < 1e-12);
    assert!((det_sat.coherence[1] - 1.0).abs() < 1e-12);

    // 3. Symmetric 2 islands (normal-density uniform disks) separated by a 4*sigma edge-to-edge gap -> coherence == 0.50.
    let (ix, iy) = sample_disk_points(&mut rng, 0.0, 0.0, 2.0, 38);
    let mut gx: Vec<f64> = ix.iter().map(|&x| x - 2.85).collect();
    let mut gy: Vec<f64> = iy.clone();
    let mut glabs = vec![0u32; 38];
    gx.extend(ix.iter().map(|&x| x + 2.85));
    gy.extend(iy);
    glabs.extend(std::iter::repeat_n(0u32, 38));
    gx.extend(std::iter::repeat_n(-10.0, 462));
    gy.extend(std::iter::repeat_n(0.0, 462));
    gx.extend(std::iter::repeat_n(10.0, 462));
    gy.extend(std::iter::repeat_n(0.0, 462));
    glabs.extend(std::iter::repeat_n(1u32, 924));
    let l_gap = Layout2d::from_xy(gx, gy).unwrap();
    let det_gap = cluster_area_faithfulness_detail(
        &l_gap,
        &glabs,
        &ClusterAreaConfig::default()
            .with_g_max(20)
            .with_boundary_r95(f64::INFINITY),
    )
    .unwrap();
    assert!(
        (det_gap.coherence[0] - 0.50).abs() < 0.01,
        "expected 4-sigma gap coherence ~ 0.50, got {}",
        det_gap.coherence[0]
    );

    // 4. Symmetric 2 islands separated by a dense foreign-cluster barrier -> coherence == 0.50.
    let mut bx = vec![-1.0; 20];
    let mut by = vec![0.0; 20];
    let mut blabs = vec![0u32; 20];
    bx.extend(std::iter::repeat_n(1.0, 20));
    by.extend(std::iter::repeat_n(0.0, 20));
    blabs.extend(std::iter::repeat_n(0u32, 20));
    // Dense foreign cluster 1 barrier right in the middle at (0, 0) plus anchors at ±10.
    bx.extend(std::iter::repeat_n(0.0, 200));
    by.extend(std::iter::repeat_n(0.0, 200));
    bx.extend(std::iter::repeat_n(-10.0, 180));
    by.extend(std::iter::repeat_n(0.0, 180));
    bx.extend(std::iter::repeat_n(10.0, 180));
    by.extend(std::iter::repeat_n(0.0, 180));
    blabs.extend(std::iter::repeat_n(1u32, 560));
    let l_bar = Layout2d::from_xy(bx, by).unwrap();
    let det_bar = cluster_area_faithfulness_detail(
        &l_bar,
        &blabs,
        &ClusterAreaConfig::default()
            .with_g_max(20)
            .with_boundary_r95(f64::INFINITY),
    )
    .unwrap();
    assert!(
        (det_bar.coherence[0] - 0.50).abs() < 1e-12,
        "expected barrier coherence == 0.50, got {}",
        det_bar.coherence[0]
    );
}

#[test]
fn test_f4_stevens_gamma3_ideal_90_10_and_gamma1_zero_ulp() {
    // 90:10 clusters with area ratio p_0 : p_1 = 0.9^3 : 0.1^3 = 729 : 1 -> radius ratio 27 : 1.
    let mut rng = DeterministicRng::from_seed_u64(4040);
    let r1 = 0.35_f64;
    let r0 = r1 * 27.0; // 9.45
    let (x0, y0) = sample_disk_points(&mut rng, -6.0, 0.0, r0, 9000);
    let (x1, y1) = sample_disk_points(&mut rng, 8.0, 0.0, r1, 1000);
    let mut xs = x0;
    xs.extend(x1);
    let mut ys = y0;
    ys.extend(y1);
    let mut labels = vec![0u32; 9000];
    labels.extend(std::iter::repeat_n(1u32, 1000));
    let layout = Layout2d::from_xy(xs, ys).unwrap();

    let det_g3 = cluster_area_faithfulness_detail(
        &layout,
        &labels,
        &ClusterAreaConfig::default().with_stevens_gamma(3.0),
    )
    .unwrap();
    assert!(
        det_g3.res_density[1] > 0.95,
        "expected F4 e_1 > 0.95, got {}",
        det_g3.res_density[1]
    );
    assert!(
        det_g3.res_deconv[1] > 0.95,
        "expected F4 d_1 > 0.95, got {}",
        det_g3.res_deconv[1]
    );

    // At gamma = 1.0, uniform node_weights = 1.0 matches unweighted Step 3/4 cleanly.
    let det_unw =
        cluster_area_faithfulness_detail(&layout, &labels, &ClusterAreaConfig::default()).unwrap();
    let nw_ones = vec![1.0_f64; 10_000];
    let det_w_ones = cluster_area_faithfulness_detail_with_weights(
        &layout,
        &labels,
        &ClusterAreaConfig::default(),
        Some(&nw_ones),
        None,
    )
    .unwrap();
    assert_report_close(&det_unw.report, &det_w_ones.report, 1e-12, "gamma1_ones");
}

#[test]
fn test_g6_target_weights_and_validation() {
    let (layout, labels) = make_ideal_10_disks(100, 1.0, 10.0);
    let n = layout.len();

    // 1. node_weights vs summed cluster_weights exact parity.
    let nw: Vec<f64> = labels.iter().map(|&lbl| ((lbl + 1) as f64) * 0.5).collect();
    let cw: Vec<f64> = (0..10u32).map(|k| 100.0 * ((k + 1) as f64) * 0.5).collect();
    let cfg = ClusterAreaConfig::default();
    let det_nw =
        cluster_area_faithfulness_detail_with_weights(&layout, &labels, &cfg, Some(&nw), None)
            .unwrap();
    let det_cw =
        cluster_area_faithfulness_detail_with_weights(&layout, &labels, &cfg, None, Some(&cw))
            .unwrap();
    assert_report_close(&det_nw.report, &det_cw.report, 1e-12, "nw_vs_cw");
    assert_eq!(det_nw.p.len(), 10);
    for k in 0..10 {
        assert!((det_nw.p[k] - det_cw.p[k]).abs() < 1e-12);
    }

    // 2. Non-contiguous arbitrary u32 labels follow sorted cluster_ids order for cluster_weights.
    let remapped: Vec<u32> = labels
        .iter()
        .map(|&k| match k {
            0 => u32::MAX,
            1 => 42,
            _ => k + 100,
        })
        .collect();
    let mut cw_sorted = vec![cw[1]];
    cw_sorted.extend_from_slice(&cw[2..10]);
    cw_sorted.push(cw[0]);
    let det_remap = cluster_area_faithfulness_detail_with_weights(
        &layout,
        &remapped,
        &cfg,
        None,
        Some(&cw_sorted),
    )
    .unwrap();
    assert_report_close(&det_cw.report, &det_remap.report, 1e-12, "cw_sorted_ids");
    assert_eq!(det_remap.cluster_ids[0], 42);
    assert_eq!(det_remap.cluster_ids[9], u32::MAX);

    // 3. Stevens gamma != 1.0 applied after weight normalization: p_k \propto (p_k^(w))^gamma.
    let cfg_g2 = cfg.clone().with_stevens_gamma(2.0);
    let det_g2 =
        cluster_area_faithfulness_detail_with_weights(&layout, &labels, &cfg_g2, None, Some(&cw))
            .unwrap();
    let sum_sq: f64 = cw.iter().map(|&w| w * w).sum();
    for (k, &w) in cw.iter().enumerate() {
        let expected_p = (w * w) / sum_sq;
        assert!((det_g2.p[k] - expected_p).abs() < 1e-12);
    }

    // 4. Huge 1e308 weights do not overflow to inf/NaN.
    let cw_huge: Vec<f64> = (1..=10).map(|k| (k as f64) * 1e307).collect();
    let det_huge =
        cluster_area_faithfulness_detail_with_weights(&layout, &labels, &cfg, None, Some(&cw_huge))
            .unwrap();
    assert_detail_unit_interval(&det_huge, "cw_1e308");
    let nw_huge = vec![1e308_f64; n];
    let det_nw_huge =
        cluster_area_faithfulness_detail_with_weights(&layout, &labels, &cfg, Some(&nw_huge), None)
            .unwrap();
    assert_detail_unit_interval(&det_nw_huge, "nw_1e308");

    // 5. All validation error paths (including n=0, n=1, k=1 before early returns).
    assert!(matches!(
        cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, Some(&nw), Some(&cw)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, Some(&nw[..n - 1]), None),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, None, Some(&cw[..9])),
        Err(CoreError::DimensionMismatch { .. })
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad_nw = nw.clone();
        bad_nw[0] = bad;
        assert!(matches!(
            cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, Some(&bad_nw), None),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let mut bad_cw = cw.clone();
        bad_cw[0] = bad;
        assert!(matches!(
            cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, None, Some(&bad_cw)),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
    for bad in [0.0, -1.0] {
        let mut bad_nw = nw.clone();
        bad_nw[0] = bad;
        assert!(matches!(
            cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, Some(&bad_nw), None),
            Err(CoreError::InvalidInput(_))
        ));
        let mut bad_cw = cw.clone();
        bad_cw[0] = bad;
        assert!(matches!(
            cluster_area_faithfulness_with_weights(&layout, &labels, &cfg, None, Some(&bad_cw)),
            Err(CoreError::InvalidInput(_))
        ));
    }

    // Degenerate N=0, N=1, K=1 still validate weights strictly before early return.
    let l0 = Layout2d::<f64>::from_xy(vec![], vec![]).unwrap();
    assert!(cluster_area_faithfulness_with_weights(&l0, &[], &cfg, Some(&[]), None).is_ok());
    assert!(cluster_area_faithfulness_with_weights(&l0, &[], &cfg, Some(&[1.0]), None).is_err());
    assert!(cluster_area_faithfulness_with_weights(&l0, &[], &cfg, None, Some(&[1.0])).is_err());
    let l1 = Layout2d::from_xy(vec![0.0], vec![0.0]).unwrap();
    assert!(cluster_area_faithfulness_with_weights(&l1, &[0], &cfg, Some(&[-1.0]), None).is_err());
    assert!(cluster_area_faithfulness_with_weights(&l1, &[0], &cfg, None, Some(&[0.0])).is_err());
    let lk1 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(
        cluster_area_faithfulness_with_weights(&lk1, &[5, 5], &cfg, None, Some(&[1.0, 2.0]))
            .is_err()
    );
    assert!(
        cluster_area_faithfulness_with_weights(&lk1, &[5, 5], &cfg, Some(&[1.0, f64::NAN]), None)
            .is_err()
    );
}
