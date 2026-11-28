use std::f64::consts::PI;

use egraph_core::{DeterministicRng, Layout2d};

use super::super::*;
use super::{make_ideal_10_disks, sample_disk_points};

#[test]
fn test_gcaf_summary_golden_values() {
    let cfg = ClusterAreaConfig::default();
    let mut rng = DeterministicRng::from_seed_u64(42);

    // 1. 20Dots monotonicity (r = 1e-4 .. 4.0)
    let mut prev_caf = -1.0_f64;
    for &r in &[1e-4, 0.05, 0.5, 2.0, 4.0] {
        let mut xs = Vec::with_capacity(10_000);
        let mut ys = Vec::with_capacity(10_000);
        let mut labs = Vec::with_capacity(10_000);
        for k in 0..20usize {
            let cx = (k % 5) as f64 * 10.0;
            let cy = (k / 5) as f64 * 10.0;
            let (dx, dy) = sample_disk_points(&mut rng, cx, cy, r, 500);
            xs.extend(dx);
            ys.extend(dy);
            labs.extend(std::iter::repeat_n(k as u32, 500));
        }
        let l = Layout2d::from_xy(xs, ys).unwrap();
        let rep = cluster_area_faithfulness(&l, &labs, &cfg).unwrap();
        assert!(
            rep.caf >= prev_caf,
            "20Dots non-monotonic at r={r}: {} < {}",
            rep.caf,
            prev_caf
        );
        prev_caf = rep.caf;
    }
    assert!(prev_caf > 0.95);

    // 2. B1a_WhiteNoise_K10 -> caf < 0.01
    let (wx, wy) = sample_disk_points(&mut rng, 0.0, 0.0, 10.0, 10_000);
    let w_labs: Vec<u32> = (0..10_000).map(|i| (i % 10) as u32).collect();
    let l_white = Layout2d::from_xy(wx, wy).unwrap();
    let rep_white = cluster_area_faithfulness(&l_white, &w_labs, &cfg).unwrap();
    assert!(rep_white.caf < 0.01, "white noise caf={}", rep_white.caf);

    // 3. D2_All10_95CrushHalo -> tight core + tiny outer halo -> caf < 0.03
    let mut dx_d2 = Vec::with_capacity(20_000);
    let mut dy_d2 = Vec::with_capacity(20_000);
    let mut labs_d2 = Vec::with_capacity(20_000);
    for k in 0..10usize {
        let ang = 2.0 * PI * (k as f64) / 10.0;
        let cx = 5.0 * ang.cos();
        let cy = 5.0 * ang.sin();
        let (cx0, cy0) = sample_disk_points(&mut rng, cx, cy, 1e-4, 1900);
        let (cx1, cy1) = sample_disk_points(&mut rng, cx, cy, 0.5, 100);
        dx_d2.extend(cx0);
        dx_d2.extend(cx1);
        dy_d2.extend(cy0);
        dy_d2.extend(cy1);
        labs_d2.extend(std::iter::repeat_n(k as u32, 2000));
    }
    let l_d2 = Layout2d::from_xy(dx_d2, dy_d2).unwrap();
    let rep_d2 = cluster_area_faithfulness(&l_d2, &labs_d2, &cfg).unwrap();
    assert!(rep_d2.caf < 0.03, "D2 caf={}", rep_d2.caf);

    // 4. G3a_Crush50_SurvEq vs G3b_Crush50_Surv5x -> G3a > G3b + 0.10
    let mut gx_a = Vec::with_capacity(10_000);
    let mut gy_a = Vec::with_capacity(10_000);
    let mut gx_b = Vec::with_capacity(10_000);
    let mut gy_b = Vec::with_capacity(10_000);
    let mut g_labs_3 = Vec::with_capacity(10_000);
    let g3_counts = [1667usize, 1667, 1666, 1667, 1667, 1666];
    let g3b_radii = [1e-4, 1e-4, 1e-4, 1.8, 1.8, 4.2];
    for k in 0..6usize {
        let ang = 2.0 * PI * (k as f64) / 6.0;
        let cx = 10.0 * ang.cos();
        let cy = 10.0 * ang.sin();
        let r_a = if k < 3 { 1e-4 } else { 2.5 };
        let (ax, ay) = sample_disk_points(&mut rng, cx, cy, r_a, g3_counts[k]);
        gx_a.extend(ax);
        gy_a.extend(ay);
        let (bx, by) = sample_disk_points(&mut rng, cx, cy, g3b_radii[k], g3_counts[k]);
        gx_b.extend(bx);
        gy_b.extend(by);
        g_labs_3.extend(std::iter::repeat_n(k as u32, g3_counts[k]));
    }
    let rep_g3a =
        cluster_area_faithfulness(&Layout2d::from_xy(gx_a, gy_a).unwrap(), &g_labs_3, &cfg)
            .unwrap();
    let rep_g3b =
        cluster_area_faithfulness(&Layout2d::from_xy(gx_b, gy_b).unwrap(), &g_labs_3, &cfg)
            .unwrap();
    assert!(
        rep_g3a.caf > rep_g3b.caf + 0.10,
        "G3a={} vs G3b={}",
        rep_g3a.caf,
        rep_g3b.caf
    );

    // 5. G4_ImpureSmudge90 (mixed 90% background + 10% cluster cores) -> caf < 0.35
    let (gx_base, gy_base) = sample_disk_points(&mut rng, 0.0, 0.0, 10.0, 9000);
    let mut gx = gx_base;
    let mut gy = gy_base;
    let mut g_labs: Vec<u32> = (0..9000).map(|i| (i % 10) as u32).collect();
    for k in 0..10usize {
        let ang = 2.0 * PI * (k as f64) / 10.0;
        let (cx, cy) = sample_disk_points(&mut rng, 5.0 * ang.cos(), 5.0 * ang.sin(), 0.5, 100);
        gx.extend(cx);
        gy.extend(cy);
        g_labs.extend(std::iter::repeat_n(k as u32, 100));
    }
    let l_g4 = Layout2d::from_xy(gx, gy).unwrap();
    let rep_g4 = cluster_area_faithfulness(&l_g4, &g_labs, &cfg).unwrap();
    assert!(rep_g4.caf < 0.35, "G4 caf={}", rep_g4.caf);

    // 6. f32 vs f64 parity (< 1e-5)
    let (l_ideal, labs_ideal) = make_ideal_10_disks(200, 1.0, 10.0);
    let l_ideal_32 = Layout2d::from_xy(
        l_ideal.x().iter().map(|&v| v as f32).collect(),
        l_ideal.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let rep64 = cluster_area_faithfulness(&l_ideal, &labs_ideal, &cfg).unwrap();
    let rep32 = cluster_area_faithfulness(&l_ideal_32, &labs_ideal, &cfg).unwrap();
    assert!(
        (rep64.caf - rep32.caf).abs() < 1e-5,
        "f32 vs f64 diff={}",
        (rep64.caf - rep32.caf).abs()
    );
}

#[test]
fn test_boundary_r95_contract() {
    let mut rng = DeterministicRng::from_seed_u64(991);
    let (mut xs, mut ys) = sample_disk_points(&mut rng, 0.0, 0.0, 99.0_f64.sqrt(), 9900);
    let (x1, y1) = sample_disk_points(&mut rng, 25.0, 0.0, 1.0, 100);
    xs.extend(x1);
    ys.extend(y1);
    let mut labels = vec![0u32; 9900];
    labels.extend(std::iter::repeat_n(1u32, 100));
    let layout = Layout2d::from_xy(xs, ys).unwrap();

    let r_b2 = cluster_area_faithfulness(&layout, &labels, &ClusterAreaConfig::default()).unwrap();
    assert!((r_b2.in_frac - 0.99).abs() < 1e-12);
    assert!((r_b2.caf - 0.99).abs() < 1e-2, "caf={}", r_b2.caf);
    assert!((r_b2.spatial_coherence - 1.0).abs() < 1e-12);
    assert_eq!(r_b2.effective_boundary_r95, 2.0);
    assert_eq!(r_b2.g_eff, 87);

    let r_inf = cluster_area_faithfulness(
        &layout,
        &labels,
        &ClusterAreaConfig::default().with_boundary_r95(f64::INFINITY),
    )
    .unwrap();
    let r_1e9 = cluster_area_faithfulness(
        &layout,
        &labels,
        &ClusterAreaConfig::default().with_boundary_r95(1e9),
    )
    .unwrap();
    assert!((r_inf.in_frac - 1.0).abs() < 1e-12);
    assert!(r_inf.caf > 0.98 && r_inf.caf.is_finite());
    assert_eq!(r_inf.g_eff, 87);
    assert!((r_inf.effective_boundary_r95 - 1020.0 / 87.0).abs() < 1e-12);

    assert!((r_1e9.in_frac - 1.0).abs() < 1e-12);
    assert!(r_1e9.caf > 0.90 && r_1e9.caf.is_finite());
    assert_eq!(r_1e9.g_eff, 12);
    assert_eq!(r_1e9.effective_boundary_r95, 85.0);
}

#[test]
fn test_f1_gmax_2048_extreme_finite_and_inf_outlier_protection() {
    // 1. g_max = 2048 with large raw g_eff: clamp g_eff instead of silently shrinking effective_b.
    let mut rng = DeterministicRng::from_seed_u64(1001);
    let n = 50_000usize;
    let (x0, y0) = sample_disk_points(&mut rng, -1.0, 0.0, 0.4, n / 2);
    let (x1, y1) = sample_disk_points(&mut rng, 1.0, 0.0, 0.4, n - n / 2);
    let mut xs = x0;
    xs.extend(x1);
    let mut ys = y0;
    ys.extend(y1);
    let mut labels = vec![0u32; n / 2];
    labels.extend(std::iter::repeat_n(1u32, n - n / 2));
    let layout_big = Layout2d::from_xy(xs, ys).unwrap();
    let cfg_2048 = ClusterAreaConfig::default()
        .with_g_max(2048)
        .with_sigma_cells(3.0)
        .with_boundary_r95(2.0);
    let rep_2048 = cluster_area_faithfulness(&layout_big, &labels, &cfg_2048).unwrap();
    // r_stencil = ceil(3.5 * 3.0) = 11, avail_h = 1012, floor(1012 / 2.0) = 506 < raw_g (688).
    assert_eq!(rep_2048.g_eff, 506);
    assert_eq!(rep_2048.effective_boundary_r95, 2.0);
    assert!((rep_2048.in_frac - 1.0).abs() < 1e-12);
    assert!(rep_2048.caf > 0.7);

    // 2. Extreme finite boundary_r95 > 85.0 at default sigma_cells = 0.85:
    // keep g_eff = 12 and clamp effective_boundary_r95 = 85.0.
    let (l_std, labs_std) = make_ideal_10_disks(200, 1.0, 10.0);
    let rep_b200 = cluster_area_faithfulness(
        &l_std,
        &labs_std,
        &ClusterAreaConfig::default().with_boundary_r95(200.0),
    )
    .unwrap();
    assert_eq!(rep_b200.g_eff, 12);
    assert_eq!(rep_b200.effective_boundary_r95, 85.0);
    assert!((rep_b200.in_frac - 1.0).abs() < 1e-12);

    // 3. boundary_r95 = +inf dense-grid outlier protection: far outlier does not crush g_eff to 12.
    let (mut ox, mut oy) = sample_disk_points(&mut rng, 0.0, 0.0, 1.0, 999);
    let mut olabs: Vec<u32> = (0..999).map(|i| if i < 500 { 0 } else { 1 }).collect();
    ox.push(500.0);
    oy.push(0.0);
    olabs.push(1);
    let l_out = Layout2d::from_xy(ox, oy).unwrap();
    let rep_out = cluster_area_faithfulness(
        &l_out,
        &olabs,
        &ClusterAreaConfig::default().with_boundary_r95(f64::INFINITY),
    )
    .unwrap();
    assert!(
        rep_out.g_eff > 12,
        "expected g_eff > 12, got {}",
        rep_out.g_eff
    );
    assert!((rep_out.effective_boundary_r95 - 1020.0 / (rep_out.g_eff as f64)).abs() < 1e-12);
    assert!((rep_out.in_frac - 0.999).abs() < 1e-12);
}

#[test]
fn test_large_k_10k_flat_scaling() {
    let cfg = ClusterAreaConfig::default();
    let n = 50_000usize;
    let mut rng = DeterministicRng::from_seed_u64(101);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut labs_10 = Vec::with_capacity(n);
    let mut labs_10k = Vec::with_capacity(n);
    for i in 0..n {
        let gx = (i % 100) as f64 * 2.0;
        let gy = (i / 100) as f64 * 2.0;
        xs.push(gx + rng.next_f64() * 0.4);
        ys.push(gy + rng.next_f64() * 0.4);
        labs_10.push((i % 10) as u32);
        labs_10k.push((i / 5) as u32);
    }
    let layout = Layout2d::from_xy(xs, ys).unwrap();

    let t1 = std::time::Instant::now();
    let rep10 = cluster_area_faithfulness(&layout, &labs_10, &cfg).unwrap();
    let d10 = t1.elapsed();

    let t2 = std::time::Instant::now();
    let rep10k = cluster_area_faithfulness(&layout, &labs_10k, &cfg).unwrap();
    let d10k = t2.elapsed();

    eprintln!(
        "large_k_10k_flat_scaling: K=10 {:?}, K=10000 {:?}",
        d10, d10k
    );
    assert_eq!(rep10.num_clusters, 10);
    assert_eq!(rep10k.num_clusters, 10_000);
    assert!(rep10k.caf.is_finite());
    // Must not scale as O(K * S^2): K=10k should remain well below 5x K=10 runtime.
    assert!(
        d10k.as_secs_f64() < (d10.as_secs_f64() * 5.0).max(0.5),
        "d10={:?}, d10k={:?}",
        d10,
        d10k
    );
}

#[test]
fn test_singleton_and_disjoint_stencil() {
    // (a) Singleton cluster with boundary_r95 = +inf
    let mut rng = DeterministicRng::from_seed_u64(55);
    let (mut xs, mut ys) = sample_disk_points(&mut rng, 0.0, 0.0, 1.0, 99);
    let mut labels = vec![0u32; 99];
    xs.push(5.0);
    ys.push(0.0);
    labels.push(1);
    let layout_s = Layout2d::from_xy(xs, ys).unwrap();
    let cfg_inf = ClusterAreaConfig::default().with_boundary_r95(f64::INFINITY);
    let det_s = cluster_area_faithfulness_detail(&layout_s, &labels, &cfg_inf).unwrap();
    assert_eq!(det_s.res_deconv[1], 1.0);
    assert_eq!(det_s.purity[1], 0.0);
    assert_eq!(det_s.caf_per_cluster[1], 0.0);
    assert!((det_s.coherence[1] - 1.0).abs() < 1e-12);

    // (b) DisjointStencil: 4 isolated points far apart -> B_k == 0 -> purity = 0 -> caf = 0
    let l_disj = Layout2d::from_xy(vec![0.0, 5.0, 0.0, 5.0], vec![0.0, 0.0, 5.0, 5.0]).unwrap();
    let det_d =
        cluster_area_faithfulness_detail(&l_disj, &[0, 0, 1, 1], &ClusterAreaConfig::default())
            .unwrap();
    assert_eq!(det_d.purity, vec![0.0, 0.0]);
    assert_eq!(det_d.report.caf, 0.0);
}

#[test]
fn test_stevens_gamma_behavior() {
    let mut rng = DeterministicRng::from_seed_u64(314);
    let (x0, y0) = sample_disk_points(&mut rng, -5.0, 0.0, 1.0, 9000);
    let (x1, y1) = sample_disk_points(&mut rng, 5.0, 0.0, 1.0, 1000);
    let mut xs = x0;
    xs.extend(x1);
    let mut ys = y0;
    ys.extend(y1);
    let mut labels = vec![0u32; 9000];
    labels.extend(std::iter::repeat_n(1u32, 1000));
    let layout = Layout2d::from_xy(xs, ys).unwrap();

    let d1 =
        cluster_area_faithfulness_detail(&layout, &labels, &ClusterAreaConfig::default()).unwrap();
    let d07 = cluster_area_faithfulness_detail(
        &layout,
        &labels,
        &ClusterAreaConfig::default().with_stevens_gamma(0.7),
    )
    .unwrap();

    assert!(d07.report.prop_vis > d1.report.prop_vis);
    assert!(d07.report.caf > d1.report.caf);
    // Step 3 purity & coherence and q-weighted scores are independent of gamma, while
    // F4 scales E_k/D_k target densities and denominators with p[k].
    assert_eq!(d07.report.purity_adj, d1.report.purity_adj);
    assert_eq!(d07.report.spatial_coherence, d1.report.spatial_coherence);
    assert_eq!(d07.q, d1.q);
    assert_eq!(d07.purity, d1.purity);
    assert!(d07.res_deconv[0] > d1.res_deconv[0]);
    assert!(d07.res_deconv[1] <= d1.res_deconv[1]);
    assert!(d07.res_density[0] >= d1.res_density[0]);
    assert_eq!(d07.coherence, d1.coherence);
}
