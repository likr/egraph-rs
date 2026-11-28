use std::f64::consts::PI;

use egraph_core::{CoreError, DeterministicRng, Layout2d};

use super::super::*;
use super::{make_ideal_10_disks, sample_disk_points};

#[test]
fn test_validation_and_degenerate() {
    let l = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let cfg = ClusterAreaConfig::default();

    // Dimension mismatch
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0], &cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Non-finite layout
    let l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        cluster_area_faithfulness(&l_nan, &[0, 1], &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let l_inf = Layout2d::from_xy(vec![0.0, f64::INFINITY], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        cluster_area_faithfulness(&l_inf, &[0, 1], &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Config validation
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_g_max(11)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_sigma_cells(0.0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_sigma_cells(-1.0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_sigma_cells(f64::NAN)),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_kappa(0.0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_tau_frac(-0.1)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_boundary_r95(0.0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_boundary_r95(-1.0)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_boundary_r95(f64::NAN)),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        cluster_area_faithfulness(&l, &[0, 1], &cfg.clone().with_stevens_gamma(0.0)),
        Err(CoreError::InvalidInput(_))
    ));

    // N = 0
    let l0 = Layout2d::<f64>::from_xy(vec![], vec![]).unwrap();
    let d0 = cluster_area_faithfulness_detail(&l0, &[], &cfg).unwrap();
    assert_eq!(d0.report, ClusterAreaReport::default());
    assert!(d0.cluster_ids.is_empty());
    assert!(d0.p.is_empty());

    // N = 1
    let l1 = Layout2d::from_xy(vec![1.0], vec![2.0]).unwrap();
    let d1 = cluster_area_faithfulness_detail(&l1, &[7], &cfg).unwrap();
    assert_eq!(d1.report.num_nodes, 1);
    assert_eq!(d1.report.num_clusters, 1);
    assert_eq!(d1.report.caf, 0.0);
    assert_eq!(d1.cluster_ids, vec![7]);
    assert_eq!(d1.counts, vec![1]);
    assert_eq!(d1.p, vec![1.0]);
    assert_eq!(d1.purity, vec![0.0]);

    // K = 1
    let l_k1 = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]).unwrap();
    let d_k1 = cluster_area_faithfulness_detail(&l_k1, &[5, 5, 5], &cfg).unwrap();
    assert_eq!(d_k1.report.num_nodes, 3);
    assert_eq!(d_k1.report.num_clusters, 1);
    assert_eq!(d_k1.report.caf, 0.0);
    assert_eq!(d_k1.cluster_ids, vec![5]);

    // All coincident
    let l_coin = Layout2d::from_xy(vec![3.0; 10], vec![-2.0; 10]).unwrap();
    let labs_coin = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let d_coin = cluster_area_faithfulness_detail(&l_coin, &labs_coin, &cfg).unwrap();
    assert_eq!(d_coin.report.caf, 0.0);
    assert_eq!(d_coin.report.num_clusters, 2);
    assert_eq!(d_coin.report.num_nodes, 10);

    // 96% coincident -> r_pre == 0.0
    let mut xs_96 = vec![0.0; 100];
    let ys_96 = vec![0.0; 100];
    let mut labs_96 = vec![0u32; 100];
    for i in 96..100 {
        xs_96[i] = 10.0;
        labs_96[i] = 1;
    }
    let l_96 = Layout2d::from_xy(xs_96, ys_96).unwrap();
    let d_96 = cluster_area_faithfulness_detail(&l_96, &labs_96, &cfg).unwrap();
    assert_eq!(d_96.report.caf, 0.0);
    assert_eq!(d_96.report.in_frac, 0.0);
    assert_eq!(d_96.report.num_clusters, 2);
    assert_eq!(d_96.report.num_nodes, 100);
}

#[test]
fn test_label_remapping_non_contiguous() {
    let cfg = ClusterAreaConfig::default();
    let l = Layout2d::from_xy(
        vec![0.0, 10.0, 20.0, 10.5, 0.5, 20.5],
        vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
    )
    .unwrap();
    let sparse = [u32::MAX, 42, 7, 42, u32::MAX, 7];
    let compact = [2, 1, 0, 1, 2, 0];

    let d_sparse = cluster_area_faithfulness_detail(&l, &sparse, &cfg).unwrap();
    let d_compact = cluster_area_faithfulness_detail(&l, &compact, &cfg).unwrap();
    assert_eq!(d_sparse.report, d_compact.report);
    assert_eq!(d_sparse.cluster_ids, vec![7, 42, u32::MAX]);
    assert_eq!(d_sparse.counts, vec![2, 2, 2]);
    assert_eq!(d_sparse.p, d_compact.p);
    assert_eq!(d_sparse.q, d_compact.q);
    assert_eq!(d_sparse.purity, d_compact.purity);
    assert_eq!(d_sparse.coherence, d_compact.coherence);

    // Also test dense non-contiguous path (> 1024 nodes for radix path too).
    let n = 1500usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut labs_dense = Vec::with_capacity(n);
    let mut labs_radix = Vec::with_capacity(n);
    let mut labs_zero = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % 3;
        xs.push((c as f64) * 10.0 + ((i % 7) as f64) * 0.1);
        ys.push(((i / 7) % 7) as f64 * 0.1);
        labs_zero.push(c as u32);
        labs_dense.push([10u32, 500, 1000][c]);
        labs_radix.push([5u32, 10_000_000, u32::MAX][c]);
    }
    let l_big = Layout2d::from_xy(xs, ys).unwrap();
    let r_zero = cluster_area_faithfulness_detail(&l_big, &labs_zero, &cfg).unwrap();
    let r_dense = cluster_area_faithfulness_detail(&l_big, &labs_dense, &cfg).unwrap();
    let r_radix = cluster_area_faithfulness_detail(&l_big, &labs_radix, &cfg).unwrap();
    assert_eq!(r_zero.report, r_dense.report);
    assert_eq!(r_zero.report, r_radix.report);
    assert_eq!(r_dense.cluster_ids, vec![10, 500, 1000]);
    assert_eq!(r_radix.cluster_ids, vec![5, 10_000_000, u32::MAX]);
}

#[test]
fn test_remap_community_labels_dense_limit_boundary() {
    // Exact dense_limit = 4_000_000 vs 4_000_001 boundary at N=10 and N=1500.
    let cfg = ClusterAreaConfig::default();

    // N = 10
    let xs10: Vec<f64> = (0..10)
        .map(|i| (i % 3) as f64 * 5.0 + (i as f64) * 0.05)
        .collect();
    let ys10: Vec<f64> = (0..10).map(|i| (i / 3) as f64 * 0.1).collect();
    let l10 = Layout2d::from_xy(xs10, ys10).unwrap();
    let base10_labs: Vec<u32> = (0..10).map(|i| (i % 3) as u32).collect();
    let base10 = cluster_area_faithfulness_detail(&l10, &base10_labs, &cfg).unwrap();

    for &(lbls3, ctx) in &[
        ([0u32, 1, 4_000_000], "n10_dense_4M"),
        ([0u32, 1, 4_000_001], "n10_sparse_4M_plus_1"),
        ([7u32, 4_000_000, 4_000_001], "n10_mixed_4M_4M1"),
    ] {
        let labs: Vec<u32> = (0..10).map(|i| lbls3[i % 3]).collect();
        let det = cluster_area_faithfulness_detail(&l10, &labs, &cfg).unwrap();
        assert_eq!(det.report, base10.report, "{ctx}");
        assert_eq!(det.cluster_ids, lbls3, "{ctx}");
        assert_eq!(det.p, base10.p, "{ctx}");
        assert_eq!(det.q, base10.q, "{ctx}");
        assert_eq!(det.purity, base10.purity, "{ctx}");
        assert_eq!(det.coherence, base10.coherence, "{ctx}");
    }

    // N = 1500 (> 1024 triggers radix_sort_u64 in sparse path)
    let n = 1500usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut base_labs = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % 3;
        xs.push((c as f64) * 10.0 + ((i % 7) as f64) * 0.1);
        ys.push(((i / 7) % 7) as f64 * 0.1);
        base_labs.push(c as u32);
    }
    let l1500 = Layout2d::from_xy(xs, ys).unwrap();
    let base1500 = cluster_area_faithfulness_detail(&l1500, &base_labs, &cfg).unwrap();

    for &(lbls3, ctx) in &[
        ([0u32, 1, 4_000_000], "n1500_dense_4M"),
        ([0u32, 1, 4_000_001], "n1500_radix_4M_plus_1"),
        ([7u32, 4_000_000, 4_000_001], "n1500_mixed_radix"),
    ] {
        let labs: Vec<u32> = (0..n).map(|i| lbls3[i % 3]).collect();
        let det = cluster_area_faithfulness_detail(&l1500, &labs, &cfg).unwrap();
        assert_eq!(det.report, base1500.report, "{ctx}");
        assert_eq!(det.cluster_ids, lbls3, "{ctx}");
        assert_eq!(det.p, base1500.p, "{ctx}");
        assert_eq!(det.q, base1500.q, "{ctx}");
        assert_eq!(det.purity, base1500.purity, "{ctx}");
        assert_eq!(det.coherence, base1500.coherence, "{ctx}");
    }
}

#[test]
fn test_scale_1e_pm_100_and_translation() {
    let cfg = ClusterAreaConfig::default();
    let (base_layout, labels) = make_ideal_10_disks(1000, 3.0, 10.0);
    let base = cluster_area_faithfulness(&base_layout, &labels, &cfg).unwrap();
    assert!(base.caf > 0.90, "base={:?}", base);

    let bx = base_layout.x();
    let by = base_layout.y();
    for &scale in &[1e-100, 1e-10, 1e-3, 1e3, 1e10, 1e100] {
        let scaled = Layout2d::from_xy(
            bx.iter().map(|&v| v * scale).collect(),
            by.iter().map(|&v| v * scale).collect(),
        )
        .unwrap();
        let rep = cluster_area_faithfulness(&scaled, &labels, &cfg).unwrap();
        assert!(
            (rep.caf - base.caf).abs() < 1e-12,
            "scale={scale}, diff={}",
            (rep.caf - base.caf).abs()
        );
    }

    // Small/moderate translation achieves < 1e-12 invariance.
    let t1e5 = Layout2d::from_xy(
        bx.iter().map(|&v| v + 1e5).collect(),
        by.iter().map(|&v| v - 1e5).collect(),
    )
    .unwrap();
    let rep_1e5 = cluster_area_faithfulness(&t1e5, &labels, &cfg).unwrap();
    assert!((rep_1e5.caf - base.caf).abs() < 1e-12);

    // +1e8 and +1e10 avoid catastrophic c3 cancellation (bounded only by f64 input quantization).
    for &shift in &[1e8, 1e10] {
        let shifted = Layout2d::from_xy(
            bx.iter().map(|&v| v + shift).collect(),
            by.iter().map(|&v| v + shift).collect(),
        )
        .unwrap();
        let rep = cluster_area_faithfulness(&shifted, &labels, &cfg).unwrap();
        assert!(
            (rep.caf - base.caf).abs() < 1e-5,
            "shift={shift}, base={}, got={}",
            base.caf,
            rep.caf
        );
    }
}

#[test]
fn test_isometry_and_permutation() {
    let cfg = ClusterAreaConfig::default();
    let mut rng = DeterministicRng::from_seed_u64(777);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut labels = Vec::new();
    // Skewed asymmetric layout (N ~ 20_000 so G_eff ~ 123): 50% cluster at (-20, 0),
    // smaller non-crushed lobes + outliers at ±60.
    let (x0, y0) = sample_disk_points(&mut rng, -20.0, 0.0, 5.0, 10_000);
    xs.extend(x0);
    ys.extend(y0);
    labels.extend(std::iter::repeat_n(0u32, 10_000));
    for k in 1..6usize {
        let cx = (k as f64) * 9.0 - 15.0;
        let cy = if k % 2 == 0 { 12.0 } else { -12.0 };
        let (xk, yk) = sample_disk_points(&mut rng, cx, cy, 3.5, 1998);
        xs.extend(xk);
        ys.extend(yk);
        labels.extend(std::iter::repeat_n(k as u32, 1998));
    }
    for i in 0..10usize {
        xs.push(if i % 2 == 0 { 60.0 } else { -60.0 });
        ys.push(if i < 5 { 60.0 } else { -60.0 });
        labels.push(((i % 5) + 1) as u32);
    }

    let base_layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let base = cluster_area_faithfulness(&base_layout, &labels, &cfg).unwrap();

    // Reflections: (x, -y), (-x, y), (-x, -y)
    for &(sx, sy) in &[(1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        let refl = Layout2d::from_xy(
            xs.iter().map(|&v| v * sx).collect(),
            ys.iter().map(|&v| v * sy).collect(),
        )
        .unwrap();
        let rep = cluster_area_faithfulness(&refl, &labels, &cfg).unwrap();
        assert!(
            (rep.caf - base.caf).abs() < 1e-12,
            "reflection ({sx},{sy}) diff={}",
            (rep.caf - base.caf).abs()
        );
    }

    // Permutation invariance
    let mut perm: Vec<usize> = (0..xs.len()).collect();
    rng.shuffle(&mut perm);
    let p_layout = Layout2d::from_xy(
        perm.iter().map(|&i| xs[i]).collect(),
        perm.iter().map(|&i| ys[i]).collect(),
    )
    .unwrap();
    let p_labels: Vec<u32> = perm.iter().map(|&i| labels[i]).collect();
    let p_rep = cluster_area_faithfulness(&p_layout, &p_labels, &cfg).unwrap();
    assert!((p_rep.caf - base.caf).abs() < 1e-12);

    // 12 rotations (30-degree steps)
    let mut min_caf = f64::INFINITY;
    let mut max_caf = f64::NEG_INFINITY;
    for step in 0..12 {
        let ang = (step as f64) * (PI / 6.0);
        let (sa, ca) = ang.sin_cos();
        let rot = Layout2d::from_xy(
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| x * ca - y * sa)
                .collect(),
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| x * sa + y * ca)
                .collect(),
        )
        .unwrap();
        let rep = cluster_area_faithfulness(&rot, &labels, &cfg).unwrap();
        min_caf = min_caf.min(rep.caf);
        max_caf = max_caf.max(rep.caf);
    }
    assert!(
        max_caf - min_caf < 3e-4,
        "rotation span={}",
        max_caf - min_caf
    );
}
