use egraph_core::{CoreError, Layout2d};

use super::super::config::H_MAX;
use super::super::grid::fill_1d_gaussian;
use super::super::mec::c3;
use super::super::*;
use super::{assert_detail_unit_interval, assert_report_close, make_ideal_10_disks};

#[test]
fn test_c3_near_collinear_slanted_collinear_and_8conn_diagonal_bridge() {
    // 3 vertex permutations of obtuse triangles ([0,0], [2,0], [1,eps])
    // straddling det.abs() <= 1e-15 * b2.max(d2), paired with a second cluster.
    let perms: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
    let cfg_tri = ClusterAreaConfig::default()
        .with_g_max(16)
        .with_sigma_cells(0.5)
        .with_boundary_r95(f64::INFINITY);
    for &eps in &[0.0_f64, 4e-16, 1.2e-15, -1.2e-15, 1e-14] {
        let tri = [(0.0_f64, 0.0_f64), (2.0_f64, 0.0_f64), (1.0_f64, eps)];
        let mut base_rep: Option<ClusterAreaReport> = None;
        for perm in &perms {
            let (c_direct, r_direct) = c3(tri[perm[0]], tri[perm[1]], tri[perm[2]]);
            assert!(c_direct.0.is_finite() && c_direct.1.is_finite() && r_direct.is_finite());

            let xs = vec![
                tri[perm[0]].0,
                tri[perm[1]].0,
                tri[perm[2]].0,
                10.0,
                12.0,
                11.0,
            ];
            let ys = vec![
                tri[perm[0]].1,
                tri[perm[1]].1,
                tri[perm[2]].1,
                0.0,
                0.0,
                1.0,
            ];
            let labs = [0u32, 0, 0, 1, 1, 1];
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let det = cluster_area_faithfulness_detail(&layout, &labs, &cfg_tri).unwrap();
            assert_detail_unit_interval(&det, &format!("c3_obtuse eps={eps:e}"));
            if let Some(ref expected) = base_rep {
                assert_report_close(
                    expected,
                    &det.report,
                    1e-12,
                    &format!("c3_obtuse perm eps={eps:e}"),
                );
            } else {
                base_rep = Some(det.report);
            }
        }
    }

    // Slanted collinear N=80 clusters (t, 2t) & (t+10, 2t+10) triggering
    // akl_toussaint_filter poly.len() < 3 fallback (pts.len() > 64).
    let mut xs_sl = Vec::with_capacity(160);
    let mut ys_sl = Vec::with_capacity(160);
    let mut labs_sl = Vec::with_capacity(160);
    for t in 0..80usize {
        let tf = t as f64;
        xs_sl.push(tf);
        ys_sl.push(2.0 * tf);
        labs_sl.push(0);
    }
    for t in 0..80usize {
        let tf = t as f64;
        xs_sl.push(tf + 10.0);
        ys_sl.push(2.0 * tf + 10.0);
        labs_sl.push(1);
    }
    let l_slanted = Layout2d::from_xy(xs_sl, ys_sl).unwrap();
    let det_slanted =
        cluster_area_faithfulness_detail(&l_slanted, &labs_sl, &ClusterAreaConfig::default())
            .unwrap();
    assert_detail_unit_interval(&det_slanted, "slanted_collinear_n80_per_cluster");
    assert!(det_slanted.report.caf > 0.2);

    // True 8-connected diagonal-only active bridge at g_max=12, sigma_cells=0.25, boundary_r95=inf.
    let mut xs_diag = vec![0.0, 0.0, 0.0, 0.2, 0.2, 0.2];
    let mut ys_diag = vec![0.0, 0.0, 0.0, 0.2, 0.2, 0.2];
    let mut labs_diag = vec![0u32; 6];
    xs_diag.extend(std::iter::repeat_n(-2.4_f64, 48));
    ys_diag.extend(std::iter::repeat_n(0.0_f64, 48));
    xs_diag.extend(std::iter::repeat_n(2.4_f64, 46));
    ys_diag.extend(std::iter::repeat_n(0.0_f64, 46));
    labs_diag.extend(std::iter::repeat_n(1u32, 94));
    let cfg_diag = ClusterAreaConfig::default()
        .with_g_max(12)
        .with_sigma_cells(0.25)
        .with_boundary_r95(f64::INFINITY);
    let det_diag = cluster_area_faithfulness_detail(
        &Layout2d::from_xy(xs_diag.clone(), ys_diag.clone()).unwrap(),
        &labs_diag,
        &cfg_diag,
    )
    .unwrap();
    assert_detail_unit_interval(&det_diag, "diag_bridge_connected");
    assert!(
        (det_diag.coherence[0] - 1.0).abs() < 1e-12,
        "expected diagonal bridge coherence[0] == 1.0, got {}",
        det_diag.coherence[0]
    );

    // Shifting (0.2,0.2) to (0.4,0.4) moves to (h+2, h+2), breaking 8-connectivity and
    // dropping coherence[0] cleanly to 0.50.
    for i in 3..6 {
        xs_diag[i] = 0.4;
        ys_diag[i] = 0.4;
    }
    let det_broken = cluster_area_faithfulness_detail(
        &Layout2d::from_xy(xs_diag, ys_diag).unwrap(),
        &labs_diag,
        &cfg_diag,
    )
    .unwrap();
    assert!(
        (det_broken.coherence[0] - 0.50).abs() < 1e-12,
        "expected broken bridge coherence[0] == 0.50, got {}",
        det_broken.coherence[0]
    );
}

#[test]
fn test_tiny_sigma_cells_kronecker_stability() {
    // Direct helper checks: parity at sigma=0.85 and Kronecker / half-tie limit as sigma -> 0+.
    let sigma_std = 0.85;
    let inv_std = -0.5 / (sigma_std * sigma_std);
    let r_std = (3.5 * sigma_std).ceil() as usize;
    let mut buf_std = vec![0.0_f64; 2 * r_std + 1];
    for &frac in &[-0.5, -0.37, 0.0, 0.25, 0.5] {
        fill_1d_gaussian(&mut buf_std, r_std, frac, sigma_std, inv_std);
        let mut raw = vec![0.0_f64; 2 * r_std + 1];
        let mut sum = 0.0_f64;
        for (i, slot) in raw.iter_mut().enumerate() {
            let dx = (i as isize - r_std as isize) as f64 - frac;
            let v = (inv_std * dx * dx).exp();
            *slot = v;
            sum += v;
        }
        for (a, b) in buf_std.iter().zip(raw.iter()) {
            assert!((*a - (*b / sum)).abs() < 1e-15);
        }
    }

    for &sigma in &[1e-160, 1e-200, f64::MIN_POSITIVE] {
        let inv = -0.5 / (sigma * sigma);
        let mut buf = vec![0.0_f64; 3];
        fill_1d_gaussian(&mut buf, 1, 0.13, sigma, inv);
        assert_eq!(buf, vec![0.0, 1.0, 0.0]);
        fill_1d_gaussian(&mut buf, 1, 0.5, sigma, inv);
        assert_eq!(buf, vec![0.0, 0.5, 0.5]);
        fill_1d_gaussian(&mut buf, 1, -0.5, sigma, inv);
        assert_eq!(buf, vec![0.5, 0.5, 0.0]);
    }

    // Non-grid-aligned 6-node layout across tiny sigma_cells values down to f64::MIN_POSITIVE.
    let layout = Layout2d::from_xy(
        vec![0.0, 0.37, 1.41, 2.53, 3.19, 4.27],
        vec![0.0, 0.49, 1.48, 2.13, 3.44, 4.49],
    )
    .unwrap();
    let labels = [0u32, 0, 0, 1, 1, 1];
    for &sigma in &[0.012, 0.01, 0.005, 1e-10, 1e-160, 1e-200, f64::MIN_POSITIVE] {
        let cfg = ClusterAreaConfig::default().with_sigma_cells(sigma);
        let detail = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
        assert_detail_unit_interval(&detail, &format!("sigma_cells={sigma:e}"));
        let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
        assert_eq!(rep, detail.report);
    }

    // Exact max_b_grid boundary clamping check.
    let max_b_grid = ((H_MAX - 4) as f64) / 128.0;
    let (l_big, labs_big) = make_ideal_10_disks(300, 1.0, 5.0);
    for &b in &[max_b_grid, max_b_grid + 1e-12] {
        let det = cluster_area_faithfulness_detail(
            &l_big,
            &labs_big,
            &ClusterAreaConfig::default().with_boundary_r95(b),
        )
        .unwrap();
        assert_detail_unit_interval(&det, &format!("max_b_grid={b}"));
    }
}

#[test]
fn test_sigma_cells_grid_upper_bound_and_max_b_grid_clamp() {
    // Exact sigma_cells upper bound (ceil(3.5 * sigma_cells) <= H_MAX - 2 = 1022).
    // 3.5 * 292.0 = 1022.0 (passes); ceil(3.5 * 292.001) = 1023 (InvalidInput).
    let layout_sep = Layout2d::from_xy(
        vec![0.0, 1.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let labels_sep = [0u32, 0, 0, 0, 1, 1, 1, 1];

    let ok_cfg = ClusterAreaConfig::default().with_sigma_cells(292.0);
    let ok_det = cluster_area_faithfulness_detail(&layout_sep, &labels_sep, &ok_cfg).unwrap();
    assert_detail_unit_interval(&ok_det, "sigma_cells=292.0");

    let bad_cfg = ClusterAreaConfig::default().with_sigma_cells(292.001);
    assert!(matches!(
        cluster_area_faithfulness(&layout_sep, &labels_sep, &bad_cfg),
        Err(CoreError::InvalidInput(_))
    ));

    // F1 grid-extent clamping: sigma_cells=100.0, boundary_r95=10.0, g_max=128
    // clamps g_eff to floor((1024 - 350 - 1) / 10.0) = 67 while preserving
    // effective_boundary_r95 = 10.0 and in_frac = 1.0.
    let clamp_cfg = ClusterAreaConfig::default()
        .with_g_max(128)
        .with_sigma_cells(100.0)
        .with_boundary_r95(10.0);
    let clamp_det = cluster_area_faithfulness_detail(&layout_sep, &labels_sep, &clamp_cfg).unwrap();
    assert_detail_unit_interval(&clamp_det, "max_b_grid_clamp");
    assert_eq!(clamp_det.report.g_eff, 67);
    assert_eq!(clamp_det.report.effective_boundary_r95, 10.0);
    assert!((clamp_det.report.in_frac - 1.0).abs() < 1e-12);
    assert!(
        clamp_det.report.caf > 0.1,
        "expected caf > 0.1 under F1 g_eff clamping, got {}",
        clamp_det.report.caf
    );
    assert_eq!(
        cluster_area_faithfulness(&layout_sep, &labels_sep, &clamp_cfg).unwrap(),
        clamp_det.report
    );
}

#[test]
fn test_stevens_gamma_numerical_extremes() {
    // Extreme stevens_gamma values on imbalanced 4-vs-3 clusters without NaN/Inf/-0.0.
    let layout = Layout2d::from_xy(
        vec![0.0, 0.2, 0.0, 0.2, 3.0, 3.2, 3.0],
        vec![0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2],
    )
    .unwrap();
    let labels = [0u32, 0, 0, 0, 1, 1, 1];
    for &gamma in &[1e-12_f64, 1e-300, 100.0, 1e4, 1e300] {
        let cfg = ClusterAreaConfig::default().with_stevens_gamma(gamma);
        let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
        let det = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
        assert_eq!(rep, det.report);
        assert_detail_unit_interval(&det, &format!("stevens_gamma={gamma:e}"));
        for v in [
            rep.caf,
            rep.purity_adj,
            rep.prop_vis,
            rep.res_density,
            rep.res_deconv,
            rep.res_combined,
            rep.spatial_coherence,
            rep.in_frac,
        ] {
            if v == 0.0 {
                assert!(!v.is_sign_negative(), "negative zero for gamma={gamma:e}");
            }
        }
        for &pv in &det.p {
            if pv == 0.0 {
                assert!(
                    !pv.is_sign_negative(),
                    "negative zero in p for gamma={gamma:e}"
                );
            }
        }
    }
}
