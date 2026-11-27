use std::f64::consts::PI;

use super::*;
use crate::error::CoreError;
use crate::layout::Layout2d;

#[test]
fn test_lorentz_distance_isometry_and_large_radius_stability() {
    // Unconstrained coordinates far outside the unit disk remain unclamped.
    let lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![0.0, 100.0, -100.0],
        vec![0.0, 100.0, 50.0],
    )
    .unwrap();
    assert_eq!(lor.x()[1], 100.0);
    assert_eq!(lor.y()[2], 50.0);

    // Exact isometric agreement with Poincaré disk model at moderate radii.
    let pairs = [
        ((0.0_f64, 0.0_f64), (1.2_f64, -0.7_f64)),
        ((-0.8, 1.5), (2.1, 0.3)),
        ((5.0, -3.0), (5.0 + 1e-4, -3.0 - 2e-4)),
    ];
    for ((x1, y1), (x2, y2)) in pairs {
        let d_lor = lorentz_distance(x1, y1, x2, y2);
        let (px1, py1) = lorentz_to_poincare_point(x1, y1);
        let (px2, py2) = lorentz_to_poincare_point(x2, y2);
        let d_poi = poincare_distance(px1, py1, px2, py2);
        assert!(
            (d_lor - d_poi).abs() < 1e-11,
            "d_lor={d_lor}, d_poi={d_poi}"
        );
    }

    // Small-perturbation accuracy where naive acosh(1 + eps^2) would underflow to 0.
    let d_tiny = lorentz_distance(5.0_f64, -3.0, 5.0 + 1e-9, -3.0 - 2e-9);
    assert!(d_tiny > 1e-9 && d_tiny < 2e-9, "d_tiny={d_tiny}");

    // Large radius + small separation without r1 - r2 radial subtraction cancellation.
    let d_large_small = lorentz_distance(100.0_f64, -80.0, 100.0 + 1e-7, -80.0 - 1e-7);
    let expected_large_small = 1.565_589_270_205_540_3e-8_f64;
    assert!(
        (d_large_small - expected_large_small).abs() < 1e-18,
        "large-radius small-step cancellation: got={d_large_small:.17e}, expected={expected_large_small:.17e}"
    );
    let (lvx, lvy) = lorentz_log_map(100.0_f64, -80.0, 100.0 + 1e-7, -80.0 - 1e-7);
    assert!(
        (lvx.hypot(lvy) - expected_large_small).abs() < 1e-18,
        "log map norm mismatch: {} vs {expected_large_small}",
        lvx.hypot(lvy)
    );

    // Large radius stability for hyp_r > 19 (naive t1*t2 - x1*x2 - y1*y2 cancels completely).
    let r1 = 1e10_f64;
    let r2 = 2e10_f64;
    let d_rad = lorentz_distance(r1, 0.0, r2, 0.0);
    assert!(
        (d_rad - 2.0_f64.ln()).abs() < 1e-10,
        "large-r collinear distance expected ln(2), got {d_rad}"
    );
    let d_same = lorentz_distance(1e10_f64, -2e10, 1e10, -2e10);
    assert!(d_same.abs() < 1e-14);
}

#[test]
fn test_lorentz_log_exp_roundtrip_and_boost_frame() {
    let pts = [(0.0_f64, 0.0_f64), (0.8, -0.4), (-2.3, 1.7), (12.0, 5.0)];
    for &(x1, y1) in &pts {
        for &(x2, y2) in &pts {
            let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
            let d = lorentz_distance(x1, y1, x2, y2);
            assert!(
                (vx.hypot(vy) - d).abs() < 1e-10,
                "norm mismatch at ({x1},{y1})->({x2},{y2}): {} vs {d}",
                vx.hypot(vy)
            );
            let (rx, ry) = lorentz_exp_map(x1, y1, vx, vy);
            assert!(
                lorentz_distance(rx, ry, x2, y2) < 1e-9,
                "exp(log) mismatch at ({x1},{y1})->({x2},{y2}): got ({rx},{ry})"
            );
        }
    }
}

#[test]
fn test_lorentz_einstein_midpoint_and_to_poincare_layout() {
    let mut lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-1.5, 0.7, 2.3, -0.4],
        vec![0.8, -1.1, 1.4, 2.0],
    )
    .unwrap();
    lor.set_scale(2.5).unwrap();

    let (mx_l, my_l) = lorentz_einstein_midpoint(lor.x(), lor.y()).unwrap();
    let (mx_lp, my_lp) = lorentz_to_poincare_point(mx_l, my_l);
    let poinc = lor.to_poincare_layout().unwrap();
    assert_eq!(poinc.kind(), ManifoldKind::HyperbolicPoincare);
    assert!((poinc.scale() - 2.5).abs() < 1e-12);
    let (mx_p, my_p) = poinc.poincare_centroid().unwrap();
    assert!((mx_lp - mx_p).abs() < 1e-11);
    assert!((my_lp - my_p).abs() < 1e-11);
    for (&px, &py) in poinc.x().iter().zip(poinc.y().iter()) {
        assert!(px.hypot(py) <= 1.0 - POINCARE_EPS_F64 + 1e-12);
    }

    assert!((POINCARE_EPS_F64 - 1e-5).abs() < 1e-15);
    assert!((POINCARE_EPS_F32 - 1e-4_f32).abs() < 1e-12);

    let sph = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 3);
    assert!(matches!(
        sph.to_poincare_layout(),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // f32 initialization with scale >= 1.0 succeeds on HyperbolicLorentz.
    let lor32 =
        ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 10, 42, 3.0).unwrap();
    lor32.validate_finite().unwrap();
    assert!(lor32.to_poincare_layout().is_ok());
}

#[test]
fn test_manifold_metric_scaling_and_rect_torus() {
    let mut sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, PI / 2.0],
        vec![PI / 2.0, PI / 2.0],
    )
    .unwrap();
    assert_eq!(
        sph.metric(),
        ManifoldMetric {
            scale: 1.0,
            aspect: 1.0
        }
    );
    assert!((sph.geodesic_distance(0, 1).unwrap() - PI / 2.0).abs() < 1e-12);
    sph.set_scale(3.0).unwrap();
    assert!((sph.geodesic_distance(0, 1).unwrap() - 1.5 * PI).abs() < 1e-12);
    let (vx, vy) = sph.tangent_delta(0, 1).unwrap();
    assert!((vx.hypot(vy) - 1.5 * PI).abs() < 1e-12);
    let mut sph_step = sph.clone();
    sph_step.apply_tangent_step(0, 0.5 * vx, 0.5 * vy).unwrap();
    assert!((sph_step.geodesic_distance(0, 1).unwrap() - 0.75 * PI).abs() < 1e-11);

    // Non-torus rejects set_torus_extents; invalid scale/aspect rejected.
    assert!(matches!(
        sph.set_torus_extents(2.0, 1.0),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        sph.set_scale(0.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sph.set_scale(f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        sph.set_aspect(-1.0),
        Err(CoreError::InvalidInput(_))
    ));

    // Rectangular torus preserves [0, 1)^2 coords while scaling (Lx, Ly).
    let mut torus =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, vec![0.1, 0.9], vec![0.2, 0.4])
            .unwrap();
    torus.set_torus_extents(4.0, 2.0).unwrap();
    assert!((torus.scale() - 2.0).abs() < 1e-12);
    assert!((torus.aspect() - 2.0).abs() < 1e-12);
    assert!((torus.x()[0] - 0.1).abs() < 1e-12);
    // dx_norm = -0.2 => lx * dx_norm = -0.8; dy_norm = 0.2 => ly * dy_norm = 0.4
    let (tdx, tdy) = torus.tangent_delta(0, 1).unwrap();
    assert!((tdx - (-0.8)).abs() < 1e-12);
    assert!((tdy - 0.4).abs() < 1e-12);
    assert!((torus.geodesic_distance(0, 1).unwrap() - (-0.8_f64).hypot(0.4)).abs() < 1e-12);
    torus.apply_tangent_step(0, 0.5 * tdx, 0.5 * tdy).unwrap();
    assert!((torus.geodesic_distance(0, 1).unwrap() - 0.5 * (-0.8_f64).hypot(0.4)).abs() < 1e-12);
}

#[test]
fn test_rescale_geodesic_all_manifolds() {
    // Collinear symmetric hyperbolic pair around origin: Einstein midpoint is (0, 0), so
    // exp_o((s_old / new_scale) * log_o(x_i)) preserves exact physical geodesic distance.
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
    ] {
        let mut h =
            ManifoldLayout2d::<f64>::from_xy(kind, vec![-0.4, 0.4], vec![0.0, 0.0]).unwrap();
        h.set_scale(2.0).unwrap();
        let d_before = h.geodesic_distance(0, 1).unwrap();
        h.rescale_geodesic(4.0).unwrap();
        assert!((h.scale() - 4.0).abs() < 1e-12);
        let d_after = h.geodesic_distance(0, 1).unwrap();
        assert!(
            (d_after - d_before).abs() < 1e-10,
            "{kind:?}: collinear rescale distance drift {d_before} vs {d_after}"
        );
    }

    // Spherical and Torus2d keep normalized coordinates unchanged under rescale_geodesic.
    let mut sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 6, 42, 1.0).unwrap();
    let sph_coords_before = sph.as_layout().clone();
    sph.rescale_geodesic(2.5).unwrap();
    assert_eq!(sph.as_layout(), &sph_coords_before);
    assert!((sph.scale() - 2.5).abs() < 1e-12);

    let mut tor = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 6, 42, 1.0).unwrap();
    let tor_coords_before: Layout2d<f64> = tor.as_layout().clone();
    tor.rescale_geodesic(3.0).unwrap();
    assert_eq!(tor.as_layout(), &tor_coords_before);
    assert!((tor.scale() - 3.0).abs() < 1e-12);
}

#[test]
fn test_lorentz_near_origin_log_map_precision() {
    let pairs = [
        ((0.0_f64, 0.0_f64), (1e-9_f64, -2e-9_f64)),
        ((1e-9, -2e-9), (0.0, 0.0)),
        ((0.0, 0.0), (1e-12, 0.0)),
        ((1e-12, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (1e-6, 0.0)),
        ((1e-6, 0.0), (0.0, 0.0)),
        ((1e-12, 0.0), (2e-12, 1e-12)),
    ];
    for ((x1, y1), (x2, y2)) in pairs {
        let d = lorentz_distance(x1, y1, x2, y2);
        let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
        let v_norm = vx.hypot(vy);
        let rel_err = (v_norm - d).abs() / d.max(1e-30);
        assert!(
            rel_err < 1e-12,
            "near-origin log_map norm mismatch for ({x1},{y1})->({x2},{y2}): norm={v_norm}, d={d}, rel_err={rel_err}"
        );
        let (rx, ry) = lorentz_exp_map(x1, y1, vx, vy);
        let roundtrip_err = lorentz_distance(rx, ry, x2, y2) / d.max(1e-30);
        assert!(
            roundtrip_err < 1e-12,
            "near-origin exp(log) roundtrip mismatch for ({x1},{y1})->({x2},{y2}): err={roundtrip_err}"
        );
    }
}

#[test]
fn test_lorentz_extreme_radius_distance_log_exp_and_poincare_projection() {
    // Large-radius distance where `r^2` overflows `f64::MAX` while `radial_den` stays finite.
    let d160 = lorentz_distance(1e160_f64, 0.0, 1.0, 0.0);
    let expected160 = 1e160_f64.asinh() - 1.0_f64.asinh();
    assert!(
        d160.is_finite() && (d160 - expected160).abs() < 1e-10,
        "d160={d160}, expected={expected160}"
    );
    let d200 = lorentz_distance(1e200_f64, 0.0, 1.0, 0.0);
    let expected200 = (2.0_f64.ln() + 200.0 * 10.0_f64.ln()) - 1.0_f64.asinh();
    assert!(
        d200.is_finite() && (d200 - expected200).abs() < 1e-10,
        "d200={d200}, expected={expected200}"
    );

    // Opposite-ray and orthogonal extreme radii where `eta - 1` itself overflows `f64::MAX`.
    let d_opp = lorentz_distance(1e200_f64, 0.0, -1e200, 0.0);
    let expected_opp = 2.0 * (2.0_f64.ln() + 200.0 * 10.0_f64.ln());
    assert!(
        d_opp.is_finite() && (d_opp - expected_opp).abs() < 1e-9,
        "d_opp={d_opp}, expected={expected_opp}"
    );
    let d_orth = lorentz_distance(1e200_f64, 0.0, 0.0, 1e200);
    assert!(d_orth.is_finite() && d_orth > 900.0);

    // Extreme-radius small tangential separation without underflow.
    let d_tang = lorentz_distance(1e200_f64, 0.0, 1e200, 1.0);
    let expected_tang = 2.0 * 0.5_f64.asinh();
    assert!(
        d_tang.is_finite() && (d_tang - expected_tang).abs() < 1e-12,
        "d_tang={d_tang}, expected={expected_tang}"
    );

    // Coordinates where `x.hypot(y)` overflows `f64::MAX` (`1.5e308, 1.5e308`).
    let d_max = lorentz_distance(1.5e308_f64, 1.5e308, 1.0, 0.0);
    assert!(d_max.is_finite() && d_max > 700.0, "d_max={d_max}");
    let d_max_origin = lorentz_distance(1.5e308_f64, 1.5e308, 0.0, 0.0);
    assert!(
        d_max_origin.is_finite() && d_max_origin > 700.0,
        "d_max_origin={d_max_origin}"
    );

    // Log map and inward exponential map at extreme radii remain finite and consistent.
    let (vx, vy) = lorentz_log_map(1e160_f64, 0.0, 1.0, 0.0);
    assert!(vx.is_finite() && vy.is_finite());
    assert!((vx.hypot(vy) - expected160).abs() < 1e-9);
    let (rx, ry) = lorentz_exp_map(1e160_f64, 0.0, -500.0, 0.0);
    assert!(rx.is_finite() && ry.is_finite() && lorentz_distance(rx, ry, 1.0, 0.0) > 100.0);

    // Extreme finite coordinates project to the Poincaré boundary disk, never `(0.0, 0.0)`.
    let (px, py) = lorentz_to_poincare_point(1.5e308_f64, 1.5e308);
    let p_norm = px.hypot(py);
    assert!(
        px > 0.7 && py > 0.7 && (p_norm - (1.0 - POINCARE_EPS_F64)).abs() < 1e-12,
        "px={px}, py={py}, norm={p_norm}"
    );
    let lor_extreme = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![1.5e308, -1.5e308],
        vec![1.5e308, 0.0],
    )
    .unwrap();
    let poinc_extreme = lor_extreme.to_poincare_layout().unwrap();
    assert!(
        (poinc_extreme.x()[0].hypot(poinc_extreme.y()[0]) - (1.0 - POINCARE_EPS_F64)).abs() < 1e-12
    );
    assert!((poinc_extreme.x()[1] + (1.0 - POINCARE_EPS_F64)).abs() < 1e-12);

    // f32 extreme finite coordinates stay finite and clamp strictly inside POINCARE_EPS_F32.
    let d32 = lorentz_distance(1e38_f32, 0.0_f32, 1.0_f32, 0.0_f32);
    assert!(d32.is_finite() && d32 > 80.0);
    let (px32, py32) = lorentz_to_poincare_point(3.0e38_f32, 3.0e38_f32);
    assert!(px32 > 0.7 && py32 > 0.7 && (px32.hypot(py32) - (1.0 - POINCARE_EPS_F32)).abs() < 1e-6);
}
