use super::*;

#[test]
fn test_lorentz_log_exp_extreme_ratio_and_roundtrips() {
    // Ratio-branch and m_max < 1e140 pairs where old `1 + t1` or `p_norm <= 1e-18` vanished.
    let pairs = [
        (
            (1e20_f64, 0.0_f64),
            (1e41_f64, 0.0_f64),
            48.354_286_952_874_96_f64,
        ),
        (
            (1e41_f64, 0.0_f64),
            (1e20_f64, 0.0_f64),
            48.354_286_952_874_96_f64,
        ),
        ((1e20_f64, 0.0_f64), (2e20_f64, 0.0_f64), 2.0_f64.ln()),
        ((2e20_f64, 0.0_f64), (1e20_f64, 0.0_f64), 2.0_f64.ln()),
        (
            (1e141_f64, 0.0_f64),
            (1e162_f64, 0.0_f64),
            48.354_286_952_874_96_f64,
        ),
        (
            (1e162_f64, 0.0_f64),
            (1e141_f64, 0.0_f64),
            48.354_286_952_874_96_f64,
        ),
    ];
    for ((x1, y1), (x2, y2), expected_d) in pairs {
        let d = lorentz_distance(x1, y1, x2, y2);
        assert!(
            (d - expected_d).abs() < 1e-10,
            "dist({x1:e}, {y1:e}, {x2:e}, {y2:e}) = {d}, expected {expected_d}"
        );
        let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
        let norm = vx.hypot(vy);
        assert!(
            (norm - expected_d).abs() < 1e-10,
            "log_map({x1:e}, {y1:e}, {x2:e}, {y2:e}) norm = {norm}, expected {expected_d}"
        );
        assert!((vx.signum() - (x2 - x1).signum()).abs() < 1e-12);
        let (rx, ry) = lorentz_exp_map(x1, y1, vx, vy);
        let roundtrip_d = lorentz_distance(rx, ry, x2, y2);
        assert!(
            roundtrip_d < 1e-9,
            "exp(log) mismatch ({x1:e}, {y1:e})->({x2:e}, {y2:e}): got ({rx:e}, {ry:e}), err={roundtrip_d}"
        );
    }

    // Extreme midpoint -> (1.5e308, 1.5e308) log_map retains non-zero direction and exact norm.
    let (mx, my) = lorentz_einstein_midpoint(&[1.5e308_f64, 0.0], &[1.5e308, 0.0]).unwrap();
    let expected_half_d = 355.320_697_260_557_04_f64;
    let (vx_m, vy_m) = lorentz_log_map(mx, my, 1.5e308, 1.5e308);
    assert!(
        (vx_m.hypot(vy_m) - expected_half_d).abs() < 1e-10 && vx_m > 0.0 && vy_m > 0.0,
        "midpoint->max log_map=({vx_m}, {vy_m}), norm={}",
        vx_m.hypot(vy_m)
    );

    // Extreme inward exp(log) roundtrip where `(1 + cos_phi)` underflows unless factored via sin_phi.
    let (vx_in, vy_in) = lorentz_log_map(1e200_f64, 0.0, 0.0, 1.0);
    let d_in = lorentz_distance(1e200_f64, 0.0, 0.0, 1.0);
    assert!((vx_in.hypot(vy_in) - d_in).abs() < 1e-10);
    let (rx_in, ry_in) = lorentz_exp_map(1e200_f64, 0.0, vx_in, vy_in);
    assert!(
        lorentz_distance(rx_in, ry_in, 0.0, 1.0) < 1e-8,
        "inward exp(log) (1e200,0)->(0,1): got ({rx_in}, {ry_in})"
    );

    // Overflow-safe `m_max * tr1` avoidance in equal-scale log_map and finite-component exp_map.
    for (x2, y2) in [
        (1.5e308_f64, 1.5e308_f64 - 1e155_f64),
        (1.5e308_f64, 1.5e308_f64 - 1e295_f64),
        (1.5e308_f64, 0.0_f64),
        (1.5e308_f64, -1.5e308_f64),
    ] {
        let d = lorentz_distance(1.5e308, 1.5e308, x2, y2);
        let (vx, vy) = lorentz_log_map(1.5e308, 1.5e308, x2, y2);
        assert!(vx.is_finite() && vy.is_finite());
        if d > 1e-15 {
            assert!(
                (vx.hypot(vy) - d).abs() < 1e-9,
                "overflow-safe log_map norm mismatch for ({x2:e}, {y2:e}): {} vs {d}",
                vx.hypot(vy)
            );
        } else {
            assert_eq!((vx, vy), (0.0, 0.0));
        }
    }

    // Tangent / slight-inward steps from `(1.5e308, 1.5e308)` where `r = hypot(x, y)` overflows.
    let (ex_tan, ey_tan) = lorentz_exp_map(1.5e308_f64, 1.5e308, -0.05, 0.05);
    assert!(ex_tan.is_finite() && ey_tan.is_finite());
    assert!(lorentz_distance(1.5e308, 1.5e308, ex_tan, ey_tan).is_finite());
    let (ex_inw, ey_inw) = lorentz_exp_map(1.5e308_f64, 1.5e308, -0.02, -0.01);
    assert!(ex_inw.is_finite() && ey_inw.is_finite());
    assert!(ex_inw < 1.5e308 && ey_inw < 1.5e308);
}

#[test]
fn test_lorentz_inward_exp_and_tiny_angle_distance() {
    // Inward exponential map at `m < 1e140` past the origin without `c * x - s * w` cancellation.
    let (rx, ry) = lorentz_exp_map(1e100_f64, 0.0, -250.0, 0.0);
    let expected_r_origin = 250.0 - 1e100_f64.asinh();
    assert!(
        rx < 0.0 && ry.abs() < 1e-15,
        "expected negative rx past origin, got ({rx:e}, {ry:e})"
    );
    assert!(
        (lorentz_distance(rx, ry, 0.0, 0.0) - expected_r_origin).abs() < 1e-10,
        "distance from origin mismatch: {} vs {expected_r_origin}",
        lorentz_distance(rx, ry, 0.0, 0.0)
    );

    // Equal-scale `c_direct` avoids `ty2 / m_max` underflow at `1e300`.
    let d_tiny_y = lorentz_distance(1e300_f64, 0.0, 1e300, 1e-30);
    assert!(
        (d_tiny_y - 1e-30).abs() < 1e-42,
        "expected ~1e-30, got {d_tiny_y:e}"
    );

    // Ratio-branch `(r_small * sin_theta)` avoids `one_minus_cos` underflow at `2e164` vs `1e200`.
    let d_ratio_tiny_ang = lorentz_distance(2e164_f64, 0.0, 1e200, 5e35);
    let expected_ratio_tiny_ang = (1.0 + 5e35_f64).acosh();
    assert!(
        (d_ratio_tiny_ang - expected_ratio_tiny_ang).abs() < 1e-10,
        "expected {expected_ratio_tiny_ang}, got {d_ratio_tiny_ang}"
    );
}

#[test]
fn test_lorentz_rescale_geodesic_extreme_cases() {
    let cases_f64: &[(&str, f64, f64, f64, f64, f64, f64)] = &[
        (
            "1.5e308_origin",
            1.5e308,
            1.5e308,
            0.0,
            0.0,
            10.0,
            710.641_394_521_114_1,
        ),
        (
            "1e200_tiny1",
            1e200,
            0.0,
            1e200,
            1.0,
            3.0,
            0.962_423_650_119_206_9,
        ),
        (
            "1e200_5e49",
            1e200,
            5e49,
            1e200,
            -5e49,
            3.0,
            230.258_509_299_404_58,
        ),
        (
            "1e200_5e35",
            1e200,
            5e35,
            1e200,
            -5e35,
            3.0,
            165.786_126_695_571_28,
        ),
        (
            "1e100_origin",
            1e100,
            0.0,
            0.0,
            0.0,
            2.0,
            230.951_656_479_964_53,
        ),
    ];
    for &(label, x0, y0, x1, y1, new_scale, expected_d) in cases_f64 {
        let mut lay = ManifoldLayout2d::<f64>::from_xy(
            ManifoldKind::HyperbolicLorentz,
            vec![x0, x1],
            vec![y0, y1],
        )
        .unwrap();
        let d_before = lay.geodesic_distance(0, 1).unwrap();
        assert!(
            (d_before - expected_d).abs() < 1e-8,
            "{label}: d_before={d_before}, expected={expected_d}"
        );
        lay.rescale_geodesic(new_scale).unwrap();
        let d_after = lay.geodesic_distance(0, 1).unwrap();
        assert!(
            (d_after - d_before).abs() < 1e-7,
            "{label}: rescale_geodesic({new_scale}) drift {d_before} -> {d_after}"
        );
    }

    // f32 extreme-radius rescale_geodesic invariance.
    let mut lay32 = ManifoldLayout2d::<f32>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![3e38_f32, 0.0_f32],
        vec![3e38_f32, 0.0_f32],
    )
    .unwrap();
    let d32_before = lay32.geodesic_distance(0, 1).unwrap();
    lay32.rescale_geodesic(10.0).unwrap();
    let d32_after = lay32.geodesic_distance(0, 1).unwrap();
    assert!(
        d32_after.is_finite() && (d32_after - d32_before).abs() < 1e-3,
        "f32 rescale_geodesic(10.0) drift {d32_before} -> {d32_after}"
    );
}
