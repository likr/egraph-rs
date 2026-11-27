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

#[test]
fn test_lorentz_rotated_collinear_and_high_ratio_precision() {
    let ln2 = std::f64::consts::LN_2;

    // Rotated exact `p2 = 2 * p1` across radii straddling 1e130..8e307 (both < 1e140 and >= 1e140).
    for r in [
        1e10_f64, 1e100, 1e130, 1e135, 1e139, 1e140, 1e150, 1e200, 1e300, 8e307,
    ] {
        for phi in [0.3_f64, 0.7, 1.0, 1.234, 2.1] {
            let x1 = r * phi.cos();
            let y1 = r * phi.sin();
            let x2 = 2.0 * x1;
            let y2 = 2.0 * y1;

            let d_fwd = lorentz_distance(x1, y1, x2, y2);
            let d_rev = lorentz_distance(x2, y2, x1, y1);
            assert!(
                (d_fwd - ln2).abs() < 1e-12 && (d_rev - ln2).abs() < 1e-12,
                "p2=2*p1 r={r:e} phi={phi}: d_fwd={d_fwd}, d_rev={d_rev}, expected ln(2)"
            );

            let (ux1, uy1, _, _, _) = super::lorentz::lorentz_polar_scaled(x1, y1);
            let (vx_f, vy_f) = lorentz_log_map(x1, y1, x2, y2);
            let norm_f = vx_f.hypot(vy_f);
            let align_f = (vx_f * ux1 + vy_f * uy1) / norm_f;
            assert!(
                (norm_f - ln2).abs() < 1e-12 && align_f > 1.0 - 1e-12,
                "forward log_map p2=2*p1 r={r:e} phi={phi}: norm={norm_f}, align={align_f}"
            );

            let (vx_r, vy_r) = lorentz_log_map(x2, y2, x1, y1);
            let norm_r = vx_r.hypot(vy_r);
            let align_r = (vx_r * ux1 + vy_r * uy1) / norm_r;
            assert!(
                (norm_r - ln2).abs() < 1e-12 && align_r < -1.0 + 1e-12,
                "reverse log_map p2=2*p1 r={r:e} phi={phi}: norm={norm_r}, align={align_r}"
            );
        }
    }

    // Exact collinear rotated `p2 = 1.5 * p1` at `r ~ 5e129` (exact mantissa scaling).
    let scale_pow = 2.0_f64.powi(380);
    let a = ((1e15_f64 * 0.7_f64.cos()) as i64 * 2) as f64 * scale_pow;
    let b = ((1e15_f64 * 0.7_f64.sin()) as i64 * 2) as f64 * scale_pow;
    let d_15 = lorentz_distance(a, b, 1.5 * a, 1.5 * b);
    assert!(
        (d_15 - 1.5_f64.ln()).abs() < 1e-12,
        "exact p2=1.5*p1 distance={d_15}, expected ln(1.5)"
    );
    let (vx15, vy15) = lorentz_log_map(a, b, 1.5 * a, 1.5 * b);
    let (uxa, uya, _, _, _) = super::lorentz::lorentz_polar_scaled(a, b);
    assert!(
        (vx15.hypot(vy15) - 1.5_f64.ln()).abs() < 1e-12
            && (vx15 * uxa + vy15 * uya) / vx15.hypot(vy15) > 1.0 - 1e-12
    );

    // Extreme high-ratio `m_max < 1e140` (`r_big > 2 * r_small`): exact collinear `2^296 * p1`.
    let x1_hr = 1e50_f64 * 0.7_f64.cos();
    let y1_hr = 1e50_f64 * 0.7_f64.sin();
    let pow296 = 2.0_f64.powi(296);
    let d_hr_exact = lorentz_distance(x1_hr, y1_hr, x1_hr * pow296, y1_hr * pow296);
    let expected_296 = 296.0 * ln2;
    assert!(
        (d_hr_exact - expected_296).abs() < 1e-10,
        "exact 2^296 high-ratio distance={d_hr_exact}, expected={expected_296}"
    );
    let (vx_hr, vy_hr) = lorentz_log_map(x1_hr, y1_hr, x1_hr * pow296, y1_hr * pow296);
    let (ux_hr, uy_hr, _, _, _) = super::lorentz::lorentz_polar_scaled(x1_hr, y1_hr);
    assert!(
        (vx_hr.hypot(vy_hr) - expected_296).abs() < 1e-10
            && (vx_hr * ux_hr + vy_hr * uy_hr) / vx_hr.hypot(vy_hr) > 1.0 - 1e-12
    );

    // High-ratio `m_max < 1e140` pairs with 1-ULP IEEE-754 angular offset: exact 700-digit references.
    let high_ratio_cases: &[(&str, f64, f64, f64, f64, f64)] = &[
        (
            "rot_phi07_1e50_1e139",
            1e50_f64 * 0.7_f64.cos(),
            1e50_f64 * 0.7_f64.sin(),
            1e139_f64 * 0.7_f64.cos(),
            1e139_f64 * 0.7_f64.sin(),
            359.938_571_185_530_1,
        ),
        (
            "rot_34_1e50_1e139",
            3e50_f64,
            4e50_f64,
            3e139_f64,
            4e139_f64,
            363.141_345_018_554_4,
        ),
        (
            "rot_phi07_1e100_1e135",
            1e100_f64 * 0.7_f64.cos(),
            1e100_f64 * 0.7_f64.sin(),
            1e135_f64 * 0.7_f64.cos(),
            1e135_f64 * 0.7_f64.sin(),
            462.947_730_607_682_6,
        ),
        (
            "subnormal_to_1e130",
            f64::from_bits(1),
            f64::from_bits(1),
            1e130_f64,
            2e130_f64,
            300.833_928_226_002_97,
        ),
        (
            "tiny_1e-200_to_1e130",
            1e-200_f64,
            1e-200_f64,
            1e130_f64,
            2e130_f64,
            300.833_928_226_002_97,
        ),
    ];
    for &(label, x1, y1, x2, y2, expected_d) in high_ratio_cases {
        let d_fwd = lorentz_distance(x1, y1, x2, y2);
        let d_rev = lorentz_distance(x2, y2, x1, y1);
        assert!(
            (d_fwd - expected_d).abs() < 1e-10 && (d_rev - expected_d).abs() < 1e-10,
            "{label}: d_fwd={d_fwd}, d_rev={d_rev}, expected={expected_d}"
        );
        let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
        assert!(
            vx.is_finite() && vy.is_finite() && (vx.hypot(vy) - expected_d).abs() < 1e-10,
            "{label}: log_map=({vx:e}, {vy:e}), norm={}, expected={expected_d}",
            vx.hypot(vy)
        );
    }
}

#[test]
fn test_lorentz_exp_map_near_max_inward_multiplication_order() {
    // Finite `r > 0.9e308`, `cos_phi < 0` where `(r * sin_phi) * r_perp` is in `[1, 2) * f64::MAX`
    // while `(r * sin_phi) * (r_perp / (1 - cos_phi))` stays strictly finite.
    let finite_r_cases = [
        (
            "r1.5e308_cos-0.5_th1.3",
            1.5e308_f64,
            0.0_f64,
            -0.5_f64,
            1.3_f64,
        ),
        (
            "r1.7e308_cos-0.9_th2.8",
            1.7e308_f64,
            0.0_f64,
            -0.9_f64,
            2.8_f64,
        ),
        (
            "r1.5e308_cos-0.99_th5.0",
            1.5e308_f64,
            0.0_f64,
            -0.99_f64,
            5.0_f64,
        ),
    ];
    for (label, x, y, cos_phi, theta) in finite_r_cases {
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        let vx = theta * cos_phi;
        let vy = theta * sin_phi;
        let (nx, ny) = lorentz_exp_map(x, y, vx, vy);
        assert!(
            nx.is_finite() && ny.is_finite() && (nx != x || ny != y),
            "{label}: expected finite moved point, got ({nx:e}, {ny:e})"
        );
        let d = lorentz_distance(x, y, nx, ny);
        assert!(
            (d - theta).abs() < 1e-11,
            "{label}: geodesic step length mismatch {d} vs {theta}"
        );
    }

    // Overflowed `r = hypot(x, y)` with finite `m > 0.9e308` (`else` scaled path using `m`).
    let overflow_r_cases = [
        (
            "m1.5e308_cos-0.5_th1.3",
            1.5e308_f64,
            1.5e308_f64,
            -0.5_f64,
            1.3_f64,
        ),
        (
            "m1.7e308_cos-0.9_th2.8",
            1.7e308_f64,
            1.7e308_f64,
            -0.9_f64,
            2.8_f64,
        ),
    ];
    for (label, x, y, cos_phi, theta) in overflow_r_cases {
        let (ux, uy, _, _, _) = super::lorentz::lorentz_polar_scaled(x, y);
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        let vx = theta * (cos_phi * ux - sin_phi * uy);
        let vy = theta * (cos_phi * uy + sin_phi * ux);
        let (nx, ny) = lorentz_exp_map(x, y, vx, vy);
        assert!(
            nx.is_finite() && ny.is_finite() && (nx != x || ny != y),
            "{label}: expected finite moved point on scaled branch, got ({nx:e}, {ny:e})"
        );
    }

    // Truly overflowing inward-tilted steps (> f64::MAX even after division) still safely return `(x, y)`.
    for (x, y) in [
        (1.5e308_f64, 0.0_f64),
        (0.0_f64, 1.5e308_f64),
        (1.5e308_f64, 1.5e308_f64),
    ] {
        let (ux, uy, _, _, _) = super::lorentz::lorentz_polar_scaled(x, y);
        let cos_phi = -0.5_f64;
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        let vx = 2.0 * (cos_phi * ux - sin_phi * uy);
        let vy = 2.0 * (cos_phi * uy + sin_phi * ux);
        let (nx, ny) = lorentz_exp_map(x, y, vx, vy);
        assert!(nx.is_finite() && ny.is_finite());
        assert_eq!((nx, ny), (x, y));
    }

    // Exact rotated inward collinear steps (`v = -2^k * p`) shrink cleanly without angular blow-up.
    for (r, phi, shift) in [(1e130_f64, 0.7_f64, -424_i32), (1.2e308, 0.7, -1016)] {
        let (x, y) = (r * phi.cos(), r * phi.sin());
        let (vx, vy) = (
            -super::lorentz::scale_pow2(x, shift),
            -super::lorentz::scale_pow2(y, shift),
        );
        let theta = vx.hypot(vy);
        let (nx, ny) = lorentz_exp_map(x, y, vx, vy);
        let expected_r = r * (-theta).exp();
        assert!(
            ((nx.hypot(ny) - expected_r) / expected_r).abs() < 1e-12,
            "rotated inward r={r:e} theta={theta}: got ({nx:e}, {ny:e}), expected r={expected_r:e}"
        );
    }

    // Overflowing tangent `hypot(vx, vy)` retains true direction instead of collapsing to `(0, 0)`.
    let (hx, hy) = lorentz_exp_map(1e200_f64, 1e200_f64, -1.5e308, -1.5e308);
    assert!(hx < -1e100 && hy < -1e100 && (hx - hy).abs() <= hx.abs() * 1e-12);
}

#[test]
fn test_lorentz_scaled_cross_and_pow2_boundaries() {
    use super::lorentz::{frexp_norm, lorentz_scaled_cross, scale_pow2};

    for v in [
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        -1.0,
        1.5,
        f64::MAX,
        -f64::MAX,
    ] {
        let (m, exp) = frexp_norm(v);
        assert!((1.0..2.0).contains(&m.abs()), "bad mantissa {m} for {v:e}");
        assert_eq!(scale_pow2(m, exp), v, "roundtrip mismatch for {v:e}");
    }
    // Single-step IEEE-754 subnormal rounding (no double rounding when `|v| < 1` and `exp < -1022`).
    for k in [5.0_f64, 11.0] {
        let v = 0.25_f64 + k * 2.0_f64.powi(-54);
        assert_eq!(scale_pow2(v, -1023).to_bits(), 0x0002_0000_0000_0001);
    }
    assert_eq!(scale_pow2(1.0, -1075), 0.0);
    assert_eq!(scale_pow2(1.0 + f64::EPSILON, -1075).to_bits(), 1_u64);

    // Tiny/subnormal pair continuity and f32 rotated `p2 = 2*p1` accuracy.
    assert!((lorentz_distance(f64::from_bits(1), 0.0, 1e-200, 0.0) - 1e-200).abs() <= 1e-215);
    assert!((lorentz_distance(1e-200_f64, 0.0, 2e-200, 0.0) - 1e-200).abs() <= 1e-215);
    let (x32, y32) = (1e30_f32 * 0.7_f32.cos(), 1e30_f32 * 0.7_f32.sin());
    assert!(
        (lorentz_distance(x32, y32, 2.0 * x32, 2.0 * y32) - std::f32::consts::LN_2).abs() < 1e-5
    );

    // Exact zero cancellation on scaled collinear vectors and extreme overflow/underflow safety.
    let (x1, y1) = (1e130_f64 * 0.7_f64.cos(), 1e130_f64 * 0.7_f64.sin());
    assert_eq!(lorentz_scaled_cross(x1, y1, 2.0 * x1, 2.0 * y1), (0.0, 0));
    let (dm, de) = lorentz_scaled_cross(f64::MAX, 0.0, 0.0, f64::MAX);
    assert!(dm.is_finite() && dm >= 1.0 && de >= 2046);
    assert_eq!(scale_pow2(1.0, -6000), 0.0);
    assert_eq!(scale_pow2(0.0, 6000), 0.0);
    assert!(scale_pow2(1.0, 6000).is_infinite());
}
