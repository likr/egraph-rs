use std::f64::consts::LN_2;

use crate::scalar::Scalar;

pub use super::lorentz_maps::{lorentz_exp_map, lorentz_log_map, lorentz_to_poincare_point};
pub use super::lorentz_midpoint::lorentz_einstein_midpoint;

/// Threshold above which `r^2` approaches `f64::MAX`, switching to scaled/asymptotic paths.
pub(super) const LARGE_RADIUS_THRESHOLD: f64 = 1e140;

/// Threshold on `log(r_max / r_min)` above which `r_min / r_max < 1e-20`.
pub(super) const RATIO_LOG_THRESHOLD: f64 = 46.0;

/// Computes the implicit time coordinate $t = \sqrt{1 + x^2 + y^2}$ on the upper sheet of $\mathbb{H}^2$.
#[inline]
pub(super) fn lorentz_time_and_radius(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    let t = 1.0_f64.hypot(r);
    (t, r)
}

/// Decomposes a finite non-zero `f64` into `(m, exp)` with `|m| in [1, 2)` and `v == m * 2^exp`.
#[inline]
pub(super) fn frexp_norm(v: f64) -> (f64, i32) {
    debug_assert!(v.is_finite() && v != 0.0);
    const TWO_POW_54: f64 = f64::from_bits((1023 + 54) << 52);
    let (vn, bias) = if v.abs() < f64::MIN_POSITIVE {
        (v * TWO_POW_54, 54)
    } else {
        (v, 0)
    };
    let bits = vn.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let m = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023_u64 << 52));
    (m, raw_exp - 1023 - bias)
}

/// Scales a finite `f64` by `2^exp` with exact IEEE-754 single-step rounding and
/// overflow/underflow safety.
#[inline]
pub(super) fn scale_pow2(v: f64, exp: i32) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return v;
    }
    let (m, ev) = frexp_norm(v);
    let total_exp = (ev as i64) + (exp as i64);
    if total_exp > 1023 {
        return f64::INFINITY.copysign(v);
    }
    if total_exp >= -1022 {
        return f64::from_bits(
            (m.to_bits() & 0x800f_ffff_ffff_ffff) | (((total_exp + 1023) as u64) << 52),
        );
    }
    if total_exp >= -1075 {
        // Place `m` at minimum normal exponent `2^-1022` (bit-exact, no rounding), then apply a
        // single IEEE-754 multiply by `2^(total_exp + 1022)` (`exp in [-53, -1]`) to avoid
        // subnormal double-rounding.
        let at_min_normal = f64::from_bits((m.to_bits() & 0x800f_ffff_ffff_ffff) | (1_u64 << 52));
        let sub_scale = f64::from_bits(((total_exp + 1022 + 1023) as u64) << 52);
        return at_min_normal * sub_scale;
    }
    0.0_f64.copysign(v)
}

/// Computes the exact-exponent-scaled 2x2 cross product `x1 * y2 - y1 * x2 = det_m * 2^det_exp`
/// with `|det_m| in [1, 2)` (or `(0.0, 0)` when mathematically zero), using Kahan's 2-FMA
/// determinant so cancellation, high ratios, and subnormal/extreme coordinates stay accurate to
/// `<= 2 ULP` without overflow or underflow.
#[inline]
pub(super) fn lorentz_scaled_cross(x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, i32) {
    let has_p = x1 != 0.0 && y2 != 0.0;
    let has_q = y1 != 0.0 && x2 != 0.0;
    if !has_p && !has_q {
        return (0.0, 0);
    }
    if !has_q {
        let (mx1, ex1) = frexp_norm(x1);
        let (my2, ey2) = frexp_norm(y2);
        let (mp, ea) = frexp_norm(mx1 * my2);
        return (mp, ex1 + ey2 + ea);
    }
    if !has_p {
        let (my1, ey1) = frexp_norm(y1);
        let (mx2, ex2) = frexp_norm(x2);
        let (mq, ea) = frexp_norm(-(my1 * mx2));
        return (mq, ey1 + ex2 + ea);
    }

    let (mx1, ex1) = frexp_norm(x1);
    let (my2, ey2) = frexp_norm(y2);
    let (my1, ey1) = frexp_norm(y1);
    let (mx2, ex2) = frexp_norm(x2);
    let ep = ex1 + ey2;
    let eq = ey1 + ex2;
    let diff = ep - eq;

    if diff > 60 {
        let (mp, ea) = frexp_norm(mx1 * my2);
        return (mp, ep + ea);
    }
    if diff < -60 {
        let (mq, ea) = frexp_norm(-(my1 * mx2));
        return (mq, eq + ea);
    }

    let (a, b, c, d, e_base) = if diff >= 0 {
        let shift = f64::from_bits(((1023 - diff) as u64) << 52);
        (mx1, my2, my1 * shift, mx2, ep)
    } else {
        let shift = f64::from_bits(((1023 + diff) as u64) << 52);
        (mx1 * shift, my2, my1, mx2, eq)
    };
    let w = c * d;
    let err = c.mul_add(d, -w);
    let f = a.mul_add(b, -w);
    let det_m = f - err;
    if det_m == 0.0 {
        return (0.0, 0);
    }
    let (m_norm, e_adj) = frexp_norm(det_m);
    (m_norm, e_base + e_adj)
}

/// Computes overflow- and subnormal-safe `(ux, uy, r_or_inf, mr, er)` for any finite non-zero
/// `(x, y)` such that `r = mr * 2^er` with `mr in [1, 2)`.
#[inline]
pub(super) fn lorentz_polar_scaled(x: f64, y: f64) -> (f64, f64, f64, f64, i32) {
    let m = x.abs().max(y.abs());
    let sx = x / m;
    let sy = y / m;
    let sr = sx.hypot(sy);
    let (mm, em) = frexp_norm(m);
    let (mr, e_adj) = frexp_norm(mm * sr);
    (sx / sr, sy / sr, m * sr, mr, em + e_adj)
}

/// Computes overflow-safe `(ux, uy, r_or_inf, log_r)` for any finite non-zero `(x, y)`.
#[inline]
pub(super) fn lorentz_unit_and_log_r(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let m = x.abs().max(y.abs());
    let sx = x / m;
    let sy = y / m;
    let sr = sx.hypot(sy);
    (sx / sr, sy / sr, m * sr, m.ln() + sr.ln())
}

/// Computes `asinh(r)` stably even when `x.hypot(y)` overflows `f64::MAX`.
#[inline]
fn lorentz_asinh_from_xy(x: f64, y: f64) -> f64 {
    let m = x.abs().max(y.abs());
    if m == 0.0 {
        return 0.0;
    }
    if m < LARGE_RADIUS_THRESHOLD {
        x.hypot(y).asinh()
    } else {
        let (_, _, _, log_r) = lorentz_unit_and_log_r(x, y);
        LN_2 + log_r
    }
}

/// Computes `(eta - 1, log_eta_fallback, t1, r1, t2, r2)` stably without catastrophic
/// cancellation for near-coincident points or overflow at extreme radii up to `f64::MAX`.
///
/// When `eta - 1` exceeds `f64::MAX`, returns `f64::INFINITY` for `eta - 1` alongside the
/// finite natural logarithm `log_eta_fallback = ln(eta)` so `lorentz_distance` remains finite.
#[inline]
fn lorentz_eta_or_log(x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64, f64, f64, f64) {
    let m1 = x1.abs().max(y1.abs());
    let m2 = x2.abs().max(y2.abs());
    if m1 == 0.0 && m2 == 0.0 {
        return (0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
    }
    if m1 == 0.0 {
        let (t2, r2) = lorentz_time_and_radius(x2, y2);
        let eta_m1 = if m2 >= LARGE_RADIUS_THRESHOLD {
            t2 - 1.0
        } else {
            (r2 * r2) / (t2 + 1.0)
        };
        let (_, _, _, log_r2) = lorentz_unit_and_log_r(x2, y2);
        return (eta_m1, log_r2, 1.0, 0.0, t2, r2);
    }
    if m2 == 0.0 {
        let (t1, r1) = lorentz_time_and_radius(x1, y1);
        let eta_m1 = if m1 >= LARGE_RADIUS_THRESHOLD {
            t1 - 1.0
        } else {
            (r1 * r1) / (t1 + 1.0)
        };
        let (_, _, _, log_r1) = lorentz_unit_and_log_r(x1, y1);
        return (eta_m1, log_r1, t1, r1, 1.0, 0.0);
    }

    let m_max = m1.max(m2);
    if m_max < LARGE_RADIUS_THRESHOLD {
        let (t1, r1) = lorentz_time_and_radius(x1, y1);
        let (t2, r2) = lorentz_time_and_radius(x2, y2);
        // Factor t1*t2 - r1*r2 = (1 + r1^2 + r2^2) / (t1*t2 + r1*r2) and
        // t1*t2 - r1*r2 - 1 = u * (u / (t1*t2 - r1*r2 + 1)) where
        // u = (x1^2 + y1^2 - x2^2 - y2^2) / (r1*t2 + r2*t1), avoiding r1 - r2 cancellation.
        let radial_den = r1 * t2 + r2 * t1;
        let radial_term = if radial_den > 0.0 {
            let r1_sq_minus_r2_sq = (x1 - x2) * (x1 + x2) + (y1 - y2) * (y1 + y2);
            let u = r1_sq_minus_r2_sq / radial_den;
            let t1t2_minus_r1r2 = (1.0 + r1 * r1 + r2 * r2) / (t1 * t2 + r1 * r2);
            u * (u / (t1t2_minus_r1r2 + 1.0))
        } else {
            0.0
        };

        // Compute r1*r2*(1 - cos(theta)) via exact-exponent Kahan 2x2 cross product, avoiding
        // unit-vector subtraction cancellation, high-ratio underflow, and rotated `p2 = 2*p1` drift.
        let (ux1, uy1, _, mr1, er1) = lorentz_polar_scaled(x1, y1);
        let (ux2, uy2, _, mr2, er2) = lorentz_polar_scaled(x2, y2);
        let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
        let angular_term = if cos_theta >= 0.0 {
            let (det_m, det_exp) = lorentz_scaled_cross(x1, y1, x2, y2);
            if det_m == 0.0 {
                0.0
            } else {
                let ang_m = (det_m * det_m) / (mr1 * mr2 * (1.0 + cos_theta));
                scale_pow2(ang_m, 2 * det_exp - er1 - er2)
            }
        } else {
            r1 * r2 * (1.0 - cos_theta)
        };

        return ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2);
    }

    // Extreme-radius path (`m_max >= 1e140`): compute overflow-safe polar components.
    let (ux1, uy1, r1, mr1, er1) = lorentz_polar_scaled(x1, y1);
    let (ux2, uy2, r2, mr2, er2) = lorentz_polar_scaled(x2, y2);
    let log_r1 = mr1.ln() + (er1 as f64) * LN_2;
    let log_r2 = mr2.ln() + (er2 as f64) * LN_2;
    let t1 = if m1 >= LARGE_RADIUS_THRESHOLD {
        r1
    } else {
        1.0_f64.hypot(r1)
    };
    let t2 = if m2 >= LARGE_RADIUS_THRESHOLD {
        r2
    } else {
        1.0_f64.hypot(r2)
    };

    let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let (det_m, det_exp) = lorentz_scaled_cross(x1, y1, x2, y2);

    if (log_r1 - log_r2).abs() > RATIO_LOG_THRESHOLD {
        // One radius dominates by > 1e20 (`t_big = r_big` to >280 digits):
        // eta = r_big * (1 / (t_small + r_small) + r_small * (1 - cos(theta))).
        let (r_big, log_big, r_small, t_small, mr_big, er_big, mr_small, er_small) =
            if log_r1 > log_r2 {
                (r1, log_r1, r2, t2, mr1, er1, mr2, er2)
            } else {
                (r2, log_r2, r1, t1, mr2, er2, mr1, er1)
            };
        let ang = if cos_theta < 0.0 {
            r_small * (1.0 - cos_theta)
        } else if det_m == 0.0 {
            0.0
        } else {
            let ang_m = (det_m * det_m) / ((mr_big * mr_big) * mr_small * (1.0 + cos_theta));
            scale_pow2(ang_m, 2 * det_exp - 2 * er_big - er_small)
        };
        let f = 1.0 / (t_small + r_small) + ang;
        let log_eta = log_big + f.ln();
        let eta = r_big * f;
        let eta_m1 = if eta.is_finite() {
            (eta - 1.0).max(0.0)
        } else {
            f64::INFINITY
        };
        return (eta_m1, log_eta, t1, r1, t2, r2);
    }

    // Both points have r >= 1e120: scale coordinates by `1 / m_max` into `[-1, 1]`.
    let tx1 = x1 / m_max;
    let ty1 = y1 / m_max;
    let tx2 = x2 / m_max;
    let ty2 = y2 / m_max;
    let tr1 = tx1.hypot(ty1);
    let tr2 = tx2.hypot(ty2);
    let tr_diff = ((tx1 - tx2) * (tx1 + tx2) + (ty1 - ty2) * (ty1 + ty2)) / (tr1 + tr2);
    let radial_term = (tr_diff * tr_diff) / (2.0 * tr1 * tr2);

    if cos_theta >= 0.0 {
        if det_m == 0.0 {
            return (radial_term.max(0.0), 0.0, t1, r1, t2, r2);
        }
        let ang_m = (det_m * det_m) / (mr1 * mr2 * (1.0 + cos_theta));
        let ang_exp = 2 * det_exp - er1 - er2;
        let angular_term = scale_pow2(ang_m, ang_exp);
        if angular_term.is_finite() {
            ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2)
        } else {
            let log_eta = ang_m.ln() + (ang_exp as f64) * LN_2;
            (f64::INFINITY, log_eta, t1, r1, t2, r2)
        }
    } else {
        let a_tilde = tr1 * tr2 * (1.0 - cos_theta);
        let c_scaled = m_max * a_tilde.sqrt();
        let angular_term = c_scaled * c_scaled;
        if angular_term.is_finite() {
            ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2)
        } else {
            let log_eta = 2.0 * m_max.ln() + a_tilde.ln();
            (f64::INFINITY, log_eta, t1, r1, t2, r2)
        }
    }
}

#[inline]
pub(super) fn lorentz_distance_f64(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    if x1 == 0.0 && y1 == 0.0 {
        return lorentz_asinh_from_xy(x2, y2);
    }
    if x2 == 0.0 && y2 == 0.0 {
        return lorentz_asinh_from_xy(x1, y1);
    }
    // Tiny/subnormal regime: `t_i = 1` to > 200 digits, so Euclidean distance avoids `eta - 1`
    // underflow below `1e-162` while remaining exact down to subnormal `f64::from_bits(1)`.
    if x1.abs().max(y1.abs()).max(x2.abs()).max(y2.abs()) < 1e-100 {
        return (x1 - x2).hypot(y1 - y2);
    }
    let (eta_m1, log_eta, _, _, _, _) = lorentz_eta_or_log(x1, y1, x2, y2);
    if eta_m1.is_finite() {
        2.0 * (0.5 * eta_m1).sqrt().asinh()
    } else {
        (LN_2 + log_eta).max(0.0)
    }
}

/// Computes the unit hyperbolic geodesic distance between spatial hyperboloid coordinates
/// `(x1, y1)` and `(x2, y2)` in $\mathbb{R}^2$.
///
/// Uses a factored `arcsinh` formulation that avoids floating-point cancellation both near
/// zero distance and at large hyperbolic radii ($r_{\mathbb{H}} > 19$), and stays overflow-safe
/// across all finite IEEE-754 inputs.
pub fn lorentz_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let x1f = x1.to_f64();
    let y1f = y1.to_f64();
    let x2f = x2.to_f64();
    let y2f = y2.to_f64();
    if !x1f.is_finite() || !y1f.is_finite() || !x2f.is_finite() || !y2f.is_finite() {
        return S::ZERO;
    }
    S::from_f64(lorentz_distance_f64(x1f, y1f, x2f, y2f))
}
