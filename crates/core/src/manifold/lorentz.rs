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

        // Compute r1*r2*(1 - cos(theta)) stably without unit-vector subtraction cancellation.
        let ux1 = x1 / r1;
        let uy1 = y1 / r1;
        let ux2 = x2 / r2;
        let uy2 = y2 / r2;
        let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
        let angular_term = if cos_theta >= 0.0 {
            if r1 <= r2 {
                let cross_r2 = ux1 * (y2 - y1) - uy1 * (x2 - x1);
                (cross_r2 * (cross_r2 * (r1 / r2))) / (1.0 + cos_theta)
            } else {
                let cross_r1 = ux2 * (y1 - y2) - uy2 * (x1 - x2);
                (cross_r1 * (cross_r1 * (r2 / r1))) / (1.0 + cos_theta)
            }
        } else {
            r1 * r2 * (1.0 - cos_theta)
        };

        return ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2);
    }

    // Extreme-radius path (`m_max >= 1e140`): compute overflow-safe polar components.
    let (ux1, uy1, r1, log_r1) = lorentz_unit_and_log_r(x1, y1);
    let (ux2, uy2, r2, log_r2) = lorentz_unit_and_log_r(x2, y2);
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
    let sin_theta = ux1 * uy2 - uy1 * ux2;

    if (log_r1 - log_r2).abs() > RATIO_LOG_THRESHOLD {
        // One radius dominates by > 1e20 (`t_big = r_big` to >280 digits):
        // eta = r_big * (1 / (t_small + r_small) + r_small * (1 - cos(theta))).
        let (r_big, log_big, r_small, t_small) = if log_r1 > log_r2 {
            (r1, log_r1, r2, t2)
        } else {
            (r2, log_r2, r1, t1)
        };
        let ang = if cos_theta < 0.0 {
            r_small * (1.0 - cos_theta)
        } else {
            ((r_small * sin_theta) * sin_theta) / (1.0 + cos_theta)
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
        let (ratio_min_over_max, tr_big, c_direct) = if tr1 <= tr2 {
            let c_dir = if 2.0 * tr1 >= tr2 {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let cp = ux1 * dy - uy1 * dx;
                if dx.is_finite() && dy.is_finite() && cp.is_finite() {
                    Some(cp)
                } else {
                    None
                }
            } else {
                None
            };
            (tr1 / tr2, tr2, c_dir)
        } else {
            let c_dir = if 2.0 * tr2 >= tr1 {
                let dx = x1 - x2;
                let dy = y1 - y2;
                let cp = ux2 * dy - uy2 * dx;
                if dx.is_finite() && dy.is_finite() && cp.is_finite() {
                    Some(cp)
                } else {
                    None
                }
            } else {
                None
            };
            (tr2 / tr1, tr1, c_dir)
        };
        let ratio_sqrt = ratio_min_over_max.sqrt();
        if let Some(c_dir) = c_direct {
            let c_scaled = c_dir * ratio_sqrt;
            let angular_term = (c_scaled * c_scaled) / (1.0 + cos_theta);
            if angular_term.is_finite() {
                return ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2);
            }
            let log_eta = 2.0 * c_scaled.abs().ln() - (1.0 + cos_theta).ln();
            return (f64::INFINITY, log_eta, t1, r1, t2, r2);
        }
        // Use `sin_theta = ux1 * uy2 - uy1 * ux2` directly when radii differ by > 2x or diffs
        // overflow, avoiding `ty / m_max` underflow and `x1 - x2` rounding amplification.
        let m_sin = m_max * sin_theta;
        let c_scaled = (m_sin * ratio_sqrt) * tr_big;
        let angular_term = (c_scaled * c_scaled) / (1.0 + cos_theta);
        if m_sin.is_finite() && c_scaled.is_finite() && angular_term.is_finite() {
            ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2)
        } else {
            let log_eta =
                2.0 * m_max.ln() + 2.0 * (tr_big * sin_theta.abs()).ln() + ratio_min_over_max.ln()
                    - (1.0 + cos_theta).ln();
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
