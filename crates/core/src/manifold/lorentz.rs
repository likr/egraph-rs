use std::f64::consts::LN_2;

use crate::scalar::Scalar;

use super::poincare::clamp_poincare_point;

pub use super::lorentz_midpoint::lorentz_einstein_midpoint;

/// Threshold above which `r^2` approaches `f64::MAX`, switching to scaled/asymptotic paths.
pub(super) const LARGE_RADIUS_THRESHOLD: f64 = 1e140;

/// Threshold on `log(r_max / r_min)` above which `r_min / r_max < 1e-20`.
const RATIO_LOG_THRESHOLD: f64 = 46.0;

/// Computes the implicit time coordinate $t = \sqrt{1 + x^2 + y^2}$ on the upper sheet of $\mathbb{H}^2$.
#[inline]
pub(super) fn lorentz_time_and_radius(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    let t = 1.0_f64.hypot(r);
    (t, r)
}

/// Computes overflow-safe `(ux, uy, r_or_inf, log_r)` for any finite non-zero `(x, y)`.
#[inline]
fn lorentz_unit_and_log_r(x: f64, y: f64) -> (f64, f64, f64, f64) {
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

    let dux = ux1 - ux2;
    let duy = uy1 - uy2;
    let one_minus_cos = 0.5 * (dux * dux + duy * duy);

    if (log_r1 - log_r2).abs() > RATIO_LOG_THRESHOLD {
        // One radius dominates by > 1e20 (`t_big = r_big` to >280 digits):
        // eta = r_big * (1 / (t_small + r_small) + r_small * (1 - cos(theta))).
        let (r_big, log_big, r_small, t_small) = if log_r1 > log_r2 {
            (r1, log_r1, r2, t2)
        } else {
            (r2, log_r2, r1, t1)
        };
        let f = 1.0 / (t_small + r_small) + r_small * one_minus_cos;
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

    let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    if cos_theta >= 0.0 {
        let (c_tilde, ratio_min_over_max) = if tr1 <= tr2 {
            (ux1 * (ty2 - ty1) - uy1 * (tx2 - tx1), tr1 / tr2)
        } else {
            (ux2 * (ty1 - ty2) - uy2 * (tx1 - tx2), tr2 / tr1)
        };
        let ratio_sqrt = ratio_min_over_max.sqrt();
        let c_scaled = (m_max * c_tilde) * ratio_sqrt;
        let angular_term = (c_scaled * c_scaled) / (1.0 + cos_theta);
        if angular_term.is_finite() {
            ((radial_term + angular_term).max(0.0), 0.0, t1, r1, t2, r2)
        } else {
            let log_eta = 2.0 * m_max.ln() + 2.0 * c_tilde.abs().ln() + ratio_min_over_max.ln()
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
fn lorentz_distance_f64(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
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

/// Computes the Riemannian logarithmic map $\log_u(v)$ at $u = (x_1, y_1)$ pointing toward
/// $v = (x_2, y_2)$ in the Lorentz boost orthonormal tangent frame `(e1, e2)`.
///
/// The returned 2D tangent vector has Euclidean norm equal to [`lorentz_distance`] everywhere,
/// including at the origin `(0, 0)`.
pub fn lorentz_log_map<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let x1f = x1.to_f64();
    let y1f = y1.to_f64();
    let x2f = x2.to_f64();
    let y2f = y2.to_f64();
    if !x1f.is_finite() || !y1f.is_finite() || !x2f.is_finite() || !y2f.is_finite() {
        return (S::ZERO, S::ZERO);
    }

    let dist = lorentz_distance_f64(x1f, y1f, x2f, y2f);
    if !dist.is_finite() || dist <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }

    let m1 = x1f.abs().max(y1f.abs());
    let m2 = x2f.abs().max(y2f.abs());
    if m1 == 0.0 {
        let (ux2, uy2, _, _) = lorentz_unit_and_log_r(x2f, y2f);
        return (S::from_f64(ux2 * dist), S::from_f64(uy2 * dist));
    }
    if m2 == 0.0 {
        let (ux1, uy1, _, _) = lorentz_unit_and_log_r(x1f, y1f);
        return (S::from_f64(-ux1 * dist), S::from_f64(-uy1 * dist));
    }

    let m_max = m1.max(m2);
    if m_max < LARGE_RADIUS_THRESHOLD {
        let (eta_m1, _, t1, _, t2, _) = lorentz_eta_or_log(x1f, y1f, x2f, y2f);
        if eta_m1.is_finite() {
            // Compute <v, e_a>_L = (x2 - x1) - (x1 / (1 + t1)) * ((t2 - t1) + (eta - 1)) stably.
            let t2_minus_t1 = if t1 + t2 > 0.0 {
                ((x2f - x1f) * (x2f + x1f) + (y2f - y1f) * (y2f + y1f)) / (t2 + t1)
            } else {
                0.0
            };
            let corr = (t2_minus_t1 + eta_m1) / (1.0 + t1);
            let p1 = (x2f - x1f) - x1f * corr;
            let p2 = (y2f - y1f) - y1f * corr;
            let p_norm = p1.hypot(p2);
            if !p_norm.is_finite() || p_norm <= 1e-18 {
                return (S::ZERO, S::ZERO);
            }
            let scale = dist / p_norm;
            return (S::from_f64(p1 * scale), S::from_f64(p2 * scale));
        }
    }

    // Overflow-safe radial/tangential boost decomposition for extreme radii.
    let (ux1, uy1, r1, log_r1) = lorentz_unit_and_log_r(x1f, y1f);
    let (ux2, uy2, r2, log_r2) = lorentz_unit_and_log_r(x2f, y2f);
    let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let sin_theta = ux1 * uy2 - uy1 * ux2;
    let dux = ux1 - ux2;
    let duy = uy1 - uy2;
    let one_minus_cos = 0.5 * (dux * dux + duy * duy);

    let (p_par, p_perp) = if log_r1 - log_r2 > RATIO_LOG_THRESHOLD {
        let t2 = 1.0_f64.hypot(r2);
        let p_par_s = -(1.0 / (t2 + r2) + r2 * one_minus_cos);
        let p_perp_s = (log_r2 - log_r1).exp() * sin_theta;
        (p_par_s, p_perp_s)
    } else if log_r2 - log_r1 > RATIO_LOG_THRESHOLD {
        let t1 = 1.0_f64.hypot(r1);
        let p_par_s = cos_theta / (t1 + r1) - r1 * one_minus_cos;
        (p_par_s, sin_theta)
    } else {
        let tx1 = x1f / m_max;
        let ty1 = y1f / m_max;
        let tx2 = x2f / m_max;
        let ty2 = y2f / m_max;
        let tr1 = tx1.hypot(ty1);
        let tr2 = tx2.hypot(ty2);
        let radial_part =
            ((tx2 - tx1) * (tx2 + tx1) + (ty2 - ty1) * (ty2 + ty1)) / (2.0 * tr1 * tr2);
        let ang_sqrt = m_max * (tr1 * tr2 * one_minus_cos).sqrt();
        let ang_par = ang_sqrt * ang_sqrt;
        if ang_par.is_finite() {
            (radial_part - ang_par, m_max * (tr2 * sin_theta))
        } else {
            (-1.0, sin_theta / (m_max * tr1 * one_minus_cos))
        }
    };

    let p1 = p_par * ux1 - p_perp * uy1;
    let p2 = p_par * uy1 + p_perp * ux1;
    let p_norm = p1.hypot(p2);
    if !p_norm.is_finite() || p_norm <= 1e-18 {
        return (S::ZERO, S::ZERO);
    }
    let scale = dist / p_norm;
    (S::from_f64(p1 * scale), S::from_f64(p2 * scale))
}

/// Computes the Riemannian exponential map $\exp_u(v)$ advancing spatial hyperboloid point
/// $u = (x, y)$ by tangent vector `(vx, vy)` expressed in the Lorentz boost frame `(e1, e2)`.
pub fn lorentz_exp_map<S: Scalar>(x: S, y: S, vx: S, vy: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();
    if !xf.is_finite() || !yf.is_finite() || !vxf.is_finite() || !vyf.is_finite() {
        return (S::ZERO, S::ZERO);
    }

    let theta = vxf.hypot(vyf);
    if theta <= 1e-15 {
        return (x, y);
    }

    // Cap step to prevent IEEE-754 overflow in cosh/sinh while remaining far past f32/f64 scales.
    let max_theta = if std::mem::size_of::<S>() == 4 {
        80.0
    } else {
        700.0
    };
    let eff_theta = theta.min(max_theta);
    let dx = vxf / theta;
    let dy = vyf / theta;

    let m = xf.abs().max(yf.abs());
    let (nx, ny) = if m < LARGE_RADIUS_THRESHOLD {
        let (t, _) = lorentz_time_and_radius(xf, yf);
        let s_dot = (xf * dx + yf * dy) / (1.0 + t);
        let wx = dx + xf * s_dot;
        let wy = dy + yf * s_dot;
        let c = eff_theta.cosh();
        let s = eff_theta.sinh();
        (c * xf + s * wx, c * yf + s * wy)
    } else {
        // Decompose along `(u, u_perp)` to avoid `inf - inf = NaN` cancellation when stepping
        // inward from extreme hyperbolic radii (`r >= 1e140`).
        let (ux, uy, r, log_r) = lorentz_unit_and_log_r(xf, yf);
        let cos_phi = (ux * dx + uy * dy).clamp(-1.0, 1.0);
        let sin_phi = ux * dy - uy * dx;
        let c = eff_theta.cosh();
        let s = eff_theta.sinh();
        let r_par = if cos_phi >= 0.0 {
            r * c + (if r.is_finite() { r } else { f64::INFINITY }) * cos_phi * s
        } else {
            let one_plus_cos = (1.0 + cos_phi).max(0.0);
            let term1 = if r.is_finite() {
                r * (-eff_theta).exp()
            } else {
                (log_r - eff_theta).exp()
            };
            let term2 = if one_plus_cos > 0.0 {
                if r.is_finite() {
                    r * s * one_plus_cos
                } else {
                    (log_r + s.ln() + one_plus_cos.ln()).exp()
                }
            } else {
                0.0
            };
            let term3 = if r.is_finite() {
                (cos_phi * s) / (2.0 * r)
            } else {
                0.5 * cos_phi * (s.ln() - log_r).exp()
            };
            term1 + term2 + term3
        };
        let r_perp = sin_phi * s;
        (r_par * ux - r_perp * uy, r_par * uy + r_perp * ux)
    };
    let rx = S::from_f64(nx);
    let ry = S::from_f64(ny);
    if rx.is_finite() && ry.is_finite() {
        (rx, ry)
    } else {
        (x, y)
    }
}

/// Maps a spatial hyperboloid point `(x, y)` to Poincaré disk coordinates `(x/(1+t), y/(1+t))`,
/// clamped strictly inside the open unit disk.
#[inline]
pub fn lorentz_to_poincare_point<S: Scalar>(x: S, y: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    if !xf.is_finite() || !yf.is_finite() {
        return (S::ZERO, S::ZERO);
    }
    if xf == 0.0 && yf == 0.0 {
        return (S::ZERO, S::ZERO);
    }
    let (t, _) = lorentz_time_and_radius(xf, yf);
    let denom = 1.0 + t;
    if !denom.is_finite() {
        // When `r = hypot(x, y)` overflows `f64::MAX`, `x/(1+t)` equals the unit direction,
        // which clamps strictly inside `1 - POINCARE_EPS`.
        let (ux, uy, _, _) = lorentz_unit_and_log_r(xf, yf);
        return clamp_poincare_point(S::from_f64(ux), S::from_f64(uy));
    }
    clamp_poincare_point(S::from_f64(xf / denom), S::from_f64(yf / denom))
}
