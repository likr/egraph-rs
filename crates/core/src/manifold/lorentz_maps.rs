use std::f64::consts::LN_2;

use crate::scalar::Scalar;

use super::lorentz::{
    LARGE_RADIUS_THRESHOLD, RATIO_LOG_THRESHOLD, frexp_norm, lorentz_distance_f64,
    lorentz_polar_scaled, lorentz_scaled_cross, lorentz_time_and_radius, lorentz_unit_and_log_r,
    scale_pow2,
};
use super::poincare::clamp_poincare_point;

/// Computes the Riemannian logarithmic map $\log_u(v)$ at $u = (x_1, y_1)$ pointing toward
/// $v = (x_2, y_2)$ in the Lorentz boost orthonormal tangent frame `(e1, e2)`.
///
/// The returned 2D tangent vector has Euclidean norm equal to `lorentz_distance` everywhere,
/// including at the origin `(0, 0)` and across extreme IEEE-754 coordinates without overflow,
/// underflow, or radial cancellation.
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
    // Tiny/subnormal regime: `t_i = 1` to > 200 digits, so Cartesian difference avoids r^2 underflow.
    if m_max < 1e-100 {
        let p1 = x2f - x1f;
        let p2 = y2f - y1f;
        let m_p = p1.abs().max(p2.abs());
        if !m_p.is_finite() || m_p == 0.0 {
            return (S::ZERO, S::ZERO);
        }
        let n1 = p1 / m_p;
        let n2 = p2 / m_p;
        let scale = dist / n1.hypot(n2);
        return (S::from_f64(n1 * scale), S::from_f64(n2 * scale));
    }

    let (ux1, uy1, r1, mr1, er1) = lorentz_polar_scaled(x1f, y1f);
    let (ux2, uy2, r2, mr2, er2) = lorentz_polar_scaled(x2f, y2f);
    let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let (det_m, det_exp) = lorentz_scaled_cross(x1f, y1f, x2f, y2f);

    let (p_par, p_perp) = if m_max < LARGE_RADIUS_THRESHOLD {
        // Branch A: all `r_i, t_i < 1.42e140`, so products `< 2.1e280` are overflow-free.
        let t1 = 1.0_f64.hypot(r1);
        let t2 = 1.0_f64.hypot(r2);
        let p_perp_v = if det_m == 0.0 {
            0.0
        } else {
            scale_pow2(det_m / mr1, det_exp - er1)
        };
        let p_par_v = if cos_theta < 0.0 {
            (t1 * r2) * cos_theta - r1 * t2
        } else {
            let a = ((x2f - x1f) * (x2f + x1f) + (y2f - y1f) * (y2f + y1f)) / (t1 * r2 + r1 * t2);
            let b = if det_m == 0.0 {
                0.0
            } else {
                let (mt1, et1) = frexp_norm(t1);
                scale_pow2(
                    (mt1 * det_m * det_m) / ((mr1 * mr1) * mr2 * (1.0 + cos_theta)),
                    et1 + 2 * det_exp - 2 * er1 - er2,
                )
            };
            a - b
        };
        (p_par_v, p_perp_v)
    } else {
        let log_r1 = mr1.ln() + (er1 as f64) * LN_2;
        let log_r2 = mr2.ln() + (er2 as f64) * LN_2;
        if log_r1 - log_r2 > RATIO_LOG_THRESHOLD {
            // Branch B1: `r1 / r2 > e^46` (`r2, t2` are always finite). Divide `(p_par, p_perp)` by `r1`.
            let t2 = 1.0_f64.hypot(r2);
            let p_perp_s = if det_m == 0.0 {
                0.0
            } else {
                scale_pow2(det_m / (mr1 * mr1), det_exp - 2 * er1)
            };
            let p_par_s = if cos_theta < 0.0 {
                r2 * cos_theta - t2
            } else {
                let ang_s = if det_m == 0.0 {
                    0.0
                } else {
                    scale_pow2(
                        (det_m * det_m) / ((mr1 * mr1) * mr2 * (1.0 + cos_theta)),
                        2 * det_exp - 2 * er1 - er2,
                    )
                };
                -(1.0 / (t2 + r2) + ang_s)
            };
            (p_par_s, p_perp_s)
        } else if log_r2 - log_r1 > RATIO_LOG_THRESHOLD {
            // Branch B2: `r2 / r1 > e^46` (`r1, t1` are always finite). Divide `(p_par, p_perp)` by `r2`.
            let t1 = 1.0_f64.hypot(r1);
            let p_perp_s = if det_m == 0.0 {
                0.0
            } else {
                scale_pow2(det_m / (mr1 * mr2), det_exp - er1 - er2)
            };
            let p_par_s = if cos_theta < 0.0 {
                t1 * cos_theta - r1
            } else {
                let ang_s = if det_m == 0.0 {
                    0.0
                } else {
                    let (mt1, et1) = frexp_norm(t1);
                    scale_pow2(
                        (mt1 * det_m * det_m) / ((mr1 * mr1) * (mr2 * mr2) * (1.0 + cos_theta)),
                        et1 + 2 * det_exp - 2 * er1 - 2 * er2,
                    )
                };
                1.0 / (t1 + r1) - ang_s
            };
            (p_par_s, p_perp_s)
        } else {
            // Branch C: extreme equal-scale regime (`m_max >= 1e140`, `|log(r1/r2)| <= 46`).
            let tx1 = x1f / m_max;
            let ty1 = y1f / m_max;
            let tx2 = x2f / m_max;
            let ty2 = y2f / m_max;
            let tr1 = tx1.hypot(ty1);
            let tr2 = tx2.hypot(ty2);

            if cos_theta < 0.0 {
                // Normalize by `r1 * r2 * (1 - cos_theta)` without overflowing `r1 * r2`.
                let p_perp_s = if det_m == 0.0 {
                    0.0
                } else {
                    scale_pow2(
                        det_m / ((mr1 * mr1) * mr2 * (1.0 - cos_theta)),
                        det_exp - 2 * er1 - er2,
                    )
                };
                (-1.0, p_perp_s)
            } else {
                let radial_part =
                    ((tx2 - tx1) * (tx2 + tx1) + (ty2 - ty1) * (ty2 + ty1)) / (2.0 * tr1 * tr2);
                if det_m == 0.0 {
                    (radial_part, 0.0)
                } else {
                    let ang_par = scale_pow2(
                        (det_m * det_m) / (mr1 * mr2 * (1.0 + cos_theta)),
                        2 * det_exp - er1 - er2,
                    );
                    let p_perp_v = scale_pow2(det_m / mr1, det_exp - er1);
                    if ang_par.is_finite() && p_perp_v.is_finite() {
                        (radial_part - ang_par, p_perp_v)
                    } else {
                        (
                            -1.0,
                            scale_pow2((mr2 * (1.0 + cos_theta)) / det_m, er2 - det_exp),
                        )
                    }
                }
            }
        }
    };

    // Pre-normalize by `m_p = max(|p_par|, |p_perp|)` so non-zero extreme/tiny vectors never
    // underflow or trigger an artificial magnitude cutoff.
    let m_p = p_par.abs().max(p_perp.abs());
    if !m_p.is_finite() || m_p == 0.0 {
        return (S::ZERO, S::ZERO);
    }
    let np_par = p_par / m_p;
    let np_perp = p_perp / m_p;
    let p1 = np_par * ux1 - np_perp * uy1;
    let p2 = np_par * uy1 + np_perp * ux1;
    let p_norm = p1.hypot(p2);
    if !p_norm.is_finite() || p_norm == 0.0 {
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

    let mv = vxf.abs().max(vyf.abs());
    if mv == 0.0 {
        return (x, y);
    }
    let (dx, dy, theta, mtheta, etheta) = lorentz_polar_scaled(vxf, vyf);
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

    let m = xf.abs().max(yf.abs());
    let (nx, ny) = if m == 0.0 {
        let s = eff_theta.sinh();
        (s * dx, s * dy)
    } else {
        // Unified polar boost decomposition `(r_par, r_perp)` along `(u, u_perp)` prevents inward
        // `c * x - s * w` cancellation across all `m > 0` and `(1 + cos_phi)` underflow.
        let (ux, uy, r, mr, er) = lorentz_polar_scaled(xf, yf);
        let sx = xf / m;
        let sy = yf / m;
        let sr = sx.hypot(sy);
        let cos_phi = (ux * dx + uy * dy).clamp(-1.0, 1.0);
        let (det_m, det_exp) = lorentz_scaled_cross(xf, yf, vxf, vyf);
        let c = eff_theta.cosh();
        let s = eff_theta.sinh();
        let (sin_phi, r_perp) = if det_m == 0.0 {
            (0.0, 0.0)
        } else {
            let sp = scale_pow2(det_m / (mr * mtheta), det_exp - er - etheta).clamp(-1.0, 1.0);
            let (ms, es) = frexp_norm(s);
            let rp = scale_pow2((det_m * ms) / (mr * mtheta), det_exp + es - er - etheta);
            (sp, rp)
        };

        if cos_phi >= 0.0 {
            let t = if m >= LARGE_RADIUS_THRESHOLD {
                r
            } else {
                1.0_f64.hypot(r)
            };
            let r_par = r * c + (t * cos_phi) * s;
            if r_par.is_finite() {
                (r_par * ux - r_perp * uy, r_par * uy + r_perp * ux)
            } else {
                let t_over_r = if r.is_finite() { t / r } else { 1.0 };
                let k = c + (t_over_r * cos_phi) * s;
                (xf * k - r_perp * uy, yf * k + r_perp * ux)
            }
        } else {
            let one_minus_cos = 1.0 - cos_phi;
            let term1 = if r.is_finite() {
                r * (-eff_theta).exp()
            } else {
                f64::INFINITY
            };
            let term2 = if r.is_finite() {
                (r * sin_phi) * (r_perp / one_minus_cos)
            } else {
                f64::INFINITY
            };
            let term3 = if m >= LARGE_RADIUS_THRESHOLD {
                ((0.5 * cos_phi) * (s / m)) / sr
            } else {
                let t = 1.0_f64.hypot(r);
                (cos_phi * s) / (t + r)
            };
            let r_par = term1 + term2 + term3;
            if r_par.is_finite() {
                (r_par * ux - r_perp * uy, r_par * uy + r_perp * ux)
            } else {
                let t1_over_sr = m * (-eff_theta).exp();
                let t2_over_sr = (m * sin_phi) * (r_perp / one_minus_cos);
                let t3_over_sr = if m >= LARGE_RADIUS_THRESHOLD {
                    ((0.5 * cos_phi) * (s / m)) / (sr * sr)
                } else {
                    let t = 1.0_f64.hypot(r);
                    (cos_phi * s) / ((t + r) * sr)
                };
                let r_par_over_sr = t1_over_sr + t2_over_sr + t3_over_sr;
                (
                    r_par_over_sr * sx - r_perp * uy,
                    r_par_over_sr * sy + r_perp * ux,
                )
            }
        }
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
