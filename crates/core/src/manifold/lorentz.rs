use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

use super::poincare::clamp_poincare_point;

/// Computes the implicit time coordinate $t = \sqrt{1 + x^2 + y^2}$ on the upper sheet of $\mathbb{H}^2$.
#[inline]
fn lorentz_time_and_radius(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    let t = 1.0_f64.hypot(r);
    (t, r)
}

/// Computes `eta - 1 = (-<u, v>_L) - 1 >= 0` stably without catastrophic cancellation for both
/// near-coincident points and large hyperbolic radii (`hyp_r > 19`).
#[inline]
fn lorentz_eta_minus_one(x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64, f64, f64) {
    let (t1, r1) = lorentz_time_and_radius(x1, y1);
    let (t2, r2) = lorentz_time_and_radius(x2, y2);

    if r1 == 0.0 && r2 == 0.0 {
        return (0.0, t1, r1, t2, r2);
    }
    if r1 == 0.0 {
        return (t2 - 1.0, t1, r1, t2, r2);
    }
    if r2 == 0.0 {
        return (t1 - 1.0, t1, r1, t2, r2);
    }

    // Factor t1*t2 - r1*r2 = (1 + r1^2 + r2^2) / (t1*t2 + r1*r2) and
    // t1*t2 - r1*r2 - 1 = u^2 / (t1*t2 - r1*r2 + 1) where u = (r1 - r2)*(r1 + r2) / (r1*t2 + r2*t1).
    let radial_den = r1 * t2 + r2 * t1;
    let radial_term = if radial_den > 0.0 && radial_den.is_finite() {
        let u = ((r1 - r2) * (r1 + r2)) / radial_den;
        let t1t2_minus_r1r2 = (1.0 + r1 * r1 + r2 * r2) / (t1 * t2 + r1 * r2);
        (u * u) / (t1t2_minus_r1r2 + 1.0)
    } else {
        // Fallback for extreme radii where r^2 would overflow f64: use log-ratio approximation.
        let ratio = if r1 >= r2 { r1 / r2 } else { r2 / r1 };
        let half_cosh = 0.5 * (ratio + ratio.recip());
        (half_cosh - 1.0).max(0.0)
    };

    // Compute r1*r2*(1 - cos(theta)) stably using half-angle / cross-product identity.
    let ux1 = x1 / r1;
    let uy1 = y1 / r1;
    let ux2 = x2 / r2;
    let uy2 = y2 / r2;
    let cos_theta = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let one_minus_cos = if cos_theta >= 0.0 {
        let sin_theta = ux1 * uy2 - uy1 * ux2;
        (sin_theta * sin_theta) / (1.0 + cos_theta)
    } else {
        1.0 - cos_theta
    };

    let angular_term = r1 * r2 * one_minus_cos;
    ((radial_term + angular_term).max(0.0), t1, r1, t2, r2)
}

/// Computes the unit hyperbolic geodesic distance between spatial hyperboloid coordinates
/// `(x1, y1)` and `(x2, y2)` in $\mathbb{R}^2$.
///
/// Uses a factored `arcsinh` formulation that avoids floating-point cancellation both near
/// zero distance and at large hyperbolic radii ($r_{\mathbb{H}} > 19$).
pub fn lorentz_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let x1f = x1.to_f64();
    let y1f = y1.to_f64();
    let x2f = x2.to_f64();
    let y2f = y2.to_f64();
    if !x1f.is_finite() || !y1f.is_finite() || !x2f.is_finite() || !y2f.is_finite() {
        return S::ZERO;
    }

    let (_, r1) = lorentz_time_and_radius(x1f, y1f);
    let (_, r2) = lorentz_time_and_radius(x2f, y2f);
    if r1 == 0.0 {
        return S::from_f64(r2.asinh());
    }
    if r2 == 0.0 {
        return S::from_f64(r1.asinh());
    }

    let (eta_m1, _, _, _, _) = lorentz_eta_minus_one(x1f, y1f, x2f, y2f);
    if !eta_m1.is_finite() {
        // Extreme radius fallback via logarithmic polar asymptotic.
        let ux1 = x1f / r1;
        let uy1 = y1f / r1;
        let ux2 = x2f / r2;
        let uy2 = y2f / r2;
        let cos_t = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
        let sin_half = ((1.0 - cos_t) * 0.5).max(0.0).sqrt();
        let d = if sin_half > 0.0 {
            r1.ln() + r2.ln() + 2.0 * (2.0 * sin_half).ln()
        } else {
            (r1 / r2).ln().abs()
        };
        return S::from_f64(d.max(0.0));
    }
    let dist = 2.0 * (0.5 * eta_m1).sqrt().asinh();
    S::from_f64(dist)
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

    let (eta_m1, t1, r1, t2, r2) = lorentz_eta_minus_one(x1f, y1f, x2f, y2f);
    let dist = if eta_m1.is_finite() {
        2.0 * (0.5 * eta_m1).sqrt().asinh()
    } else {
        lorentz_distance(x1, y1, x2, y2).to_f64()
    };
    if dist <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }

    // Compute <v, e_a>_L = (x2 - x1) - (x1 / (1 + t1)) * ((t2 - t1) + (eta - 1)) stably.
    let t2_minus_t1 = if t1 + t2 > 0.0 {
        ((r2 - r1) * (r2 + r1)) / (t2 + t1)
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

    let (t, _) = lorentz_time_and_radius(xf, yf);
    let s_dot = (xf * dx + yf * dy) / (1.0 + t);
    let wx = dx + xf * s_dot;
    let wy = dy + yf * s_dot;

    let c = eff_theta.cosh();
    let s = eff_theta.sinh();
    let nx = c * xf + s * wx;
    let ny = c * yf + s * wy;
    let rx = S::from_f64(nx);
    let ry = S::from_f64(ny);
    if rx.is_finite() && ry.is_finite() {
        (rx, ry)
    } else {
        (x, y)
    }
}

/// Computes the hyperbolic Einstein midpoint of spatial hyperboloid points `(xs, ys)`.
pub fn lorentz_einstein_midpoint<S: Scalar>(xs: &[S], ys: &[S]) -> CoreResult<(S, S)> {
    if xs.len() != ys.len() {
        return Err(CoreError::dimension_mismatch(
            xs.len(),
            ys.len(),
            "lorentz_einstein_midpoint::ys",
        ));
    }
    if xs.is_empty() {
        return Ok((S::ZERO, S::ZERO));
    }

    let mut sum_t = 0.0_f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_r = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err(CoreError::non_finite("lorentz_einstein_midpoint"));
        }
        let xf = x.to_f64();
        let yf = y.to_f64();
        let (t, r) = lorentz_time_and_radius(xf, yf);
        sum_t += t;
        sum_x += xf;
        sum_y += yf;
        sum_r += r;
    }

    let inv_n = 1.0 / (xs.len() as f64);
    let t_avg = sum_t * inv_n;
    let x_avg = sum_x * inv_n;
    let y_avg = sum_y * inv_n;
    let r_avg_sum = sum_r * inv_n;
    let r_vec = x_avg.hypot(y_avg);

    // Since t_i - r_i = 1/(t_i + r_i), accumulate t_avg - r_vec without large-t cancellation.
    let mut sum_t_minus_r = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (t, r) = lorentz_time_and_radius(x.to_f64(), y.to_f64());
        sum_t_minus_r += 1.0 / (t + r);
    }
    let t_minus_r = (sum_t_minus_r * inv_n) + (r_avg_sum - r_vec).max(0.0);
    let minkowski_norm_sq = (t_minus_r * (t_avg + r_vec)).max(1.0);
    let norm = minkowski_norm_sq.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Ok((S::ZERO, S::ZERO));
    }
    Ok((S::from_f64(x_avg / norm), S::from_f64(y_avg / norm)))
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
    let (t, _) = lorentz_time_and_radius(xf, yf);
    let denom = 1.0 + t;
    clamp_poincare_point(S::from_f64(xf / denom), S::from_f64(yf / denom))
}
