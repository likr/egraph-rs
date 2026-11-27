use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

use super::lorentz::{LARGE_RADIUS_THRESHOLD, lorentz_time_and_radius};

#[inline]
fn to_finite_scalar<S: Scalar>(v: f64, sign_source: f64) -> S {
    let s_max = if std::mem::size_of::<S>() == 4 {
        f32::MAX as f64
    } else {
        f64::MAX
    };
    let clamped = if v.is_finite() {
        v.clamp(-s_max, s_max)
    } else {
        sign_source.signum() * s_max
    };
    S::from_f64(clamped)
}

/// Computes the hyperbolic Einstein midpoint of spatial hyperboloid points `(xs, ys)`.
///
/// Uses an `m_max`-scaled, cancellation-free angular/radial decomposition so results remain
/// accurate and overflow-/underflow-free across all finite IEEE-754 inputs up to `(f64::MAX, f64::MAX)`.
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

    let mut m_max = 0.0_f64;
    let mut all_equal = true;
    let x0 = xs[0];
    let y0 = ys[0];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err(CoreError::non_finite("lorentz_einstein_midpoint"));
        }
        if x != x0 || y != y0 {
            all_equal = false;
        }
        m_max = m_max.max(x.to_f64().abs()).max(y.to_f64().abs());
    }
    if m_max == 0.0 {
        return Ok((S::ZERO, S::ZERO));
    }
    if all_equal {
        return Ok((x0, y0));
    }

    let inv_n = 1.0 / (xs.len() as f64);

    // Small/unit regime (`m_max <= 1.0`): avoid `1.0 / m_max` overflow on subnormal/tiny inputs.
    if m_max <= 1.0 {
        let mut x_avg = 0.0_f64;
        let mut y_avg = 0.0_f64;
        let mut t_sum = 0.0_f64;
        let mut r_sum = 0.0_f64;
        let mut t_minus_r_sum = 0.0_f64;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let xf = x.to_f64();
            let yf = y.to_f64();
            let (t, r) = lorentz_time_and_radius(xf, yf);
            x_avg += xf * inv_n;
            y_avg += yf * inv_n;
            t_sum += t;
            r_sum += r;
            t_minus_r_sum += 1.0 / (t + r);
        }
        let t_avg = t_sum * inv_n;
        let r_avg = r_sum * inv_n;
        let t_minus_r_avg = t_minus_r_sum * inv_n;
        let r_vec = x_avg.hypot(y_avg);
        let t_minus_r = t_minus_r_avg + (r_avg - r_vec).max(0.0);
        let norm = (t_minus_r * (t_avg + r_vec)).max(1.0).sqrt();
        return Ok((
            to_finite_scalar(x_avg / norm, x_avg),
            to_finite_scalar(y_avg / norm, y_avg),
        ));
    }

    // Scaled regime (`m_max > 1.0`): `inv_m = 1 / m_max <= 1.0` keeps `st`, `sr`, and `1/(st+sr)`
    // bounded without overflowing `hypot(x, y)` or `norm_sq`.
    let inv_m = 1.0 / m_max;
    let mut x_avg = 0.0_f64;
    let mut y_avg = 0.0_f64;
    let mut st_sum = 0.0_f64;
    let mut inv_st_plus_sr_avg = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let xf = x.to_f64();
        let yf = y.to_f64();
        x_avg += xf * inv_n;
        y_avg += yf * inv_n;
        let sx = xf / m_max;
        let sy = yf / m_max;
        let sr = sx.hypot(sy);
        let st = inv_m.hypot(sr);
        st_sum += st;
        // Mathematically `m_max / (t_i + r_i) <= m_max`; clamp to `m_max` to prevent `1 / inv_m`
        // overflow when `inv_m` rounds down at `f64::MAX`.
        let inv_st_plus_sr = (1.0 / (st + sr)).min(m_max);
        inv_st_plus_sr_avg += inv_st_plus_sr * inv_n;
    }

    let sx_avg = x_avg / m_max;
    let sy_avg = y_avg / m_max;
    let st_avg = st_sum * inv_n;
    let sr_vec = sx_avg.hypot(sy_avg);
    if sr_vec == 0.0 {
        return Ok((S::ZERO, S::ZERO));
    }

    let ux_vec = sx_avg / sr_vec;
    let uy_vec = sy_avg / sr_vec;
    let st_plus_sr_vec = st_avg + sr_vec;

    // `q_small = m_max * p_small` where `m_max` cancels algebraically inside `1 / (st + sr)`.
    let q_small = inv_st_plus_sr_avg * st_plus_sr_vec;

    // Cancellation-free angular discrepancy: `sr_avg - sr_vec = (1 / n) * sum_i sr_i * (1 - cos(theta_i))`.
    let mut p_ang_unscaled = 0.0_f64;
    let mut tiny_sin_max = 0.0_f64;
    let mut tiny_sin_sq_norm = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let sx = x.to_f64() / m_max;
        let sy = y.to_f64() / m_max;
        let sr = sx.hypot(sy);
        if sr == 0.0 {
            continue;
        }
        let ux = sx / sr;
        let uy = sy / sr;
        let cos_theta = (ux * ux_vec + uy * uy_vec).clamp(-1.0, 1.0);
        if cos_theta < 0.0 {
            p_ang_unscaled += sr * (1.0 - cos_theta) * inv_n;
        } else {
            let sin_theta = (ux * uy_vec - uy * ux_vec).abs();
            if sin_theta >= 1e-150 {
                p_ang_unscaled += sr * ((sin_theta * sin_theta) / (1.0 + cos_theta)) * inv_n;
            } else if sin_theta > 0.0 && m_max >= LARGE_RADIUS_THRESHOLD {
                // Keep `m_max * sin_theta <= f64::MAX * 1e-150 ~ 1.8e158` overflow-safe via max scaling.
                let term = (m_max * sin_theta) * (sr / (1.0 + cos_theta)).sqrt();
                if term > tiny_sin_max {
                    let ratio = tiny_sin_max / term;
                    tiny_sin_sq_norm = tiny_sin_sq_norm * (ratio * ratio) + 1.0;
                    tiny_sin_max = term;
                } else {
                    let ratio = term / tiny_sin_max;
                    tiny_sin_sq_norm += ratio * ratio;
                }
            }
        }
    }

    let p_ang = p_ang_unscaled * st_plus_sr_vec;
    let rms_ang_tiny = if tiny_sin_max > 0.0 {
        tiny_sin_max * (tiny_sin_sq_norm * inv_n * st_plus_sr_vec).sqrt()
    } else {
        0.0
    };

    if m_max <= LARGE_RADIUS_THRESHOLD {
        let norm = (q_small + (m_max * m_max) * p_ang).max(1.0).sqrt();
        return Ok((
            to_finite_scalar(x_avg / norm, x_avg),
            to_finite_scalar(y_avg / norm, y_avg),
        ));
    }

    if p_ang == 0.0 && rms_ang_tiny == 0.0 {
        let norm = q_small.max(1.0).sqrt();
        return Ok((
            to_finite_scalar(x_avg / norm, x_avg),
            to_finite_scalar(y_avg / norm, y_avg),
        ));
    }

    let m_p_ang = m_max * p_ang;
    if p_ang > 0.0 && (!m_p_ang.is_finite() || m_p_ang > q_small.max(1.0) * 1e16) {
        let s_norm = p_ang.sqrt();
        return Ok((
            to_finite_scalar(sx_avg / s_norm, x_avg),
            to_finite_scalar(sy_avg / s_norm, y_avg),
        ));
    }

    if q_small <= 1e300 && m_p_ang <= 1e-160 && rms_ang_tiny <= 1e140 {
        let norm = (q_small + m_max * m_p_ang + rms_ang_tiny * rms_ang_tiny)
            .max(1.0)
            .sqrt();
        return Ok((
            to_finite_scalar(x_avg / norm, x_avg),
            to_finite_scalar(y_avg / norm, y_avg),
        ));
    }

    let denom_sq = (q_small.max(1.0) / m_max) + m_p_ang + (rms_ang_tiny / m_max) * rms_ang_tiny;
    let denom = denom_sq.sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return Ok((
            to_finite_scalar(x_avg, x_avg),
            to_finite_scalar(y_avg, y_avg),
        ));
    }
    let scale_num = m_max.sqrt();
    Ok((
        to_finite_scalar((x_avg / scale_num) / denom, x_avg),
        to_finite_scalar((y_avg / scale_num) / denom, y_avg),
    ))
}
