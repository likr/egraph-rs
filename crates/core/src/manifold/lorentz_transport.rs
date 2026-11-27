use crate::scalar::Scalar;

use super::lorentz::{
    LARGE_RADIUS_THRESHOLD, frexp_norm, lorentz_polar_scaled, lorentz_scaled_cross, scale_pow2,
};

pub(super) fn lorentz_step_and_transport<S: Scalar>(
    x: S,
    y: S,
    d0: f64,
    d1: f64,
    vec_opt: Option<&mut [f64]>,
    compute_q: bool,
) -> (S, S) {
    let mv = d0.abs().max(d1.abs());
    if mv == 0.0 {
        return (x, y);
    }
    let (dx, dy, theta, mtheta, etheta) = lorentz_polar_scaled(d0, d1);
    if theta <= 1e-15 {
        return (x, y);
    }
    let max_theta = if std::mem::size_of::<S>() == 4 {
        80.0
    } else {
        700.0
    };
    let eff_theta = theta.min(max_theta);

    let xf = x.to_f64();
    let yf = y.to_f64();
    let m = xf.abs().max(yf.abs());
    if m == 0.0 {
        if !compute_q {
            return (x, y);
        }
        let s = eff_theta.sinh();
        let rx = S::from_f64(s * dx);
        let ry = S::from_f64(s * dy);
        return if rx.is_finite() && ry.is_finite() {
            (rx, ry)
        } else {
            (x, y)
        };
    }

    let (ux, uy, r, mr, er) = lorentz_polar_scaled(xf, yf);
    let cos_phi = (ux * dx + uy * dy).clamp(-1.0, 1.0);
    let (det_m, det_exp) = lorentz_scaled_cross(xf, yf, d0, d1);
    let sin_phi = if det_m == 0.0 {
        0.0
    } else {
        scale_pow2(det_m / (mr * mtheta), det_exp - er - etheta).clamp(-1.0, 1.0)
    };

    if let Some(vec) = vec_opt {
        let tau_p = if er < -50 {
            scale_pow2(0.5 * mr, er)
        } else if m < LARGE_RADIUS_THRESHOLD {
            r / (1.0 + 1.0_f64.hypot(r))
        } else {
            1.0
        };
        let tau_d = if etheta < -50 {
            scale_pow2(0.5 * mtheta, etheta)
        } else if eff_theta < 20.0 {
            (0.5 * eff_theta).tanh()
        } else {
            1.0
        };
        let rho = tau_p * tau_d;
        let a_i = -rho * sin_phi;
        let a_r = if cos_phi >= 0.0 {
            1.0 + rho * cos_phi
        } else {
            let one_minus_tau_p = if m >= LARGE_RADIUS_THRESHOLD {
                if r.is_finite() {
                    1.0 / r
                } else {
                    scale_pow2(1.0 / mr, -er)
                }
            } else {
                let t = 1.0_f64.hypot(r);
                (1.0 + 1.0 / (t + r)) / (1.0 + t)
            };
            let exp_neg = (-eff_theta).exp();
            let one_minus_tau_d = (2.0 * exp_neg) / (1.0 + exp_neg);
            let one_minus_rho = one_minus_tau_p + tau_p * one_minus_tau_d;
            one_minus_rho + rho * (sin_phi * sin_phi) / (1.0 - cos_phi)
        };
        let m_a = a_r.abs().max(a_i.abs());
        if m_a > 0.0 && m_a.is_finite() {
            let sa_r = a_r / m_a;
            let sa_i = a_i / m_a;
            let denom = sa_r * sa_r + sa_i * sa_i;
            let cos_psi = (sa_r * sa_r - sa_i * sa_i) / denom;
            let sin_psi = (2.0 * sa_r * sa_i) / denom;
            let w0 = vec[0];
            let w1 = vec[1];
            vec[0] = cos_psi * w0 - sin_psi * w1;
            vec[1] = sin_psi * w0 + cos_psi * w1;
        }
    }

    if !compute_q {
        return (x, y);
    }

    let c = eff_theta.cosh();
    let s = eff_theta.sinh();
    let r_perp = if det_m == 0.0 {
        0.0
    } else {
        let (ms, es) = frexp_norm(s);
        scale_pow2((det_m * ms) / (mr * mtheta), det_exp + es - er - etheta)
    };
    let (nx, ny) = if cos_phi >= 0.0 {
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
        let sx = xf / m;
        let sy = yf / m;
        let sr = sx.hypot(sy);
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
    };
    let rx = S::from_f64(nx);
    let ry = S::from_f64(ny);
    if rx.is_finite() && ry.is_finite() {
        (rx, ry)
    } else {
        (x, y)
    }
}
