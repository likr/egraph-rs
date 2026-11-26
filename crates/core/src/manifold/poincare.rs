use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

/// Maximum radius enforced inside the open Poincaré unit disk to keep `1 - ||p||^2 > 0`.
const POINCARE_EPS_F64: f64 = 1e-5;
const POINCARE_EPS_F32: f64 = 1e-4;

#[inline]
pub(super) fn poincare_max_radius<S: Scalar>() -> f64 {
    if std::mem::size_of::<S>() == 4 {
        1.0 - POINCARE_EPS_F32
    } else {
        1.0 - POINCARE_EPS_F64
    }
}

/// Clamps a point `(x, y)` strictly inside the open Poincaré unit disk (`||p|| <= 1 - eps`).
#[inline]
pub fn clamp_poincare_point<S: Scalar>(x: S, y: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    if !xf.is_finite() || !yf.is_finite() {
        return (S::ZERO, S::ZERO);
    }
    let max_r = poincare_max_radius::<S>();
    let r = xf.hypot(yf);
    if r > max_r {
        (S::from_f64((xf / r) * max_r), S::from_f64((yf / r) * max_r))
    } else {
        (x, y)
    }
}

/// Computes the Poincaré disk Möbius (gyrovector) addition $a \oplus_M b = \frac{a + b}{1 + \bar{a} b}$
/// in complex disk coordinates, clamping inputs and outputs inside the open unit disk.
pub fn poincare_mobius_add<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let (cx1, cy1) = clamp_poincare_point(x1, y1);
    let (cx2, cy2) = clamp_poincare_point(x2, y2);
    let ax = cx1.to_f64();
    let ay = cy1.to_f64();
    let bx = cx2.to_f64();
    let by = cy2.to_f64();

    let num_x = ax + bx;
    let num_y = ay + by;
    let den_r = 1.0 + ax * bx + ay * by;
    let den_i = ax * by - ay * bx;
    let den_sq = den_r * den_r + den_i * den_i;
    if den_sq <= 1e-24 {
        return (S::ZERO, S::ZERO);
    }
    let rx = (num_x * den_r + num_y * den_i) / den_sq;
    let ry = (num_y * den_r - num_x * den_i) / den_sq;
    clamp_poincare_point(S::from_f64(rx), S::from_f64(ry))
}

/// Computes the closed-form Möbius-equivariant hyperbolic centroid (Einstein midpoint via the
/// Lorentz hyperboloid lift) of Poincaré disk points `(xs, ys)`.
pub fn poincare_einstein_midpoint<S: Scalar>(xs: &[S], ys: &[S]) -> CoreResult<(S, S)> {
    if xs.len() != ys.len() {
        return Err(CoreError::dimension_mismatch(
            xs.len(),
            ys.len(),
            "poincare_einstein_midpoint::ys",
        ));
    }
    if xs.is_empty() {
        return Ok((S::ZERO, S::ZERO));
    }

    let mut sum_t = 0.0_f64;
    let mut sum_sx = 0.0_f64;
    let mut sum_sy = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err(CoreError::non_finite("poincare_einstein_midpoint"));
        }
        let (cx, cy) = clamp_poincare_point(x, y);
        let px = cx.to_f64();
        let py = cy.to_f64();
        let r2 = px * px + py * py;
        let denom = (1.0 - r2).max(1e-18);
        sum_t += (1.0 + r2) / denom;
        sum_sx += (2.0 * px) / denom;
        sum_sy += (2.0 * py) / denom;
    }

    let inv_n = 1.0 / (xs.len() as f64);
    let t_avg = sum_t * inv_n;
    let sx_avg = sum_sx * inv_n;
    let sy_avg = sum_sy * inv_n;
    let s_norm2 = sx_avg * sx_avg + sy_avg * sy_avg;
    let t0 = (t_avg * t_avg - s_norm2).max(1.0).sqrt();
    let denom_back = t_avg + t0;
    if denom_back <= 0.0 || !denom_back.is_finite() {
        return Ok((S::ZERO, S::ZERO));
    }
    Ok(clamp_poincare_point(
        S::from_f64(sx_avg / denom_back),
        S::from_f64(sy_avg / denom_back),
    ))
}

/// Computes the Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
pub fn poincare_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let (u0, u1) = clamp_poincare_point(x1, y1);
    let (v0, v1) = clamp_poincare_point(x2, y2);
    let ux = u0.to_f64();
    let uy = u1.to_f64();
    let vx = v0.to_f64();
    let vy = v1.to_f64();

    let dx = ux - vx;
    let dy = uy - vy;
    let diff_sq = dx * dx + dy * dy;
    let denom = (1.0 - (ux * ux + uy * uy)) * (1.0 - (vx * vx + vy * vy));
    let arg = (1.0 + 2.0 * diff_sq / denom.max(1e-18)).max(1.0);
    S::from_f64(arg.acosh())
}

/// Computes the Poincaré disk logarithmic map $\log_x(y)$ in the tangent space at `x = (x1, y1)`.
///
/// The returned vector points toward `y = (x2, y2)` via the Möbius translation $-x \oplus y$
/// and has Euclidean norm equal to [`poincare_distance`].
pub fn poincare_log_map<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let (cx1, cy1) = clamp_poincare_point(x1, y1);
    let (cx2, cy2) = clamp_poincare_point(x2, y2);
    let xx = cx1.to_f64();
    let xy = cy1.to_f64();
    let yx = cx2.to_f64();
    let yy = cy2.to_f64();

    let dx = yx - xx;
    let dy = yy - xy;
    let dr = 1.0 - xx * yx - xy * yy;
    let di = xy * yx - xx * yy;
    let denom = dr * dr + di * di;
    if denom <= 1e-24 {
        return (S::ZERO, S::ZERO);
    }
    let zx = (dr * dx + di * dy) / denom;
    let zy = (dr * dy - di * dx) / denom;
    let z_norm = zx.hypot(zy);
    if z_norm <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }
    let z_clamped = z_norm.min(1.0 - 1e-15);
    let dist = 2.0 * z_clamped.atanh();
    let scale = dist / z_norm;
    (S::from_f64(zx * scale), S::from_f64(zy * scale))
}

/// Computes the Poincaré disk exponential map $\exp_x(v)$ advancing `x = (x, y)` by tangent
/// vector `v = (vx, vy)` and clamping the result inside the open unit disk.
pub fn poincare_exp_map<S: Scalar>(x: S, y: S, vx: S, vy: S) -> (S, S) {
    let (cx, cy) = clamp_poincare_point(x, y);
    let xx = cx.to_f64();
    let xy = cy.to_f64();
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();

    let v_norm = vxf.hypot(vyf);
    if v_norm <= 1e-15 {
        return (cx, cy);
    }
    let scale = (0.5 * v_norm).tanh() / v_norm;
    let ux = vxf * scale;
    let uy = vyf * scale;

    // Complex Mobius addition (x + u) / (1 + conj(x) * u), exact inverse of `poincare_log_map`.
    let num_x = xx + ux;
    let num_y = xy + uy;
    let den_r = 1.0 + xx * ux + xy * uy;
    let den_i = xx * uy - xy * ux;
    let den_sq = den_r * den_r + den_i * den_i;
    if den_sq <= 1e-24 {
        return clamp_poincare_point(S::from_f64(ux), S::from_f64(uy));
    }
    let rx = (num_x * den_r + num_y * den_i) / den_sq;
    let ry = (num_y * den_r - num_x * den_i) / den_sq;
    clamp_poincare_point(S::from_f64(rx), S::from_f64(ry))
}
