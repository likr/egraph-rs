use crate::scalar::Scalar;

/// Wraps a scalar coordinate into the half-open unit torus interval `[0, 1)`.
#[inline]
pub fn wrap_torus_coord<S: Scalar>(value: S) -> S {
    let f = value.to_f64();
    if !f.is_finite() {
        return S::ZERO;
    }
    let w = f.rem_euclid(1.0);
    let wrapped = if w >= 1.0 { 0.0 } else { w };
    let res = S::from_f64(wrapped);
    // Guard f32 rounding where values like `-1e-9` can round up to `1.0_f32`.
    if res >= S::ONE { S::ZERO } else { res }
}

/// Wraps a 2D coordinate pair onto the flat unit torus `[0, 1)^2`.
#[inline]
pub fn wrap_torus_point<S: Scalar>(x: S, y: S) -> (S, S) {
    (wrap_torus_coord(x), wrap_torus_coord(y))
}

/// Computes the shortest signed displacement vector `(dx, dy)` in `[-0.5, 0.5]^2`
/// from `(x2, y2)` to `(x1, y1)` on the unit torus `[0, 1)^2`.
#[inline]
pub fn torus_delta<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let dx = wrap_1d_signed_delta(wrap_torus_coord(x1).to_f64() - wrap_torus_coord(x2).to_f64());
    let dy = wrap_1d_signed_delta(wrap_torus_coord(y1).to_f64() - wrap_torus_coord(y2).to_f64());
    (S::from_f64(dx), S::from_f64(dy))
}

#[inline]
fn wrap_1d_signed_delta(mut d: f64) -> f64 {
    if d > 0.5 {
        d -= 1.0;
    } else if d < -0.5 {
        d += 1.0;
    }
    d
}

/// Computes the geodesic distance on the flat unit torus `[0, 1)^2`.
#[inline]
pub fn torus_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let (dx, dy) = torus_delta(x1, y1, x2, y2);
    dx.hypot(dy)
}
