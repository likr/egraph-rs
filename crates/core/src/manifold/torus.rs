use crate::scalar::Scalar;

/// Wraps a scalar coordinate into the half-open unit torus interval `[0, 1)`.
#[inline]
pub fn wrap_torus_coord<S: Scalar>(value: S) -> S {
    let f = value.to_f64();
    if !f.is_finite() {
        return S::ZERO;
    }
    let w = f.rem_euclid(1.0);
    let wrapped = if w >= 1.0 || w == 0.0 { 0.0 } else { w };
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

/// Finds the optimal 1D circular cut `c in [0, 1)` (Chen et al. 2021) minimizing the number
/// of boundary-wrapped edges and breaking ties by maximizing the empty gap width.
pub(super) fn optimal_torus_cut_1d<S: Scalar>(coords: &[S], edges: &[(u32, u32)]) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }
    let wrapped: Vec<f64> = coords
        .iter()
        .map(|&c| {
            let w = wrap_torus_coord(c).to_f64();
            if w == 0.0 { 0.0 } else { w }
        })
        .collect();

    let mut unique = wrapped.clone();
    unique.sort_unstable_by(f64::total_cmp);
    unique.dedup();
    let m = unique.len();
    if m <= 1 {
        return (unique[0] + 0.5).rem_euclid(1.0);
    }

    let mut diff = vec![0_isize; m];
    for &(u, v) in edges {
        let p = wrapped[u as usize];
        let q = wrapped[v as usize];
        let (a, b) = if p <= q { (p, q) } else { (q, p) };
        let span = b - a;
        if span == 0.0 {
            continue;
        }
        let ia = unique
            .binary_search_by(|val| val.total_cmp(&a))
            .unwrap_or_else(|i| i.min(m - 1));
        let ib = unique
            .binary_search_by(|val| val.total_cmp(&b))
            .unwrap_or_else(|i| i.min(m - 1));
        if ia == ib {
            continue;
        }
        if span <= 0.5 {
            diff[ia] += 1;
            diff[ib] -= 1;
        } else {
            diff[0] += 1;
            diff[ia] -= 1;
            diff[ib] += 1;
        }
    }

    let mut cur = 0_isize;
    let mut best_wrap = isize::MAX;
    let mut best_width = -1.0_f64;
    let mut best_cut = 0.0_f64;
    for k in 0..m {
        cur += diff[k];
        let width = if k + 1 < m {
            unique[k + 1] - unique[k]
        } else {
            1.0 + unique[0] - unique[m - 1]
        };
        if cur < best_wrap || (cur == best_wrap && width > best_width) {
            best_wrap = cur;
            best_width = width;
            best_cut = (unique[k] + 0.5 * width).rem_euclid(1.0);
        }
    }
    if best_cut >= 1.0 || best_cut == 0.0 {
        0.0
    } else {
        best_cut
    }
}
