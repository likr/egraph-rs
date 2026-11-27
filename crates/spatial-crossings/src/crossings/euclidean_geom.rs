use std::cmp::Ordering;

use super::exact_limbs::{ExactFactor, exact_deg2_sign};
use super::geometry::{STEEP_SLOPE_THRESHOLD, Seg, seg_y_err};
use super::torus_exact::{cmp_post_x_exact_f64, intersection_x_dd_f64, segs_intersect_exact_f64};

#[inline]
fn is_tiny_nonzero(v: f64) -> bool {
    v != 0.0 && v.abs() < 1e-150
}

#[inline]
pub(super) fn same_sign_nonzero(a: f64, b: f64) -> bool {
    (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)
}

#[inline]
pub(super) fn segments_geometrically_intersect(a: &Seg, b: &Seg) -> bool {
    if a.x_max < b.x_min || b.x_max < a.x_min || a.y_max < b.y_min || b.y_max < a.y_min {
        return false;
    }
    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = dx1 * (b.y1 - a.y1) - dy1 * (b.x1 - a.x1);
    let t1 = dx1 * (b.y2 - a.y1) - dy1 * (b.x2 - a.x1);
    if same_sign_nonzero(s1, t1) {
        return false;
    }

    let s2 = dx2 * (a.y1 - b.y1) - dy2 * (a.x1 - b.x1);
    let t2 = dx2 * (a.y2 - b.y1) - dy2 * (a.x2 - b.x1);
    if same_sign_nonzero(s2, t2) {
        return false;
    }

    if (s1 == 0.0 || t1 == 0.0 || s2 == 0.0 || t2 == 0.0)
        && (is_tiny_nonzero(dx1)
            || is_tiny_nonzero(dy1)
            || is_tiny_nonzero(dx2)
            || is_tiny_nonzero(dy2)
            || !s1.is_finite()
            || !t1.is_finite()
            || !s2.is_finite()
            || !t2.is_finite())
    {
        return segs_intersect_exact_f64([a.x1, a.y1, a.x2, a.y2], [b.x1, b.y1, b.x2, b.y2]);
    }
    true
}

#[inline]
pub(super) fn segments_cross(a: &Seg, b: &Seg) -> bool {
    if a.shares_endpoint(b) {
        return false;
    }
    segments_geometrically_intersect(a, b)
}

#[inline]
pub(super) fn cmp_slope_euclidean(a: &Seg, b: &Seg) -> Ordering {
    if a.slope.abs() < STEEP_SLOPE_THRESHOLD
        && b.slope.abs() < STEEP_SLOPE_THRESHOLD
        && a.slope != b.slope
    {
        return a.slope.total_cmp(&b.slope);
    }
    exact_deg2_sign(&[
        (
            ExactFactor::from_pair(a.y2, a.y1),
            ExactFactor::from_pair(b.x2, b.x1),
            1,
        ),
        (
            ExactFactor::from_pair(b.y2, b.y1),
            ExactFactor::from_pair(a.x2, a.x1),
            -1,
        ),
    ])
}

#[inline]
pub(super) fn intersection_x(a: &Seg, b: &Seg) -> Option<f64> {
    let dx_a = a.x2 - a.x1;
    let dx_b = b.x2 - b.x1;
    let (p, q) = if dx_a < dx_b || (dx_a == dx_b && a.id <= b.id) {
        (a, b)
    } else {
        (b, a)
    };
    let dx1 = p.x2 - p.x1;
    let dy1 = p.y2 - p.y1;
    let dx2 = q.x2 - q.x1;
    let dy2 = q.y2 - q.y1;
    let det = dx1 * dy2 - dy1 * dx2;
    let lo = p.x_min.max(q.x_min);
    let hi = p.x_max.min(q.x_max);
    if !det.is_finite()
        || (det != 0.0 && det.abs() <= 1e-260)
        || is_tiny_nonzero(dx1)
        || is_tiny_nonzero(dy1)
        || is_tiny_nonzero(dx2)
        || is_tiny_nonzero(dy2)
    {
        let max_coord =
            p.x1.abs()
                .max(p.x2.abs())
                .max(p.y1.abs())
                .max(p.y2.abs())
                .max(q.x1.abs())
                .max(q.x2.abs())
                .max(q.y1.abs())
                .max(q.y2.abs());
        if max_coord <= 1e100 {
            return intersection_x_dd_f64(
                [p.x1, p.y1, p.x2, p.y2],
                [q.x1, q.y1, q.x2, q.y2],
                lo,
                hi,
            );
        }
    }
    if det == 0.0 {
        return None;
    }
    let t = ((q.x1 - p.x1) * dy2 - (q.y1 - p.y1) * dx2) / det;
    let x = p.x1 + t * dx1;
    Some(x.clamp(lo, hi))
}

/// Exact post-`x_evt` vertical order of two active non-vertical segments spanning `x_evt`.
///
/// Uses bounding-box y-disjointness and endpoint orientation signs (`s1, t1, s2, t2`) for
/// non-intersecting pairs, shared-endpoint slope checks, scale-safe `det == 0.0` fallback,
/// exact subnormal/overflow fallback, and canonicalized `intersection_x`.
#[inline]
pub(super) fn cmp_post_x(a: &Seg, b: &Seg, x_evt: f64) -> Ordering {
    if a.id == b.id {
        return Ordering::Equal;
    }
    if a.y_max < b.y_min {
        return Ordering::Less;
    }
    if b.y_max < a.y_min {
        return Ordering::Greater;
    }

    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = dx1 * (b.y1 - a.y1) - dy1 * (b.x1 - a.x1);
    let t1 = dx1 * (b.y2 - a.y1) - dy1 * (b.x2 - a.x1);
    if same_sign_nonzero(s1, t1) {
        return if s1 > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }

    let s2 = dx2 * (a.y1 - b.y1) - dy2 * (a.x1 - b.x1);
    let t2 = dx2 * (a.y2 - b.y1) - dy2 * (a.x2 - b.x1);
    if same_sign_nonzero(s2, t2) {
        return if s2 > 0.0 {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }

    let det = dx1 * dy2 - dy1 * dx2;
    if (s1 == 0.0 || t1 == 0.0 || s2 == 0.0 || t2 == 0.0 || det == 0.0 || !det.is_finite())
        && (is_tiny_nonzero(dx1)
            || is_tiny_nonzero(dy1)
            || is_tiny_nonzero(dx2)
            || is_tiny_nonzero(dy2)
            || !det.is_finite())
    {
        let max_coord =
            a.x1.abs()
                .max(a.x2.abs())
                .max(a.y1.abs())
                .max(a.y2.abs())
                .max(b.x1.abs())
                .max(b.x2.abs())
                .max(b.y1.abs())
                .max(b.y2.abs())
                .max(x_evt.abs());
        if max_coord <= 1e100 {
            return cmp_post_x_exact_f64(
                a.id,
                [a.x1, a.y1, a.x2, a.y2],
                b.id,
                [b.x1, b.y1, b.x2, b.y2],
                x_evt,
            );
        }
    }

    if a.x2 == b.x2 && a.y2 == b.y2 && a.x2 > x_evt {
        return cmp_slope_euclidean(b, a).then_with(|| a.id.cmp(&b.id));
    }
    if a.x1 == b.x1 && a.y1 == b.y1 && a.x1 <= x_evt {
        return cmp_slope_euclidean(a, b).then_with(|| a.id.cmp(&b.id));
    }

    if det == 0.0 {
        let ya = a.y_at(x_evt);
        let yb = b.y_at(x_evt);
        let rel_tol = seg_y_err(a, x_evt).max(seg_y_err(b, x_evt));
        if (ya - yb).abs() > rel_tol {
            return ya.total_cmp(&yb);
        }
        return cmp_slope_euclidean(a, b).then_with(|| a.id.cmp(&b.id));
    }

    let xi = intersection_x(a, b).unwrap_or(x_evt);
    let post_crossing = xi <= x_evt;
    if post_crossing {
        cmp_slope_euclidean(a, b)
            .then_with(|| 0.0_f64.total_cmp(&det))
            .then_with(|| a.id.cmp(&b.id))
    } else {
        cmp_slope_euclidean(b, a)
            .then_with(|| det.total_cmp(&0.0))
            .then_with(|| a.id.cmp(&b.id))
    }
}
