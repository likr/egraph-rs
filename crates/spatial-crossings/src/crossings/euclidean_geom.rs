use std::cmp::Ordering;

use super::exact_limbs::{ExactFactor, exact_deg2_sign};
use super::geometry::{STEEP_SLOPE_THRESHOLD, Seg, seg_y_err};
use super::torus_exact::{cmp_post_x_exact_f64, intersection_x_dd_f64, segs_intersect_exact_f64};

#[inline]
fn is_tiny_nonzero(v: f64) -> bool {
    v != 0.0 && v.abs() < 1e-150
}

#[inline]
fn det_sign_filtered(dx1: f64, dy1: f64, dx2: f64, dy2: f64) -> Option<Ordering> {
    let left = dx1 * dy2;
    let right = dy1 * dx2;
    let det = left - right;
    let sum = left.abs() + right.abs();
    if sum.is_finite() && sum > 1e-260 {
        let err = 8.0 * f64::EPSILON * sum;
        if det.abs() > err && det.abs() > 1e-260 {
            return Some(if det > 0.0 {
                Ordering::Greater
            } else {
                Ordering::Less
            });
        }
    }
    None
}

#[inline]
fn share_exact_endpoint(a: &Seg, b: &Seg) -> bool {
    (a.x1 == b.x1 && a.y1 == b.y1)
        || (a.x1 == b.x2 && a.y1 == b.y2)
        || (a.x2 == b.x1 && a.y2 == b.y1)
        || (a.x2 == b.x2 && a.y2 == b.y2)
}

#[inline]
pub(super) fn segments_geometrically_intersect(a: &Seg, b: &Seg) -> bool {
    if a.x_max < b.x_min || b.x_max < a.x_min || a.y_max < b.y_min || b.y_max < a.y_min {
        return false;
    }
    if share_exact_endpoint(a, b) {
        return true;
    }
    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = det_sign_filtered(dx1, dy1, b.x1 - a.x1, b.y1 - a.y1);
    let t1 = det_sign_filtered(dx1, dy1, b.x2 - a.x1, b.y2 - a.y1);
    if let (Some(so), Some(to)) = (s1, t1)
        && so == to
    {
        return false;
    }

    let s2 = det_sign_filtered(dx2, dy2, a.x1 - b.x1, a.y1 - b.y1);
    let t2 = det_sign_filtered(dx2, dy2, a.x2 - b.x1, a.y2 - b.y1);
    if let (Some(so), Some(to)) = (s2, t2)
        && so == to
    {
        return false;
    }

    if s1.is_none() || t1.is_none() || s2.is_none() || t2.is_none() {
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
    let max_s = a.slope.abs().max(b.slope.abs());
    if max_s < STEEP_SLOPE_THRESHOLD
        && (a.slope - b.slope).abs() > 8.0 * f64::EPSILON * max_s
        && !is_tiny_nonzero(a.y2 - a.y1)
        && !is_tiny_nonzero(a.x2 - a.x1)
        && !is_tiny_nonzero(b.y2 - b.y1)
        && !is_tiny_nonzero(b.x2 - b.x1)
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
fn max_pair_coord(a: &Seg, b: &Seg, extra: f64) -> f64 {
    a.x1.abs()
        .max(a.x2.abs())
        .max(a.y1.abs())
        .max(a.y2.abs())
        .max(b.x1.abs())
        .max(b.x2.abs())
        .max(b.y1.abs())
        .max(b.y2.abs())
        .max(extra.abs())
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
    let left = dx1 * dy2;
    let right = dy1 * dx2;
    let det = left - right;
    let det_sum = left.abs() + right.abs();
    let lo = p.x_min.max(q.x_min);
    let hi = p.x_max.min(q.x_max);
    let det_uncertain = !det_sum.is_finite()
        || (det_sum > 0.0 && (det.abs() <= 8.0 * f64::EPSILON * det_sum || det.abs() <= 1e-260))
        || is_tiny_nonzero(dx1)
        || is_tiny_nonzero(dy1)
        || is_tiny_nonzero(dx2)
        || is_tiny_nonzero(dy2);
    if det_uncertain && max_pair_coord(p, q, 0.0) <= 1e100 {
        return intersection_x_dd_f64([p.x1, p.y1, p.x2, p.y2], [q.x1, q.y1, q.x2, q.y2], lo, hi);
    }
    if det == 0.0 {
        return None;
    }
    let num_l = (q.x1 - p.x1) * dy2;
    let num_r = (q.y1 - p.y1) * dx2;
    let num = num_l - num_r;
    let num_sum = num_l.abs() + num_r.abs();
    let num_uncertain = !num_sum.is_finite()
        || (num_sum > 0.0 && (num.abs() <= 8.0 * f64::EPSILON * num_sum || num.abs() <= 1e-260));
    let ep_uncertain = !share_exact_endpoint(p, q)
        && (det_sign_filtered(dx1, dy1, q.x1 - p.x1, q.y1 - p.y1).is_none()
            || det_sign_filtered(dx1, dy1, q.x2 - p.x1, q.y2 - p.y1).is_none()
            || det_sign_filtered(dx2, dy2, p.x1 - q.x1, p.y1 - q.y1).is_none()
            || det_sign_filtered(dx2, dy2, p.x2 - q.x1, p.y2 - q.y1).is_none());
    if (num_uncertain || ep_uncertain) && max_pair_coord(p, q, 0.0) <= 1e100 {
        return intersection_x_dd_f64([p.x1, p.y1, p.x2, p.y2], [q.x1, q.y1, q.x2, q.y2], lo, hi);
    }
    let t = num / det;
    let x = p.x1 + t * dx1;
    Some(x.clamp(lo, hi))
}

/// Exact post-`x_evt` vertical order of two active non-vertical segments spanning `x_evt`.
///
/// Uses bounding-box y-disjointness, exact shared endpoints, Shewchuk-style error-bound
/// orientation filters, exact fallback on cancellation/subnormal inputs, and canonicalized
/// `intersection_x`.
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
    if a.x2 == b.x2 && a.y2 == b.y2 && a.x2 > x_evt {
        return cmp_slope_euclidean(b, a).then_with(|| a.id.cmp(&b.id));
    }
    if a.x1 == b.x1 && a.y1 == b.y1 && a.x1 <= x_evt {
        return cmp_slope_euclidean(a, b).then_with(|| a.id.cmp(&b.id));
    }

    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = det_sign_filtered(dx1, dy1, b.x1 - a.x1, b.y1 - a.y1);
    let t1 = det_sign_filtered(dx1, dy1, b.x2 - a.x1, b.y2 - a.y1);
    if let (Some(so), Some(to)) = (s1, t1)
        && so == to
    {
        return if so == Ordering::Greater {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }

    let s2 = det_sign_filtered(dx2, dy2, a.x1 - b.x1, a.y1 - b.y1);
    let t2 = det_sign_filtered(dx2, dy2, a.x2 - b.x1, a.y2 - b.y1);
    if let (Some(so), Some(to)) = (s2, t2)
        && so == to
    {
        return if so == Ordering::Greater {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }

    let det_ord = det_sign_filtered(dx1, dy1, dx2, dy2);
    if (s1.is_none() || t1.is_none() || s2.is_none() || t2.is_none() || det_ord.is_none())
        && max_pair_coord(a, b, x_evt) <= 1e100
    {
        return cmp_post_x_exact_f64(
            a.id,
            [a.x1, a.y1, a.x2, a.y2],
            b.id,
            [b.x1, b.y1, b.x2, b.y2],
            x_evt,
        );
    }

    let det = dx1 * dy2 - dy1 * dx2;
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
