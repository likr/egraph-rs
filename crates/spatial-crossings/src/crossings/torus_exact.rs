use std::cmp::Ordering;

use super::double_double::{Dd, dd_add, dd_div, dd_mul, dd_neg, dd_sub};
use super::exact_limbs::{ExactFactor, exact_collinear_in_box, exact_deg2_sign, exact_deg3_sign};
pub(super) use super::shifted_coords::{ShiftedF64, ShiftedPt, ShiftedSeg};

#[inline]
pub(super) fn det_sign_shifted(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
) -> Ordering {
    let ds_x1 = (a2.x.shift - a1.x.shift) as f64;
    let ds_y1 = (a2.y.shift - a1.y.shift) as f64;
    let ds_x2 = (b2.x.shift - b1.x.shift) as f64;
    let ds_y2 = (b2.y.shift - b1.y.shift) as f64;

    let dx1 = ds_x1 + (a2.x.base - a1.x.base);
    let dy1 = ds_y1 + (a2.y.base - a1.y.base);
    let dx2 = ds_x2 + (b2.x.base - b1.x.base);
    let dy2 = ds_y2 + (b2.y.base - b1.y.base);

    let left = dx1 * dy2;
    let right = dy1 * dx2;
    let det = left - right;
    let sum = left.abs() + right.abs();
    if sum.is_finite() && sum > 1e-260 {
        let ex1 = if ds_x1 == 0.0 { dx1.abs() } else { 4.0 };
        let ey1 = if ds_y1 == 0.0 { dy1.abs() } else { 4.0 };
        let ex2 = if ds_x2 == 0.0 { dx2.abs() } else { 4.0 };
        let ey2 = if ds_y2 == 0.0 { dy2.abs() } else { 4.0 };
        let err = 16.0
            * f64::EPSILON
            * (ex1 * dy2.abs() + ey2 * dx1.abs() + ey1 * dx2.abs() + ex2 * dy1.abs() + sum);
        if det.abs() > err && det.abs() > 1e-260 {
            return if det > 0.0 {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
    }

    exact_deg2_sign(&[
        (
            ExactFactor::from_delta(a1.x, a2.x),
            ExactFactor::from_delta(b1.y, b2.y),
            1,
        ),
        (
            ExactFactor::from_delta(a1.y, a2.y),
            ExactFactor::from_delta(b1.x, b2.x),
            -1,
        ),
    ])
}

#[inline]
pub(super) fn orient2d_shifted(o: ShiftedPt, a: ShiftedPt, b: ShiftedPt) -> Ordering {
    det_sign_shifted(o, a, o, b)
}

#[inline]
pub(super) fn cmp_slope_shifted(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
) -> Ordering {
    // For dx_a > 0, dx_b > 0: dy_a/dx_a <=> dy_b/dx_b iff dy_a*dx_b - dy_b*dx_a <=> 0.
    det_sign_shifted(b1, b2, a1, a2)
}

#[inline]
pub(super) fn torus_segments_intersect(a: &ShiftedSeg, b: &ShiftedSeg) -> bool {
    if a.x_max.cmp_exact(b.x_min) == Ordering::Less
        || b.x_max.cmp_exact(a.x_min) == Ordering::Less
        || a.y_max.cmp_exact(b.y_min) == Ordering::Less
        || b.y_max.cmp_exact(a.y_min) == Ordering::Less
    {
        return false;
    }
    let s1 = orient2d_shifted(a.p1, a.p2, b.p1);
    let t1 = orient2d_shifted(a.p1, a.p2, b.p2);
    if (s1 == Ordering::Greater && t1 == Ordering::Greater)
        || (s1 == Ordering::Less && t1 == Ordering::Less)
    {
        return false;
    }
    let s2 = orient2d_shifted(b.p1, b.p2, a.p1);
    let t2 = orient2d_shifted(b.p1, b.p2, a.p2);
    !((s2 == Ordering::Greater && t2 == Ordering::Greater)
        || (s2 == Ordering::Less && t2 == Ordering::Less))
}

#[inline]
pub(super) fn segs_intersect_exact_f64(a: [f64; 4], b: [f64; 4]) -> bool {
    let max_abs = a
        .iter()
        .chain(b.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs > 1e300 { 0.125 } else { 1.0 };
    let sa = ShiftedSeg::from_shifted(
        ShiftedPt::from_f64(a[0] * scale, a[1] * scale),
        ShiftedPt::from_f64(a[2] * scale, a[3] * scale),
    );
    let sb = ShiftedSeg::from_shifted(
        ShiftedPt::from_f64(b[0] * scale, b[1] * scale),
        ShiftedPt::from_f64(b[2] * scale, b[3] * scale),
    );
    torus_segments_intersect(&sa, &sb)
}

#[cold]
#[inline(never)]
pub(super) fn intersection_x_dd_f64(a: [f64; 4], b: [f64; 4], lo: f64, hi: f64) -> Option<f64> {
    let pa1 = ShiftedPt::from_f64(a[0], a[1]);
    let pa2 = ShiftedPt::from_f64(a[2], a[3]);
    let pb1 = ShiftedPt::from_f64(b[0], b[1]);
    let pb2 = ShiftedPt::from_f64(b[2], b[3]);
    let (xi, _) = dd_intersection_xy(pa1, pa2, pb1, pb2)?;
    let x = xi.to_f64().clamp(lo, hi);
    Some(super::cross_pred::refine_xi_ceil(
        pa1, pa2, pb1, pb2, x, lo, hi,
    ))
}

#[cold]
#[inline(never)]
pub(super) fn cmp_post_x_exact_f64(
    id_a: usize,
    a: [f64; 4],
    id_b: usize,
    b: [f64; 4],
    x_evt: f64,
) -> Ordering {
    let ma = ShiftedSeg::from_shifted(
        ShiftedPt::from_f64(a[0], a[1]),
        ShiftedPt::from_f64(a[2], a[3]),
    );
    let mb = ShiftedSeg::from_shifted(
        ShiftedPt::from_f64(b[0], b[1]),
        ShiftedPt::from_f64(b[2], b[3]),
    );
    let s1 = orient2d_shifted(ma.p1, ma.p2, mb.p1);
    let t1 = orient2d_shifted(ma.p1, ma.p2, mb.p2);
    if s1 != Ordering::Equal && s1 == t1 {
        return if s1 == Ordering::Greater {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let s2 = orient2d_shifted(mb.p1, mb.p2, ma.p1);
    let t2 = orient2d_shifted(mb.p1, mb.p2, ma.p2);
    if s2 != Ordering::Equal && s2 == t2 {
        return if s2 == Ordering::Greater {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }
    if ma.p2 == mb.p2 && a[2] > x_evt {
        return cmp_slope_shifted(mb.p1, mb.p2, ma.p1, ma.p2).then_with(|| id_a.cmp(&id_b));
    }
    if ma.p1 == mb.p1 && a[0] <= x_evt {
        return cmp_slope_shifted(ma.p1, ma.p2, mb.p1, mb.p2).then_with(|| id_a.cmp(&id_b));
    }
    let det_ord = det_sign_shifted(ma.p1, ma.p2, mb.p1, mb.p2);
    if det_ord == Ordering::Equal {
        return id_a.cmp(&id_b);
    }
    let post_crossing = exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x_evt) != Some(Ordering::Greater);
    if post_crossing {
        cmp_slope_shifted(ma.p1, ma.p2, mb.p1, mb.p2)
            .then_with(|| Ordering::Equal.cmp(&det_ord))
            .then_with(|| id_a.cmp(&id_b))
    } else {
        cmp_slope_shifted(mb.p1, mb.p2, ma.p1, ma.p2)
            .then_with(|| det_ord)
            .then_with(|| id_a.cmp(&id_b))
    }
}

#[inline]
fn exact_coord_cmp_axis(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    bound: f64,
    is_y: bool,
) -> Option<Ordering> {
    let (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) = if !is_y {
        (a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y)
    } else {
        (a1.y, a1.x, a2.y, a2.x, b1.y, b1.x, b2.y, b2.x)
    };

    // Fast 106-bit DD filter for `N_x(X) = (bx1 - X)*dxa*dyb - (ax1 - X)*dya*dxb - (by1 - ay1)*dxa*dxb`.
    let dxa = ax1.delta_dd(ax2);
    let dya = ay1.delta_dd(ay2);
    let dxb = bx1.delta_dd(bx2);
    let dyb = by1.delta_dd(by2);
    let p_l = dd_mul(dxa, dyb);
    let p_r = dd_mul(dya, dxb);
    let det_dd = dd_sub(p_l, p_r);
    let det_sum = p_l.hi.abs() + p_r.hi.abs();

    if det_sum > 1e-260 && det_dd.hi.abs() > 1e-25 * det_sum {
        let u = bx1.sub_f64_dd(bound);
        let v = ax1.sub_f64_dd(bound);
        let w = ay1.delta_dd(by1);
        let t1 = dd_mul(u, dd_mul(dxa, dyb));
        let t2 = dd_mul(v, dd_mul(dya, dxb));
        let t3 = dd_mul(w, dd_mul(dxa, dxb));
        let num = dd_sub(dd_sub(t1, t2), t3);
        let num_sum = t1.hi.abs() + t2.hi.abs() + t3.hi.abs();
        if num_sum > 1e-260 && num.hi.abs() > 1e-25 * num_sum {
            let same_sign = (num.hi > 0.0) == (det_dd.hi > 0.0);
            return Some(if same_sign {
                Ordering::Greater
            } else {
                Ordering::Less
            });
        }
    }

    let det_ord = det_sign_shifted(a1, a2, b1, b2);
    if det_ord == Ordering::Equal {
        return None;
    }

    let f_dxa = ExactFactor::from_delta(ax1, ax2);
    let f_dya = ExactFactor::from_delta(ay1, ay2);
    let f_dxb = ExactFactor::from_delta(bx1, bx2);
    let f_dyb = ExactFactor::from_delta(by1, by2);
    let f_u = ExactFactor::from_shifted_minus_f64(bx1, bound);
    let f_v = ExactFactor::from_shifted_minus_f64(ax1, bound);
    let f_w = ExactFactor::from_delta(ay1, by1);
    let num_ord = exact_deg3_sign(&[
        (f_u, f_dxa, f_dyb, 1),
        (f_v, f_dya, f_dxb, -1),
        (f_w, f_dxa, f_dxb, -1),
    ]);
    if num_ord == Ordering::Equal {
        return Some(Ordering::Equal);
    }
    let mut gt = num_ord == det_ord;
    if is_y {
        gt = !gt;
    }
    Some(if gt {
        Ordering::Greater
    } else {
        Ordering::Less
    })
}

#[inline]
pub(super) fn exact_xi_cmp(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    x_bound: f64,
) -> Option<Ordering> {
    exact_coord_cmp_axis(a1, a2, b1, b2, x_bound, false)
}

#[inline]
pub(super) fn exact_yi_cmp(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    y_bound: f64,
) -> Option<Ordering> {
    exact_coord_cmp_axis(a1, a2, b1, b2, y_bound, true)
}

fn exact_intersection_in_box(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    (x_hi, x_hi_open): (f64, bool),
    (y_hi, y_hi_open): (f64, bool),
) -> bool {
    let sa = ShiftedSeg::from_shifted(a1, a2);
    let sb = ShiftedSeg::from_shifted(b1, b2);
    if !torus_segments_intersect(&sa, &sb) {
        return false;
    }
    let sx_hi = if x_hi == 1.0 {
        ShiftedF64::ONE
    } else {
        ShiftedF64::new(x_hi, 0)
    };
    let sy_hi = if y_hi == 1.0 {
        ShiftedF64::ONE
    } else {
        ShiftedF64::new(y_hi, 0)
    };
    match (
        exact_xi_cmp(a1, a2, b1, b2, 0.0),
        exact_xi_cmp(a1, a2, b1, b2, x_hi),
        exact_yi_cmp(a1, a2, b1, b2, 0.0),
        exact_yi_cmp(a1, a2, b1, b2, y_hi),
    ) {
        (Some(cx0), Some(cx1), Some(cy0), Some(cy1)) => {
            let ok_x1 = if x_hi_open {
                cx1 == Ordering::Less
            } else {
                cx1 != Ordering::Greater
            };
            let ok_y1 = if y_hi_open {
                cy1 == Ordering::Less
            } else {
                cy1 != Ordering::Greater
            };
            cx0 != Ordering::Less && ok_x1 && cy0 != Ordering::Less && ok_y1
        }
        _ => {
            let o1 = if sa.p1.cmp_exact(sb.p1) == Ordering::Greater {
                sa.p1
            } else {
                sb.p1
            };
            let o2 = if sa.p2.cmp_exact(sb.p2) == Ordering::Less {
                sa.p2
            } else {
                sb.p2
            };
            exact_collinear_in_box(o1, o2, sx_hi, x_hi_open, sy_hi, y_hi_open)
        }
    }
}

#[inline]
pub(super) fn exact_intersection_in_unit_box(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
) -> bool {
    exact_intersection_in_box(a1, a2, b1, b2, (1.0, true), (1.0, true))
}

#[inline]
pub(super) fn exact_intersection_in_torus_max_box(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    torus_max: f64,
) -> bool {
    exact_intersection_in_box(a1, a2, b1, b2, (torus_max, false), (torus_max, false))
}

/// Computes the intersection point `(xi, yi)` in 106-bit DD using the nearest endpoint among `{a1, a2, b1, b2}`.
pub(super) fn dd_intersection_xy(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
) -> Option<(Dd, Dd)> {
    if det_sign_shifted(a1, a2, b1, b2) == Ordering::Equal {
        return None;
    }
    let dxa = a1.x.delta_dd(a2.x);
    let dya = a1.y.delta_dd(a2.y);
    let dxb = b1.x.delta_dd(b2.x);
    let dyb = b1.y.delta_dd(b2.y);
    let det = dd_sub(dd_mul(dxa, dyb), dd_mul(dya, dxb));
    if det.is_zero() {
        let sa = ShiftedSeg::from_shifted(a1, a2);
        let sb = ShiftedSeg::from_shifted(b1, b2);
        let x_lo = if sa.x_min.cmp_exact(sb.x_min) == Ordering::Greater {
            sa.x_min
        } else {
            sb.x_min
        };
        let x_hi = if sa.x_max.cmp_exact(sb.x_max) == Ordering::Less {
            sa.x_max
        } else {
            sb.x_max
        };
        let y_lo = if sa.y_min.cmp_exact(sb.y_min) == Ordering::Greater {
            sa.y_min
        } else {
            sb.y_min
        };
        let y_hi = if sa.y_max.cmp_exact(sb.y_max) == Ordering::Less {
            sa.y_max
        } else {
            sb.y_max
        };
        let half = Dd::from_f64(0.5);
        return Some((
            dd_mul(dd_add(x_lo.to_dd(), x_hi.to_dd()), half),
            dd_mul(dd_add(y_lo.to_dd(), y_hi.to_dd()), half),
        ));
    }
    let cross_dd = |ux: Dd, uy: Dd, vx: Dd, vy: Dd| dd_sub(dd_mul(ux, vy), dd_mul(uy, vx));

    let t = dd_div(
        cross_dd(a1.x.delta_dd(b1.x), a1.y.delta_dd(b1.y), dxb, dyb),
        det,
    );
    let u = dd_div(
        cross_dd(b1.x.delta_dd(a2.x), b1.y.delta_dd(a2.y), dxb, dyb),
        det,
    );
    let s = dd_div(
        cross_dd(a1.x.delta_dd(b1.x), a1.y.delta_dd(b1.y), dxa, dya),
        det,
    );
    let v = dd_div(
        cross_dd(b2.x.delta_dd(a1.x), b2.y.delta_dd(a1.y), dxa, dya),
        det,
    );

    let len_a = dxa.hi.hypot(dya.hi);
    let len_b = dxb.hi.hypot(dyb.hi);
    let d_t = t.hi.abs() * len_a;
    let d_u = u.hi.abs() * len_a;
    let d_s = s.hi.abs() * len_b;
    let d_v = v.hi.abs() * len_b;

    let dd_is_finite =
        |val: Dd| val.hi.is_finite() && val.lo.is_finite() && val.to_f64().is_finite();
    let finite_or_far = |base: Dd, step: Dd| {
        let sum = dd_add(base, step);
        if dd_is_finite(sum) {
            sum
        } else {
            let sign = if step.hi != 0.0 {
                step.hi.signum()
            } else {
                base.hi.signum()
            };
            Dd::from_f64(sign * 1e300)
        }
    };

    let (xi, yi) = if d_t <= d_u && d_t <= d_s && d_t <= d_v {
        (
            finite_or_far(a1.x.to_dd(), dd_mul(t, dxa)),
            finite_or_far(a1.y.to_dd(), dd_mul(t, dya)),
        )
    } else if d_u <= d_s && d_u <= d_v {
        (
            finite_or_far(a2.x.to_dd(), dd_mul(dd_neg(u), dxa)),
            finite_or_far(a2.y.to_dd(), dd_mul(dd_neg(u), dya)),
        )
    } else if d_s <= d_v {
        (
            finite_or_far(b1.x.to_dd(), dd_mul(s, dxb)),
            finite_or_far(b1.y.to_dd(), dd_mul(s, dyb)),
        )
    } else {
        (
            finite_or_far(b2.x.to_dd(), dd_mul(dd_neg(v), dxb)),
            finite_or_far(b2.y.to_dd(), dd_mul(dd_neg(v), dyb)),
        )
    };
    Some((xi, yi))
}
