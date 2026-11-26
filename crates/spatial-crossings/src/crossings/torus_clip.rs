use std::cmp::Ordering;
use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, CoreResult, Scalar, wrap_torus_coord};

use super::double_double::{
    Dd, dd_add, dd_cmp, dd_div, dd_mul, dd_mul_f64, dd_neg, dd_sub, two_diff,
};
use super::torus_exact::{ShiftedF64, ShiftedPt};

/// Largest IEEE-754 `f64` strictly below `1.0` so clipped endpoints stay in `[0, 1)` with zero ULP gap.
pub(super) const TORUS_MAX: f64 = 1.0 - 0.5 * f64::EPSILON;

#[derive(Debug, Clone, Copy)]
pub(super) struct ClippedTorusSeg {
    pub(super) coords: [f64; 4],
    pub(super) free_dd: [Option<Dd>; 4],
}

#[inline]
pub(super) fn canon_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
pub(super) fn clamp_unit(v: f64) -> f64 {
    canon_zero(v.clamp(0.0, TORUS_MAX))
}

#[inline]
pub(super) fn wrap_torus_pos<S: Scalar>(value: S) -> f64 {
    canon_zero(wrap_torus_coord(value).to_f64())
}

#[inline]
pub(super) fn best_axis_shift(p0: f64, p1: f64) -> i32 {
    let d = p1 - p0;
    if d > 0.5 {
        -1
    } else if d < -0.5 {
        1
    } else {
        0
    }
}

#[inline]
pub(super) fn dd_unwrapped_delta(p0: f64, p1: f64, shift: i32) -> Dd {
    let (d_hi, d_lo) = two_diff(p1, p0);
    dd_add(Dd::from_f64(shift as f64), Dd { hi: d_hi, lo: d_lo })
}

#[inline]
fn clamp_dd_unit(v: Dd) -> Dd {
    if dd_cmp(v, Dd::ZERO) == Ordering::Less {
        Dd::ZERO
    } else if dd_cmp(v, Dd::ONE) == Ordering::Greater {
        Dd::ONE
    } else {
        v
    }
}

#[inline]
fn axis_cut_dd(p0: f64, d: Dd, shift: i32) -> Option<Dd> {
    match shift {
        -1 => Some(clamp_dd_unit(dd_div(Dd::from_f64(-p0), d))),
        1 => Some(clamp_dd_unit(dd_div(Dd::from_f64(1.0 - p0), d))),
        _ => None,
    }
}

#[inline]
fn axis_cut_complement_dd(p1_w: f64, d: Dd, shift: i32) -> Dd {
    match shift {
        1 => clamp_dd_unit(dd_div(Dd::from_f64(p1_w), d)),
        -1 => clamp_dd_unit(dd_div(Dd::from_f64(p1_w - 1.0), d)),
        _ => Dd::ZERO,
    }
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into 1, 2, or 3 Euclidean segments in `[0, 1)^2`
/// under unit `[0, 1)^2` aspect ratio.
pub fn torus_edge_segments<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> CoreResult<Vec<[f64; 4]>> {
    torus_edge_segments_with_extents(x1, y1, x2, y2, 1.0, 1.0)
}

pub(super) fn torus_edge_segments_with_extents<S: Scalar>(
    x1: S,
    y1: S,
    x2: S,
    y2: S,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<[f64; 4]>> {
    if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments"));
    }
    if !lx.is_finite() || !ly.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments::extents"));
    }
    if lx <= 0.0 || ly <= 0.0 {
        return Err(CoreError::invalid_input(
            "torus extents lx and ly must be strictly positive",
        ));
    }

    let x0 = wrap_torus_pos(x1);
    let y0 = wrap_torus_pos(y1);
    let x1_w = wrap_torus_pos(x2);
    let y1_w = wrap_torus_pos(y2);

    // Since the torus box is axis-aligned, minimizing physical distance decouples per axis.
    let best_dx = best_axis_shift(x0, x1_w);
    let best_dy = best_axis_shift(y0, y1_w);

    if best_dx == 0 && best_dy == 0 {
        return Ok(vec![[x0, y0, x1_w, y1_w]]);
    }

    let dx = dd_unwrapped_delta(x0, x1_w, best_dx);
    let dy = dd_unwrapped_delta(y0, y1_w, best_dy);

    let mut cuts = [Dd::ZERO, Dd::ONE, Dd::ZERO, Dd::ZERO];
    let mut n_cuts = 2usize;
    let tx = axis_cut_dd(x0, dx, best_dx);
    let ty = axis_cut_dd(y0, dy, best_dy);
    for t in [tx, ty].into_iter().flatten() {
        cuts[n_cuts] = t;
        n_cuts += 1;
    }
    cuts[..n_cuts].sort_unstable_by(|&a, &b| dd_cmp(a, b));

    let eval_axis = |t: Dd,
                     p0: f64,
                     p1_w: f64,
                     d: Dd,
                     shift: i32,
                     cell: i32,
                     tc: Option<Dd>,
                     other_cut: Option<Dd>,
                     other_p1_w: f64,
                     other_d: Dd,
                     other_shift: i32| {
        if tc.is_some_and(|cut| dd_cmp(cut, t) == Ordering::Equal) {
            return match (shift, cell) {
                (1, 0) | (-1, -1) => TORUS_MAX,
                (1, 1) | (-1, 0) => 0.0,
                _ => clamp_unit(
                    dd_add(ShiftedF64::new(p0, -cell as i8).to_dd(), dd_mul(t, d)).to_f64(),
                ),
            };
        }
        if t.is_zero() {
            return clamp_unit(ShiftedF64::new(p0, -cell as i8).to_f64());
        }
        if dd_cmp(t, Dd::ONE) == Ordering::Equal {
            return clamp_unit(ShiftedF64::new(p1_w, (shift - cell) as i8).to_f64());
        }
        let val_dd = if t.hi <= 0.5 {
            dd_add(ShiftedF64::new(p0, -cell as i8).to_dd(), dd_mul(t, d))
        } else {
            let u = if other_cut.is_some_and(|oc| dd_cmp(oc, t) == Ordering::Equal) {
                axis_cut_complement_dd(other_p1_w, other_d, other_shift)
            } else {
                dd_sub(Dd::ONE, t)
            };
            dd_sub(
                ShiftedF64::new(p1_w, (shift - cell) as i8).to_dd(),
                dd_mul(u, d),
            )
        };
        clamp_unit(val_dd.to_f64())
    };

    let mut pieces = Vec::with_capacity(n_cuts - 1);
    for w in cuts[..n_cuts].windows(2) {
        let (t_a, t_b) = (w[0], w[1]);
        if dd_cmp(t_b, t_a) != Ordering::Greater {
            continue;
        }
        let cx = tx.map_or(0, |tc| {
            if dd_cmp(t_a, tc) != Ordering::Less {
                best_dx
            } else {
                0
            }
        });
        let cy = ty.map_or(0, |tc| {
            if dd_cmp(t_a, tc) != Ordering::Less {
                best_dy
            } else {
                0
            }
        });
        pieces.push([
            eval_axis(t_a, x0, x1_w, dx, best_dx, cx, tx, ty, y1_w, dy, best_dy),
            eval_axis(t_a, y0, y1_w, dy, best_dy, cy, ty, tx, x1_w, dx, best_dx),
            eval_axis(t_b, x0, x1_w, dx, best_dx, cx, tx, ty, y1_w, dy, best_dy),
            eval_axis(t_b, y0, y1_w, dy, best_dy, cy, ty, tx, x1_w, dx, best_dx),
        ]);
    }
    Ok(pieces)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClipBound {
    None,
    Left,
    Right,
    Bottom,
    Top,
}

#[inline]
fn eval_other_at_bound(
    a_axis: ShiftedF64,
    b_axis: ShiftedF64,
    a_other: ShiftedF64,
    b_other: ShiftedF64,
    d_axis: Dd,
    d_other: Dd,
    bound: f64,
) -> (f64, Dd) {
    let da = a_axis.sub_f64_dd(bound);
    let db = b_axis.sub_f64_dd(bound);
    let val_dd = if da.hi.abs() <= db.hi.abs() {
        let t = dd_div(dd_neg(da), d_axis);
        dd_add(a_other.to_dd(), dd_mul(t, d_other))
    } else {
        let u = dd_div(db, d_axis);
        dd_sub(b_other.to_dd(), dd_mul(u, d_other))
    };
    (clamp_unit(val_dd.to_f64()), val_dd)
}

#[inline]
fn eval_clipped_endpoint(
    a: ShiftedPt,
    b: ShiftedPt,
    dx: Dd,
    dy: Dd,
    is_end: bool,
    bound: ClipBound,
) -> (f64, f64, Option<Dd>, Option<Dd>) {
    match bound {
        ClipBound::None => {
            let (px, py) = if !is_end { (a.x, a.y) } else { (b.x, b.y) };
            (clamp_unit(px.to_f64()), clamp_unit(py.to_f64()), None, None)
        }
        ClipBound::Left => {
            let (y, y_dd) = eval_other_at_bound(a.x, b.x, a.y, b.y, dx, dy, 0.0);
            (0.0, y, None, Some(y_dd))
        }
        ClipBound::Right => {
            let (y, y_dd) = eval_other_at_bound(a.x, b.x, a.y, b.y, dx, dy, TORUS_MAX);
            (TORUS_MAX, y, None, Some(y_dd))
        }
        ClipBound::Bottom => {
            let (x, x_dd) = eval_other_at_bound(a.y, b.y, a.x, b.x, dy, dx, 0.0);
            (x, 0.0, Some(x_dd), None)
        }
        ClipBound::Top => {
            let (x, x_dd) = eval_other_at_bound(a.y, b.y, a.x, b.x, dy, dx, TORUS_MAX);
            (x, TORUS_MAX, Some(x_dd), None)
        }
    }
}

/// Clips an exact shifted segment `(p1 -> p2)` translated by `(kx, ky)` against `[0.0, TORUS_MAX]^2`
/// using 106-bit DD Liang-Barsky bounds and near-endpoint boundary evaluation.
pub(super) fn clip_shifted_seg_to_torus(
    p1: ShiftedPt,
    p2: ShiftedPt,
    kx: i8,
    ky: i8,
) -> Option<ClippedTorusSeg> {
    let ax = p1.x.shifted_by(kx);
    let ay = p1.y.shifted_by(ky);
    let bx = p2.x.shifted_by(kx);
    let by = p2.y.shifted_by(ky);
    let dx = ax.delta_dd(bx);
    let dy = ay.delta_dd(by);

    let q_left = ax.sub_f64_dd(0.0);
    let q_right = dd_neg(ax.sub_f64_dd(TORUS_MAX));
    let q_bottom = ay.sub_f64_dd(0.0);
    let q_top = dd_neg(ay.sub_f64_dd(TORUS_MAX));

    let mut t0 = Dd::ZERO;
    let mut t1 = Dd::ONE;
    let mut bound0 = ClipBound::None;
    let mut bound1 = ClipBound::None;

    for (p, q, bound) in [
        (dd_neg(dx), q_left, ClipBound::Left),
        (dx, q_right, ClipBound::Right),
        (dd_neg(dy), q_bottom, ClipBound::Bottom),
        (dy, q_top, ClipBound::Top),
    ] {
        if p.is_zero() {
            if q.to_f64() < -1e-26 {
                return None;
            }
        } else {
            let r = dd_div(q, p);
            if p.is_neg() {
                if dd_cmp(r, t0) == Ordering::Greater {
                    t0 = r;
                    bound0 = bound;
                }
            } else if dd_cmp(r, t1) == Ordering::Less {
                t1 = r;
                bound1 = bound;
            }
        }
    }

    let diff = dd_sub(t0, t1).to_f64();
    if diff > 1e-26 {
        return None;
    }

    if diff > 0.0 {
        let t_mid = clamp_dd_unit(dd_mul_f64(dd_add(t0, t1), 0.5));
        let eval_mid = |a: ShiftedF64, b: ShiftedF64, d: Dd| {
            if t_mid.hi <= 0.5 {
                dd_add(a.to_dd(), dd_mul(t_mid, d))
            } else {
                dd_sub(b.to_dd(), dd_mul(dd_sub(Dd::ONE, t_mid), d))
            }
        };
        let (x, x_dd) = if bound0 == ClipBound::Left || bound1 == ClipBound::Left {
            (0.0, None)
        } else if bound0 == ClipBound::Right || bound1 == ClipBound::Right {
            (TORUS_MAX, None)
        } else {
            let xd = eval_mid(ax, bx, dx);
            (clamp_unit(xd.to_f64()), Some(xd))
        };
        let (y, y_dd) = if bound0 == ClipBound::Bottom || bound1 == ClipBound::Bottom {
            (0.0, None)
        } else if bound0 == ClipBound::Top || bound1 == ClipBound::Top {
            (TORUS_MAX, None)
        } else {
            let yd = eval_mid(ay, by, dy);
            (clamp_unit(yd.to_f64()), Some(yd))
        };
        return Some(ClippedTorusSeg {
            coords: [x, y, x, y],
            free_dd: [x_dd, y_dd, x_dd, y_dd],
        });
    }

    let a_sh = ShiftedPt::new(ax, ay);
    let b_sh = ShiftedPt::new(bx, by);
    let (x0, y0, fx0, fy0) = eval_clipped_endpoint(a_sh, b_sh, dx, dy, false, bound0);
    let (x1, y1, fx1, fy1) = eval_clipped_endpoint(a_sh, b_sh, dx, dy, true, bound1);
    Some(ClippedTorusSeg {
        coords: [x0, y0, x1, y1],
        free_dd: [fx0, fy0, fx1, fy1],
    })
}

#[inline]
pub(super) fn torus_crossing_angle_metrics(
    (dx1_dd, dy1_dd): (Dd, Dd),
    (dx2_dd, dy2_dd): (Dd, Dd),
) -> (Option<f64>, f64) {
    let (dx1, dy1) = (dx1_dd.to_f64(), dy1_dd.to_f64());
    let (dx2, dy2) = (dx2_dd.to_f64(), dy2_dd.to_f64());
    let (len1, len2) = (dx1.hypot(dy1), dx2.hypot(dy2));
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let (ux1, uy1) = (dx1 / len1, dy1 / len1);
    let (ux2, uy2) = (dx2 / len2, dy2 / len2);
    let dot = ux1 * ux2 + uy1 * uy2;
    let cross = ux1 * uy2 - uy1 * ux2;
    let acute = cross.abs().atan2(dot.abs()).min(FRAC_PI_2);
    let cos_val = dot.clamp(-1.0, 1.0);
    (Some(canon_zero(acute)), canon_zero(cos_val * cos_val))
}
