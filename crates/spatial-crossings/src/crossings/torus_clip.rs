use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, CoreResult, Scalar, wrap_torus_coord};

use super::geometry::Seg;

/// Largest IEEE-754 `f64` strictly below `1.0` so clipped endpoints stay in `[0, 1)` with zero ULP gap.
pub(super) const TORUS_MAX: f64 = 1.0 - 0.5 * f64::EPSILON;

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
pub(super) fn unwrapped_delta(p0: f64, p1: f64, shift: i32) -> f64 {
    match shift {
        0 => p1 - p0,
        1 => (1.0 - p0) + p1,
        _ => (p1 - 1.0) - p0,
    }
}

#[inline]
fn axis_cut(p0: f64, d: f64, shift: i32) -> Option<f64> {
    match shift {
        -1 => Some(canon_zero((-p0 / d).clamp(0.0, 1.0))),
        1 => Some(canon_zero(((1.0 - p0) / d).clamp(0.0, 1.0))),
        _ => None,
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

    let dx = unwrapped_delta(x0, x1_w, best_dx);
    let dy = unwrapped_delta(y0, y1_w, best_dy);

    let mut cuts = [0.0_f64, 1.0, 0.0, 0.0];
    let mut n_cuts = 2usize;
    let tx = axis_cut(x0, dx, best_dx);
    let ty = axis_cut(y0, dy, best_dy);
    for t in [tx, ty].into_iter().flatten() {
        cuts[n_cuts] = t;
        n_cuts += 1;
    }
    cuts[..n_cuts].sort_unstable_by(f64::total_cmp);

    let eval_axis = |t: f64, p0: f64, p1_w: f64, d: f64, shift: i32, cell: i32, tc: Option<f64>| {
        if tc == Some(t) {
            return match (shift, cell) {
                (1, 0) | (-1, -1) => TORUS_MAX,
                (1, 1) | (-1, 0) => 0.0,
                _ => clamp_unit(p0 - cell as f64 + t * d),
            };
        }
        let raw = if t == 0.0 {
            p0 - cell as f64
        } else if t == 1.0 {
            p1_w + (shift - cell) as f64
        } else {
            (p0 - cell as f64) + t * d
        };
        clamp_unit(raw)
    };

    let mut pieces = Vec::with_capacity(n_cuts - 1);
    for w in cuts[..n_cuts].windows(2) {
        let (t_a, t_b) = (w[0], w[1]);
        if t_b <= t_a {
            continue;
        }
        let (cx, cy) = (
            tx.map_or(0, |tc| if t_a >= tc { best_dx } else { 0 }),
            ty.map_or(0, |tc| if t_a >= tc { best_dy } else { 0 }),
        );
        pieces.push([
            eval_axis(t_a, x0, x1_w, dx, best_dx, cx, tx),
            eval_axis(t_a, y0, y1_w, dy, best_dy, cy, ty),
            eval_axis(t_b, x0, x1_w, dx, best_dx, cx, tx),
            eval_axis(t_b, y0, y1_w, dy, best_dy, cy, ty),
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
fn eval_clipped_point(
    p0: (f64, f64),
    p1: (f64, f64),
    d: (f64, f64),
    shift: (f64, f64),
    t: f64,
    bound: ClipBound,
) -> (f64, f64) {
    let ((ax, ay), (bx, by), (dx, dy), (kx, ky)) = (p0, p1, d, shift);
    match bound {
        ClipBound::None => {
            let (px, py) = if t == 0.0 {
                (ax, ay)
            } else if t == 1.0 {
                (bx, by)
            } else {
                (ax + t * dx, ay + t * dy)
            };
            (clamp_unit(px + kx), clamp_unit(py + ky))
        }
        ClipBound::Left => (0.0, clamp_unit((ay + t * dy) + ky)),
        ClipBound::Right => (TORUS_MAX, clamp_unit((ay + t * dy) + ky)),
        ClipBound::Bottom => (clamp_unit((ax + t * dx) + kx), 0.0),
        ClipBound::Top => (clamp_unit((ax + t * dx) + kx), TORUS_MAX),
    }
}

/// Clips `(ax, ay) -> (bx, by)` directly against the shifted target box
/// `[-kx, TORUS_MAX - kx] x [-ky, TORUS_MAX - ky]` and translates the result by `(kx, ky)`
/// into `[0.0, TORUS_MAX]^2`.
///
/// Clipping before applying `(kx, ky)` preserves Sterbenz-exact precision near `0` when `kx = 1`
/// or `ky = 1`, while snapping active boundary coordinates directly to `0.0` or `TORUS_MAX`
/// eliminates `O(eps)` residuals and `1.0 -> TORUS_MAX` endpoint tilt.
#[inline]
pub(super) fn clip_seg_to_shifted_box(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    kx: f64,
    ky: f64,
) -> Option<[f64; 4]> {
    let (dx, dy) = (bx - ax, by - ay);
    let x_lo = -kx;
    let x_hi = if kx == 0.0 { TORUS_MAX } else { TORUS_MAX - kx };
    let y_lo = -ky;
    let y_hi = if ky == 0.0 { TORUS_MAX } else { TORUS_MAX - ky };

    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    let (mut bound0, mut bound1) = (ClipBound::None, ClipBound::None);
    for (p, q, bound) in [
        (-dx, ax - x_lo, ClipBound::Left),
        (dx, x_hi - ax, ClipBound::Right),
        (-dy, ay - y_lo, ClipBound::Bottom),
        (dy, y_hi - ay, ClipBound::Top),
    ] {
        if p == 0.0 {
            if q < -1e-12 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t0 {
                    t0 = r;
                    bound0 = bound;
                }
            } else if r < t1 {
                t1 = r;
                bound1 = bound;
            }
        }
    }
    if t0 > t1 + 1e-12 {
        return None;
    }

    if t0 > t1 {
        let t_mid = (0.5 * (t0 + t1)).clamp(0.0, 1.0);
        let x = if bound0 == ClipBound::Left || bound1 == ClipBound::Left {
            0.0
        } else if bound0 == ClipBound::Right || bound1 == ClipBound::Right {
            TORUS_MAX
        } else {
            clamp_unit((ax + t_mid * dx) + kx)
        };
        let y = if bound0 == ClipBound::Bottom || bound1 == ClipBound::Bottom {
            0.0
        } else if bound0 == ClipBound::Top || bound1 == ClipBound::Top {
            TORUS_MAX
        } else {
            clamp_unit((ay + t_mid * dy) + ky)
        };
        return Some([x, y, x, y]);
    }

    let (x0, y0) = eval_clipped_point((ax, ay), (bx, by), (dx, dy), (kx, ky), t0, bound0);
    let (x1, y1) = eval_clipped_point((ax, ay), (bx, by), (dx, dy), (kx, ky), t1, bound1);
    Some([x0, y0, x1, y1])
}

#[inline]
pub(super) fn clip_seg_to_unit_torus(ax: f64, ay: f64, bx: f64, by: f64) -> Option<[f64; 4]> {
    clip_seg_to_shifted_box(ax, ay, bx, by, 0.0, 0.0)
}

#[inline]
pub(super) fn crossing_point(a: &Seg, b: &Seg) -> (f64, f64) {
    let (dx1, dy1, dx2, dy2) = (a.x2 - a.x1, a.y2 - a.y1, b.x2 - b.x1, b.y2 - b.y1);
    let det = dx1 * dy2 - dy1 * dx2;
    if det != 0.0 {
        let t = ((b.x1 - a.x1) * dy2 - (b.y1 - a.y1) * dx2) / det;
        (
            (a.x1 + t * dx1).clamp(a.x_min.max(b.x_min), a.x_max.min(b.x_max)),
            (a.y1 + t * dy1).clamp(a.y_min.max(b.y_min), a.y_max.min(b.y_max)),
        )
    } else {
        (
            0.5 * (a.x_min.max(b.x_min) + a.x_max.min(b.x_max)),
            0.5 * (a.y_min.max(b.y_min) + a.y_max.min(b.y_max)),
        )
    }
}

#[inline]
pub(super) fn torus_crossing_angle_metrics(
    (dx1, dy1): (f64, f64),
    (dx2, dy2): (f64, f64),
) -> (Option<f64>, f64) {
    let (len1, len2) = (dx1.hypot(dy1), dx2.hypot(dy2));
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let cos_val = ((dx1 / len1) * (dx2 / len2) + (dy1 / len1) * (dy2 / len2)).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    let cos_sq = cos_val * cos_val;
    (Some(canon_zero(acute)), canon_zero(cos_sq))
}
