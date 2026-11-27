use std::cmp::Ordering;
use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};

use super::double_double::dd_div;
use super::torus_exact::ShiftedPt;

#[derive(Debug, Clone, Copy)]
pub(super) struct Seg {
    pub(super) id: usize,
    pub(super) u: u32,
    pub(super) v: u32,
    pub(super) x1: f64,
    pub(super) y1: f64,
    pub(super) x2: f64,
    pub(super) y2: f64,
    pub(super) sx1: i8,
    pub(super) sy1: i8,
    pub(super) sx2: i8,
    pub(super) sy2: i8,
    pub(super) x_min: f64,
    pub(super) x_max: f64,
    pub(super) y_min: f64,
    pub(super) y_max: f64,
    pub(super) slope: f64,
    pub(super) is_vertical: bool,
}

impl Seg {
    pub(super) fn from_endpoints(
        id: usize,
        u: u32,
        v: u32,
        ax: f64,
        ay: f64,
        bx: f64,
        by: f64,
    ) -> Self {
        let (x1, y1, x2, y2) = if ax < bx || (ax == bx && ay <= by) {
            (ax, ay, bx, by)
        } else {
            (bx, by, ax, ay)
        };
        let is_vertical = x1 == x2;
        let slope = if is_vertical {
            0.0
        } else {
            let dy = y2 - y1;
            let dx = x2 - x1;
            let s = dy / dx;
            if s.is_finite() && s.abs() <= 1e290 {
                s
            } else if dy == 0.0 {
                0.0
            } else {
                dy.signum() * dx.signum() * 1e290
            }
        };
        Self {
            id,
            u,
            v,
            x1,
            y1,
            x2,
            y2,
            sx1: 0,
            sy1: 0,
            sx2: 0,
            sy2: 0,
            x_min: x1.min(x2),
            x_max: x1.max(x2),
            y_min: y1.min(y2),
            y_max: y1.max(y2),
            slope,
            is_vertical,
        }
    }

    pub(super) fn from_shifted(id: usize, u: u32, v: u32, a: ShiftedPt, b: ShiftedPt) -> Self {
        let (p1, p2) = if a.cmp_exact(b) != Ordering::Greater {
            (a, b)
        } else {
            (b, a)
        };
        let is_vertical = p1.x.cmp_exact(p2.x) == Ordering::Equal;
        let x1 = p1.x.to_f64();
        let y1 = p1.y.to_f64();
        let x2 = p2.x.to_f64();
        let y2 = p2.y.to_f64();
        let slope = if is_vertical {
            0.0
        } else {
            let dy = p1.y.delta_dd(p2.y);
            let dx = p1.x.delta_dd(p2.x);
            let s = dd_div(dy, dx).to_f64();
            if s.is_finite() && s.abs() <= 1e290 {
                s
            } else if dy.is_zero() {
                0.0
            } else {
                (dy.hi.signum() * dx.hi.signum()) * 1e290
            }
        };
        Self {
            id,
            u,
            v,
            x1,
            y1,
            x2,
            y2,
            sx1: p1.x.shift,
            sy1: p1.y.shift,
            sx2: p2.x.shift,
            sy2: p2.y.shift,
            x_min: x1.min(x2),
            x_max: x1.max(x2),
            y_min: y1.min(y2),
            y_max: y1.max(y2),
            slope,
            is_vertical,
        }
    }

    #[inline]
    pub(super) fn shares_endpoint(&self, other: &Self) -> bool {
        self.u == other.u
            || self.u == other.v
            || self.v == other.u
            || self.v == other.v
            || self.u == self.v
            || other.u == other.v
    }

    #[inline]
    pub(super) fn y_at(&self, x: f64) -> f64 {
        if x <= self.x1 {
            self.y1
        } else if x >= self.x2 {
            self.y2
        } else {
            self.y1 + self.slope * (x - self.x1)
        }
    }
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
    !same_sign_nonzero(s2, t2)
}

#[inline]
pub(super) fn segments_cross(a: &Seg, b: &Seg) -> bool {
    if a.shares_endpoint(b) {
        return false;
    }
    segments_geometrically_intersect(a, b)
}

#[inline]
pub(super) fn seg_y_err(s: &Seg, x: f64) -> f64 {
    let x_scale = x.abs().max(s.x1.abs()).max(s.x2.abs());
    let y_scale = s.y1.abs().max(s.y2.abs());
    32.0 * f64::EPSILON * (s.slope.abs() * x_scale + y_scale + 1.0)
}

#[inline]
pub(super) fn within_band(seed: &Seg, other: &Seg, x_evt: f64, eps_y: f64) -> bool {
    let tol = 2.0 * eps_y + seg_y_err(seed, x_evt) + seg_y_err(other, x_evt);
    (other.y_at(x_evt) - seed.y_at(x_evt)).abs() <= tol
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
    if det == 0.0 {
        return None;
    }
    let t = ((q.x1 - p.x1) * dy2 - (q.y1 - p.y1) * dx2) / det;
    let x = p.x1 + t * dx1;
    let lo = p.x_min.max(q.x_min);
    let hi = p.x_max.min(q.x_max);
    Some(x.clamp(lo, hi))
}

/// Exact post-`x_evt` vertical order of two active non-vertical segments spanning `x_evt`.
///
/// Uses bounding-box y-disjointness and exact endpoint orientation signs (`s1, t1, s2, t2`) for
/// non-intersecting pairs (avoiding `y_at(x_evt)` cancellation and scalar `dt_probe` overshoot),
/// shared-endpoint slope checks, scale-safe `det == 0.0` fallback, and canonicalized
/// `intersection_x` to separate completed crossings (`xi <= x_evt`) from future ones (`xi > x_evt`).
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

    if a.x2 == b.x2 && a.y2 == b.y2 && a.x2 > x_evt {
        return b.slope.total_cmp(&a.slope).then_with(|| a.id.cmp(&b.id));
    }
    if a.x1 == b.x1 && a.y1 == b.y1 && a.x1 <= x_evt {
        return a.slope.total_cmp(&b.slope).then_with(|| a.id.cmp(&b.id));
    }

    let det = dx1 * dy2 - dy1 * dx2;
    if det == 0.0 {
        let ya = a.y_at(x_evt);
        let yb = b.y_at(x_evt);
        let x_scale = x_evt
            .abs()
            .max(a.x1.abs())
            .max(a.x2.abs())
            .max(b.x1.abs())
            .max(b.x2.abs());
        let y_scale = a.y1.abs().max(a.y2.abs()).max(b.y1.abs()).max(b.y2.abs());
        let rel_tol = 32.0 * f64::EPSILON * (a.slope.abs().max(b.slope.abs()) * x_scale + y_scale);
        if (ya - yb).abs() > rel_tol {
            return ya.total_cmp(&yb);
        }
        return a.slope.total_cmp(&b.slope).then_with(|| a.id.cmp(&b.id));
    }

    let xi = intersection_x(a, b).unwrap_or(x_evt);
    let post_crossing = xi <= x_evt;
    if post_crossing {
        a.slope
            .total_cmp(&b.slope)
            .then_with(|| 0.0_f64.total_cmp(&det))
            .then_with(|| a.id.cmp(&b.id))
    } else {
        b.slope
            .total_cmp(&a.slope)
            .then_with(|| det.total_cmp(&0.0))
            .then_with(|| a.id.cmp(&b.id))
    }
}

#[inline]
pub(super) fn crossing_angle_metrics(a: &Seg, b: &Seg) -> (Option<f64>, f64) {
    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;
    let len1 = dx1.hypot(dy1);
    let len2 = dx2.hypot(dy2);
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let ux1 = dx1 / len1;
    let uy1 = dy1 / len1;
    let ux2 = dx2 / len2;
    let uy2 = dy2 / len2;
    let cos_val = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    (Some(acute), cos_val * cos_val)
}

pub(super) fn extract_csr_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut edges = Vec::with_capacity(graph.num_edges());
    for u in 0..n {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if directed || u32_u < v {
                edges.push((u32_u, v));
            }
        }
    }
    edges
}

pub(super) fn build_segments<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> CoreResult<Vec<Seg>> {
    if edges.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut segs = Vec::with_capacity(edges.len());

    for (id, &(u, v)) in edges.iter().enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(
                ui,
                n,
                "SweepLineCrossings::edge_source",
            ));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(
                vi,
                n,
                "SweepLineCrossings::edge_target",
            ));
        }
        segs.push(Seg::from_endpoints(
            id,
            u,
            v,
            xs[ui].to_f64(),
            ys[ui].to_f64(),
            xs[vi].to_f64(),
            ys[vi].to_f64(),
        ));
    }
    Ok(segs)
}
