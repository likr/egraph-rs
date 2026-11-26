use std::cmp::Ordering;

use super::double_double::{Dd, dd_cmp};
use super::geometry::{Seg, cmp_post_x, intersection_x, segments_geometrically_intersect};
use super::torus_exact::{
    ShiftedF64, ShiftedSeg, cmp_slope_shifted, dd_intersection_xy, det_sign_shifted, exact_xi_cmp,
    orient2d_shifted, segs_intersect_exact_f64, torus_segments_intersect,
};

pub(super) trait CrossPred {
    fn intersect(&self, a: &Seg, b: &Seg) -> bool;
    #[inline]
    fn cross(&self, a: &Seg, b: &Seg) -> bool {
        !a.shares_endpoint(b) && self.intersect(a, b)
    }
    fn cmp_post_x(&self, a: &Seg, b: &Seg, x_evt: f64) -> Ordering;
    fn intersection_x(&self, a: &Seg, b: &Seg) -> Option<f64>;
    fn cmp_slope(&self, a: &Seg, b: &Seg) -> Ordering;
    fn same_right_endpoint(&self, a: &Seg, b: &Seg) -> bool;
    fn is_collinear_det_zero(&self, a: &Seg, b: &Seg) -> bool;
    fn xi_lt_sweep(&self, a: &Seg, b: &Seg, xi: f64, sweep_x: f64) -> bool;
    fn xi_eq_sweep(&self, a: &Seg, b: &Seg, xi: f64, sweep_x: f64) -> bool;
}

pub(super) struct EuclideanPred;

impl CrossPred for EuclideanPred {
    #[inline]
    fn intersect(&self, a: &Seg, b: &Seg) -> bool {
        segments_geometrically_intersect(a, b)
    }

    #[inline]
    fn cmp_post_x(&self, a: &Seg, b: &Seg, x_evt: f64) -> Ordering {
        cmp_post_x(a, b, x_evt)
    }

    #[inline]
    fn intersection_x(&self, a: &Seg, b: &Seg) -> Option<f64> {
        intersection_x(a, b)
    }

    #[inline]
    fn cmp_slope(&self, a: &Seg, b: &Seg) -> Ordering {
        a.slope.total_cmp(&b.slope)
    }

    #[inline]
    fn same_right_endpoint(&self, a: &Seg, b: &Seg) -> bool {
        a.x2 == b.x2 && a.y2 == b.y2
    }

    #[inline]
    fn is_collinear_det_zero(&self, a: &Seg, b: &Seg) -> bool {
        a.slope == b.slope && (a.x2 - a.x1) * (b.y2 - b.y1) - (a.y2 - a.y1) * (b.x2 - b.x1) == 0.0
    }

    #[inline]
    fn xi_lt_sweep(&self, _a: &Seg, _b: &Seg, xi: f64, sweep_x: f64) -> bool {
        xi < sweep_x
    }

    #[inline]
    fn xi_eq_sweep(&self, _a: &Seg, _b: &Seg, xi: f64, sweep_x: f64) -> bool {
        xi == sweep_x
    }
}

#[inline]
fn next_up_f64(v: f64) -> f64 {
    if v.is_nan() || v == f64::INFINITY {
        return v;
    }
    if v == 0.0 {
        return f64::from_bits(1);
    }
    let bits = v.to_bits();
    if v > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        let next = f64::from_bits(bits - 1);
        if next == 0.0 { 0.0 } else { next }
    }
}

#[inline]
fn cmp_shifted_with_f64(a: ShiftedF64, b: f64) -> Ordering {
    if a.shift == 0 {
        a.base.total_cmp(&b)
    } else {
        dd_cmp(a.sub_f64_dd(b), Dd::ZERO)
    }
}

pub(super) struct TorusExactPred<'a>(pub(super) &'a [ShiftedSeg]);

impl CrossPred for TorusExactPred<'_> {
    #[inline]
    fn intersect(&self, a: &Seg, b: &Seg) -> bool {
        if (a.sx1 | a.sy1 | a.sx2 | a.sy2 | b.sx1 | b.sy1 | b.sx2 | b.sy2) == 0 {
            segs_intersect_exact_f64([a.x1, a.y1, a.x2, a.y2], [b.x1, b.y1, b.x2, b.y2])
        } else {
            torus_segments_intersect(&self.0[a.id], &self.0[b.id])
        }
    }

    #[inline]
    fn cmp_post_x(&self, a: &Seg, b: &Seg, x_evt: f64) -> Ordering {
        if a.id == b.id {
            return Ordering::Equal;
        }
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        if ma.y_max.cmp_exact(mb.y_min) == Ordering::Less {
            return Ordering::Less;
        }
        if mb.y_max.cmp_exact(ma.y_min) == Ordering::Less {
            return Ordering::Greater;
        }

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

        if ma.p2 == mb.p2 && cmp_shifted_with_f64(ma.p2.x, x_evt) == Ordering::Greater {
            return cmp_slope_shifted(mb.p1, mb.p2, ma.p1, ma.p2).then_with(|| a.id.cmp(&b.id));
        }
        if ma.p1 == mb.p1 && cmp_shifted_with_f64(ma.p1.x, x_evt) != Ordering::Greater {
            return cmp_slope_shifted(ma.p1, ma.p2, mb.p1, mb.p2).then_with(|| a.id.cmp(&b.id));
        }

        let det_ord = det_sign_shifted(ma.p1, ma.p2, mb.p1, mb.p2);
        if det_ord == Ordering::Equal {
            return a.id.cmp(&b.id);
        }

        let post_crossing =
            exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x_evt) != Some(Ordering::Greater);
        if post_crossing {
            cmp_slope_shifted(ma.p1, ma.p2, mb.p1, mb.p2)
                .then_with(|| Ordering::Equal.cmp(&det_ord))
                .then_with(|| a.id.cmp(&b.id))
        } else {
            cmp_slope_shifted(mb.p1, mb.p2, ma.p1, ma.p2)
                .then_with(|| det_ord)
                .then_with(|| a.id.cmp(&b.id))
        }
    }

    #[inline]
    fn intersection_x(&self, a: &Seg, b: &Seg) -> Option<f64> {
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        let (xi, _) = dd_intersection_xy(ma.p1, ma.p2, mb.p1, mb.p2)?;
        let lo = a.x_min.max(b.x_min);
        let hi = a.x_max.min(b.x_max);
        let mut x = xi.to_f64().clamp(lo, hi);
        // Ensure `x >= xi_exact` (within [lo, hi]) so `cmp_post_x` agrees that `xi <= x`
        // and `reorder_surviving_bands_at_x` never reverts a completed intersection swap.
        for _ in 0..4 {
            if x >= hi || exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x) != Some(Ordering::Greater) {
                break;
            }
            x = next_up_f64(x).min(hi);
        }
        Some(x)
    }

    #[inline]
    fn cmp_slope(&self, a: &Seg, b: &Seg) -> Ordering {
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        cmp_slope_shifted(ma.p1, ma.p2, mb.p1, mb.p2)
    }

    #[inline]
    fn same_right_endpoint(&self, a: &Seg, b: &Seg) -> bool {
        self.0[a.id].p2 == self.0[b.id].p2
    }

    #[inline]
    fn is_collinear_det_zero(&self, a: &Seg, b: &Seg) -> bool {
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        det_sign_shifted(ma.p1, ma.p2, mb.p1, mb.p2) == Ordering::Equal
    }

    #[inline]
    fn xi_lt_sweep(&self, a: &Seg, b: &Seg, _xi: f64, sweep_x: f64) -> bool {
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, sweep_x) == Some(Ordering::Less)
    }

    #[inline]
    fn xi_eq_sweep(&self, a: &Seg, b: &Seg, _xi: f64, sweep_x: f64) -> bool {
        let ma = &self.0[a.id];
        let mb = &self.0[b.id];
        exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, sweep_x) == Some(Ordering::Equal)
    }
}
