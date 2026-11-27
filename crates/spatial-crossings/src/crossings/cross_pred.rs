use std::cmp::Ordering;

use super::double_double::{Dd, dd_add, dd_cmp, dd_div, dd_mul, dd_neg, dd_sub};
use super::euclidean_geom::{
    cmp_post_x, cmp_slope_euclidean, intersection_x, segments_geometrically_intersect,
};
use super::geometry::{STEEP_SLOPE_THRESHOLD, Seg};
use super::torus_exact::{
    ShiftedF64, ShiftedPt, ShiftedSeg, cmp_slope_shifted, dd_intersection_xy, det_sign_shifted,
    exact_xi_cmp, orient2d_shifted, segs_intersect_exact_f64, torus_segments_intersect,
};

pub(super) trait CrossPred {
    fn intersect(&self, a: &Seg, b: &Seg) -> bool;
    #[inline]
    fn cross(&self, a: &Seg, b: &Seg) -> bool {
        !a.shares_endpoint(b) && self.intersect(a, b)
    }
    fn y_at(&self, s: &Seg, x: f64) -> f64;
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
    fn y_at(&self, s: &Seg, x: f64) -> f64 {
        s.y_at(x)
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
        cmp_slope_euclidean(a, b)
    }

    #[inline]
    fn same_right_endpoint(&self, a: &Seg, b: &Seg) -> bool {
        a.x2 == b.x2 && a.y2 == b.y2
    }

    #[inline]
    fn is_collinear_det_zero(&self, a: &Seg, b: &Seg) -> bool {
        cmp_slope_euclidean(a, b) == Ordering::Equal
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
fn prev_down_f64(v: f64) -> f64 {
    if v.is_nan() || v == f64::NEG_INFINITY {
        return v;
    }
    if v == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = v.to_bits();
    if v > 0.0 {
        let prev = f64::from_bits(bits - 1);
        if prev == 0.0 { 0.0 } else { prev }
    } else {
        f64::from_bits(bits + 1)
    }
}

#[inline]
fn f64_to_sortable_u64(x: f64) -> u64 {
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    if (bits >> 63) != 0 {
        !bits
    } else {
        bits ^ (1_u64 << 63)
    }
}

#[inline]
fn sortable_u64_to_f64(k: u64) -> f64 {
    let bits = if (k >> 63) != 0 {
        k ^ (1_u64 << 63)
    } else {
        !k
    };
    let v = f64::from_bits(bits);
    if v == 0.0 { 0.0 } else { v }
}

/// Binary-searches `[lo_x, hi_x]` in IEEE-754 sortable bit space for the smallest `f64` `x >= xi_exact`
/// (`<= 64` exact predicate probes) when 106-bit DD intersection coordinates collapse across deep exponents.
#[cold]
#[inline(never)]
pub(super) fn binary_search_xi_ceil(
    a1: ShiftedPt,
    a2: ShiftedPt,
    b1: ShiftedPt,
    b2: ShiftedPt,
    lo_x: f64,
    hi_x: f64,
) -> f64 {
    let mut k_lo = f64_to_sortable_u64(lo_x);
    let mut k_hi = f64_to_sortable_u64(hi_x);
    while k_lo < k_hi {
        let mid = k_lo + ((k_hi - k_lo) >> 1);
        let x_mid = sortable_u64_to_f64(mid);
        if exact_xi_cmp(a1, a2, b1, b2, x_mid) != Some(Ordering::Greater) {
            k_hi = mid;
        } else {
            k_lo = mid + 1;
        }
    }
    sortable_u64_to_f64(k_hi).clamp(lo_x, hi_x)
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
    fn y_at(&self, s: &Seg, x: f64) -> f64 {
        if s.slope.abs() < STEEP_SLOPE_THRESHOLD && (s.sx1 | s.sy1 | s.sx2 | s.sy2) == 0 {
            return s.y_at(x);
        }
        let ma = &self.0[s.id];
        if cmp_shifted_with_f64(ma.p1.x, x) != Ordering::Less {
            return s.y1;
        }
        if cmp_shifted_with_f64(ma.p2.x, x) != Ordering::Greater {
            return s.y2;
        }
        if ma.p1.y == ma.p2.y {
            return s.y1;
        }
        let dx = ma.p1.x.delta_dd(ma.p2.x);
        let dy = ma.p1.y.delta_dd(ma.p2.y);
        let dl = dd_neg(ma.p1.x.sub_f64_dd(x));
        let dr = ma.p2.x.sub_f64_dd(x);
        let y = if dd_cmp(dl, dr) != Ordering::Greater {
            let t = dd_div(dl, dx);
            dd_add(ma.p1.y.to_dd(), dd_mul(t, dy)).to_f64()
        } else {
            let u = dd_div(dr, dx);
            dd_sub(ma.p2.y.to_dd(), dd_mul(u, dy)).to_f64()
        };
        if y.is_finite() {
            y.clamp(s.y_min, s.y_max)
        } else {
            s.y_at(x)
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
        // Ensure `x` equals the exact ceiling `f64` of `xi_exact` within `[lo, hi]` so
        // `cmp_post_x` never reverts a scheduled swap and intermediate events stay ordered.
        match exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x) {
            Some(Ordering::Greater) => {
                for _ in 0..4 {
                    if x >= hi
                        || exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x) != Some(Ordering::Greater)
                    {
                        break;
                    }
                    x = next_up_f64(x).min(hi);
                }
                if x < hi && exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, x) == Some(Ordering::Greater)
                {
                    if exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, hi) != Some(Ordering::Greater) {
                        x = binary_search_xi_ceil(ma.p1, ma.p2, mb.p1, mb.p2, x, hi);
                    } else {
                        x = hi;
                    }
                }
            }
            Some(Ordering::Less) if x > lo => {
                let prev = prev_down_f64(x).max(lo);
                match exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, prev) {
                    Some(Ordering::Equal) => x = prev,
                    Some(Ordering::Less) => {
                        x = prev;
                        for _ in 0..3 {
                            if x <= lo {
                                break;
                            }
                            let p = prev_down_f64(x).max(lo);
                            match exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, p) {
                                Some(Ordering::Less) => x = p,
                                Some(Ordering::Equal) => {
                                    x = p;
                                    break;
                                }
                                _ => break,
                            }
                        }
                        if x > lo
                            && exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, prev_down_f64(x).max(lo))
                                == Some(Ordering::Less)
                        {
                            if exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, lo)
                                != Some(Ordering::Greater)
                            {
                                x = lo;
                            } else {
                                x = binary_search_xi_ceil(ma.p1, ma.p2, mb.p1, mb.p2, lo, x);
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
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
