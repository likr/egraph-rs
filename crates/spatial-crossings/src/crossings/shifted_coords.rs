use std::cmp::Ordering;

use super::double_double::{Dd, dd_add, two_diff, two_sum};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ShiftedF64 {
    pub(super) base: f64,
    pub(super) shift: i8,
}

impl ShiftedF64 {
    pub(super) const ZERO: Self = Self {
        base: 0.0,
        shift: 0,
    };
    pub(super) const ONE: Self = Self {
        base: 0.0,
        shift: 1,
    };

    #[inline]
    pub(super) fn new(base: f64, shift: i8) -> Self {
        Self {
            base: if base == 0.0 { 0.0 } else { base },
            shift,
        }
    }

    #[inline]
    pub(super) fn shifted_by(self, delta: i8) -> Self {
        Self {
            base: self.base,
            shift: self.shift + delta,
        }
    }

    #[inline]
    pub(super) fn cmp_exact(self, other: Self) -> Ordering {
        self.shift
            .cmp(&other.shift)
            .then_with(|| self.base.total_cmp(&other.base))
    }

    #[inline]
    pub(super) fn to_f64(self) -> f64 {
        let v = if self.shift == 0 {
            self.base
        } else {
            self.shift as f64 + self.base
        };
        if v == 0.0 { 0.0 } else { v }
    }

    #[inline]
    pub(super) fn to_dd(self) -> Dd {
        if self.shift == 0 {
            Dd::from_f64(self.base)
        } else {
            let (hi, lo) = two_sum(self.shift as f64, self.base);
            Dd { hi, lo }
        }
    }

    #[inline]
    pub(super) fn delta_dd(self, other: Self) -> Dd {
        let ds = (other.shift as i32 - self.shift as i32) as f64;
        let (d_hi, d_lo) = two_diff(other.base, self.base);
        dd_add(Dd::from_f64(ds), Dd { hi: d_hi, lo: d_lo })
    }

    #[inline]
    pub(super) fn sub_f64_dd(self, bound: f64) -> Dd {
        let (d_hi, d_lo) = two_diff(self.base, bound);
        dd_add(Dd::from_f64(self.shift as f64), Dd { hi: d_hi, lo: d_lo })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ShiftedPt {
    pub(super) x: ShiftedF64,
    pub(super) y: ShiftedF64,
}

impl ShiftedPt {
    #[inline]
    pub(super) const fn new(x: ShiftedF64, y: ShiftedF64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub(super) fn from_f64(x: f64, y: f64) -> Self {
        Self {
            x: ShiftedF64::new(x, 0),
            y: ShiftedF64::new(y, 0),
        }
    }

    #[inline]
    pub(super) fn cmp_exact(self, other: Self) -> Ordering {
        self.x
            .cmp_exact(other.x)
            .then_with(|| self.y.cmp_exact(other.y))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ShiftedSeg {
    pub(super) p1: ShiftedPt,
    pub(super) p2: ShiftedPt,
    pub(super) x_min: ShiftedF64,
    pub(super) x_max: ShiftedF64,
    pub(super) y_min: ShiftedF64,
    pub(super) y_max: ShiftedF64,
}

impl ShiftedSeg {
    #[inline]
    pub(super) fn from_shifted(a: ShiftedPt, b: ShiftedPt) -> Self {
        let (p1, p2) = if a.cmp_exact(b) != Ordering::Greater {
            (a, b)
        } else {
            (b, a)
        };
        let (x_min, x_max) = if p1.x.cmp_exact(p2.x) != Ordering::Greater {
            (p1.x, p2.x)
        } else {
            (p2.x, p1.x)
        };
        let (y_min, y_max) = if p1.y.cmp_exact(p2.y) != Ordering::Greater {
            (p1.y, p2.y)
        } else {
            (p2.y, p1.y)
        };
        Self {
            p1,
            p2,
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }
}
