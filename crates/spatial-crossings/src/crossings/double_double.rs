use std::cmp::Ordering;

const VELTKAMP_SPLIT: f64 = 134_217_729.0; // 2^27 + 1

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Dd {
    pub(super) hi: f64,
    pub(super) lo: f64,
}

impl Dd {
    pub(super) const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub(super) const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    #[inline]
    pub(super) const fn from_f64(v: f64) -> Self {
        Self { hi: v, lo: 0.0 }
    }

    #[inline]
    pub(super) fn to_f64(self) -> f64 {
        let sum = self.hi + self.lo;
        if sum == 0.0 { 0.0 } else { sum }
    }

    #[inline]
    pub(super) fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }

    #[inline]
    pub(super) fn is_neg(self) -> bool {
        self.hi + self.lo < 0.0
    }
}

/// Error-free Knuth `TwoSum` for `a + b`.
#[inline]
pub(super) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virt = s - a;
    let a_virt = s - b_virt;
    let b_err = b - b_virt;
    let a_err = a - a_virt;
    (s, a_err + b_err)
}

/// Error-free Knuth `TwoDiff` for `a - b`.
#[inline]
pub(super) fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    let b_virt = s - a;
    let a_virt = s - b_virt;
    let b_err = (-b) - b_virt;
    let a_err = a - a_virt;
    (s, a_err + b_err)
}

/// Deterministic pure-arithmetic Veltkamp-Dekker `TwoProd` for `a * b` (identical on native and `wasm32`).
#[inline]
pub(super) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let ca = VELTKAMP_SPLIT * a;
    let a_hi = ca - (ca - a);
    let a_lo = a - a_hi;
    let cb = VELTKAMP_SPLIT * b;
    let b_hi = cb - (cb - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
pub(super) fn dd_neg(a: Dd) -> Dd {
    Dd {
        hi: -a.hi,
        lo: -a.lo,
    }
}

#[inline]
pub(super) fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s1, s2) = two_sum(a.hi, b.hi);
    let (t1, t2) = two_sum(a.lo, b.lo);
    let (s2, t1) = two_sum(s2, t1);
    let (vh, vl) = two_sum(s1, s2);
    let (hi, lo) = two_sum(vh, vl + t1 + t2);
    Dd { hi, lo }
}

#[inline]
pub(super) fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, dd_neg(b))
}

#[inline]
pub(super) fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p1, p2) = two_prod(a.hi, b.hi);
    let p2 = p2 + (a.hi * b.lo + a.lo * b.hi);
    let (hi, lo) = two_sum(p1, p2);
    Dd { hi, lo }
}

#[inline]
pub(super) fn dd_mul_f64(a: Dd, b: f64) -> Dd {
    let (p1, p2) = two_prod(a.hi, b);
    let (hi, lo) = two_sum(p1, p2 + a.lo * b);
    Dd { hi, lo }
}

#[inline]
pub(super) fn dd_div(a: Dd, b: Dd) -> Dd {
    let q1 = a.hi / b.hi;
    let r = dd_sub(a, dd_mul_f64(b, q1));
    let q2 = r.hi / b.hi;
    let r = dd_sub(r, dd_mul_f64(b, q2));
    let q3 = r.hi / b.hi;
    let (hi, lo) = two_sum(q1, q2);
    dd_add(Dd { hi, lo }, Dd::from_f64(q3))
}

#[inline]
pub(super) fn dd_cmp(a: Dd, b: Dd) -> Ordering {
    let d = dd_sub(a, b);
    let sum = d.hi + d.lo;
    let s = if sum == 0.0 { 0.0 } else { sum };
    s.total_cmp(&0.0)
}
