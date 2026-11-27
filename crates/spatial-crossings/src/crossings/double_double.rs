use std::cmp::Ordering;

const VELTKAMP_SPLIT: f64 = 134_217_729.0; // 2^27 + 1
const POW2_995: f64 = f64::from_bits(((1023 + 995) as u64) << 52);
const POW2_M968: f64 = f64::from_bits(((1023 - 968) as u64) << 52);
const POW2_512: f64 = f64::from_bits(((1023 + 512) as u64) << 52);
const POW2_M512: f64 = f64::from_bits(((1023 - 512) as u64) << 52);

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

#[inline]
fn two_prod_core(a: f64, b: f64, p: f64) -> (f64, f64) {
    let ca = VELTKAMP_SPLIT * a;
    let a_hi = ca - (ca - a);
    let a_lo = a - a_hi;
    let cb = VELTKAMP_SPLIT * b;
    let b_hi = cb - (cb - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[cold]
#[inline(never)]
fn two_prod_scaled(a: f64, b: f64) -> (f64, f64) {
    let mut sa = a;
    let mut sb = b;
    let mut exp_shift = 0_i32;

    if sa.abs() > POW2_995 {
        sa *= POW2_M512;
        exp_shift += 512;
    } else if sa.abs() < POW2_M968 {
        sa *= POW2_512;
        exp_shift -= 512;
    }

    if sb.abs() > POW2_995 {
        sb *= POW2_M512;
        exp_shift += 512;
    } else if sb.abs() < POW2_M968 {
        sb *= POW2_512;
        exp_shift -= 512;
    }

    let mut sp = sa * sb;
    if exp_shift == 0 {
        if sp.abs() > POW2_995 {
            if sa.abs() >= sb.abs() {
                sa *= POW2_M512;
            } else {
                sb *= POW2_M512;
            }
            exp_shift += 512;
            sp = sa * sb;
        } else if sp.abs() < POW2_M968 {
            if sa.abs() <= sb.abs() {
                sa *= POW2_512;
            } else {
                sb *= POW2_512;
            }
            exp_shift -= 512;
            sp = sa * sb;
        }
    }

    let (p, err) = two_prod_core(sa, sb, sp);
    match exp_shift {
        512 => (p * POW2_512, err * POW2_512),
        -512 => (p * POW2_M512, err * POW2_M512),
        0 => (p, err),
        _ => {
            let scale = 2.0_f64.powi(exp_shift);
            (p * scale, err * scale)
        }
    }
}

/// Deterministic pure-arithmetic Veltkamp-Dekker `TwoProd` for `a * b` (identical on native and `wasm32`).
/// Rescales extreme exponents by exact powers of two so Veltkamp splitting never overflows or loses subnormal precision.
#[inline]
pub(super) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if !p.is_finite() || p == 0.0 {
        return (p, 0.0);
    }
    let aa = a.abs();
    let bb = b.abs();
    let pa = p.abs();
    if (POW2_M968..=POW2_995).contains(&aa)
        && (POW2_M968..=POW2_995).contains(&bb)
        && (POW2_M968..=POW2_995).contains(&pa)
    {
        return two_prod_core(a, b, p);
    }
    two_prod_scaled(a, b)
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
fn dd_div_core(a: Dd, b: Dd, q1: f64) -> Dd {
    let r = dd_sub(a, dd_mul_f64(b, q1));
    let q2 = r.hi / b.hi;
    let r = dd_sub(r, dd_mul_f64(b, q2));
    let q3 = r.hi / b.hi;
    let (hi, lo) = two_sum(q1, q2);
    dd_add(Dd { hi, lo }, Dd::from_f64(q3))
}

#[inline]
fn normalize_dd_pow2(x: Dd) -> (Dd, i32) {
    let (mut hi, mut lo) = if x.hi == 0.0 && x.lo != 0.0 {
        two_sum(x.hi, x.lo)
    } else {
        (x.hi, x.lo)
    };
    let mut shift = 0_i32;
    if hi.abs() < POW2_M968 {
        hi *= POW2_512;
        lo *= POW2_512;
        shift -= 512;
    } else if hi.abs() > POW2_995 {
        hi *= POW2_M512;
        lo *= POW2_M512;
        shift += 512;
    }
    let exp = ((hi.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let inv = 2.0_f64.powi(-exp);
    (
        Dd {
            hi: hi * inv,
            lo: lo * inv,
        },
        shift + exp,
    )
}

#[cold]
#[inline(never)]
fn dd_div_scaled(a: Dd, b: Dd) -> Dd {
    let (na, ea) = normalize_dd_pow2(a);
    let (nb, eb) = normalize_dd_pow2(b);
    let q1 = na.hi / nb.hi;
    let q = dd_div_core(na, nb, q1);
    let shift = ea - eb;
    if shift > 1024 {
        return Dd::from_f64(q.hi.signum() * f64::MAX);
    }
    if shift < -1076 {
        return Dd::ZERO;
    }
    let s1 = shift.clamp(-512, 512);
    let s2 = shift - s1;
    let f1 = 2.0_f64.powi(s1);
    let f2 = 2.0_f64.powi(s2);
    let hi = (q.hi * f1) * f2;
    let lo = (q.lo * f1) * f2;
    if !hi.is_finite() || !(hi + lo).is_finite() {
        return Dd::from_f64(q.hi.signum() * f64::MAX);
    }
    Dd {
        hi,
        lo: if lo.is_finite() { lo } else { 0.0 },
    }
}

#[inline]
pub(super) fn dd_div(a: Dd, b: Dd) -> Dd {
    if !a.hi.is_finite() || !a.lo.is_finite() || !b.hi.is_finite() || !b.lo.is_finite() {
        return Dd::from_f64(a.hi / b.hi);
    }
    if b.hi == 0.0 && b.lo == 0.0 {
        if a.is_zero() {
            return Dd::from_f64(f64::NAN);
        }
        let a_lead = if a.hi != 0.0 { a.hi } else { a.lo };
        return Dd::from_f64(a_lead.signum() * b.hi.signum() * f64::MAX);
    }
    if a.is_zero() {
        return Dd::ZERO;
    }
    if a.hi == 0.0 || b.hi == 0.0 {
        return dd_div_scaled(a, b);
    }
    let q1 = a.hi / b.hi;
    let aa = a.hi.abs();
    let ba = b.hi.abs();
    let qa = q1.abs();
    if (POW2_M968..=POW2_995).contains(&aa)
        && (POW2_M968..=POW2_995).contains(&ba)
        && qa <= POW2_995
        && (qa >= POW2_M968 || q1 == 0.0)
    {
        return dd_div_core(a, b, q1);
    }
    dd_div_scaled(a, b)
}

#[inline]
pub(super) fn dd_cmp(a: Dd, b: Dd) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let d = dd_sub(a, b);
    let sum = d.hi + d.lo;
    if sum.is_finite() {
        let s = if sum == 0.0 { 0.0 } else { sum };
        s.total_cmp(&0.0)
    } else {
        a.hi.total_cmp(&b.hi).then_with(|| a.lo.total_cmp(&b.lo))
    }
}
