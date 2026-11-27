use std::cmp::Ordering;

use super::double_double::two_diff;
use super::shifted_coords::{ShiftedF64, ShiftedPt};

const LIMBS: usize = 68;

#[inline]
fn f64_to_mant_exp(v: f64) -> Option<(i64, i32)> {
    if v == 0.0 {
        return None;
    }
    let bits = v.to_bits();
    let neg = (bits >> 63) != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1_u64 << 52) - 1);
    let (mag, exp) = if exp_bits == 0 {
        (frac as i64, -1074)
    } else {
        ((frac | (1_u64 << 52)) as i64, exp_bits - 1075)
    };
    Some((if neg { -mag } else { mag }, exp))
}

#[derive(Clone, Copy)]
pub(super) struct ExactFactor {
    terms: [(i64, i32); 3],
    len: usize,
}

impl ExactFactor {
    #[inline]
    pub(super) fn from_delta(a: ShiftedF64, b: ShiftedF64) -> Self {
        let ds = (b.shift as i32 - a.shift as i32) as f64;
        let (d_hi, d_lo) = two_diff(b.base, a.base);
        Self::from_triple([ds, d_hi, d_lo])
    }

    #[inline]
    pub(super) fn from_shifted_minus_f64(a: ShiftedF64, bound: f64) -> Self {
        let ds = a.shift as f64;
        let (d_hi, d_lo) = two_diff(a.base, bound);
        Self::from_triple([ds, d_hi, d_lo])
    }

    #[inline]
    pub(super) fn from_one() -> Self {
        Self::from_triple([1.0, 0.0, 0.0])
    }

    #[inline]
    pub(super) fn from_pair(pos: f64, neg: f64) -> Self {
        Self::from_triple([pos, -neg, 0.0])
    }

    #[inline]
    pub(super) fn from_triple(vals: [f64; 3]) -> Self {
        let mut terms = [(0_i64, 0_i32); 3];
        let mut len = 0usize;
        for v in vals {
            if let Some(me) = f64_to_mant_exp(v) {
                terms[len] = me;
                len += 1;
            }
        }
        Self { terms, len }
    }
}

#[inline]
fn add_limb_quad(limbs: &mut [u64; LIMBS], word: usize, w: [u64; 4]) {
    let mut carry = 0_u64;
    for (offset, &val) in w.iter().enumerate() {
        let idx = word + offset;
        if idx >= LIMBS {
            break;
        }
        let (s1, c1) = limbs[idx].overflowing_add(val);
        let (s2, c2) = s1.overflowing_add(carry);
        limbs[idx] = s2;
        carry = u64::from(c1) + u64::from(c2);
    }
    let mut idx = word + 4;
    while carry != 0 && idx < LIMBS {
        let (s, c) = limbs[idx].overflowing_add(carry);
        limbs[idx] = s;
        carry = u64::from(c);
        idx += 1;
    }
}

pub(super) fn exact_deg2_sign(pairs: &[(ExactFactor, ExactFactor, i8)]) -> Ordering {
    let mut min_e = i32::MAX;
    let mut max_e = i32::MIN;
    let mut any = false;
    for &(f1, f2, _) in pairs {
        for &(_, e1) in &f1.terms[..f1.len] {
            for &(_, e2) in &f2.terms[..f2.len] {
                let e = e1 + e2;
                min_e = min_e.min(e);
                max_e = max_e.max(e);
                any = true;
            }
        }
    }
    if !any {
        return Ordering::Equal;
    }
    let num_limbs = ((max_e - min_e) as usize / 64 + 5).min(LIMBS);
    let mut pos = [0_u64; LIMBS];
    let mut neg = [0_u64; LIMBS];

    for &(f1, f2, sign) in pairs {
        for &(m1, e1) in &f1.terms[..f1.len] {
            for &(m2, e2) in &f2.terms[..f2.len] {
                let prod = (m1 as i128) * (m2 as i128) * (sign as i128);
                let shift = (e1 + e2 - min_e) as usize;
                let word = shift / 64;
                let bit = (shift % 64) as u32;
                let abs_p = prod.unsigned_abs();
                let w0 = (abs_p as u64) << bit;
                let rest = abs_p >> (64 - bit);
                let w = [w0, rest as u64, (rest >> 64) as u64, 0];
                if prod > 0 {
                    add_limb_quad(&mut pos, word, w);
                } else {
                    add_limb_quad(&mut neg, word, w);
                }
            }
        }
    }

    for idx in (0..num_limbs).rev() {
        match pos[idx].cmp(&neg[idx]) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    Ordering::Equal
}

pub(super) fn exact_deg3_sign(
    triples: &[(ExactFactor, ExactFactor, ExactFactor, i8); 3],
) -> Ordering {
    let mut min_e = i32::MAX;
    let mut max_e = i32::MIN;
    let mut any = false;
    for &(f1, f2, f3, _) in triples {
        for &(_, e1) in &f1.terms[..f1.len] {
            for &(_, e2) in &f2.terms[..f2.len] {
                for &(_, e3) in &f3.terms[..f3.len] {
                    let e = e1 + e2 + e3;
                    min_e = min_e.min(e);
                    max_e = max_e.max(e);
                    any = true;
                }
            }
        }
    }
    if !any {
        return Ordering::Equal;
    }
    let num_limbs = ((max_e - min_e) as usize / 64 + 6).min(LIMBS);
    let mut pos = [0_u64; LIMBS];
    let mut neg = [0_u64; LIMBS];

    for &(f1, f2, f3, sign) in triples {
        for &(m1, e1) in &f1.terms[..f1.len] {
            for &(m2, e2) in &f2.terms[..f2.len] {
                let ab = (m1.unsigned_abs() as u128) * (m2.unsigned_abs() as u128);
                let ab_lo = ab as u64 as u128;
                let ab_hi = ab >> 64;
                for &(m3, e3) in &f3.terms[..f3.len] {
                    let c = m3.unsigned_abs() as u128;
                    let p0 = ab_lo * c;
                    let p1 = ab_hi * c + (p0 >> 64);
                    let (w0, w1, w2) = (p0 as u64, p1 as u64, (p1 >> 64) as u64);
                    let shift = (e1 + e2 + e3 - min_e) as usize;
                    let word = shift / 64;
                    let bit = (shift % 64) as u32;
                    let sw = if bit == 0 {
                        [w0, w1, w2, 0]
                    } else {
                        [
                            w0 << bit,
                            (w1 << bit) | (w0 >> (64 - bit)),
                            (w2 << bit) | (w1 >> (64 - bit)),
                            w2 >> (64 - bit),
                        ]
                    };
                    let is_neg = (m1 < 0) ^ (m2 < 0) ^ (m3 < 0) ^ (sign < 0);
                    if !is_neg {
                        add_limb_quad(&mut pos, word, sw);
                    } else {
                        add_limb_quad(&mut neg, word, sw);
                    }
                }
            }
        }
    }

    for idx in (0..num_limbs).rev() {
        match pos[idx].cmp(&neg[idx]) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    Ordering::Equal
}

#[derive(Clone, Copy)]
struct RatBound {
    num: ExactFactor,
    den: ExactFactor,
    open: bool,
}

#[inline]
fn cmp_rat(a: RatBound, b: RatBound) -> Ordering {
    exact_deg2_sign(&[(a.num, b.den, 1), (b.num, a.den, -1)])
}

#[inline]
fn max_rat(a: RatBound, b: RatBound) -> RatBound {
    match cmp_rat(a, b) {
        Ordering::Greater => a,
        Ordering::Less => b,
        Ordering::Equal => RatBound {
            num: a.num,
            den: a.den,
            open: a.open || b.open,
        },
    }
}

#[inline]
fn min_rat(a: RatBound, b: RatBound) -> RatBound {
    match cmp_rat(a, b) {
        Ordering::Less => a,
        Ordering::Greater => b,
        Ordering::Equal => RatBound {
            num: a.num,
            den: a.den,
            open: a.open || b.open,
        },
    }
}

pub(super) fn exact_collinear_in_box(
    o1: ShiftedPt,
    o2: ShiftedPt,
    x_hi: ShiftedF64,
    x_hi_open: bool,
    y_hi: ShiftedF64,
    y_hi_open: bool,
) -> bool {
    let in_box = |p: ShiftedPt| {
        p.x.cmp_exact(ShiftedF64::ZERO) != Ordering::Less
            && if x_hi_open {
                p.x.cmp_exact(x_hi) == Ordering::Less
            } else {
                p.x.cmp_exact(x_hi) != Ordering::Greater
            }
            && p.y.cmp_exact(ShiftedF64::ZERO) != Ordering::Less
            && if y_hi_open {
                p.y.cmp_exact(y_hi) == Ordering::Less
            } else {
                p.y.cmp_exact(y_hi) != Ordering::Greater
            }
    };
    if in_box(o1) || in_box(o2) {
        return true;
    }
    if o1 == o2 {
        return false;
    }

    let one = ExactFactor::from_one();
    let mut lo = RatBound {
        num: ExactFactor::from_triple([0.0, 0.0, 0.0]),
        den: one,
        open: false,
    };
    let mut hi = RatBound {
        num: one,
        den: one,
        open: false,
    };

    for (c1, c2, c_hi, hi_open) in [(o1.x, o2.x, x_hi, x_hi_open), (o1.y, o2.y, y_hi, y_hi_open)] {
        match c1.cmp_exact(c2) {
            Ordering::Equal => {
                let in_axis = c1.cmp_exact(ShiftedF64::ZERO) != Ordering::Less
                    && if hi_open {
                        c1.cmp_exact(c_hi) == Ordering::Less
                    } else {
                        c1.cmp_exact(c_hi) != Ordering::Greater
                    };
                if !in_axis {
                    return false;
                }
            }
            Ordering::Less => {
                let d = ExactFactor::from_delta(c1, c2);
                let b_lo = RatBound {
                    num: ExactFactor::from_delta(c1, ShiftedF64::ZERO),
                    den: d,
                    open: false,
                };
                let b_hi = RatBound {
                    num: ExactFactor::from_delta(c1, c_hi),
                    den: d,
                    open: hi_open,
                };
                lo = max_rat(lo, b_lo);
                hi = min_rat(hi, b_hi);
            }
            Ordering::Greater => {
                let d = ExactFactor::from_delta(c2, c1);
                let b_lo = RatBound {
                    num: ExactFactor::from_delta(c_hi, c1),
                    den: d,
                    open: hi_open,
                };
                let b_hi = RatBound {
                    num: ExactFactor::from_delta(ShiftedF64::ZERO, c1),
                    den: d,
                    open: false,
                };
                lo = max_rat(lo, b_lo);
                hi = min_rat(hi, b_hi);
            }
        }
    }

    match cmp_rat(lo, hi) {
        Ordering::Less => true,
        Ordering::Equal => !lo.open && !hi.open,
        Ordering::Greater => false,
    }
}
