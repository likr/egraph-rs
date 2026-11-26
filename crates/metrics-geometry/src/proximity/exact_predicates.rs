use std::cmp::Ordering;

use egraph_core::{Layout2d, Scalar};

#[inline]
pub(super) fn dist_sq_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Error-free Knuth `TwoSum` for `a - b`, exact for all finite `f64` when `|a|, |b| <= 0.25 * f64::MAX`.
#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    let b_virt = s - a;
    let a_virt = s - b_virt;
    let b_err = (-b) - b_virt;
    let a_err = a - a_virt;
    (s, a_err + b_err)
}

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

#[inline]
fn push_prod_term(u: f64, v: f64, sign: i8, terms: &mut [(i128, i32); 16], len: &mut usize) {
    if let (Some((mu, eu)), Some((mv, ev))) = (f64_to_mant_exp(u), f64_to_mant_exp(v)) {
        let prod = (mu as i128) * (mv as i128) * (sign as i128);
        terms[*len] = (prod, eu + ev);
        *len += 1;
    }
}

fn push_cross_terms(
    o: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    sign: i8,
    terms: &mut [(i128, i32); 16],
    len: &mut usize,
) {
    let (dx1_hi, dx1_lo) = two_diff(a[0], o[0]);
    let (dy1_hi, dy1_lo) = two_diff(a[1], o[1]);
    let (dx2_hi, dx2_lo) = two_diff(b[0], o[0]);
    let (dy2_hi, dy2_lo) = two_diff(b[1], o[1]);

    push_prod_term(dx1_hi, dy2_hi, sign, terms, len);
    push_prod_term(dx1_hi, dy2_lo, sign, terms, len);
    push_prod_term(dx1_lo, dy2_hi, sign, terms, len);
    push_prod_term(dx1_lo, dy2_lo, sign, terms, len);

    push_prod_term(dy1_hi, dx2_hi, -sign, terms, len);
    push_prod_term(dy1_hi, dx2_lo, -sign, terms, len);
    push_prod_term(dy1_lo, dx2_hi, -sign, terms, len);
    push_prod_term(dy1_lo, dx2_lo, -sign, terms, len);
}

#[inline]
fn add_limb_triple(limbs: &mut [u64; 68], word: usize, w0: u64, w1: u64, w2: u64) {
    let (sum0, c0) = limbs[word].overflowing_add(w0);
    limbs[word] = sum0;
    let (sum1a, c1a) = limbs[word + 1].overflowing_add(w1);
    let (sum1b, c1b) = sum1a.overflowing_add(u64::from(c0));
    limbs[word + 1] = sum1b;
    let (sum2a, c2a) = limbs[word + 2].overflowing_add(w2);
    let (sum2b, c2b) = sum2a.overflowing_add(u64::from(c1a) + u64::from(c1b));
    limbs[word + 2] = sum2b;
    let mut carry = u64::from(c2a) + u64::from(c2b);
    let mut idx = word + 3;
    while carry != 0 && idx < limbs.len() {
        let (sum, c) = limbs[idx].overflowing_add(carry);
        limbs[idx] = sum;
        carry = u64::from(c);
        idx += 1;
    }
}

/// Evaluates the exact sign of `\sum_k m_k * 2^{e_k}` with zero floating-point error,
/// covering the full IEEE-754 finite exponent range (`|coords| <= 0.25 * f64::MAX`).
fn exact_sum_sign(terms: &[(i128, i32)]) -> Ordering {
    if terms.is_empty() {
        return Ordering::Equal;
    }
    let mut min_e = i32::MAX;
    let mut max_e = i32::MIN;
    for &(_, e) in terms {
        min_e = min_e.min(e);
        max_e = max_e.max(e);
    }
    let num_limbs = ((max_e - min_e) as usize / 64 + 4).min(68);
    let mut pos = [0_u64; 68];
    let mut neg = [0_u64; 68];

    for &(m, e) in terms {
        if m == 0 {
            continue;
        }
        let shift = (e - min_e) as usize;
        let word = shift / 64;
        let bit = (shift % 64) as u32;
        let abs_m = m.unsigned_abs();
        let w0 = (abs_m as u64) << bit;
        let rest = abs_m >> (64 - bit);
        let w1 = rest as u64;
        let w2 = (rest >> 64) as u64;
        if m > 0 {
            add_limb_triple(&mut pos, word, w0, w1, w2);
        } else {
            add_limb_triple(&mut neg, word, w0, w1, w2);
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

fn orient2d_exact(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Ordering {
    let mut terms = [(0_i128, 0_i32); 16];
    let mut len = 0usize;
    push_cross_terms(o, a, b, 1, &mut terms, &mut len);
    exact_sum_sign(&terms[..len])
}

/// Exact orientation sign of `(a - o) x (b - o)` with a fast `f64` error-bound filter.
#[inline]
pub(super) fn orient2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Ordering {
    let det_left = (a[0] - o[0]) * (b[1] - o[1]);
    let det_right = (a[1] - o[1]) * (b[0] - o[0]);
    let det = det_left - det_right;
    let det_sum = det_left.abs() + det_right.abs();
    if det_sum.is_finite() && (det_sum == 0.0 || det_sum > 1e-280) {
        let err_bound = 8.0 * f64::EPSILON * det_sum;
        if det.abs() > err_bound {
            return if det > 0.0 {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
    }
    orient2d_exact(o, a, b)
}

/// Exact comparison of `|(a - o) x (b1 - o)|` vs `|(a - o) x (b2 - o)|` with a fast `f64` filter.
#[inline]
pub(super) fn cross_abs_cmp(o: [f64; 2], a: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> Ordering {
    let dx = a[0] - o[0];
    let dy = a[1] - o[1];

    let l1 = dx * (b1[1] - o[1]);
    let r1 = dy * (b1[0] - o[0]);
    let det1 = l1 - r1;
    let sum1 = l1.abs() + r1.abs();

    let l2 = dx * (b2[1] - o[1]);
    let r2 = dy * (b2[0] - o[0]);
    let det2 = l2 - r2;
    let sum2 = l2.abs() + r2.abs();

    if sum1.is_finite()
        && sum2.is_finite()
        && (sum1 == 0.0 || sum1 > 1e-280)
        && (sum2 == 0.0 || sum2 > 1e-280)
    {
        let err1 = 8.0 * f64::EPSILON * sum1;
        let err2 = 8.0 * f64::EPSILON * sum2;
        let abs1 = det1.abs();
        let abs2 = det2.abs();
        if abs1 - err1 > abs2 + err2 {
            return Ordering::Greater;
        }
        if abs2 - err2 > abs1 + err1 {
            return Ordering::Less;
        }
    }

    let s1 = orient2d_exact(o, a, b1);
    let s2 = orient2d_exact(o, a, b2);
    match (s1, s2) {
        (Ordering::Equal, Ordering::Equal) => Ordering::Equal,
        (Ordering::Equal, _) => Ordering::Less,
        (_, Ordering::Equal) => Ordering::Greater,
        _ => {
            let mut terms = [(0_i128, 0_i32); 16];
            let mut len = 0usize;
            let sign1 = if s1 == Ordering::Greater { 1 } else { -1 };
            let sign2 = if s2 == Ordering::Greater { -1 } else { 1 };
            push_cross_terms(o, a, b1, sign1, &mut terms, &mut len);
            push_cross_terms(o, a, b2, sign2, &mut terms, &mut len);
            exact_sum_sign(&terms[..len])
        }
    }
}

#[inline]
fn dist_2d_safe(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let sq = dx * dx + dy * dy;
    if sq.is_finite() && (sq > 0.0 || (dx == 0.0 && dy == 0.0)) {
        sq.sqrt()
    } else {
        dx.hypot(dy)
    }
}

/// Computes the exact Euclidean diameter $\max_{i, j} \|p_i - p_j\|_2$ in $O(N \log N)$ time
/// via exact-predicate Andrew's monotone chain convex hull and rotating calipers.
pub(super) fn euclidean_diameter_2d<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    if n <= 64 {
        let mut best_d = 0.0_f64;
        for i in 0..n {
            let p_i = [xs[i].to_f64(), ys[i].to_f64()];
            for j in (i + 1)..n {
                let p_j = [xs[j].to_f64(), ys[j].to_f64()];
                best_d = best_d.max(dist_2d_safe(p_i, p_j));
            }
        }
        return best_d;
    }

    // Canonicalize `-0.0` to `+0.0` so `total_cmp` and `dedup_by` (`==`) agree
    // and never leave non-adjacent `(0.0, 0.0)` duplicates that collapse the hull.
    let mut max_abs = 0.0_f64;
    let mut pts: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let x = xs[i].to_f64() + 0.0;
            let y = ys[i].to_f64() + 0.0;
            max_abs = max_abs.max(x.abs()).max(y.abs());
            [x, y]
        })
        .collect();
    if max_abs == 0.0 {
        return 0.0;
    }
    // Scale by an exact power of 2 when coordinates exceed `0.25 * f64::MAX` so `two_diff`
    // and intermediate differences inside exact orientation predicates can never overflow.
    let pre_scale = if max_abs > 1e300 { 0.125 } else { 1.0 };
    if pre_scale != 1.0 {
        for p in &mut pts {
            p[0] *= pre_scale;
            p[1] *= pre_scale;
        }
    }
    let inv_scale = 1.0 / pre_scale;

    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|b, a| a[0] == b[0] && a[1] == b[1]);

    if pts.len() <= 1 {
        return 0.0;
    }
    if pts.len() == 2 {
        return dist_2d_safe(pts[0], pts[1]) * inv_scale;
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while hull.len() >= 2
            && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], p) != Ordering::Greater
        {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len
            && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], p) != Ordering::Greater
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();

    let m = hull.len();
    if m <= 1 {
        return 0.0;
    }
    if m == 2 {
        return dist_2d_safe(hull[0], hull[1]) * inv_scale;
    }
    if m <= 64 {
        let mut best_d = 0.0_f64;
        for i in 0..m {
            for j in (i + 1)..m {
                best_d = best_d.max(dist_2d_safe(hull[i], hull[j]));
            }
        }
        return best_d * inv_scale;
    }

    // Standard rotating calipers with exact area comparisons on the strictly convex CCW hull.
    let mut best_d = 0.0_f64;
    let mut j = 1usize;
    for i in 0..m {
        let next_i = (i + 1) % m;
        best_d = best_d.max(dist_2d_safe(hull[i], hull[next_i]));
        let mut steps = 0usize;
        while steps < m {
            let next_j = (j + 1) % m;
            match cross_abs_cmp(hull[i], hull[next_i], hull[next_j], hull[j]) {
                Ordering::Greater => {
                    j = next_j;
                    steps += 1;
                }
                Ordering::Equal => {
                    best_d = best_d.max(dist_2d_safe(hull[i], hull[next_j]));
                    best_d = best_d.max(dist_2d_safe(hull[next_i], hull[next_j]));
                    break;
                }
                Ordering::Less => break,
            }
        }
        best_d = best_d.max(dist_2d_safe(hull[i], hull[j]));
        best_d = best_d.max(dist_2d_safe(hull[next_i], hull[j]));
    }
    best_d * inv_scale
}
