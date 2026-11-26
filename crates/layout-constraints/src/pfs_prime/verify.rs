use std::cmp::Ordering;

use egraph_core::{CoreResult, Layout2d, Scalar};
use egraph_spatial_trees::{Rect2d, RectangleOverlapSweep};

#[inline]
pub(super) fn normalize_neg_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

pub(super) fn find_f64_exact_overlaps<S: Scalar>(
    layout: &Layout2d<S>,
    eff_hw_req: &[f64],
    eff_hh_req: &[f64],
) -> CoreResult<Vec<(usize, usize)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut rects = Vec::with_capacity(n);
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let hw = eff_hw_req[i];
        let hh = eff_hh_req[i];
        let mx = 4.0 * (xi.abs() + hw) * f64::EPSILON + f64::MIN_POSITIVE;
        let my = 4.0 * (yi.abs() + hh) * f64::EPSILON + f64::MIN_POSITIVE;
        let x_min = (xi - hw - mx).clamp(-f64::MAX, f64::MAX);
        let x_max = (xi + hw + mx).clamp(-f64::MAX, f64::MAX);
        let y_min = (yi - hh - my).clamp(-f64::MAX, f64::MAX);
        let y_max = (yi + hh + my).clamp(-f64::MAX, f64::MAX);
        rects.push(Rect2d::new(x_min, x_max, y_min, y_max)?);
    }

    let mut candidates = RectangleOverlapSweep::find_overlaps(&rects)?;
    candidates.retain(|&(i, j)| {
        let req_x = eff_hw_req[i] + eff_hw_req[j];
        let req_y = eff_hh_req[i] + eff_hh_req[j];
        if req_x <= 0.0 || req_y <= 0.0 {
            return false;
        }
        let dx = (xs[i].to_f64() - xs[j].to_f64()).abs();
        let dy = (ys[i].to_f64() - ys[j].to_f64()).abs();
        dx < req_x && dy < req_y
    });
    candidates.sort_unstable();
    candidates.dedup();
    Ok(candidates)
}

pub(super) fn count_union_pairs(a: &[(usize, usize)], b: &[(usize, usize)]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                count += 1;
                i += 1;
            }
            Ordering::Greater => {
                count += 1;
                j += 1;
            }
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count + (a.len() - i) + (b.len() - j)
}

/// Checks in O(N log N) time whether `layout` introduces any strict orthogonal order inversion
/// relative to `(orig_x, orig_y)` by verifying `prev_group_max <= curr_group_min` across
/// consecutive `orig_x` tie-groups and `orig_y` tie-groups.
pub(super) fn has_orthogonal_order_inversion<S: Scalar>(
    orig_x: &[f64],
    orig_y: &[f64],
    layout: &Layout2d<S>,
) -> bool {
    let n = orig_x.len().min(orig_y.len()).min(layout.len());
    if n < 2 {
        return false;
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for (orig, new_s) in [(&orig_x[..n], layout.x()), (&orig_y[..n], layout.y())] {
        order.clear();
        order.extend(0..n);
        order.sort_unstable_by(|&a, &b| orig[a].total_cmp(&orig[b]).then_with(|| a.cmp(&b)));

        let mut prev_group_max: Option<f64> = None;
        let mut pos = 0usize;
        while pos < n {
            let base = orig[order[pos]];
            let mut end = pos + 1;
            while end < n && orig[order[end]] == base {
                end += 1;
            }
            let mut curr_group_min = f64::INFINITY;
            let mut curr_group_max = f64::NEG_INFINITY;
            for &node in &order[pos..end] {
                let val = new_s[node].to_f64();
                if val < curr_group_min {
                    curr_group_min = val;
                }
                if val > curr_group_max {
                    curr_group_max = val;
                }
            }
            if prev_group_max.is_some_and(|prev_max| prev_max > curr_group_min) {
                return true;
            }
            prev_group_max =
                Some(prev_group_max.map_or(curr_group_max, |pm| pm.max(curr_group_max)));
            pos = end;
        }
    }
    false
}

#[inline]
pub(super) fn bump_coord_s_to_clear<S: Scalar>(
    mut cand_s: S,
    u_coord_s: S,
    u_ext_s: S,
    v_ext_s: S,
    u_coord_f64: f64,
    req_f64: f64,
) -> S {
    let u_upper_s = u_coord_s + u_ext_s;
    if cand_s.to_f64() - u_coord_f64 >= req_f64 && (cand_s - v_ext_s) >= u_upper_s {
        return cand_s;
    }
    let target = (u_coord_f64 + req_f64).max((u_upper_s + v_ext_s).to_f64());
    if cand_s.to_f64() < target {
        let target_s = S::from_f64(target);
        if target_s.is_finite() {
            cand_s = target_s;
        }
    }
    let mut step = (cand_s.to_f64().abs() * S::EPSILON.to_f64()).max(S::MIN_POSITIVE.to_f64());
    while cand_s.to_f64() - u_coord_f64 < req_f64 || (cand_s - v_ext_s) < u_upper_s {
        let next = S::from_f64(cand_s.to_f64() + step);
        if !next.is_finite() || !step.is_finite() {
            break;
        }
        if next > cand_s {
            cand_s = next;
        } else {
            step *= 2.0;
        }
    }
    cand_s
}
