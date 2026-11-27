use super::double_double::{Dd, dd_cmp};
use super::euclidean_geom::segments_geometrically_intersect;
use super::geometry::Seg;
use super::torus_clip::{ClippedTorusSeg, TORUS_MAX, clamp_unit};
use super::torus_exact::segs_intersect_exact_f64;

#[inline]
pub(super) fn both_intersect(a: [f64; 4], b: [f64; 4]) -> bool {
    segs_intersect_exact_f64(a, b)
        && segments_geometrically_intersect(
            &Seg::from_endpoints(0, 0, 1, a[0], a[1], a[2], a[3]),
            &Seg::from_endpoints(1, 2, 3, b[0], b[1], b[2], b[3]),
        )
}

#[inline]
fn step_torus_ulp(v: f64, delta: i32) -> Option<f64> {
    let bits = if v == 0.0 { 0_u64 } else { v.to_bits() };
    let target = (bits as i128) + (delta as i128);
    if target < 0 || target > TORUS_MAX.to_bits() as i128 {
        None
    } else {
        let f = f64::from_bits(target as u64);
        Some(if f == 0.0 { 0.0 } else { f })
    }
}

#[inline]
fn bracket_dd(val: Dd) -> [f64; 2] {
    let f = clamp_unit(val.to_f64());
    match dd_cmp(Dd::from_f64(f), val) {
        std::cmp::Ordering::Equal => [f, f],
        std::cmp::Ordering::Less => [f, step_torus_ulp(f, 1).unwrap_or(f)],
        std::cmp::Ordering::Greater => [step_torus_ulp(f, -1).unwrap_or(f), f],
    }
}

#[inline]
fn ulp_cost(orig: [f64; 4], cand: [f64; 4]) -> u64 {
    let mut sum = 0_u64;
    for i in 0..4 {
        let b0 = if orig[i] == 0.0 { 0 } else { orig[i].to_bits() };
        let b1 = if cand[i] == 0.0 { 0 } else { cand[i].to_bits() };
        sum += b0.abs_diff(b1);
    }
    sum
}

/// Repairs clipped `[0.0, TORUS_MAX]^2` segments for a `non_gap` crossing so both exact and
/// `f64` segment-intersection predicates hold with minimal (typically 0-1 ULP) perturbation.
pub(super) fn ensure_intersecting_pair(
    ca: &ClippedTorusSeg,
    cb: &ClippedTorusSeg,
    xi_s: Dd,
    yi_s: Dd,
) -> ([f64; 4], [f64; 4]) {
    if both_intersect(ca.coords, cb.coords) {
        return (ca.coords, cb.coords);
    }

    // Tier 1: 1-ULP bracket search over rounded free coordinates from DD clipping.
    let mut free_slots: [(bool, usize, [f64; 2]); 4] = [(false, 0, [0.0, 0.0]); 4];
    let mut n_free = 0usize;
    for (is_b, seg) in [(false, ca), (true, cb)] {
        for (idx, opt_dd) in seg.free_dd.iter().enumerate() {
            if let Some(val_dd) = *opt_dd
                && n_free < 4
            {
                free_slots[n_free] = (is_b, idx, bracket_dd(val_dd));
                n_free += 1;
            }
        }
    }
    if n_free > 0 {
        let mut best: Option<(u64, [f64; 4], [f64; 4])> = None;
        let combos = 1usize << n_free;
        for mask in 1..combos {
            let mut ta = ca.coords;
            let mut tb = cb.coords;
            for (bit, &(is_b, idx, br)) in free_slots[..n_free].iter().enumerate() {
                let pick = br[(mask >> bit) & 1];
                if is_b {
                    tb[idx] = pick;
                } else {
                    ta[idx] = pick;
                }
            }
            if both_intersect(ta, tb) {
                let cost = ulp_cost(ca.coords, ta) + ulp_cost(cb.coords, tb);
                if best.as_ref().is_none_or(|(bc, _, _)| cost < *bc) {
                    best = Some((cost, ta, tb));
                }
            }
        }
        if let Some((_, ta, tb)) = best {
            return (ta, tb);
        }
    }

    // Tier 2: 1-2 ULP single-coordinate nudge (preferring non-boundary coords first).
    let is_boundary = |v: f64| v == 0.0 || v == TORUS_MAX;
    for prefer_interior in [true, false] {
        for mag in [1_i32, -1, 2, -2] {
            for is_b in [false, true] {
                for idx in 0..4 {
                    let orig = if is_b { cb.coords[idx] } else { ca.coords[idx] };
                    if prefer_interior && is_boundary(orig) {
                        continue;
                    }
                    let Some(nudged) = step_torus_ulp(orig, mag) else {
                        continue;
                    };
                    let mut ta = ca.coords;
                    let mut tb = cb.coords;
                    if is_b {
                        tb[idx] = nudged;
                    } else {
                        ta[idx] = nudged;
                    }
                    if both_intersect(ta, tb) {
                        return (ta, tb);
                    }
                }
            }
        }
    }

    // Tier 3: unconditional exact crossing anchor fallback.
    let ix = clamp_unit(xi_s.to_f64());
    let iy = clamp_unit(yi_s.to_f64());
    let snap_closest = |mut s: [f64; 4]| {
        let d0 = (s[0] - ix).hypot(s[1] - iy);
        let d1 = (s[2] - ix).hypot(s[3] - iy);
        if d0 <= d1 {
            s[0] = ix;
            s[1] = iy;
        } else {
            s[2] = ix;
            s[3] = iy;
        }
        s
    };
    (snap_closest(ca.coords), snap_closest(cb.coords))
}
