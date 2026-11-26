/// Linear-interpolation $O(N)$ quantile matching `np.quantile(buf, q)`.
fn quantile_linear(buf: &mut [f64], q: f64) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return buf[0];
    }
    let h = ((n - 1) as f64) * q;
    let lo = (h.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = h - (lo as f64);
    let (_, v_lo, right) = buf.select_nth_unstable_by(lo, f64::total_cmp);
    let v_lo = *v_lo;
    if hi == lo || frac == 0.0 {
        return v_lo;
    }
    let v_hi = right.iter().copied().min_by(f64::total_cmp).unwrap_or(v_lo);
    v_lo + frac * (v_hi - v_lo)
}

#[inline]
fn median_linear(buf: &mut [f64]) -> f64 {
    quantile_linear(buf, 0.5)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn cross_2d(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

#[inline]
fn cmp_point(a: (f64, f64), b: (f64, f64)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
}

#[inline]
fn c2(a: (f64, f64), b: (f64, f64)) -> ((f64, f64), f64) {
    let cx = 0.5 * (a.0 + b.0);
    let cy = 0.5 * (a.1 + b.1);
    let r = 0.5 * (a.0 - b.0).hypot(a.1 - b.1);
    ((cx, cy), r)
}

/// Vertex-`a`-local circumcircle of `(a, b, d)` avoiding catastrophic cancellation
/// under large coordinate offsets, with relative collinearity fallback.
#[inline]
pub(super) fn c3(a: (f64, f64), b: (f64, f64), d: (f64, f64)) -> ((f64, f64), f64) {
    let bx = b.0 - a.0;
    let by = b.1 - a.1;
    let dx = d.0 - a.0;
    let dy = d.1 - a.1;
    let b2 = bx * bx + by * by;
    let d2 = dx * dx + dy * dy;
    let det = 2.0 * (bx * dy - by * dx);
    if det.abs() <= 1e-15 * b2.max(d2) {
        let (c_ab, r_ab) = c2(a, b);
        let (c_bd, r_bd) = c2(b, d);
        let (c_ad, r_ad) = c2(a, d);
        if r_ab >= r_bd && r_ab >= r_ad {
            (c_ab, r_ab)
        } else if r_bd >= r_ad {
            (c_bd, r_bd)
        } else {
            (c_ad, r_ad)
        }
    } else {
        let ux = (dy * b2 - by * d2) / det;
        let uy = (bx * d2 - dx * b2) / det;
        ((a.0 + ux, a.1 + uy), ux.hypot(uy))
    }
}

#[inline]
fn in_circle(c: (f64, f64), r: f64, p: (f64, f64)) -> bool {
    (p.0 - c.0).hypot(p.1 - c.1) <= r * (1.0 + 1e-12)
}

/// Akl-Toussaint 8-extreme interior point rejection with relative edge tolerance
/// and permutation-invariant tie-breaking.
fn akl_toussaint_filter(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if pts.len() <= 64 {
        return pts.to_vec();
    }
    let mut extremes = [pts[0]; 8];
    let mut keys = [
        pts[0].0,
        pts[0].0 + pts[0].1,
        pts[0].1,
        pts[0].0 - pts[0].1,
        pts[0].0,
        pts[0].0 + pts[0].1,
        pts[0].1,
        pts[0].0 - pts[0].1,
    ];
    for &p in &pts[1..] {
        let p_keys = [
            p.0,
            p.0 + p.1,
            p.1,
            p.0 - p.1,
            p.0,
            p.0 + p.1,
            p.1,
            p.0 - p.1,
        ];
        // Min indices: 0 (x), 3 (x-y), 5 (x+y), 6 (y)
        for &idx in &[0usize, 3, 5, 6] {
            let ord = p_keys[idx]
                .total_cmp(&keys[idx])
                .then_with(|| cmp_point(p, extremes[idx]));
            if ord == std::cmp::Ordering::Less {
                keys[idx] = p_keys[idx];
                extremes[idx] = p;
            }
        }
        // Max indices: 1 (x+y), 2 (y), 4 (x), 7 (x-y)
        for &idx in &[1usize, 2, 4, 7] {
            let ord = p_keys[idx]
                .total_cmp(&keys[idx])
                .then_with(|| cmp_point(p, extremes[idx]));
            if ord == std::cmp::Ordering::Greater {
                keys[idx] = p_keys[idx];
                extremes[idx] = p;
            }
        }
    }

    let mut poly: Vec<(f64, f64)> = extremes.to_vec();
    poly.sort_unstable_by(|&a, &b| cmp_point(a, b));
    poly.dedup();
    if poly.len() < 3 {
        return pts.to_vec();
    }

    let inv_len = 1.0 / (poly.len() as f64);
    let cx = poly.iter().map(|p| p.0).sum::<f64>() * inv_len;
    let cy = poly.iter().map(|p| p.1).sum::<f64>() * inv_len;
    poly.sort_unstable_by(|&a, &b| {
        (a.1 - cy)
            .atan2(a.0 - cx)
            .total_cmp(&(b.1 - cy).atan2(b.0 - cx))
            .then_with(|| cmp_point(a, b))
    });

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(px, py) in &poly {
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }
    let poly_scale = (max_x - min_x).max(max_y - min_y).max(1e-300);

    let m = poly.len();
    let mut edges = Vec::with_capacity(m);
    for i in 0..m {
        let a = poly[i];
        let b = poly[(i + 1) % m];
        let ex = b.0 - a.0;
        let ey = b.1 - a.1;
        let tol = 1e-12 * ex.hypot(ey) * poly_scale;
        edges.push((a.0, a.1, ex, ey, tol));
    }

    let mut kept = Vec::with_capacity(pts.len().min(512));
    for &p in pts {
        let mut strictly_inside = true;
        for &(ax, ay, ex, ey, tol) in &edges {
            let c = ex * (p.1 - ay) - ey * (p.0 - ax);
            if c <= tol {
                strictly_inside = false;
                break;
            }
        }
        if !strictly_inside {
            kept.push(p);
        }
    }

    if kept.len() < 3 { pts.to_vec() } else { kept }
}

/// Computes the exact Minimum Enclosing Circle `(center, radius)` of `pts` via
/// Akl-Toussaint filtering, Andrew's monotone chain convex hull, lexicographical
/// canonicalization, fixed-seed SplitMix64 shuffling, and Welzl's incremental algorithm.
pub(super) fn exact_mec(pts: &[(f64, f64)]) -> ((f64, f64), f64) {
    if pts.is_empty() {
        return ((0.0, 0.0), 0.0);
    }
    let mut cand = akl_toussaint_filter(pts);
    cand.sort_unstable_by(|&a, &b| cmp_point(a, b));
    cand.dedup();

    if cand.len() == 1 {
        return (cand[0], 0.0);
    }
    if cand.len() == 2 {
        return c2(cand[0], cand[1]);
    }

    let mut lower = Vec::with_capacity(cand.len());
    for &p in &cand {
        while lower.len() >= 2 && cross_2d(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::with_capacity(cand.len());
    for &p in cand.iter().rev() {
        while upper.len() >= 2 && cross_2d(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    let mut hull = lower;
    hull.extend_from_slice(&upper);

    if hull.is_empty() {
        return (cand[0], 0.0);
    }
    if hull.len() == 1 {
        return (hull[0], 0.0);
    }

    // Explicit lexicographical canonicalization guarantees identical shuffle order
    // regardless of input permutation or polygon orientation.
    hull.sort_unstable_by(|&a, &b| cmp_point(a, b));
    let mut rng_state = 42_u64;
    for i in (1..hull.len()).rev() {
        let j = (((splitmix64(&mut rng_state) as u128) * ((i + 1) as u128)) >> 64) as usize;
        hull.swap(i, j);
    }

    let mut c = hull[0];
    let mut r = 0.0_f64;
    for i in 1..hull.len() {
        let pi = hull[i];
        if !in_circle(c, r, pi) {
            c = pi;
            r = 0.0;
            for j in 0..i {
                let pj = hull[j];
                if !in_circle(c, r, pj) {
                    (c, r) = c2(pi, pj);
                    for &pk in &hull[0..j] {
                        if !in_circle(c, r, pk) {
                            (c, r) = c3(pi, pj, pk);
                        }
                    }
                }
            }
        }
    }
    (c, r)
}

pub(super) struct NormalizedFrame {
    pub(super) x_pre: Vec<f64>,
    pub(super) y_pre: Vec<f64>,
    pub(super) d_buf: Vec<f64>,
    pub(super) mu: (f64, f64),
    pub(super) r95: f64,
}

pub(super) fn normalize_and_center_mec(xs: &[f64], ys: &[f64]) -> Option<NormalizedFrame> {
    let n = xs.len();
    // Step 1a: Pre-normalize coordinates by median + Q_0.95 to guarantee full
    // scale invariance across 1e-100..1e100 and translation stability.
    let mut scratch = xs.to_vec();
    let m_pre_x = median_linear(&mut scratch);
    scratch.copy_from_slice(ys);
    let m_pre_y = median_linear(&mut scratch);

    for i in 0..n {
        scratch[i] = (xs[i] - m_pre_x).hypot(ys[i] - m_pre_y);
    }
    let r_pre = quantile_linear(&mut scratch, 0.95);
    if !r_pre.is_finite() || r_pre <= 0.0 || !(1.0 / r_pre).is_finite() {
        return None;
    }

    let mut x_pre = Vec::with_capacity(n);
    let mut y_pre = Vec::with_capacity(n);
    for i in 0..n {
        x_pre.push((xs[i] - m_pre_x) / r_pre);
        y_pre.push((ys[i] - m_pre_y) / r_pre);
    }

    // Step 1b: 8-step Weiszfeld L1 median initialized at coordinate-wise median.
    scratch.copy_from_slice(&x_pre);
    let mut mx = median_linear(&mut scratch);
    scratch.copy_from_slice(&y_pre);
    let mut my = median_linear(&mut scratch);
    for _ in 0..8 {
        let mut w_sum = 0.0_f64;
        let mut wx_sum = 0.0_f64;
        let mut wy_sum = 0.0_f64;
        for i in 0..n {
            let d = (x_pre[i] - mx).hypot(y_pre[i] - my);
            let w = 1.0 / d.max(1e-12);
            w_sum += w;
            wx_sum += w * x_pre[i];
            wy_sum += w * y_pre[i];
        }
        if w_sum > 0.0 && w_sum.is_finite() {
            mx = wx_sum / w_sum;
            my = wy_sum / w_sum;
        }
    }

    let mut d_buf = vec![0.0_f64; n];
    for i in 0..n {
        d_buf[i] = (x_pre[i] - mx).hypot(y_pre[i] - my);
    }
    scratch.copy_from_slice(&d_buf);
    let r0 = quantile_linear(&mut scratch, 0.95);
    if !r0.is_finite() || r0 <= 1e-12 {
        return None;
    }

    // Step 1c: Two-stage exact MEC centering.
    // Match `in_circle`'s 1e-12 relative tolerance across all hypot-derived radial
    // thresholds so 1-ULP platform `hypot` differences never split tied boundary points.
    let mut mec_pts = Vec::with_capacity(n);
    let r0_cut = (1.5 * r0) * (1.0 + 1e-12);
    for i in 0..n {
        if d_buf[i] <= r0_cut {
            mec_pts.push((x_pre[i], y_pre[i]));
        }
    }
    let (mu1, _) = exact_mec(&mec_pts);

    for i in 0..n {
        d_buf[i] = (x_pre[i] - mu1.0).hypot(y_pre[i] - mu1.1);
    }
    scratch.copy_from_slice(&d_buf);
    let r1 = quantile_linear(&mut scratch, 0.95);
    let r1_cut = r1 * (1.0 + 1e-12);

    mec_pts.clear();
    for i in 0..n {
        if d_buf[i] <= r1_cut {
            mec_pts.push((x_pre[i], y_pre[i]));
        }
    }
    let (mu, _) = exact_mec(&mec_pts);

    for i in 0..n {
        d_buf[i] = (x_pre[i] - mu.0).hypot(y_pre[i] - mu.1);
    }
    scratch.copy_from_slice(&d_buf);
    let r95 = quantile_linear(&mut scratch, 0.95);
    if !r95.is_finite() || r95 <= 1e-12 {
        return None;
    }

    Some(NormalizedFrame {
        x_pre,
        y_pre,
        d_buf,
        mu,
        r95,
    })
}
