use super::MIN_EDGE_LEN;

/// Computes the Euclidean norm $\|a - b\|_2$ with overflow-safe scaling.
#[inline]
pub(super) fn norm_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let d = (va - vb).abs();
        if !d.is_finite() {
            return d;
        }
        if d > max_abs {
            max_abs = d;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let s = (va - vb) / max_abs;
        sum_sq += s * s;
    }
    max_abs * sum_sq.sqrt()
}

/// Computes the Euclidean norm $\|v\|_2$ with overflow-safe scaling.
#[inline]
pub(super) fn norm_vec(v: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for &val in v {
        let a = val.abs();
        if !a.is_finite() {
            return a;
        }
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for &val in v {
        let s = val / max_abs;
        sum_sq += s * s;
    }
    max_abs * sum_sq.sqrt()
}

/// Computes Holten-Van Wijk edge compatibility `(c_e, cos_theta)` for two segments
/// `(p1, p2)` and `(q1, q2)` in arbitrary Euclidean dimension.
pub(super) fn edge_compatibility(
    p1: &[f64],
    p2: &[f64],
    p_norm: f64,
    q1: &[f64],
    q2: &[f64],
    q_norm: f64,
) -> (f64, f64) {
    let dims = p1.len();
    let mut dot_unit = 0.0_f64;
    let mut max_mid_diff = 0.0_f64;

    for d in 0..dims {
        let up = (p2[d] - p1[d]) / p_norm;
        let uq = (q2[d] - q1[d]) / q_norm;
        dot_unit += up * uq;

        let pm = 0.5 * (p1[d] + p2[d]);
        let qm = 0.5 * (q1[d] + q2[d]);
        let md = (qm - pm).abs();
        if !md.is_finite() {
            return (0.0, 0.0);
        }
        if md > max_mid_diff {
            max_mid_diff = md;
        }
    }

    let cos_theta = dot_unit.clamp(-1.0, 1.0);
    let c_a = cos_theta.abs();

    let l_min = p_norm.min(q_norm);
    let l_max = p_norm.max(q_norm);
    let ratio = l_min / l_max;
    let c_s = 2.0 / ((1.0 + ratio) / (2.0 * ratio) + 2.0 / (1.0 + ratio));

    let mid_dist = if max_mid_diff == 0.0 {
        0.0
    } else {
        let mut mid_sq = 0.0_f64;
        for d in 0..dims {
            let pm = 0.5 * (p1[d] + p2[d]);
            let qm = 0.5 * (q1[d] + q2[d]);
            let s = (qm - pm) / max_mid_diff;
            mid_sq += s * s;
        }
        max_mid_diff * mid_sq.sqrt()
    };

    let l_avg = 0.5 * l_max * (1.0 + ratio);
    let denom = l_avg + mid_dist;
    let c_p = if denom.is_finite() {
        if denom == 0.0 { 1.0 } else { l_avg / denom }
    } else {
        let scale = l_avg.max(mid_dist);
        (l_avg / scale) / (l_avg / scale + mid_dist / scale)
    };

    let vp = visibility_compatibility(p1, p2, p_norm, q1, q2);
    let vq = visibility_compatibility(q1, q2, q_norm, p1, p2);
    let c_v = vp.min(vq);

    ((c_a * c_s * c_p * c_v).clamp(0.0, 1.0), cos_theta)
}

fn visibility_compatibility(p1: &[f64], p2: &[f64], p_norm: f64, q1: &[f64], q2: &[f64]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    for d in 0..p1.len() {
        let up = (p2[d] - p1[d]) / p_norm;
        s0 += (q1[d] - p1[d]) * up;
        s1 += (q2[d] - p1[d]) * up;
    }
    let i_dist = (s1 - s0).abs();
    if !i_dist.is_finite() || i_dist <= MIN_EDGE_LEN {
        return 0.0;
    }
    let m_dist = (0.5 * (s0 + s1) - 0.5 * p_norm).abs();
    if !m_dist.is_finite() {
        return 0.0;
    }
    (1.0 - 2.0 * m_dist / i_dist).max(0.0)
}
