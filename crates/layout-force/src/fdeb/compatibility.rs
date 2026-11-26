use super::MIN_EDGE_LEN;

#[allow(clippy::too_many_arguments)]
pub(super) fn edge_compatibility(
    p1x: f64,
    p1y: f64,
    p2x: f64,
    p2y: f64,
    p_norm: f64,
    q1x: f64,
    q1y: f64,
    q2x: f64,
    q2y: f64,
    q_norm: f64,
) -> (f64, f64) {
    let pdx = p2x - p1x;
    let pdy = p2y - p1y;
    let qdx = q2x - q1x;
    let qdy = q2y - q1y;

    let dot = pdx * qdx + pdy * qdy;
    let cos_theta = (dot / (p_norm * q_norm)).clamp(-1.0, 1.0);
    let c_a = cos_theta.abs();

    let l_avg = 0.5 * (p_norm + q_norm);
    let l_min = p_norm.min(q_norm);
    let l_max = p_norm.max(q_norm);
    let c_s = 2.0 / (l_avg / l_min + l_max / l_avg);

    let pmx = 0.5 * (p1x + p2x);
    let pmy = 0.5 * (p1y + p2y);
    let qmx = 0.5 * (q1x + q2x);
    let qmy = 0.5 * (q1y + q2y);
    let mid_dist = (qmx - pmx).hypot(qmy - pmy);
    let c_p = l_avg / (l_avg + mid_dist);

    let vp = visibility_compatibility(p1x, p1y, pdx, pdy, p_norm, pmx, pmy, q1x, q1y, q2x, q2y);
    let vq = visibility_compatibility(q1x, q1y, qdx, qdy, q_norm, qmx, qmy, p1x, p1y, p2x, p2y);
    let c_v = vp.min(vq);

    (c_a * c_s * c_p * c_v, cos_theta)
}

#[allow(clippy::too_many_arguments)]
fn visibility_compatibility(
    p1x: f64,
    p1y: f64,
    pdx: f64,
    pdy: f64,
    p_norm: f64,
    pmx: f64,
    pmy: f64,
    q1x: f64,
    q1y: f64,
    q2x: f64,
    q2y: f64,
) -> f64 {
    let p_norm_sq = p_norm * p_norm;
    let r0 = ((q1x - p1x) * pdx + (q1y - p1y) * pdy) / p_norm_sq;
    let r1 = ((q2x - p1x) * pdx + (q2y - p1y) * pdy) / p_norm_sq;

    let i0x = p1x + r0 * pdx;
    let i0y = p1y + r0 * pdy;
    let i1x = p1x + r1 * pdx;
    let i1y = p1y + r1 * pdy;

    let i_dist = (i1x - i0x).hypot(i1y - i0y);
    if i_dist <= MIN_EDGE_LEN {
        return 0.0;
    }
    let imx = 0.5 * (i0x + i1x);
    let imy = 0.5 * (i0y + i1y);
    let m_dist = (imx - pmx).hypot(imy - pmy);
    (1.0 - 2.0 * m_dist / i_dist).max(0.0)
}
