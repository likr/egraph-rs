use egraph_core::{CoreError, CoreResult, RiemannianLayout, Scalar, VisitStamp};

use super::pairs::{HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair};

pub(super) struct ScratchAccum<'a> {
    pub dim: usize,
    pub grad: &'a mut [f64],
    pub hess_diag: &'a mut [f64],
    pub hess_yy: &'a mut [f64],
    pub hess_xy: &'a mut [f64],
    pub tan_buf: &'a mut [f64],
    pub touched_stamp: &'a mut VisitStamp,
    pub touched_nodes: &'a mut Vec<u32>,
}

#[inline]
fn tangent_norm(vec: &[f64]) -> f64 {
    if vec.len() == 2 {
        return vec[0].hypot(vec[1]);
    }
    let mut max_abs = 0.0_f64;
    for &v in vec {
        let a = v.abs();
        if !a.is_finite() {
            return f64::INFINITY;
        }
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for &v in vec {
        let r = v / max_abs;
        sum_sq += r * r;
    }
    max_abs * sum_sq.sqrt()
}

#[inline]
fn accumulate_endpoint<L: RiemannianLayout>(
    layout: &L,
    u: usize,
    v: usize,
    d: f64,
    w: f64,
    is_hinge: bool,
    sc: &mut ScratchAccum<'_>,
) -> CoreResult<()> {
    let dim = sc.dim;
    let tan = &mut sc.tan_buf[..dim];
    layout.log_map_into(u, v, tan)?;
    let norm = tangent_norm(tan);
    if !norm.is_finite() || norm <= 0.0 || (is_hinge && norm >= d) {
        return Ok(());
    }
    let err = (norm - d) / norm;
    if dim == 2 {
        let tx = tan[0];
        let ty = tan[1];
        let gx = -w * err * tx;
        let gy = -w * err * ty;
        let ux = tx / norm;
        let uy = ty / norm;
        // Full 2x2 Gauss-Newton tangent Hessian: w * (I + u u^T) is SO(2)-equivariant.
        let cxx = w * (1.0 + ux * ux);
        let cyy = w * (1.0 + uy * uy);
        let cxy = w * (ux * uy);
        if gx.is_finite() && gy.is_finite() && cxx.is_finite() && cyy.is_finite() && cxy.is_finite()
        {
            sc.grad[2 * u] += gx;
            sc.grad[2 * u + 1] += gy;
            sc.hess_diag[u] += cxx;
            sc.hess_yy[u] += cyy;
            sc.hess_xy[u] += cxy;
            if sc.touched_stamp.visit(u) {
                sc.touched_nodes.push(u as u32);
            }
        }
    } else {
        // Isotropic curvature trace w * (d + 1) / d is SO(d)-invariant for arbitrary tangent_dim.
        let c_iso = w * (((dim as f64) + 1.0) / (dim as f64));
        let ok = c_iso.is_finite() && tan.iter().all(|&ta| (-w * err * ta).is_finite());
        if ok {
            let base = u * dim;
            for (axis, &ta) in tan.iter().enumerate() {
                sc.grad[base + axis] += -w * err * ta;
            }
            sc.hess_diag[u] += c_iso;
            if sc.touched_stamp.visit(u) {
                sc.touched_nodes.push(u as u32);
            }
        }
    }
    Ok(())
}

#[inline]
pub(super) fn accumulate_sym<L: RiemannianLayout>(
    layout: &L,
    sc: &mut ScratchAccum<'_>,
    p: PackedSymPair<L::Scalar>,
) -> CoreResult<()> {
    let d = p.d.to_f64();
    let w = p.w.to_f64();
    if !d.is_finite() || d <= 0.0 || !w.is_finite() || w <= 0.0 {
        return Ok(());
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    accumulate_endpoint(layout, i, j, d, w, is_hinge, sc)?;
    accumulate_endpoint(layout, j, i, d, w, is_hinge, sc)
}

#[inline]
pub(super) fn accumulate_gen<L: RiemannianLayout>(
    layout: &L,
    sc: &mut ScratchAccum<'_>,
    p: &PackedGenPair<L::Scalar>,
) -> CoreResult<()> {
    let dij = p.dij.to_f64();
    let dji = p.dji.to_f64();
    let wij = p.wij.to_f64();
    let wji = p.wji.to_f64();
    if !dij.is_finite() || dij <= 0.0 || !dji.is_finite() || dji <= 0.0 {
        return Ok(());
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    if wij.is_finite() && wij > 0.0 {
        accumulate_endpoint(layout, i, j, dij, wij, is_hinge, sc)?;
    }
    if wji.is_finite() && wji > 0.0 {
        accumulate_endpoint(layout, j, i, dji, wji, is_hinge, sc)?;
    }
    Ok(())
}

#[inline]
pub(super) fn flush_batch<L: RiemannianLayout>(
    layout: &mut L,
    sc: &mut ScratchAccum<'_>,
    alpha_t: f64,
    damping: f64,
) -> CoreResult<()> {
    let dim = sc.dim;
    for &u_u32 in sc.touched_nodes.iter() {
        let u = u_u32 as usize;
        if dim == 2 {
            let gx = sc.grad[2 * u];
            let gy = sc.grad[2 * u + 1];
            let a = sc.hess_diag[u] + damping;
            let b = sc.hess_yy[u] + damping;
            let c = sc.hess_xy[u];
            let det = a * b - c * c;
            sc.grad[2 * u] = 0.0;
            sc.grad[2 * u + 1] = 0.0;
            sc.hess_diag[u] = 0.0;
            sc.hess_yy[u] = 0.0;
            sc.hess_xy[u] = 0.0;
            if !det.is_finite() || det <= 0.0 {
                return Err(CoreError::numerical_breakdown(
                    "Sgd2Solver tangent Hessian became singular",
                ));
            }
            let sx = -alpha_t * (b * gx - c * gy) / det;
            let sy = -alpha_t * (-c * gx + a * gy) / det;
            if !sx.is_finite() || !sy.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "Sgd2Solver step became non-finite",
                ));
            }
            layout.exp_map(u, &[sx, sy])?;
        } else {
            let denom = sc.hess_diag[u] + damping;
            sc.hess_diag[u] = 0.0;
            if !denom.is_finite() || denom <= 0.0 {
                return Err(CoreError::numerical_breakdown(
                    "Sgd2Solver isotropic curvature became non-finite",
                ));
            }
            let base = u * dim;
            let delta = &mut sc.tan_buf[..dim];
            for (axis, d_out) in delta.iter_mut().enumerate() {
                let step = -alpha_t * sc.grad[base + axis] / denom;
                sc.grad[base + axis] = 0.0;
                if !step.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "Sgd2Solver step became non-finite",
                    ));
                }
                *d_out = step;
            }
            layout.exp_map(u, delta)?;
        }
    }
    sc.touched_nodes.clear();
    sc.touched_stamp.clear();
    Ok(())
}
