use egraph_core::{CoreResult, DeterministicRng, Layout2d, LayoutNd, Scalar, VisitStamp};

use super::config::Sgd2Config;
use super::nd_variants::{Sgd2NdScratch, nd_norm_f64};
use super::pairs::{
    HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair, PairStorage, try_for_each_shuffled_pair,
};

#[derive(Debug, Clone)]
pub(super) struct Sgd2State {
    batch_size: usize,
    damping: f64,
    grad_x: Vec<f64>,
    grad_y: Vec<f64>,
    h_x: Vec<f64>,
    h_y: Vec<f64>,
    touched_stamp: VisitStamp,
    touched_nodes: Vec<u32>,
    pub(super) nd_scratch: Option<Sgd2NdScratch>,
}

pub(super) fn build_sgd2_state(num_nodes: usize, config: Sgd2Config) -> Sgd2State {
    let touched_cap = (config.batch_size.saturating_mul(2)).min(num_nodes);
    Sgd2State {
        batch_size: config.batch_size,
        damping: config.damping,
        grad_x: vec![0.0; num_nodes],
        grad_y: vec![0.0; num_nodes],
        h_x: vec![0.0; num_nodes],
        h_y: vec![0.0; num_nodes],
        touched_stamp: VisitStamp::new(num_nodes),
        touched_nodes: Vec::with_capacity(touched_cap),
        nd_scratch: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn step_sgd2_state<S: Scalar>(
    st: &mut Sgd2State,
    pairs: &mut PairStorage<S>,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    eta_f64: f64,
    eta_max: f64,
    step_scale: f64,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    if num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 && eta_max > 0.0 {
        let alpha_t = (eta_f64 / eta_max).clamp(0.0, step_scale.max(1.0));
        let batch_size = st.batch_size;
        let damping = st.damping;
        let (xs, ys) = layout.xy_mut();
        let mut batch_count = 0usize;

        match pairs {
            PairStorage::Symmetric(sym_pairs) => {
                try_for_each_shuffled_pair(sym_pairs, block_order, rng, |&p| {
                    accumulate_sgd2_sym_pair(xs, ys, st, p);
                    batch_count += 1;
                    if batch_count >= batch_size {
                        flush_sgd2_batch(xs, ys, st, alpha_t, damping);
                        batch_count = 0;
                    }
                    Ok(())
                })?;
            }
            PairStorage::General(gen_pairs) => {
                try_for_each_shuffled_pair(gen_pairs, block_order, rng, |p| {
                    accumulate_sgd2_gen_pair(xs, ys, st, p);
                    batch_count += 1;
                    if batch_count >= batch_size {
                        flush_sgd2_batch(xs, ys, st, alpha_t, damping);
                        batch_count = 0;
                    }
                    Ok(())
                })?;
            }
        }
        if batch_count > 0 {
            flush_sgd2_batch(xs, ys, st, alpha_t, damping);
        }
    }
    Ok(())
}

#[inline]
fn record_sgd2_touch(st: &mut Sgd2State, idx: usize) {
    if st.touched_stamp.visit(idx) {
        st.touched_nodes.push(idx as u32);
    }
}

#[inline]
fn accumulate_sgd2_sym_pair<S: Scalar>(
    xs: &[S],
    ys: &[S],
    st: &mut Sgd2State,
    p: PackedSymPair<S>,
) {
    let d = p.d.to_f64();
    let w = p.w.to_f64();
    if !d.is_finite() || d <= 0.0 || !w.is_finite() || w <= 0.0 {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let dx = xs[i].to_f64() - xs[j].to_f64();
    let dy = ys[i].to_f64() - ys[j].to_f64();
    let norm = dx.hypot(dy);
    if !norm.is_finite() || norm <= 0.0 {
        return;
    }
    if is_hinge && norm >= d {
        return;
    }
    let err = (norm - d) / norm;
    let gx = w * err * dx;
    let gy = w * err * dy;
    let inv_n2 = 1.0 / (norm * norm);
    let cx = 2.0 * w * (0.5 + 0.5 * (dx * dx) * inv_n2);
    let cy = 2.0 * w * (0.5 + 0.5 * (dy * dy) * inv_n2);
    if !gx.is_finite() || !gy.is_finite() || !cx.is_finite() || !cy.is_finite() {
        return;
    }
    st.grad_x[i] += gx;
    st.grad_y[i] += gy;
    st.h_x[i] += cx;
    st.h_y[i] += cy;
    st.grad_x[j] -= gx;
    st.grad_y[j] -= gy;
    st.h_x[j] += cx;
    st.h_y[j] += cy;
    record_sgd2_touch(st, i);
    record_sgd2_touch(st, j);
}

#[inline]
fn accumulate_sgd2_gen_pair<S: Scalar>(
    xs: &[S],
    ys: &[S],
    st: &mut Sgd2State,
    p: &PackedGenPair<S>,
) {
    let dij = p.dij.to_f64();
    let dji = p.dji.to_f64();
    let wij = p.wij.to_f64();
    let wji = p.wji.to_f64();
    if !dij.is_finite() || dij <= 0.0 || !dji.is_finite() || dji <= 0.0 {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let dx = xs[i].to_f64() - xs[j].to_f64();
    let dy = ys[i].to_f64() - ys[j].to_f64();
    let norm = dx.hypot(dy);
    if !norm.is_finite() || norm <= 0.0 {
        return;
    }
    let inv_n2 = 1.0 / (norm * norm);
    let dir_x = 0.5 + 0.5 * (dx * dx) * inv_n2;
    let dir_y = 0.5 + 0.5 * (dy * dy) * inv_n2;

    if wij.is_finite() && wij > 0.0 && (!is_hinge || norm < dij) {
        let err_i = (norm - dij) / norm;
        let gx = wij * err_i * dx;
        let gy = wij * err_i * dy;
        let cx = 2.0 * wij * dir_x;
        let cy = 2.0 * wij * dir_y;
        if gx.is_finite() && gy.is_finite() && cx.is_finite() && cy.is_finite() {
            st.grad_x[i] += gx;
            st.grad_y[i] += gy;
            st.h_x[i] += cx;
            st.h_y[i] += cy;
            record_sgd2_touch(st, i);
        }
    }
    if wji.is_finite() && wji > 0.0 && (!is_hinge || norm < dji) {
        let err_j = (norm - dji) / norm;
        let gx = wji * err_j * dx;
        let gy = wji * err_j * dy;
        let cx = 2.0 * wji * dir_x;
        let cy = 2.0 * wji * dir_y;
        if gx.is_finite() && gy.is_finite() && cx.is_finite() && cy.is_finite() {
            st.grad_x[j] -= gx;
            st.grad_y[j] -= gy;
            st.h_x[j] += cx;
            st.h_y[j] += cy;
            record_sgd2_touch(st, j);
        }
    }
}

#[inline]
fn flush_sgd2_batch<S: Scalar>(
    xs: &mut [S],
    ys: &mut [S],
    st: &mut Sgd2State,
    alpha_t: f64,
    damping: f64,
) {
    for &u_u32 in &st.touched_nodes {
        let u = u_u32 as usize;
        let sx = alpha_t * st.grad_x[u] / (st.h_x[u] + damping);
        let sy = alpha_t * st.grad_y[u] / (st.h_y[u] + damping);
        if sx.is_finite() {
            let nx = xs[u] - S::from_f64(sx);
            if nx.is_finite() {
                xs[u] = nx;
            }
        }
        if sy.is_finite() {
            let ny = ys[u] - S::from_f64(sy);
            if ny.is_finite() {
                ys[u] = ny;
            }
        }
        st.grad_x[u] = 0.0;
        st.grad_y[u] = 0.0;
        st.h_x[u] = 0.0;
        st.h_y[u] = 0.0;
    }
    st.touched_nodes.clear();
    st.touched_stamp.clear();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn step_sgd2_state_nd<S: Scalar>(
    st: &mut Sgd2State,
    pairs: &mut PairStorage<S>,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    eta_f64: f64,
    eta_max: f64,
    step_scale: f64,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let dims = layout.dims();
    let total_len = num_nodes * dims;
    let sc = st
        .nd_scratch
        .get_or_insert_with(|| Sgd2NdScratch::new(dims, total_len));
    if num_nodes <= 1 || !eta_f64.is_finite() || eta_f64 <= 0.0 || eta_max <= 0.0 {
        return Ok(());
    }
    let alpha_t = (eta_f64 / eta_max).clamp(0.0, step_scale.max(1.0));
    let batch_size = st.batch_size;
    let damping = st.damping;
    let touched_stamp = &mut st.touched_stamp;
    let touched_nodes = &mut st.touched_nodes;
    let coords = layout.coords_mut();
    let mut batch_count = 0usize;

    match pairs {
        PairStorage::Symmetric(sym_pairs) => {
            try_for_each_shuffled_pair(sym_pairs, block_order, rng, |&p| {
                accumulate_sgd2_sym_nd(coords, dims, sc, touched_stamp, touched_nodes, p);
                batch_count += 1;
                if batch_count >= batch_size {
                    flush_sgd2_nd(
                        coords,
                        dims,
                        sc,
                        touched_stamp,
                        touched_nodes,
                        alpha_t,
                        damping,
                    );
                    batch_count = 0;
                }
                Ok(())
            })?;
        }
        PairStorage::General(gen_pairs) => {
            try_for_each_shuffled_pair(gen_pairs, block_order, rng, |p| {
                accumulate_sgd2_gen_nd(coords, dims, sc, touched_stamp, touched_nodes, p);
                batch_count += 1;
                if batch_count >= batch_size {
                    flush_sgd2_nd(
                        coords,
                        dims,
                        sc,
                        touched_stamp,
                        touched_nodes,
                        alpha_t,
                        damping,
                    );
                    batch_count = 0;
                }
                Ok(())
            })?;
        }
    }
    if batch_count > 0 {
        flush_sgd2_nd(
            coords,
            dims,
            sc,
            touched_stamp,
            touched_nodes,
            alpha_t,
            damping,
        );
    }
    Ok(())
}

struct Sgd2DirUpdate {
    bi: usize,
    bj: usize,
    w: f64,
    err: f64,
    inv_n2: f64,
    sign: f64,
    target_node: usize,
}

#[inline]
fn try_accum_sgd2_endpoint_nd<S: Scalar>(
    coords: &[S],
    dims: usize,
    upd: Sgd2DirUpdate,
    sc: &mut Sgd2NdScratch,
    touched_stamp: &mut VisitStamp,
    touched_nodes: &mut Vec<u32>,
) {
    let ok = (0..dims).all(|a| {
        let da = coords[upd.bi + a].to_f64() - coords[upd.bj + a].to_f64();
        let ga = upd.w * upd.err * da;
        let ca = 2.0 * upd.w * (0.5 + 0.5 * (da * da) * upd.inv_n2);
        ga.is_finite() && ca.is_finite()
    });
    if !ok {
        return;
    }
    let bt = upd.target_node * dims;
    for a in 0..dims {
        let da = coords[upd.bi + a].to_f64() - coords[upd.bj + a].to_f64();
        sc.grad[bt + a] += upd.sign * upd.w * upd.err * da;
        sc.h[bt + a] += 2.0 * upd.w * (0.5 + 0.5 * (da * da) * upd.inv_n2);
    }
    if touched_stamp.visit(upd.target_node) {
        touched_nodes.push(upd.target_node as u32);
    }
}

#[inline]
fn accumulate_sgd2_sym_nd<S: Scalar>(
    coords: &[S],
    dims: usize,
    sc: &mut Sgd2NdScratch,
    touched_stamp: &mut VisitStamp,
    touched_nodes: &mut Vec<u32>,
    p: PackedSymPair<S>,
) {
    let d = p.d.to_f64();
    let w = p.w.to_f64();
    if !d.is_finite() || d <= 0.0 || !w.is_finite() || w <= 0.0 {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let bi = i * dims;
    let bj = j * dims;
    let norm = nd_norm_f64(coords, bi, bj, dims);
    if !norm.is_finite() || norm <= 0.0 || (is_hinge && norm >= d) {
        return;
    }
    let err = (norm - d) / norm;
    let inv_n2 = 1.0 / (norm * norm);
    let ok = (0..dims).all(|a| {
        let da = coords[bi + a].to_f64() - coords[bj + a].to_f64();
        (w * err * da).is_finite() && (2.0 * w * (0.5 + 0.5 * (da * da) * inv_n2)).is_finite()
    });
    if !ok {
        return;
    }
    for a in 0..dims {
        let da = coords[bi + a].to_f64() - coords[bj + a].to_f64();
        let ga = w * err * da;
        let ca = 2.0 * w * (0.5 + 0.5 * (da * da) * inv_n2);
        sc.grad[bi + a] += ga;
        sc.h[bi + a] += ca;
        sc.grad[bj + a] -= ga;
        sc.h[bj + a] += ca;
    }
    if touched_stamp.visit(i) {
        touched_nodes.push(i as u32);
    }
    if touched_stamp.visit(j) {
        touched_nodes.push(j as u32);
    }
}

#[inline]
fn accumulate_sgd2_gen_nd<S: Scalar>(
    coords: &[S],
    dims: usize,
    sc: &mut Sgd2NdScratch,
    touched_stamp: &mut VisitStamp,
    touched_nodes: &mut Vec<u32>,
    p: &PackedGenPair<S>,
) {
    let dij = p.dij.to_f64();
    let dji = p.dji.to_f64();
    let wij = p.wij.to_f64();
    let wji = p.wji.to_f64();
    if !dij.is_finite() || dij <= 0.0 || !dji.is_finite() || dji <= 0.0 {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let bi = i * dims;
    let bj = j * dims;
    let norm = nd_norm_f64(coords, bi, bj, dims);
    if !norm.is_finite() || norm <= 0.0 {
        return;
    }
    let inv_n2 = 1.0 / (norm * norm);
    if wij.is_finite() && wij > 0.0 && (!is_hinge || norm < dij) {
        let err_i = (norm - dij) / norm;
        try_accum_sgd2_endpoint_nd(
            coords,
            dims,
            Sgd2DirUpdate {
                bi,
                bj,
                w: wij,
                err: err_i,
                inv_n2,
                sign: 1.0,
                target_node: i,
            },
            sc,
            touched_stamp,
            touched_nodes,
        );
    }
    if wji.is_finite() && wji > 0.0 && (!is_hinge || norm < dji) {
        let err_j = (norm - dji) / norm;
        try_accum_sgd2_endpoint_nd(
            coords,
            dims,
            Sgd2DirUpdate {
                bi,
                bj,
                w: wji,
                err: err_j,
                inv_n2,
                sign: -1.0,
                target_node: j,
            },
            sc,
            touched_stamp,
            touched_nodes,
        );
    }
}

#[inline]
fn flush_sgd2_nd<S: Scalar>(
    coords: &mut [S],
    dims: usize,
    sc: &mut Sgd2NdScratch,
    touched_stamp: &mut VisitStamp,
    touched_nodes: &mut Vec<u32>,
    alpha_t: f64,
    damping: f64,
) {
    for &u_u32 in touched_nodes.iter() {
        let base = (u_u32 as usize) * dims;
        for a in 0..dims {
            let idx = base + a;
            let step = alpha_t * sc.grad[idx] / (sc.h[idx] + damping);
            if step.is_finite() {
                let next_c = coords[idx] - S::from_f64(step);
                if next_c.is_finite() {
                    coords[idx] = next_c;
                }
            }
            sc.grad[idx] = 0.0;
            sc.h[idx] = 0.0;
        }
    }
    touched_nodes.clear();
    touched_stamp.clear();
}
