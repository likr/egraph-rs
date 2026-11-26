use egraph_core::{CoreResult, DeterministicRng, ManifoldLayout2d, Scalar};

use super::pairs::{
    BLOCK_SHUFFLE_THRESHOLD, BLOCK_SIZE, HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair,
    PairStorage, try_for_each_shuffled_pair,
};
use super::sgd2::Sgd2State;
use super::snap_t_fdp::SnapTFdpState;
use super::update::apply_clamped_manifold_vertex_step;

pub(super) fn step_snap_t_fdp_state_manifold<S: Scalar>(
    st: &mut SnapTFdpState,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let total = st.config.iterations;
    if num_nodes <= 1 || st.edges_u.is_empty() || total == 0 || iter >= total {
        return Ok(());
    }
    let eta_f64 = st.config.learning_rate * (1.0 - (iter as f64) / (total as f64));
    if !eta_f64.is_finite() || eta_f64 <= 0.0 {
        return Ok(());
    }
    let progress = if total <= 1 {
        0.0
    } else {
        (iter as f64) / ((total - 1) as f64)
    };
    let rho = st.config.rho_start
        + (st.config.rho_end - st.config.rho_start) * progress.powf(st.config.rho_power);
    let base_attr = S::from_f64(eta_f64 * 0.1 * st.config.alpha);
    let base_rep = S::from_f64(eta_f64 * rho * st.config.repulsion_scale);
    let k_neg = st.config.negatives_per_edge;
    let bounded_attraction = st.config.bounded_attraction;
    let len = st.edge_order.len();

    if len <= BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(&mut st.edge_order);
        apply_snap_edge_slice_manifold(
            layout,
            &st.edge_order,
            &st.edges_u,
            &st.edges_v,
            st.edge_weights.as_deref(),
            st.deg_scales.as_deref(),
            bounded_attraction,
            num_nodes,
            k_neg,
            base_attr,
            base_rep,
            rng,
        )?;
    } else {
        rng.shuffle(block_order);
        for &b in block_order.iter() {
            let start = (b as usize) * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(len);
            let chunk = &mut st.edge_order[start..end];
            rng.shuffle(chunk);
            apply_snap_edge_slice_manifold(
                layout,
                chunk,
                &st.edges_u,
                &st.edges_v,
                st.edge_weights.as_deref(),
                st.deg_scales.as_deref(),
                bounded_attraction,
                num_nodes,
                k_neg,
                base_attr,
                base_rep,
                rng,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_snap_edge_slice_manifold<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    order_slice: &[u32],
    edges_u: &[u32],
    edges_v: &[u32],
    edge_weights: Option<&[f32]>,
    deg_scales: Option<&[f32]>,
    bounded_attraction: bool,
    num_nodes: usize,
    negatives_per_edge: usize,
    base_attr: S,
    base_rep: S,
    rng: &mut DeterministicRng,
) -> CoreResult<()> {
    let eight = S::from_f64(8.0);
    let max_attr = S::from_f64(0.49);
    let max_rep = S::from_f64(0.5);
    for (pos, &edge_idx) in order_slice.iter().enumerate() {
        let idx = edge_idx as usize;
        let u = edges_u[idx] as usize;
        let v = edges_v[idx] as usize;
        let w_e = match edge_weights {
            Some(ws) => S::from_f32(ws[idx]),
            None => S::ONE,
        };
        let deg_factor = match deg_scales {
            Some(ds) => S::from_f32(ds[u]) * S::from_f32(ds[v]),
            None => S::ONE,
        };

        if w_e > S::ZERO && base_attr > S::ZERO && deg_factor > S::ZERO {
            let (tux, tuy) = layout.tangent_delta(u, v)?;
            let d2 = tux * tux + tuy * tuy;
            let inv_1_d2 = if d2.is_finite() {
                S::ONE / (S::ONE + d2)
            } else {
                S::ZERO
            };
            let kernel = if bounded_attraction {
                eight * inv_1_d2
            } else {
                S::ONE + eight * inv_1_d2
            };
            let attr = base_attr * w_e * deg_factor * kernel;
            if attr.is_finite() && attr > S::ZERO {
                let coeff = attr.min(max_attr);
                let (tvx, tvy) = layout.tangent_delta(v, u)?;
                apply_clamped_manifold_vertex_step(layout, u, tux * coeff, tuy * coeff)?;
                apply_clamped_manifold_vertex_step(layout, v, tvx * coeff, tvy * coeff)?;
            }
        }

        if base_rep > S::ZERO {
            for neg_idx in 0..negatives_per_edge {
                let anchor = if ((pos + neg_idx) & 1) == 0 { u } else { v };
                let w = rng.gen_range_usize(num_nodes);
                if w == anchor {
                    continue;
                }
                let (tax, tay) = layout.tangent_delta(anchor, w)?;
                let rd2 = tax * tax + tay * tay;
                if rd2.is_finite() {
                    let denom = (S::ONE + rd2) * (S::ONE + rd2);
                    if denom.is_finite() && denom > S::ZERO {
                        let rep = (base_rep / denom).min(max_rep);
                        if rep.is_finite() && rep > S::ZERO {
                            let (twx, twy) = layout.tangent_delta(w, anchor)?;
                            apply_clamped_manifold_vertex_step(
                                layout,
                                anchor,
                                -tax * rep,
                                -tay * rep,
                            )?;
                            apply_clamped_manifold_vertex_step(layout, w, -twx * rep, -twy * rep)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn step_sgd2_state_manifold<S: Scalar>(
    st: &mut Sgd2State,
    pairs: &mut PairStorage<S>,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    eta_f64: f64,
    eta_max: f64,
    step_scale: f64,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    if num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 && eta_max > 0.0 {
        let alpha_t = (eta_f64 / eta_max).clamp(0.0, step_scale.max(1.0));
        let batch_size = st.batch_size;
        let damping = st.damping;
        let mut batch_count = 0usize;

        match pairs {
            PairStorage::Symmetric(sym_pairs) => {
                try_for_each_shuffled_pair(sym_pairs, block_order, rng, |&p| {
                    accumulate_sgd2_sym_manifold(layout, st, p)?;
                    batch_count += 1;
                    if batch_count >= batch_size {
                        flush_sgd2_batch_manifold(layout, st, alpha_t, damping)?;
                        batch_count = 0;
                    }
                    Ok(())
                })?;
            }
            PairStorage::General(gen_pairs) => {
                try_for_each_shuffled_pair(gen_pairs, block_order, rng, |p| {
                    accumulate_sgd2_gen_manifold(layout, st, p)?;
                    batch_count += 1;
                    if batch_count >= batch_size {
                        flush_sgd2_batch_manifold(layout, st, alpha_t, damping)?;
                        batch_count = 0;
                    }
                    Ok(())
                })?;
            }
        }
        if batch_count > 0 {
            flush_sgd2_batch_manifold(layout, st, alpha_t, damping)?;
        }
    }
    Ok(())
}

fn accumulate_sgd2_sym_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    st: &mut Sgd2State,
    p: PackedSymPair<S>,
) -> CoreResult<()> {
    let d = p.d.to_f64();
    let w = p.w.to_f64();
    if !d.is_finite() || d <= 0.0 || !w.is_finite() || w <= 0.0 {
        return Ok(());
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;

    let (tix, tiy) = layout.tangent_delta(i, j)?;
    let tixf = tix.to_f64();
    let tiyf = tiy.to_f64();
    let norm = tixf.hypot(tiyf);
    if !norm.is_finite() || norm <= 0.0 || (is_hinge && norm >= d) {
        return Ok(());
    }
    let err = (norm - d) / norm;
    let gx_i = -w * err * tixf;
    let gy_i = -w * err * tiyf;
    let inv_n2 = 1.0 / (norm * norm);
    let cx_i = 2.0 * w * (0.5 + 0.5 * (tixf * tixf) * inv_n2);
    let cy_i = 2.0 * w * (0.5 + 0.5 * (tiyf * tiyf) * inv_n2);

    let (tjx, tjy) = layout.tangent_delta(j, i)?;
    let tjxf = tjx.to_f64();
    let tjyf = tjy.to_f64();
    let norm_j = tjxf.hypot(tjyf);
    if !norm_j.is_finite() || norm_j <= 0.0 {
        return Ok(());
    }
    let err_j = (norm_j - d) / norm_j;
    let gx_j = -w * err_j * tjxf;
    let gy_j = -w * err_j * tjyf;
    let inv_nj2 = 1.0 / (norm_j * norm_j);
    let cx_j = 2.0 * w * (0.5 + 0.5 * (tjxf * tjxf) * inv_nj2);
    let cy_j = 2.0 * w * (0.5 + 0.5 * (tjyf * tjyf) * inv_nj2);

    if gx_i.is_finite()
        && gy_i.is_finite()
        && cx_i.is_finite()
        && cy_i.is_finite()
        && gx_j.is_finite()
        && gy_j.is_finite()
        && cx_j.is_finite()
        && cy_j.is_finite()
    {
        st.grad_x[i] += gx_i;
        st.grad_y[i] += gy_i;
        st.h_x[i] += cx_i;
        st.h_y[i] += cy_i;
        st.grad_x[j] += gx_j;
        st.grad_y[j] += gy_j;
        st.h_x[j] += cx_j;
        st.h_y[j] += cy_j;
        if st.touched_stamp.visit(i) {
            st.touched_nodes.push(i as u32);
        }
        if st.touched_stamp.visit(j) {
            st.touched_nodes.push(j as u32);
        }
    }
    Ok(())
}

fn accumulate_sgd2_gen_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    st: &mut Sgd2State,
    p: &PackedGenPair<S>,
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
        let (tix, tiy) = layout.tangent_delta(i, j)?;
        let tixf = tix.to_f64();
        let tiyf = tiy.to_f64();
        let norm = tixf.hypot(tiyf);
        if norm.is_finite() && norm > 0.0 && (!is_hinge || norm < dij) {
            let err_i = (norm - dij) / norm;
            let gx = -wij * err_i * tixf;
            let gy = -wij * err_i * tiyf;
            let inv_n2 = 1.0 / (norm * norm);
            let cx = 2.0 * wij * (0.5 + 0.5 * (tixf * tixf) * inv_n2);
            let cy = 2.0 * wij * (0.5 + 0.5 * (tiyf * tiyf) * inv_n2);
            if gx.is_finite() && gy.is_finite() && cx.is_finite() && cy.is_finite() {
                st.grad_x[i] += gx;
                st.grad_y[i] += gy;
                st.h_x[i] += cx;
                st.h_y[i] += cy;
                if st.touched_stamp.visit(i) {
                    st.touched_nodes.push(i as u32);
                }
            }
        }
    }
    if wji.is_finite() && wji > 0.0 {
        let (tjx, tjy) = layout.tangent_delta(j, i)?;
        let tjxf = tjx.to_f64();
        let tjyf = tjy.to_f64();
        let norm = tjxf.hypot(tjyf);
        if norm.is_finite() && norm > 0.0 && (!is_hinge || norm < dji) {
            let err_j = (norm - dji) / norm;
            let gx = -wji * err_j * tjxf;
            let gy = -wji * err_j * tjyf;
            let inv_n2 = 1.0 / (norm * norm);
            let cx = 2.0 * wji * (0.5 + 0.5 * (tjxf * tjxf) * inv_n2);
            let cy = 2.0 * wji * (0.5 + 0.5 * (tjyf * tjyf) * inv_n2);
            if gx.is_finite() && gy.is_finite() && cx.is_finite() && cy.is_finite() {
                st.grad_x[j] += gx;
                st.grad_y[j] += gy;
                st.h_x[j] += cx;
                st.h_y[j] += cy;
                if st.touched_stamp.visit(j) {
                    st.touched_nodes.push(j as u32);
                }
            }
        }
    }
    Ok(())
}

fn flush_sgd2_batch_manifold<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    st: &mut Sgd2State,
    alpha_t: f64,
    damping: f64,
) -> CoreResult<()> {
    for &u_u32 in &st.touched_nodes {
        let u = u_u32 as usize;
        let sx = -alpha_t * st.grad_x[u] / (st.h_x[u] + damping);
        let sy = -alpha_t * st.grad_y[u] / (st.h_y[u] + damping);
        if sx.is_finite() && sy.is_finite() {
            apply_clamped_manifold_vertex_step(layout, u, S::from_f64(sx), S::from_f64(sy))?;
        }
        st.grad_x[u] = 0.0;
        st.grad_y[u] = 0.0;
        st.h_x[u] = 0.0;
        st.h_y[u] = 0.0;
    }
    st.touched_nodes.clear();
    st.touched_stamp.clear();
    Ok(())
}
