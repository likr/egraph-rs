use egraph_core::{CoreResult, DeterministicRng, ManifoldLayout2d, Scalar};

use super::config::SgdScheduleConfig;
use super::drgraph_coarsen::DrGraphState;
use super::graph_cne::GraphCneState;
use super::largevis::LargeVisState;
use super::pacmap::PacmapState;
use super::update::{
    apply_clamped_manifold_vertex_step, scatter_collapsed_manifold_layout_if_needed,
};
use super::variant_config::GraphCneObjective;

pub(super) fn step_pacmap_state_manifold<S: Scalar>(
    st: &mut PacmapState,
    num_nodes: usize,
    iter: usize,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let total = st.config.total_iterations();
    if num_nodes <= 1 || total == 0 || iter >= total {
        return Ok(());
    }

    let [p1_iters, p2_iters, _] = st.config.phase_iterations;
    let (w_nb, w_mn, w_fp) = if iter < p1_iters {
        let p1 = if p1_iters <= 1 {
            0.0
        } else {
            (iter as f64) / ((p1_iters - 1) as f64)
        };
        (2.0, 1000.0 * (1.0 - p1) + 3.0 * p1, 1.0)
    } else if iter < p1_iters + p2_iters {
        (3.0, 3.0, 1.0)
    } else {
        (1.0, 0.0, 1.0)
    };

    st.grad_x.fill(0.0);
    st.grad_y.fill(0.0);

    if w_nb > 0.0 {
        let scale_nb = 20.0 * w_nb;
        let has_weights = !st.nb_weights.is_empty();
        for (idx, &(u, v)) in st.nb_pairs.iter().enumerate() {
            let ui = u as usize;
            let vi = v as usize;
            let (tux, tuy) = layout.tangent_delta(ui, vi)?;
            let (tvx, tvy) = layout.tangent_delta(vi, ui)?;
            let tuxf = tux.to_f64();
            let tuyf = tuy.to_f64();
            let d2 = tuxf * tuxf + tuyf * tuyf;
            if d2.is_finite() {
                let denom = 10.0 + d2;
                let pair_w = if has_weights { st.nb_weights[idx] } else { 1.0 };
                let coeff = (scale_nb * pair_w) / (denom * denom);
                if coeff.is_finite() {
                    st.grad_x[ui] -= coeff * tuxf;
                    st.grad_y[ui] -= coeff * tuyf;
                    st.grad_x[vi] -= coeff * tvx.to_f64();
                    st.grad_y[vi] -= coeff * tvy.to_f64();
                }
            }
        }
    }
    if w_mn > 0.0 {
        let scale_mn = 20_000.0 * w_mn;
        for &(u, v) in &st.mn_pairs {
            let ui = u as usize;
            let vi = v as usize;
            let (tux, tuy) = layout.tangent_delta(ui, vi)?;
            let (tvx, tvy) = layout.tangent_delta(vi, ui)?;
            let tuxf = tux.to_f64();
            let tuyf = tuy.to_f64();
            let d2 = tuxf * tuxf + tuyf * tuyf;
            if d2.is_finite() {
                let denom = 10_000.0 + d2;
                let coeff = scale_mn / (denom * denom);
                if coeff.is_finite() {
                    st.grad_x[ui] -= coeff * tuxf;
                    st.grad_y[ui] -= coeff * tuyf;
                    st.grad_x[vi] -= coeff * tvx.to_f64();
                    st.grad_y[vi] -= coeff * tvy.to_f64();
                }
            }
        }
    }
    if w_fp > 0.0 {
        let scale_fp = -2.0 * w_fp;
        for &(u, v) in &st.fp_pairs {
            let ui = u as usize;
            let vi = v as usize;
            let (tux, tuy) = layout.tangent_delta(ui, vi)?;
            let (tvx, tvy) = layout.tangent_delta(vi, ui)?;
            let tuxf = tux.to_f64();
            let tuyf = tuy.to_f64();
            let d2 = tuxf * tuxf + tuyf * tuyf;
            if d2.is_finite() {
                let denom = 1.0 + d2;
                let coeff = scale_fp / (denom * denom);
                if coeff.is_finite() {
                    st.grad_x[ui] -= coeff * tuxf;
                    st.grad_y[ui] -= coeff * tuyf;
                    st.grad_x[vi] -= coeff * tvx.to_f64();
                    st.grad_y[vi] -= coeff * tvy.to_f64();
                }
            }
        }
    }

    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const ADAM_EPS: f64 = 1e-7;
    let t_step = (iter + 1) as i32;
    let bc1 = (1.0 - BETA1.powi(t_step)).max(1e-12);
    let bc2 = (1.0 - BETA2.powi(t_step)).max(1e-12);
    let lr_t = st.config.learning_rate * bc2.sqrt() / bc1;

    for u in 0..num_nodes {
        let gx = st.grad_x[u];
        let gy = st.grad_y[u];
        st.m_x[u] = BETA1 * st.m_x[u] + (1.0 - BETA1) * gx;
        st.m_y[u] = BETA1 * st.m_y[u] + (1.0 - BETA1) * gy;
        st.v_x[u] = BETA2 * st.v_x[u] + (1.0 - BETA2) * gx * gx;
        st.v_y[u] = BETA2 * st.v_y[u] + (1.0 - BETA2) * gy * gy;

        let step_x = -lr_t * st.m_x[u] / (st.v_x[u].sqrt() + ADAM_EPS);
        let step_y = -lr_t * st.m_y[u] / (st.v_y[u].sqrt() + ADAM_EPS);
        if step_x.is_finite() && step_y.is_finite() {
            apply_clamped_manifold_vertex_step(
                layout,
                u,
                S::from_f64(step_x),
                S::from_f64(step_y),
            )?;
        }
    }
    Ok(())
}

pub(super) fn step_largevis_state_manifold<S: Scalar>(
    st: &LargeVisState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return Ok(());
    }
    if iter == 0 {
        scatter_collapsed_manifold_layout_if_needed(layout, 0.02, rng)?;
    }
    let Some(edge_alias) = &st.edge_alias else {
        return Ok(());
    };
    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    if !lr.is_finite() || lr <= 0.0 {
        return Ok(());
    }
    let m = st.edges_u.len();
    let gamma = st.config.gamma;
    let neg_samples = st.config.negative_samples;
    let grad_cap = 4.0 * layout.scale();
    for _ in 0..m {
        let e_idx = edge_alias.sample(rng);
        let u_raw = st.edges_u[e_idx] as usize;
        let v_raw = st.edges_v[e_idx] as usize;
        let (u, v) = if (rng.next_u32() & 1) == 0 {
            (u_raw, v_raw)
        } else {
            (v_raw, u_raw)
        };

        let (tux, tuy) = layout.tangent_delta(u, v)?;
        let (tvx, tvy) = layout.tangent_delta(v, u)?;
        let tuxf = tux.to_f64();
        let tuyf = tuy.to_f64();
        let d2 = tuxf * tuxf + tuyf * tuyf;
        if d2.is_finite() {
            let pos_coeff = (2.0 * lr) / (1.0 + d2);
            let sux = (pos_coeff * tuxf).clamp(-grad_cap, grad_cap);
            let suy = (pos_coeff * tuyf).clamp(-grad_cap, grad_cap);
            let svx = (pos_coeff * tvx.to_f64()).clamp(-grad_cap, grad_cap);
            let svy = (pos_coeff * tvy.to_f64()).clamp(-grad_cap, grad_cap);
            if sux.is_finite() && suy.is_finite() && svx.is_finite() && svy.is_finite() {
                apply_clamped_manifold_vertex_step(layout, u, S::from_f64(sux), S::from_f64(suy))?;
                apply_clamped_manifold_vertex_step(layout, v, S::from_f64(svx), S::from_f64(svy))?;
            }
        }

        if let Some(neg_alias) = &st.neg_alias
            && gamma > 0.0
        {
            for _ in 0..neg_samples {
                let k = neg_alias.sample(rng);
                if k == u {
                    continue;
                }
                let (tukx, tuky) = layout.tangent_delta(u, k)?;
                let tukxf = tukx.to_f64();
                let tukyf = tuky.to_f64();
                let rd2 = tukxf * tukxf + tukyf * tukyf;
                if rd2.is_finite() && rd2 > 0.0 {
                    let grad = (2.0 * gamma) / ((rd2 + 1e-6) * (1.0 + rd2));
                    let sx = (-lr * grad * tukxf).clamp(-grad_cap, grad_cap);
                    let sy = (-lr * grad * tukyf).clamp(-grad_cap, grad_cap);
                    if sx.is_finite() && sy.is_finite() {
                        apply_clamped_manifold_vertex_step(
                            layout,
                            u,
                            S::from_f64(sx),
                            S::from_f64(sy),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn step_graph_cne_state_manifold<S: Scalar>(
    st: &mut GraphCneState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return Ok(());
    }
    if iter == 0 {
        scatter_collapsed_manifold_layout_if_needed(layout, 0.02, rng)?;
    }
    let Some(edge_alias) = &st.edge_alias else {
        return Ok(());
    };
    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    if !lr.is_finite() || lr <= 0.0 {
        return Ok(());
    }
    let m = st.edges_u.len();
    let neg_samples = st.config.negative_samples;
    let rho = st.config.repulsion_weight;
    let objective = st.config.objective;
    let grad_cap = 4.0 * layout.scale();

    for _ in 0..m {
        let e_idx = edge_alias.sample(rng);
        let u_raw = st.edges_u[e_idx] as usize;
        let v_raw = st.edges_v[e_idx] as usize;
        let (u, v) = if (rng.next_u32() & 1) == 0 {
            (u_raw, v_raw)
        } else {
            (v_raw, u_raw)
        };

        st.neg_scratch.clear();
        for _ in 0..neg_samples {
            let k = rng.gen_range_usize(num_nodes);
            if k != u {
                st.neg_scratch.push(k as u32);
            }
        }

        let (tux, tuy) = layout.tangent_delta(u, v)?;
        let (tvx, tvy) = layout.tangent_delta(v, u)?;
        let tuxf = tux.to_f64();
        let tuyf = tuy.to_f64();
        let d2_uv = tuxf * tuxf + tuyf * tuyf;
        if !d2_uv.is_finite() {
            continue;
        }
        let w_uv = 1.0 / (1.0 + d2_uv);

        st.neg_tx.clear();
        st.neg_ty.clear();
        st.neg_w.clear();
        let mut sum_neg_w = 0.0_f64;
        if rho > 0.0 {
            for &k_u32 in &st.neg_scratch {
                let (tkx, tky) = layout.tangent_delta(u, k_u32 as usize)?;
                let tkxf = tkx.to_f64();
                let tkyf = tky.to_f64();
                let d2 = tkxf * tkxf + tkyf * tkyf;
                let w_uk = if d2.is_finite() {
                    1.0 / (1.0 + d2)
                } else {
                    0.0
                };
                st.neg_tx.push(tkxf);
                st.neg_ty.push(tkyf);
                st.neg_w.push(w_uk);
                sum_neg_w += w_uk;
            }
        }

        let (attr_coeff, mut acc_rep_x, mut acc_rep_y) = match objective {
            GraphCneObjective::InfoNce => {
                let denom = w_uv + rho * sum_neg_w;
                if !denom.is_finite() || denom <= 0.0 {
                    continue;
                }
                let mut rx = 0.0_f64;
                let mut ry = 0.0_f64;
                if rho > 0.0 {
                    for ((&tkxf, &tkyf), &w_uk) in
                        st.neg_tx.iter().zip(st.neg_ty.iter()).zip(st.neg_w.iter())
                    {
                        if w_uk > 0.0 {
                            let rep_coeff = 2.0 * lr * ((rho * w_uk) / denom) * w_uk;
                            rx += (-rep_coeff * tkxf).clamp(-grad_cap, grad_cap);
                            ry += (-rep_coeff * tkyf).clamp(-grad_cap, grad_cap);
                        }
                    }
                }
                (2.0 * lr * (1.0 - w_uv / denom) * w_uv, rx, ry)
            }
            GraphCneObjective::NegTSne => {
                let mut rx = 0.0_f64;
                let mut ry = 0.0_f64;
                if rho > 0.0 && !st.neg_tx.is_empty() {
                    let rep_scale = rho / (st.neg_tx.len() as f64);
                    for ((&tkxf, &tkyf), &w_uk) in
                        st.neg_tx.iter().zip(st.neg_ty.iter()).zip(st.neg_w.iter())
                    {
                        if w_uk > 0.0 {
                            let rep_coeff = 2.0 * lr * rep_scale * w_uk * w_uk;
                            rx += (-rep_coeff * tkxf).clamp(-grad_cap, grad_cap);
                            ry += (-rep_coeff * tkyf).clamp(-grad_cap, grad_cap);
                        }
                    }
                }
                (2.0 * lr * w_uv, rx, ry)
            }
        };

        let pos_ux = (attr_coeff * tuxf).clamp(-grad_cap, grad_cap);
        let pos_uy = (attr_coeff * tuyf).clamp(-grad_cap, grad_cap);
        let pos_vx = (attr_coeff * tvx.to_f64()).clamp(-grad_cap, grad_cap);
        let pos_vy = (attr_coeff * tvy.to_f64()).clamp(-grad_cap, grad_cap);
        if !acc_rep_x.is_finite() {
            acc_rep_x = 0.0;
        }
        if !acc_rep_y.is_finite() {
            acc_rep_y = 0.0;
        }
        if pos_ux.is_finite() && pos_uy.is_finite() && pos_vx.is_finite() && pos_vy.is_finite() {
            apply_clamped_manifold_vertex_step(
                layout,
                u,
                S::from_f64(pos_ux + acc_rep_x),
                S::from_f64(pos_uy + acc_rep_y),
            )?;
            apply_clamped_manifold_vertex_step(
                layout,
                v,
                S::from_f64(pos_vx),
                S::from_f64(pos_vy),
            )?;
        }
    }
    Ok(())
}

pub(super) fn scale_pacmap_phase_iterations(
    old_phases: [usize; 3],
    target_total: usize,
) -> [usize; 3] {
    if target_total == 0 {
        return [0, 0, 0];
    }
    let old_total = old_phases[0] + old_phases[1] + old_phases[2];
    if old_total == target_total {
        return old_phases;
    }
    if old_total == 0 {
        let p1 = (target_total * 2) / 9;
        let p2 = (target_total * 2) / 9;
        return [p1, p2, target_total - p1 - p2];
    }
    let p1 = (old_phases[0] * target_total) / old_total;
    let p2 = (old_phases[1] * target_total) / old_total;
    [p1, p2, target_total - p1 - p2]
}

pub(super) fn reset_drgraph_schedule(st: &mut DrGraphState, cfg: &SgdScheduleConfig, seed: u64) {
    st.config.iterations = cfg.iterations;
    st.config.initial_lr = cfg.step_scale;
    st.config.min_lr = (cfg.step_scale * cfg.eps).max(0.0);
    st.config.seed = seed;
    let num_levels = st.levels.len();
    st.level_iters.fill(0);
    if cfg.iterations > 0 && num_levels > 0 {
        let base = cfg.iterations / num_levels;
        let rem = cfg.iterations % num_levels;
        for (l, slot) in st.level_iters.iter_mut().enumerate() {
            *slot = base + usize::from(l < rem);
        }
    }
    st.current_level = st.level_iters.iter().rposition(|&c| c > 0).unwrap_or(0);
    st.level_iter = 0;
    st.initialized_coords = false;
    st.active_manifold = None;
    st.nd_scratch = None;
}
