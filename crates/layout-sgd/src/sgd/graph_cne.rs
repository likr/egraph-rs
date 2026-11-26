use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, SolverStatus,
};
use egraph_neighborhood::graph_row_normalized_probabilities;

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::graph_sampling::{AliasTable, extract_canonical_undirected_edges};
use super::pairs::{MAX_PACKED_NODES, validate_graph_weights};
use super::update::scatter_collapsed_layout_if_needed;
use super::variant_config::{GraphCneConfig, GraphCneObjective};

#[derive(Debug, Clone)]
pub(super) struct GraphCneState {
    pub(super) config: GraphCneConfig,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edge_alias: Option<AliasTable>,
    neg_scratch: Vec<u32>,
}

pub(super) struct BuiltGraphCne {
    pub(super) num_nodes: usize,
    pub(super) num_springs: usize,
    pub(super) state: GraphCneState,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

pub(super) fn build_graph_cne_state<W: Scalar>(
    graph: &CsrGraph<W>,
    config: GraphCneConfig,
) -> CoreResult<BuiltGraphCne> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::graph_cne")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }

    let schedule_cfg = SgdScheduleConfig {
        iterations: config.iterations,
        eps: 0.01,
        kind: SgdScheduleKind::Linear,
        step_scale: config.initial_lr,
    };
    let status = if config.iterations == 0 {
        SolverStatus::Completed
    } else {
        SolverStatus::Initialized
    };

    let mut edges_u = Vec::new();
    let mut edges_v = Vec::new();
    let mut edge_weights = Vec::new();

    if n > 1 && graph.nnz() > 0 {
        if config.row_normalize {
            let probs = graph_row_normalized_probabilities(graph)?;
            for u in 0..n {
                let u_u32 = u as u32;
                let (nbrs, vals) = probs.row(u);
                for (&v, &p_uv) in nbrs.iter().zip(vals.iter()) {
                    if u_u32 < v && p_uv.is_finite() && p_uv > 0.0 {
                        edges_u.push(u_u32);
                        edges_v.push(v);
                        edge_weights.push(p_uv);
                    }
                }
            }
        } else {
            for (u, v, w) in extract_canonical_undirected_edges(graph) {
                if w.is_finite() && w > 0.0 {
                    edges_u.push(u);
                    edges_v.push(v);
                    edge_weights.push(w);
                }
            }
        }
    }

    let edge_alias = AliasTable::from_weights(&edge_weights);
    let m = edges_u.len();

    Ok(BuiltGraphCne {
        num_nodes: n,
        num_springs: m,
        state: GraphCneState {
            config,
            edges_u,
            edges_v,
            edge_alias,
            neg_scratch: Vec::with_capacity(config.negative_samples),
        },
        schedule_cfg,
        status,
        rng: DeterministicRng::from_seed_u64(config.seed),
    })
}

pub(super) fn step_graph_cne_state<S: Scalar>(
    st: &mut GraphCneState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut Layout2d<S>,
) {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }
    if iter == 0 {
        scatter_collapsed_layout_if_needed(layout, 0.02, rng);
    }
    let Some(edge_alias) = &st.edge_alias else {
        return;
    };

    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    if lr.is_finite() && lr > 0.0 {
        let m = st.edges_u.len();
        let neg_samples = st.config.negative_samples;
        let rho = st.config.repulsion_weight;
        let objective = st.config.objective;
        let (xs, ys) = layout.xy_mut();
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

            let ux_cur = xs[u].to_f64();
            let uy_cur = ys[u].to_f64();
            let dx_uv = ux_cur - xs[v].to_f64();
            let dy_uv = uy_cur - ys[v].to_f64();
            let d2_uv = dx_uv * dx_uv + dy_uv * dy_uv;
            if !d2_uv.is_finite() {
                continue;
            }
            let w_uv = 1.0 / (1.0 + d2_uv);

            match objective {
                GraphCneObjective::InfoNce => {
                    let mut sum_neg_w = 0.0_f64;
                    if rho > 0.0 {
                        for &k_u32 in &st.neg_scratch {
                            let k = k_u32 as usize;
                            let dx = ux_cur - xs[k].to_f64();
                            let dy = uy_cur - ys[k].to_f64();
                            let d2 = dx * dx + dy * dy;
                            if d2.is_finite() {
                                sum_neg_w += 1.0 / (1.0 + d2);
                            }
                        }
                    }
                    let denom = w_uv + rho * sum_neg_w;
                    if !denom.is_finite() || denom <= 0.0 {
                        continue;
                    }
                    let p_uv = w_uv / denom;
                    let attr_coeff = 2.0 * lr * (1.0 - p_uv) * w_uv;
                    let sx_pos = (attr_coeff * dx_uv).clamp(-4.0, 4.0);
                    let sy_pos = (attr_coeff * dy_uv).clamp(-4.0, 4.0);

                    let mut acc_rep_x = 0.0_f64;
                    let mut acc_rep_y = 0.0_f64;
                    if rho > 0.0 {
                        for &k_u32 in &st.neg_scratch {
                            let k = k_u32 as usize;
                            let dx = ux_cur - xs[k].to_f64();
                            let dy = uy_cur - ys[k].to_f64();
                            let d2 = dx * dx + dy * dy;
                            if d2.is_finite() {
                                let w_uk = 1.0 / (1.0 + d2);
                                let q_k = (rho * w_uk) / denom;
                                let rep_coeff = 2.0 * lr * q_k * w_uk;
                                acc_rep_x += (rep_coeff * dx).clamp(-4.0, 4.0);
                                acc_rep_y += (rep_coeff * dy).clamp(-4.0, 4.0);
                            }
                        }
                    }

                    if sx_pos.is_finite()
                        && sy_pos.is_finite()
                        && acc_rep_x.is_finite()
                        && acc_rep_y.is_finite()
                    {
                        let ux_new = xs[u] + S::from_f64(acc_rep_x - sx_pos);
                        let uy_new = ys[u] + S::from_f64(acc_rep_y - sy_pos);
                        let vx_new = xs[v] + S::from_f64(sx_pos);
                        let vy_new = ys[v] + S::from_f64(sy_pos);
                        if ux_new.is_finite()
                            && uy_new.is_finite()
                            && vx_new.is_finite()
                            && vy_new.is_finite()
                        {
                            xs[u] = ux_new;
                            ys[u] = uy_new;
                            xs[v] = vx_new;
                            ys[v] = vy_new;
                        }
                    }
                }
                GraphCneObjective::NegTSne => {
                    let attr_coeff = 2.0 * lr * w_uv;
                    let sx_pos = (attr_coeff * dx_uv).clamp(-4.0, 4.0);
                    let sy_pos = (attr_coeff * dy_uv).clamp(-4.0, 4.0);

                    let mut acc_rep_x = 0.0_f64;
                    let mut acc_rep_y = 0.0_f64;
                    if rho > 0.0 && !st.neg_scratch.is_empty() {
                        let rep_scale = rho / (st.neg_scratch.len() as f64);
                        for &k_u32 in &st.neg_scratch {
                            let k = k_u32 as usize;
                            let dx = ux_cur - xs[k].to_f64();
                            let dy = uy_cur - ys[k].to_f64();
                            let d2 = dx * dx + dy * dy;
                            if d2.is_finite() {
                                let w_uk = 1.0 / (1.0 + d2);
                                let rep_coeff = 2.0 * lr * rep_scale * w_uk * w_uk;
                                acc_rep_x += (rep_coeff * dx).clamp(-4.0, 4.0);
                                acc_rep_y += (rep_coeff * dy).clamp(-4.0, 4.0);
                            }
                        }
                    }

                    if sx_pos.is_finite()
                        && sy_pos.is_finite()
                        && acc_rep_x.is_finite()
                        && acc_rep_y.is_finite()
                    {
                        let ux_new = xs[u] + S::from_f64(acc_rep_x - sx_pos);
                        let uy_new = ys[u] + S::from_f64(acc_rep_y - sy_pos);
                        let vx_new = xs[v] + S::from_f64(sx_pos);
                        let vy_new = ys[v] + S::from_f64(sy_pos);
                        if ux_new.is_finite()
                            && uy_new.is_finite()
                            && vx_new.is_finite()
                            && vy_new.is_finite()
                        {
                            xs[u] = ux_new;
                            ys[u] = uy_new;
                            xs[v] = vx_new;
                            ys[v] = vy_new;
                        }
                    }
                }
            }
        }
    }
}
