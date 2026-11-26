use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, LayoutNd, Scalar, SolverStatus,
};

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::graph_sampling::{AliasTable, extract_canonical_undirected_edges};
use super::nd::scatter_collapsed_layout_nd_if_needed;
use super::nd_variants::nd_dist_sq_f64;
use super::pairs::{MAX_PACKED_NODES, validate_graph_weights};
use super::update::scatter_collapsed_layout_if_needed;
use super::variant_config::LargeVisConfig;

#[derive(Debug, Clone)]
pub(super) struct LargeVisState {
    pub(super) config: LargeVisConfig,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edge_alias: Option<AliasTable>,
    pub(super) neg_alias: Option<AliasTable>,
}

pub(super) struct BuiltLargeVis {
    pub(super) num_nodes: usize,
    pub(super) num_springs: usize,
    pub(super) state: LargeVisState,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

pub(super) fn build_largevis_state<W: Scalar>(
    graph: &CsrGraph<W>,
    config: LargeVisConfig,
) -> CoreResult<BuiltLargeVis> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::largevis")?;

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
        max_mu: None,
    };
    let status = if config.iterations == 0 {
        SolverStatus::Completed
    } else {
        SolverStatus::Initialized
    };

    let mut edges_u = Vec::new();
    let mut edges_v = Vec::new();
    let mut edge_weights = Vec::new();
    let mut deg_w = vec![0.0_f64; n];

    if n > 1 && graph.nnz() > 0 {
        for (u, v, w) in extract_canonical_undirected_edges(graph) {
            if w.is_finite() && w > 0.0 {
                edges_u.push(u);
                edges_v.push(v);
                edge_weights.push(w);
                deg_w[u as usize] += w;
                deg_w[v as usize] += w;
            }
        }
    }

    let edge_alias = AliasTable::from_weights(&edge_weights);
    let neg_weights: Vec<f64> = deg_w
        .into_iter()
        .map(|dw| {
            if dw > 0.0 {
                dw.powf(config.degree_exponent)
            } else {
                0.0
            }
        })
        .collect();
    let neg_alias = AliasTable::from_weights(&neg_weights);
    let m = edges_u.len();

    Ok(BuiltLargeVis {
        num_nodes: n,
        num_springs: m,
        state: LargeVisState {
            config,
            edges_u,
            edges_v,
            edge_alias,
            neg_alias,
        },
        schedule_cfg,
        status,
        rng: DeterministicRng::from_seed_u64(config.seed),
    })
}

pub(super) fn step_largevis_state<S: Scalar>(
    st: &LargeVisState,
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
        let gamma = st.config.gamma;
        let neg_samples = st.config.negative_samples;
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

            let dx = xs[u].to_f64() - xs[v].to_f64();
            let dy = ys[u].to_f64() - ys[v].to_f64();
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() {
                let grad = -2.0 / (1.0 + d2);
                let sx = (lr * grad * dx).clamp(-4.0, 4.0);
                let sy = (lr * grad * dy).clamp(-4.0, 4.0);
                if sx.is_finite() && sy.is_finite() {
                    let ux = xs[u] + S::from_f64(sx);
                    let uy = ys[u] + S::from_f64(sy);
                    let vx = xs[v] - S::from_f64(sx);
                    let vy = ys[v] - S::from_f64(sy);
                    if ux.is_finite() && uy.is_finite() && vx.is_finite() && vy.is_finite() {
                        xs[u] = ux;
                        ys[u] = uy;
                        xs[v] = vx;
                        ys[v] = vy;
                    }
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
                    let rdx = xs[u].to_f64() - xs[k].to_f64();
                    let rdy = ys[u].to_f64() - ys[k].to_f64();
                    let rd2 = rdx * rdx + rdy * rdy;
                    if rd2.is_finite() && rd2 > 0.0 {
                        let grad = (2.0 * gamma) / ((rd2 + 1e-6) * (1.0 + rd2));
                        let sx = (lr * grad * rdx).clamp(-4.0, 4.0);
                        let sy = (lr * grad * rdy).clamp(-4.0, 4.0);
                        if sx.is_finite() && sy.is_finite() {
                            let ux = xs[u] + S::from_f64(sx);
                            let uy = ys[u] + S::from_f64(sy);
                            if ux.is_finite() && uy.is_finite() {
                                xs[u] = ux;
                                ys[u] = uy;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn step_largevis_state_nd<S: Scalar>(
    st: &LargeVisState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut LayoutNd<S>,
) {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }
    if iter == 0 {
        scatter_collapsed_layout_nd_if_needed(layout, 0.02, rng);
    }
    let Some(edge_alias) = &st.edge_alias else {
        return;
    };
    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    if !lr.is_finite() || lr <= 0.0 {
        return;
    }
    let m = st.edges_u.len();
    let gamma = st.config.gamma;
    let neg_samples = st.config.negative_samples;
    let dims = layout.dims();
    let coords = layout.coords_mut();

    for _ in 0..m {
        let e_idx = edge_alias.sample(rng);
        let u_raw = st.edges_u[e_idx] as usize;
        let v_raw = st.edges_v[e_idx] as usize;
        let (u, v) = if (rng.next_u32() & 1) == 0 {
            (u_raw, v_raw)
        } else {
            (v_raw, u_raw)
        };

        let (bu, bv) = (u * dims, v * dims);
        let d2 = nd_dist_sq_f64(coords, bu, bv, dims);
        if d2.is_finite() {
            let grad = -2.0 / (1.0 + d2);
            let ok = (0..dims).all(|a| {
                let da = coords[bu + a].to_f64() - coords[bv + a].to_f64();
                let sa = (lr * grad * da).clamp(-4.0, 4.0);
                sa.is_finite()
                    && (coords[bu + a] + S::from_f64(sa)).is_finite()
                    && (coords[bv + a] - S::from_f64(sa)).is_finite()
            });
            if ok {
                for a in 0..dims {
                    let da = coords[bu + a].to_f64() - coords[bv + a].to_f64();
                    let sa = S::from_f64((lr * grad * da).clamp(-4.0, 4.0));
                    coords[bu + a] += sa;
                    coords[bv + a] -= sa;
                }
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
                let bk = k * dims;
                let rd2 = nd_dist_sq_f64(coords, bu, bk, dims);
                if rd2.is_finite() && rd2 > 0.0 {
                    let grad = (2.0 * gamma) / ((rd2 + 1e-6) * (1.0 + rd2));
                    let ok = (0..dims).all(|a| {
                        let da = coords[bu + a].to_f64() - coords[bk + a].to_f64();
                        let sa = (lr * grad * da).clamp(-4.0, 4.0);
                        sa.is_finite() && (coords[bu + a] + S::from_f64(sa)).is_finite()
                    });
                    if ok {
                        for a in 0..dims {
                            let da = coords[bu + a].to_f64() - coords[bk + a].to_f64();
                            let sa = (lr * grad * da).clamp(-4.0, 4.0);
                            coords[bu + a] += S::from_f64(sa);
                        }
                    }
                }
            }
        }
    }
}
