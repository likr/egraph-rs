use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, SolverStatus,
};

use super::config::{SgdScheduleConfig, SgdScheduleKind, SnapTFdpConfig};
use super::pairs::{
    BLOCK_SHUFFLE_THRESHOLD, BLOCK_SIZE, MAX_PACKED_NODES, init_pair_blocking,
    validate_graph_weights,
};

#[derive(Debug, Clone)]
pub(super) struct SnapTFdpState {
    pub(super) config: SnapTFdpConfig,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edge_weights: Option<Vec<f32>>,
    pub(super) deg_scales: Option<Vec<f32>>,
    pub(super) edge_order: Vec<u32>,
}

pub(super) struct BuiltSnapTFdp {
    pub(super) num_nodes: usize,
    pub(super) num_springs: usize,
    pub(super) block_order: Vec<u32>,
    pub(super) state: SnapTFdpState,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

pub(super) fn build_snap_t_fdp_state<W: Scalar>(
    graph: &CsrGraph<W>,
    config: SnapTFdpConfig,
) -> CoreResult<BuiltSnapTFdp> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::snap_t_fdp")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }

    let mut edges_u: Vec<u32>;
    let mut edges_v: Vec<u32>;
    let mut edge_weights: Option<Vec<f32>>;

    if n <= 1 || graph.nnz() == 0 {
        edges_u = Vec::new();
        edges_v = Vec::new();
        edge_weights = None;
    } else if !graph.is_directed() {
        // Stream directly into flat u32 arrays without temporary tuple vectors to respect
        // strict 1M (<512 MB) and 4M (<1.6 GiB) Peak RSS budgets.
        let m_est = graph.num_edges();
        edges_u = Vec::with_capacity(m_est);
        edges_v = Vec::with_capacity(m_est);
        edge_weights = graph.is_weighted().then(|| Vec::with_capacity(m_est));
        let weights_opt = graph.weights();
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos];
                if u_u32 < v {
                    edges_u.push(u_u32);
                    edges_v.push(v);
                    if let (Some(ws_out), Some(ws_in)) = (edge_weights.as_mut(), weights_opt) {
                        ws_out.push(ws_in[pos].to_f32());
                    }
                }
            }
        }
    } else if let Some(ws_in) = graph.weights() {
        let mut raw: Vec<(u32, u32, f32)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for (&v, &w) in graph.indices()[start..end]
                .iter()
                .zip(ws_in[start..end].iter())
            {
                if u_u32 == v {
                    continue;
                }
                let (a, b) = if u_u32 < v { (u_u32, v) } else { (v, u_u32) };
                raw.push((a, b, w.to_f32()));
            }
        }
        raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        edges_u = Vec::with_capacity(raw.len());
        edges_v = Vec::with_capacity(raw.len());
        let mut ws_out: Vec<f32> = Vec::with_capacity(raw.len());
        for (a, b, w) in raw {
            if edges_u.last().copied() == Some(a) && edges_v.last().copied() == Some(b) {
                if let Some(last_w) = ws_out.last_mut() {
                    *last_w = (*last_w).max(w);
                }
            } else {
                edges_u.push(a);
                edges_v.push(b);
                ws_out.push(w);
            }
        }
        edge_weights = Some(ws_out);
    } else {
        let mut raw: Vec<(u32, u32)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u_u32 = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for &v in &graph.indices()[start..end] {
                if u_u32 != v {
                    raw.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                }
            }
        }
        raw.sort_unstable();
        raw.dedup();
        edges_u = Vec::with_capacity(raw.len());
        edges_v = Vec::with_capacity(raw.len());
        for (a, b) in raw {
            edges_u.push(a);
            edges_v.push(b);
        }
        edge_weights = None;
    }

    let m = edges_u.len();
    if u32::try_from(m).is_err() {
        return Err(CoreError::invalid_csr(
            "SNAP-tFDP undirected edge count exceeds u32::MAX",
        ));
    }

    let mut edge_order: Vec<u32> = (0..m as u32).collect();
    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let block_order = init_pair_blocking(&mut edge_order, &mut rng);

    // Allocate optional 4N-byte degree attenuation factors only when degree_beta > 0.
    let deg_scales = if config.degree_beta > 0.0 && n > 1 && !edges_u.is_empty() {
        let mut degs = vec![0u32; n];
        for (&u, &v) in edges_u.iter().zip(edges_v.iter()) {
            degs[u as usize] += 1;
            degs[v as usize] += 1;
        }
        let beta = config.degree_beta;
        Some(
            degs.into_iter()
                .map(|d| (d.max(1) as f64).powf(-beta) as f32)
                .collect(),
        )
    } else {
        None
    };

    let schedule_cfg = SgdScheduleConfig {
        t_max: config.iterations,
        eps: 0.01,
        kind: SgdScheduleKind::Linear,
        step_scale: config.learning_rate,
    };
    let status = if config.iterations == 0 {
        SolverStatus::Completed
    } else {
        SolverStatus::Initialized
    };

    Ok(BuiltSnapTFdp {
        num_nodes: n,
        num_springs: m,
        block_order,
        state: SnapTFdpState {
            config,
            edges_u,
            edges_v,
            edge_weights,
            deg_scales,
            edge_order,
        },
        schedule_cfg,
        status,
        rng,
    })
}

pub(super) fn step_snap_t_fdp_state<S: Scalar>(
    st: &mut SnapTFdpState,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut Layout2d<S>,
) {
    let total = st.config.iterations;
    if num_nodes > 1 && !st.edges_u.is_empty() && total > 0 && iter < total {
        let eta_f64 = st.config.learning_rate * (1.0 - (iter as f64) / (total as f64));
        if eta_f64.is_finite() && eta_f64 > 0.0 {
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
            let (xs, ys) = layout.xy_mut();
            let len = st.edge_order.len();

            if len <= BLOCK_SHUFFLE_THRESHOLD {
                rng.shuffle(&mut st.edge_order);
                apply_snap_edge_slice(
                    xs,
                    ys,
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
                );
            } else {
                rng.shuffle(block_order);
                for &b in block_order.iter() {
                    let start = (b as usize) * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(len);
                    let chunk = &mut st.edge_order[start..end];
                    rng.shuffle(chunk);
                    apply_snap_edge_slice(
                        xs,
                        ys,
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
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_snap_edge_slice<S: Scalar>(
    xs: &mut [S],
    ys: &mut [S],
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
) {
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

        // Bounded t-FDP attraction along edge (u, v) with optional degree attenuation.
        if w_e > S::ZERO && base_attr > S::ZERO && deg_factor > S::ZERO {
            let dx = xs[u] - xs[v];
            let dy = ys[u] - ys[v];
            let d2 = dx * dx + dy * dy;
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
                let sx = dx * coeff;
                let sy = dy * coeff;
                if sx.is_finite() && sy.is_finite() {
                    xs[u] -= sx;
                    ys[u] -= sy;
                    xs[v] += sx;
                    ys[v] += sy;
                }
            }
        }

        // Uniform random Student-t negative repulsion incident to u/v.
        if base_rep > S::ZERO {
            for neg_idx in 0..negatives_per_edge {
                let anchor = if ((pos + neg_idx) & 1) == 0 { u } else { v };
                let w = rng.gen_range_usize(num_nodes);
                if w == anchor {
                    continue;
                }
                let rx = xs[anchor] - xs[w];
                let ry = ys[anchor] - ys[w];
                let rd2 = rx * rx + ry * ry;
                if rd2.is_finite() {
                    let denom = (S::ONE + rd2) * (S::ONE + rd2);
                    if denom.is_finite() && denom > S::ZERO {
                        let rep = (base_rep / denom).min(max_rep);
                        if rep.is_finite() && rep > S::ZERO {
                            let sx = rx * rep;
                            let sy = ry * rep;
                            if sx.is_finite() && sy.is_finite() {
                                xs[anchor] += sx;
                                ys[anchor] += sy;
                                xs[w] -= sx;
                                ys[w] -= sy;
                            }
                        }
                    }
                }
            }
        }
    }
}
