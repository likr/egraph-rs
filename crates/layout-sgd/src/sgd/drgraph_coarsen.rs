use egraph_clustering::coarsen_graph;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, SolverStatus, VisitStamp,
};
use egraph_linalg::DijkstraWorkspace;
use egraph_neighborhood::graph_row_normalized_probabilities;

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::graph_sampling::{
    AliasTable, build_positive_canonical_csr, build_symmetric_csr_from_sorted_pairs,
};
use super::pairs::{MAX_PACKED_NODES, validate_graph_weights};
use super::variant_config::DrGraphConfig;

#[derive(Debug, Clone)]
pub(super) struct DrGraphLevel {
    pub(super) num_nodes: usize,
    pub(super) coarse_of_finer: Option<Vec<u32>>,
    pub(super) cluster_counts_of_finer: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub(super) struct DrGraphState {
    pub(super) config: DrGraphConfig,
    pub(super) pos_u: Vec<u32>,
    pub(super) pos_v: Vec<u32>,
    pub(super) edge_alias: Option<AliasTable>,
    pub(super) levels: Vec<DrGraphLevel>,
    pub(super) fine_to_level: Vec<Vec<u32>>,
    pub(super) level_iters: Vec<usize>,
    pub(super) current_level: usize,
    pub(super) level_iter: usize,
    pub(super) initialized_coords: bool,
    pub(super) active_x: Vec<f64>,
    pub(super) active_y: Vec<f64>,
    pub(super) scratch_x: Vec<f64>,
    pub(super) scratch_y: Vec<f64>,
    pub(super) scratch_cnt: Vec<f64>,
}

pub(super) struct BuiltDrGraph {
    pub(super) num_nodes: usize,
    pub(super) num_springs: usize,
    pub(super) state: DrGraphState,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

pub(super) fn compute_drgraph_similarities(
    sym_graph: &CsrGraph<f64>,
    k_order: usize,
) -> CoreResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
    let n = sym_graph.num_nodes();
    if n <= 1 || sym_graph.nnz() == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    if k_order == 1 && !sym_graph.is_weighted() {
        let probs = graph_row_normalized_probabilities(sym_graph)?;
        let mut pos_u = Vec::with_capacity(sym_graph.num_edges());
        let mut pos_v = Vec::with_capacity(sym_graph.num_edges());
        let mut pos_w = Vec::with_capacity(sym_graph.num_edges());
        for u in 0..n {
            let u_u32 = u as u32;
            let (nbrs, vals) = probs.row(u);
            for (&v, &p_uv) in nbrs.iter().zip(vals.iter()) {
                if u_u32 < v && p_uv.is_finite() && p_uv > 0.0 {
                    pos_u.push(u_u32);
                    pos_v.push(v);
                    pos_w.push(p_uv);
                }
            }
        }
        return Ok((pos_u, pos_v, pos_w));
    }

    let mut raw_pairs: Vec<(u32, u32, f64)> = Vec::new();
    let mut visited = VisitStamp::new(n);
    let mut queue: Vec<(u32, usize)> = Vec::new();
    let mut nbr_dists: Vec<(u32, f64)> = Vec::new();
    let mut dist_scratch: Vec<f64> = Vec::new();
    let mut dijk_dists = vec![f64::INFINITY; n];
    let mut dijk = (sym_graph.is_weighted() && k_order >= 2).then(|| DijkstraWorkspace::new(n));
    let weights_opt = sym_graph.weights();

    for i in 0..n {
        nbr_dists.clear();
        if k_order == 1 {
            let start = sym_graph.indptr()[i] as usize;
            let end = sym_graph.indptr()[i + 1] as usize;
            for pos in start..end {
                let j = sym_graph.indices()[pos];
                if (j as usize) != i {
                    let d = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    if d.is_finite() && d > 0.0 {
                        nbr_dists.push((j, d));
                    }
                }
            }
        } else {
            visited.clear();
            visited.mark(i);
            queue.clear();
            queue.push((i as u32, 0));
            let mut head = 0usize;
            while head < queue.len() {
                let (u_u32, depth) = queue[head];
                head += 1;
                if depth >= k_order {
                    continue;
                }
                for &v_u32 in sym_graph.neighbors(u_u32 as usize) {
                    if visited.visit(v_u32 as usize) {
                        let next_depth = depth + 1;
                        queue.push((v_u32, next_depth));
                        if !sym_graph.is_weighted() {
                            nbr_dists.push((v_u32, next_depth as f64));
                        }
                    }
                }
            }
            if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(sym_graph, i, &mut dijk_dists)?;
                for &(v_u32, depth) in &queue {
                    if depth >= 1 {
                        let d = dijk_dists[v_u32 as usize];
                        if d.is_finite() && d > 0.0 {
                            nbr_dists.push((v_u32, d));
                        }
                    }
                }
            }
        }

        if nbr_dists.is_empty() {
            continue;
        }
        dist_scratch.clear();
        for &(_, d) in &nbr_dists {
            dist_scratch.push(d);
        }
        dist_scratch.sort_unstable_by(f64::total_cmp);
        let sigma = dist_scratch[dist_scratch.len() / 2].max(1e-6);
        let inv_2s2 = 0.5 / (sigma * sigma);
        let min_d = dist_scratch[0];
        let min_d2 = min_d * min_d;

        let mut sum_w = 0.0_f64;
        for &(_, d) in &nbr_dists {
            sum_w += (-(d * d - min_d2) * inv_2s2).exp();
        }
        if !sum_w.is_finite() || sum_w <= 0.0 {
            continue;
        }
        let inv_sum = 1.0 / sum_w;
        let i_u32 = i as u32;
        for &(j_u32, d) in &nbr_dists {
            let ns_ji = (-(d * d - min_d2) * inv_2s2).exp() * inv_sum;
            let (a, b) = if i_u32 < j_u32 {
                (i_u32, j_u32)
            } else {
                (j_u32, i_u32)
            };
            raw_pairs.push((a, b, ns_ji));
        }
    }

    raw_pairs.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let inv_2n = 0.5 / (n as f64);
    let mut pos_u = Vec::with_capacity(raw_pairs.len());
    let mut pos_v = Vec::with_capacity(raw_pairs.len());
    let mut pos_w = Vec::with_capacity(raw_pairs.len());
    let mut idx = 0usize;
    while idx < raw_pairs.len() {
        let (u, v, _) = raw_pairs[idx];
        let mut sum_p = 0.0_f64;
        while idx < raw_pairs.len() && raw_pairs[idx].0 == u && raw_pairs[idx].1 == v {
            sum_p += raw_pairs[idx].2;
            idx += 1;
        }
        let ns_uv = sum_p * inv_2n;
        if ns_uv.is_finite() && ns_uv > 0.0 {
            pos_u.push(u);
            pos_v.push(v);
            pos_w.push(ns_uv);
        }
    }
    Ok((pos_u, pos_v, pos_w))
}

pub(super) fn build_drgraph_state<W: Scalar>(
    graph: &CsrGraph<W>,
    config: DrGraphConfig,
) -> CoreResult<BuiltDrGraph> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::drgraph")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }

    let schedule_cfg = SgdScheduleConfig {
        t_max: config.iterations,
        eps: 0.01,
        kind: SgdScheduleKind::Linear,
        step_scale: config.initial_lr,
    };
    let status = if config.iterations == 0 {
        SolverStatus::Completed
    } else {
        SolverStatus::Initialized
    };

    let mut rng = DeterministicRng::from_seed_u64(config.seed);

    let trivial_state = || DrGraphState {
        config,
        pos_u: Vec::new(),
        pos_v: Vec::new(),
        edge_alias: None,
        levels: vec![DrGraphLevel {
            num_nodes: n,
            coarse_of_finer: None,
            cluster_counts_of_finer: None,
        }],
        fine_to_level: vec![Vec::new()],
        level_iters: vec![config.iterations],
        current_level: 0,
        level_iter: 0,
        initialized_coords: false,
        active_x: vec![0.0; n],
        active_y: vec![0.0; n],
        scratch_x: vec![0.0; n],
        scratch_y: vec![0.0; n],
        scratch_cnt: vec![0.0; n],
    };

    if n <= 1 || graph.nnz() == 0 {
        return Ok(BuiltDrGraph {
            num_nodes: n,
            num_springs: 0,
            state: trivial_state(),
            schedule_cfg,
            status,
            rng,
        });
    }

    let (sym_graph, positive_edges) = build_positive_canonical_csr(graph)?;
    if positive_edges.is_empty() {
        return Ok(BuiltDrGraph {
            num_nodes: n,
            num_springs: 0,
            state: trivial_state(),
            schedule_cfg,
            status,
            rng,
        });
    }

    let (pos_u, pos_v, pos_weights) = compute_drgraph_similarities(&sym_graph, config.k_order)?;
    let edge_alias = AliasTable::from_weights(&pos_weights);
    let m = pos_u.len();

    // Build random-merge multilevel coarsening hierarchy stopping when
    // `|V^{l+1}| > coarsen_stop_ratio * |V^l|` (or trivial collapse <= 1).
    let unweighted_base = {
        let mut u1 = Vec::with_capacity(positive_edges.len());
        let mut v1 = Vec::with_capacity(positive_edges.len());
        for &(u, v, _) in &positive_edges {
            u1.push(u);
            v1.push(v);
        }
        let (indptr, indices) = build_symmetric_csr_from_sorted_pairs(n, &u1, &v1);
        CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, false)?
    };

    let mut level_graphs = Vec::new();
    level_graphs.push(unweighted_base);
    let mut levels = vec![DrGraphLevel {
        num_nodes: n,
        coarse_of_finer: None,
        cluster_counts_of_finer: None,
    }];
    let mut perm: Vec<u32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();

    while levels.len() < 64 {
        let cur_idx = level_graphs.len() - 1;
        let cur_n = level_graphs[cur_idx].num_nodes();
        if cur_n <= 2 {
            break;
        }
        perm.clear();
        perm.extend(0..cur_n as u32);
        rng.shuffle(&mut perm);

        labels.clear();
        labels.resize(cur_n, u32::MAX);
        let mut next_cid = 0u32;
        {
            let cur_g = &level_graphs[cur_idx];
            for &u_u32 in &perm {
                let u = u_u32 as usize;
                if labels[u] != u32::MAX {
                    continue;
                }
                labels[u] = next_cid;
                for &v_u32 in cur_g.neighbors(u) {
                    let v = v_u32 as usize;
                    if labels[v] == u32::MAX {
                        labels[v] = next_cid;
                    }
                }
                next_cid += 1;
            }
        }

        let coarse_n = next_cid as usize;
        if coarse_n <= 1
            || coarse_n >= cur_n
            || (coarse_n as f64) > config.coarsen_stop_ratio * (cur_n as f64)
        {
            break;
        }
        let coarsened = coarsen_graph(&level_graphs[cur_idx], &labels)?;
        if coarsened.num_clusters <= 1
            || coarsened.num_clusters >= cur_n
            || (coarsened.num_clusters as f64) > config.coarsen_stop_ratio * (cur_n as f64)
        {
            break;
        }
        levels[cur_idx].coarse_of_finer = Some(coarsened.node_to_coarse);
        levels[cur_idx].cluster_counts_of_finer = Some(coarsened.coarse_sizes);
        levels.push(DrGraphLevel {
            num_nodes: coarsened.num_clusters,
            coarse_of_finer: None,
            cluster_counts_of_finer: None,
        });
        level_graphs.push(coarsened.graph);
    }

    let num_levels = levels.len();
    let mut fine_to_level = Vec::with_capacity(num_levels);
    fine_to_level.push(Vec::new());
    for l in 1..num_levels {
        let mut map = Vec::with_capacity(n);
        let step_map = levels[l - 1]
            .coarse_of_finer
            .as_ref()
            .ok_or_else(|| CoreError::invalid_csr("missing coarse_of_finer mapping"))?;
        if l == 1 {
            map.extend_from_slice(step_map);
        } else {
            let prev = &fine_to_level[l - 1];
            for &idx in prev {
                map.push(step_map[idx as usize]);
            }
        }
        fine_to_level.push(map);
    }

    let mut level_iters = vec![0usize; num_levels];
    if config.iterations > 0 {
        let base = config.iterations / num_levels;
        let rem = config.iterations % num_levels;
        for (l, slot) in level_iters.iter_mut().enumerate() {
            *slot = base + usize::from(l < rem);
        }
    }
    let start_level = level_iters.iter().rposition(|&c| c > 0).unwrap_or(0);

    Ok(BuiltDrGraph {
        num_nodes: n,
        num_springs: m,
        state: DrGraphState {
            config,
            pos_u,
            pos_v,
            edge_alias,
            levels,
            fine_to_level,
            level_iters,
            current_level: start_level,
            level_iter: 0,
            initialized_coords: false,
            active_x: vec![0.0; n],
            active_y: vec![0.0; n],
            scratch_x: vec![0.0; n],
            scratch_y: vec![0.0; n],
            scratch_cnt: vec![0.0; n],
        },
        schedule_cfg,
        status,
        rng,
    })
}
