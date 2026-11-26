use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use super::config::SgdScheduleConfig;
use super::graph_sampling::sample_edges_and_random_pairs;
use super::pairs::{
    DirectedPairAccumulator, MAX_PACKED_NODES, PackedGenPair, PackedSymPair, PairStorage,
    resolve_directed_pair_params, validate_graph_weights, validate_pair_endpoints,
    validate_schedule_config,
};
use super::solver::SgdSolver;

impl<S: Scalar> SgdSolver<S> {
    /// Constructs an [`SgdSolver`] from explicit flat SoA two-sided spring arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pairs(
        num_nodes: usize,
        src: Vec<u32>,
        tgt: Vec<u32>,
        dij: Vec<S>,
        dji: Vec<S>,
        wij: Vec<S>,
        wji: Vec<S>,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        Self::from_pairs_with_hinges(
            num_nodes,
            src,
            tgt,
            dij,
            dji,
            wij,
            wji,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            schedule,
            seed,
        )
    }

    /// Constructs an [`SgdSolver`] combining flat SoA two-sided spring pairs and
    /// one-sided hinge repulsion pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pairs_with_hinges(
        num_nodes: usize,
        spring_src: Vec<u32>,
        spring_tgt: Vec<u32>,
        spring_dij: Vec<S>,
        spring_dji: Vec<S>,
        spring_wij: Vec<S>,
        spring_wji: Vec<S>,
        hinge_src: Vec<u32>,
        hinge_tgt: Vec<u32>,
        hinge_dij: Vec<S>,
        hinge_dji: Vec<S>,
        hinge_wij: Vec<S>,
        hinge_wji: Vec<S>,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;

        let validate_soa_slice = |src: &[u32],
                                  tgt: &[u32],
                                  dij: &[S],
                                  dji: &[S],
                                  wij: &[S],
                                  wji: &[S],
                                  ctx: &'static str|
         -> CoreResult<bool> {
            let len = src.len();
            for actual in [tgt.len(), dij.len(), dji.len(), wij.len(), wji.len()] {
                if actual != len {
                    return Err(CoreError::dimension_mismatch(len, actual, ctx));
                }
            }
            let mut is_sym = true;
            for idx in 0..len {
                validate_pair_endpoints(src[idx] as usize, tgt[idx] as usize, num_nodes, idx, ctx)?;
                for d in [dij[idx], dji[idx]] {
                    if !d.is_finite() {
                        return Err(CoreError::non_finite(ctx));
                    }
                    if d <= S::ZERO {
                        return Err(CoreError::invalid_input(format!(
                            "{ctx}: target distances must be > 0 at index {idx}"
                        )));
                    }
                }
                for w in [wij[idx], wji[idx]] {
                    if !w.is_finite() {
                        return Err(CoreError::non_finite(ctx));
                    }
                    if w < S::ZERO {
                        return Err(CoreError::invalid_input(format!(
                            "{ctx}: weights must be >= 0 at index {idx}"
                        )));
                    }
                }
                if dij[idx] != dji[idx] || wij[idx] != wji[idx] {
                    is_sym = false;
                }
            }
            Ok(is_sym)
        };

        let sym_s = validate_soa_slice(
            &spring_src,
            &spring_tgt,
            &spring_dij,
            &spring_dji,
            &spring_wij,
            &spring_wji,
            "SgdSolver::from_pairs_with_hinges::springs",
        )?;
        let sym_h = validate_soa_slice(
            &hinge_src,
            &hinge_tgt,
            &hinge_dij,
            &hinge_dji,
            &hinge_wij,
            &hinge_wji,
            "SgdSolver::from_pairs_with_hinges::hinges",
        )?;

        let n_springs = spring_src.len();
        let n_hinges = hinge_src.len();
        let total = n_springs + n_hinges;

        if sym_s && sym_h {
            let mut pairs = Vec::with_capacity(total);
            for idx in 0..n_springs {
                pairs.push(PackedSymPair::new(
                    spring_src[idx],
                    spring_tgt[idx],
                    false,
                    spring_dij[idx],
                    spring_wij[idx],
                ));
            }
            for idx in 0..n_hinges {
                pairs.push(PackedSymPair::new(
                    hinge_src[idx],
                    hinge_tgt[idx],
                    true,
                    hinge_dij[idx],
                    hinge_wij[idx],
                ));
            }
            Self::from_pair_storage(
                num_nodes,
                PairStorage::Symmetric(pairs),
                n_springs,
                n_hinges,
                schedule_cfg,
                seed,
            )
        } else {
            let mut pairs = Vec::with_capacity(total);
            for idx in 0..n_springs {
                pairs.push(PackedGenPair::new(
                    spring_src[idx],
                    spring_tgt[idx],
                    false,
                    spring_dij[idx],
                    spring_dji[idx],
                    spring_wij[idx],
                    spring_wji[idx],
                ));
            }
            for idx in 0..n_hinges {
                pairs.push(PackedGenPair::new(
                    hinge_src[idx],
                    hinge_tgt[idx],
                    true,
                    hinge_dij[idx],
                    hinge_dji[idx],
                    hinge_wij[idx],
                    hinge_wji[idx],
                ));
            }
            Self::from_pair_storage(
                num_nodes,
                PairStorage::General(pairs),
                n_springs,
                n_hinges,
                schedule_cfg,
                seed,
            )
        }
    }

    /// Constructs a full all-pairs shortest-path SGD solver using [`BfsWorkspace`] for
    /// unweighted graphs and [`DijkstraWorkspace`] for weighted graphs.
    pub fn full<W: Scalar>(
        graph: &CsrGraph<W>,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        validate_graph_weights(graph, "SgdSolver::full")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let cap = n.saturating_mul(n.saturating_sub(1)) / 2;
        if !graph.is_directed() {
            let mut pairs = Vec::with_capacity(cap);
            let mut dist_row = vec![f64::INFINITY; n];
            let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
            let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
            for i in 0..(n - 1) {
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, i, &mut dist_row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, i, &mut dist_row)?;
                }
                for (j, &d) in dist_row.iter().enumerate().skip(i + 1) {
                    if d.is_finite() && d > 0.0 {
                        let d_s = S::from_f64(d);
                        let w_s = S::from_f64(1.0 / (d * d));
                        if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                            pairs.push(PackedSymPair::new(i as u32, j as u32, false, d_s, w_s));
                        }
                    }
                }
            }
            let n_springs = pairs.len();
            return Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed);
        }

        let total = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("full APSP distance matrix overflow"))?;
        let mut dists = vec![f64::INFINITY; total];
        let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
        let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
        for u in 0..n {
            let row = &mut dists[u * n..(u + 1) * n];
            if let Some(ws) = bfs.as_mut() {
                ws.bfs_from(graph, u, row)?;
            } else if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(graph, u, row)?;
            }
        }
        let mut acc = DirectedPairAccumulator::with_capacity(cap);
        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                if let Some((duv, dvu, wuv, wvu)) =
                    resolve_directed_pair_params(dists[i * n + j], dists[j * n + i])
                {
                    acc.push_f64(i as u32, j as u32, duv, dvu, wuv, wvu);
                }
            }
        }
        let storage = acc.into_storage();
        let n_springs = storage.len();
        Self::from_pair_storage(n, storage, n_springs, 0, schedule, seed)
    }

    /// Constructs a random-pair sparse SGD solver combining all 1-hop edges with up to
    /// `pairs_per_node` sampled node pairs per vertex using exact shortest-path distances.
    pub fn random_pair<W: Scalar>(
        graph: &CsrGraph<W>,
        pairs_per_node: usize,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        validate_graph_weights(graph, "SgdSolver::random_pair")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let raw_pairs = sample_edges_and_random_pairs(graph, pairs_per_node, seed);
        let m = raw_pairs.len();
        let mut dist_row = vec![f64::INFINITY; n];
        let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
        let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));

        if !graph.is_directed() {
            let mut pairs = Vec::with_capacity(m);
            let mut idx = 0usize;
            while idx < m {
                let u = raw_pairs[idx].0;
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, u as usize, &mut dist_row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, u as usize, &mut dist_row)?;
                }
                while idx < m && raw_pairs[idx].0 == u {
                    let v = raw_pairs[idx].1;
                    let d = dist_row[v as usize];
                    if d.is_finite() && d > 0.0 {
                        let d_s = S::from_f64(d);
                        let w_s = S::from_f64(1.0 / (d * d));
                        if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                            pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                        }
                    }
                    idx += 1;
                }
            }
            let n_springs = pairs.len();
            return Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed);
        }

        let mut d_uv_arr = vec![f64::INFINITY; m];
        let mut d_vu_arr = vec![f64::INFINITY; m];
        let mut queries: Vec<(u32, u32, usize, bool)> = Vec::with_capacity(m * 2);
        for (slot, &(u, v)) in raw_pairs.iter().enumerate() {
            queries.push((u, v, slot, true));
            queries.push((v, u, slot, false));
        }
        queries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut q_idx = 0usize;
        while q_idx < queries.len() {
            let s = queries[q_idx].0;
            if let Some(ws) = bfs.as_mut() {
                ws.bfs_from(graph, s as usize, &mut dist_row)?;
            } else if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(graph, s as usize, &mut dist_row)?;
            }
            while q_idx < queries.len() && queries[q_idx].0 == s {
                let (_, t, slot, is_fwd) = queries[q_idx];
                let d = dist_row[t as usize];
                if is_fwd {
                    d_uv_arr[slot] = d;
                } else {
                    d_vu_arr[slot] = d;
                }
                q_idx += 1;
            }
        }

        let mut acc = DirectedPairAccumulator::with_capacity(m);
        for (slot, &(u, v)) in raw_pairs.iter().enumerate() {
            if let Some((duv, dvu, wuv, wvu)) =
                resolve_directed_pair_params(d_uv_arr[slot], d_vu_arr[slot])
            {
                acc.push_f64(u, v, duv, dvu, wuv, wvu);
            }
        }
        let storage = acc.into_storage();
        let n_springs = storage.len();
        Self::from_pair_storage(n, storage, n_springs, 0, schedule, seed)
    }

    /// Constructs an Omega SGD solver using Euclidean distances in a precomputed row-major
    /// `(num_nodes, dims)` spectral embedding for all 1-hop edges and `k` random partners per node.
    pub fn omega<W: Scalar>(
        graph: &CsrGraph<W>,
        embedding_flat: &[f64],
        dims: usize,
        k: usize,
        min_dist: f64,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        if dims == 0 {
            return Err(CoreError::invalid_input("omega dims must be >= 1"));
        }
        if !min_dist.is_finite() {
            return Err(CoreError::non_finite("SgdSolver::omega::min_dist"));
        }
        if min_dist <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "omega min_dist must be > 0, got {min_dist}"
            )));
        }

        let n = graph.num_nodes();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("omega embedding size overflow"))?;
        if embedding_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                embedding_flat.len(),
                "SgdSolver::omega::embedding_flat",
            ));
        }
        if embedding_flat.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("SgdSolver::omega::embedding_flat"));
        }
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let raw_pairs = sample_edges_and_random_pairs(graph, k, seed);
        let mut pairs = Vec::with_capacity(raw_pairs.len());
        for &(u, v) in &raw_pairs {
            let row_u = &embedding_flat[(u as usize) * dims..((u as usize) + 1) * dims];
            let row_v = &embedding_flat[(v as usize) * dims..((v as usize) + 1) * dims];
            let mut sq_sum = 0.0_f64;
            for (&a, &b) in row_u.iter().zip(row_v.iter()) {
                let diff = a - b;
                sq_sum += diff * diff;
            }
            let d = sq_sum.sqrt().max(min_dist);
            let d_s = S::from_f64(d);
            let w_s = S::from_f64(1.0 / (d * d));
            if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
            }
        }

        let n_springs = pairs.len();
        Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed)
    }
}
