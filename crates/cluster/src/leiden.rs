//! Pure-Rust Leiden community detection (Traag, Waltman, & van Eck, 2019, Algorithm A.2)
//! supporting both Modularity and Constant Potts Model (CPM) objectives over flat CSR
//! super-graphs and `VisitStamp` accumulators (zero `HashMap` in inner loops or aggregation).

use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};

/// Objective function optimized by [`leiden`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LeidenObjective {
    /// Standard Modularity $Q_\gamma$ (Traag et al., 2019, Eq. (1) / Eq. (A.5)).
    #[default]
    Modularity,
    /// Constant Potts Model (CPM) (Traag et al., 2019, Eq. (2) / Eq. (A.4)).
    Cpm,
}

/// Configuration for [`leiden`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeidenConfig {
    /// Quality objective (`Modularity` or `Cpm`).
    pub objective: LeidenObjective,
    /// Resolution parameter $\gamma \ge 0.0$ (`1.0` for standard modularity).
    pub resolution: f64,
    /// Temperature/randomness parameter $\theta > 0.0$ controlling stochastic refinement merges.
    pub randomness: f64,
    /// Maximum number of multi-level Leiden passes (local moving + refinement + aggregation).
    pub max_passes: usize,
    /// Minimum quality gain required to commit a pass and continue.
    pub min_quality_gain: f64,
    /// Deterministic PRNG seed controlling queue order and refinement sampling.
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            objective: LeidenObjective::Modularity,
            resolution: 1.0,
            randomness: 0.01,
            max_passes: 100,
            min_quality_gain: 1e-7,
            seed: 42,
        }
    }
}

/// Output of [`leiden`].
#[derive(Debug, Clone, PartialEq)]
pub struct LeidenResult {
    /// Canonical community assignment in `0..num_communities` in first-appearance vertex order.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty communities detected.
    pub num_communities: usize,
    /// Objective quality value: normalized $Q_\gamma$ for `Modularity`, or unnormalized
    /// $\sum_C [E(C, C) - \gamma \binom{\|C\|}{2}]$ (Traag Eq. (A.4)) for `Cpm`.
    pub quality: f64,
    /// Standard normalized modularity $Q_\gamma$ evaluated with `config.resolution` (`0.0` if `2m == 0`).
    pub modularity: f64,
    /// Number of multi-level passes committed.
    pub passes: usize,
}

/// Compact internal super-graph supporting weighted self-loops and super-node sizes.
#[derive(Clone)]
struct LeidenLevelGraph {
    num_nodes: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f64>,
    loop_weights: Vec<f64>,
    degrees: Vec<f64>,
    node_sizes: Vec<f64>,
    total_weight_2m: f64,
}

impl LeidenLevelGraph {
    fn from_csr<W: Scalar>(graph: &CsrGraph<W>, objective: LeidenObjective) -> CoreResult<Self> {
        let n = graph.num_nodes();
        let mut degrees = vec![0.0_f64; n];
        let mut weights = Vec::with_capacity(graph.nnz());
        let mut total_weight_2m = 0.0_f64;

        if let Some(ws) = graph.weights() {
            for (u, deg_u) in degrees.iter_mut().enumerate() {
                let start = graph.indptr()[u] as usize;
                let end = graph.indptr()[u + 1] as usize;
                let mut sum = 0.0_f64;
                for &w in &ws[start..end] {
                    let wf = w.to_f64();
                    if !wf.is_finite() {
                        return Err(CoreError::non_finite("leiden::graph::weights"));
                    }
                    if wf < 0.0 {
                        return Err(CoreError::invalid_input(
                            "leiden edge weights must be non-negative",
                        ));
                    }
                    weights.push(wf);
                    sum += wf;
                }
                if !sum.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "edge weight sum overflows f64",
                    ));
                }
                *deg_u = sum;
                total_weight_2m += sum;
                if !total_weight_2m.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "edge weight sum overflows f64",
                    ));
                }
            }
        } else {
            for (u, deg_u) in degrees.iter_mut().enumerate() {
                let d = graph.degree(u) as f64;
                *deg_u = d;
                total_weight_2m += d;
            }
            if !total_weight_2m.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "edge weight sum overflows f64",
                ));
            }
            weights.resize(graph.nnz(), 1.0);
        }

        if total_weight_2m > 0.0
            && objective == LeidenObjective::Modularity
            && !(1.0 / total_weight_2m).is_finite()
        {
            return Err(CoreError::numerical_breakdown(
                "total edge weight underflows f64 precision",
            ));
        }

        let node_sizes = match objective {
            LeidenObjective::Modularity => degrees.clone(),
            LeidenObjective::Cpm => vec![1.0_f64; n],
        };

        Ok(Self {
            num_nodes: n,
            indptr: graph.indptr().to_vec(),
            indices: graph.indices().to_vec(),
            weights,
            loop_weights: vec![0.0_f64; n],
            degrees,
            node_sizes,
            total_weight_2m,
        })
    }

    fn evaluate_quality(
        &self,
        comm: &[u32],
        num_comms: usize,
        objective: LeidenObjective,
        resolution: f64,
    ) -> f64 {
        let mut sigma_in = vec![0.0_f64; num_comms];
        let mut comm_size = vec![0.0_f64; num_comms];

        for u in 0..self.num_nodes {
            let cu = comm[u] as usize;
            sigma_in[cu] += self.loop_weights[u];
            comm_size[cu] += self.node_sizes[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                if comm[v] as usize == cu {
                    sigma_in[cu] += self.weights[pos];
                }
            }
        }

        match objective {
            LeidenObjective::Modularity => {
                if self.total_weight_2m <= 0.0 {
                    return 0.0;
                }
                let mut q = 0.0_f64;
                for c in 0..num_comms {
                    let kc = comm_size[c];
                    if kc > 0.0 || sigma_in[c] > 0.0 {
                        let frac = kc / self.total_weight_2m;
                        q += (sigma_in[c] / self.total_weight_2m) - resolution * frac * frac;
                    }
                }
                q
            }
            LeidenObjective::Cpm => {
                let mut h = 0.0_f64;
                for c in 0..num_comms {
                    let sz = comm_size[c];
                    h += 0.5 * sigma_in[c] - resolution * (sz * (sz - 1.0) * 0.5);
                }
                h
            }
        }
    }

    /// Aggregates refined communities `0..num_new` into the next-level super-graph without `HashMap`.
    fn aggregate(&self, refined_compacted: &[u32], num_new: usize) -> Self {
        let mut new_loops = vec![0.0_f64; num_new];
        let mut new_degrees = vec![0.0_f64; num_new];
        let mut new_sizes = vec![0.0_f64; num_new];
        let mut counts = vec![0usize; num_new];

        for u in 0..self.num_nodes {
            let cu = refined_compacted[u] as usize;
            new_loops[cu] += self.loop_weights[u];
            new_degrees[cu] += self.degrees[u];
            new_sizes[cu] += self.node_sizes[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = refined_compacted[v] as usize;
                if cu == cv {
                    new_loops[cu] += self.weights[pos];
                } else {
                    counts[cu] += 1;
                }
            }
        }

        let mut row_buckets: Vec<Vec<(u32, f64)>> =
            counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        for u in 0..self.num_nodes {
            let cu = refined_compacted[u] as usize;
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = refined_compacted[v];
                if cu != cv as usize {
                    row_buckets[cu].push((cv, self.weights[pos]));
                }
            }
        }

        let mut new_indptr = Vec::with_capacity(num_new + 1);
        new_indptr.push(0u32);
        let mut new_indices = Vec::new();
        let mut new_weights = Vec::new();

        for bucket in &mut row_buckets {
            bucket.sort_unstable_by_key(|&(cv, _)| cv);
            let row_start = new_indices.len();
            for &(cv, w) in bucket.iter() {
                if new_indices.len() > row_start && new_indices[new_indices.len() - 1] == cv {
                    let last_idx = new_weights.len() - 1;
                    new_weights[last_idx] += w;
                } else {
                    new_indices.push(cv);
                    new_weights.push(w);
                }
            }
            new_indptr.push(new_indices.len() as u32);
        }

        Self {
            num_nodes: num_new,
            indptr: new_indptr,
            indices: new_indices,
            weights: new_weights,
            loop_weights: new_loops,
            degrees: new_degrees,
            node_sizes: new_sizes,
            total_weight_2m: self.total_weight_2m,
        }
    }
}

/// Renumbers labels in canonical `0..c` order of first appearance by vertex index.
fn canonicalize_labels(labels: &mut [u32], max_id_bound: usize) -> usize {
    let mut remap = vec![u32::MAX; max_id_bound.max(1)];
    let mut next_id = 0u32;
    for label in labels.iter_mut() {
        let idx = *label as usize;
        if remap[idx] == u32::MAX {
            remap[idx] = next_id;
            next_id += 1;
        }
        *label = remap[idx];
    }
    next_id as usize
}

/// Queue-based `MoveNodesFast` phase (Traag et al., 2019, Algorithm A.2, lines 13-25).
///
/// Considers neighbor communities and splitting to an empty community `∅` when `u` is not a
/// singleton, enforcing a relative ULP strictness guard to prevent floating-point oscillation.
fn move_nodes_fast(
    level: &LeidenLevelGraph,
    comm: &mut [u32],
    gamma_eff: f64,
    rng: &mut DeterministicRng,
) -> bool {
    let nl = level.num_nodes;
    let mut comm_size = vec![0.0_f64; nl];
    let mut comm_members = vec![0u32; nl];
    for u in 0..nl {
        let c = comm[u] as usize;
        comm_size[c] += level.node_sizes[u];
        comm_members[c] += 1;
    }

    let mut free_comm_ids = Vec::with_capacity(nl);
    for c in (0..nl).rev() {
        if comm_members[c] == 0 {
            free_comm_ids.push(c as u32);
        }
    }

    let mut order: Vec<u32> = (0..nl as u32).collect();
    rng.shuffle(&mut order);
    let mut queue: VecDeque<u32> = VecDeque::from(order);
    let mut in_queue = vec![true; nl];

    let mut comm_stamp = VisitStamp::new(nl);
    let mut comm_weights = vec![0.0_f64; nl];
    let mut touched_comms: Vec<u32> = Vec::new();
    let mut any_moved = false;

    while let Some(u_u32) = queue.pop_front() {
        let u = u_u32 as usize;
        in_queue[u] = false;

        let c_old = comm[u];
        let c_old_idx = c_old as usize;
        let rem_old_members = comm_members[c_old_idx] - 1;

        comm_stamp.clear();
        touched_comms.clear();

        let start = level.indptr[u] as usize;
        let end = level.indptr[u + 1] as usize;
        for pos in start..end {
            let w = level.weights[pos];
            if w <= 0.0 {
                continue;
            }
            let v = level.indices[pos] as usize;
            let cv = comm[v];
            let cv_idx = cv as usize;
            if comm_stamp.visit(cv_idx) {
                touched_comms.push(cv);
            }
            comm_weights[cv_idx] += w;
        }

        if touched_comms.is_empty() && rem_old_members == 0 {
            continue;
        }

        touched_comms.sort_unstable();

        let size_u = level.node_sizes[u];
        let prev_old_size = comm_size[c_old_idx];
        let rem_old_size = if rem_old_members == 0 {
            0.0
        } else {
            (prev_old_size - size_u).max(0.0)
        };
        let e_u_old = if comm_stamp.is_visited(c_old_idx) {
            comm_weights[c_old_idx]
        } else {
            0.0
        };
        let old_term = e_u_old - gamma_eff * size_u * rem_old_size;

        let mut best_delta_h = 0.0_f64;
        let mut best_cand: Option<u32> = None;
        let mut move_to_empty = false;

        for &cand in &touched_comms {
            if cand == c_old {
                continue;
            }
            let cand_idx = cand as usize;
            let cand_term = comm_weights[cand_idx] - gamma_eff * size_u * comm_size[cand_idx];
            let delta_h = cand_term - old_term;
            let min_gain = 1e-12 * (old_term.abs() + cand_term.abs());
            if delta_h > min_gain
                && (best_cand.is_none() || delta_h > best_delta_h + 1e-12 * best_delta_h.abs())
            {
                best_delta_h = delta_h;
                best_cand = Some(cand);
            }
        }

        if rem_old_members > 0 {
            let delta_h_empty = -old_term;
            let min_gain = 1e-12 * old_term.abs();
            if delta_h_empty > min_gain
                && (best_cand.is_none()
                    || delta_h_empty > best_delta_h + 1e-12 * best_delta_h.abs())
            {
                best_cand = None;
                move_to_empty = true;
            }
        }

        for &cand in &touched_comms {
            comm_weights[cand as usize] = 0.0;
        }

        let chosen_target = if move_to_empty {
            free_comm_ids.pop()
        } else {
            best_cand
        };

        if let Some(target_c) = chosen_target {
            any_moved = true;
            comm_members[c_old_idx] = rem_old_members;
            comm_size[c_old_idx] = rem_old_size;
            if rem_old_members == 0 {
                free_comm_ids.push(c_old);
            }

            let target_idx = target_c as usize;
            comm[u] = target_c;
            comm_members[target_idx] += 1;
            comm_size[target_idx] += size_u;

            for pos in start..end {
                if level.weights[pos] <= 0.0 {
                    continue;
                }
                let v = level.indices[pos] as usize;
                if comm[v] != target_c && !in_queue[v] {
                    in_queue[v] = true;
                    queue.push_back(v as u32);
                }
            }
        }
    }

    any_moved
}

/// `RefinePartition` / `MergeNodesSubset` phase (Traag et al., 2019, Algorithm A.2, lines 26-43).
///
/// Refines each outer community `S` into internally well-connected sub-communities using
/// positive-edge filtering (`E(v, C) > 0.0`), $O(1)$ `ext_in_s` updates, deterministic sorted
/// candidate ordering, and log-sum-exp stabilized softmax sampling.
fn refine_partition(
    level: &LeidenLevelGraph,
    comm: &[u32],
    num_outer: usize,
    gamma_eff: f64,
    randomness: f64,
    rng: &mut DeterministicRng,
) -> (Vec<u32>, usize) {
    let nl = level.num_nodes;
    let mut sub_comm: Vec<u32> = (0..nl as u32).collect();
    let mut sub_size = level.node_sizes.clone();
    let mut sub_members = vec![1u32; nl];

    let mut outer_counts = vec![0usize; num_outer];
    for &c in comm {
        outer_counts[c as usize] += 1;
    }
    let mut outer_groups: Vec<Vec<u32>> = outer_counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    for (u, &c) in comm.iter().enumerate() {
        outer_groups[c as usize].push(u as u32);
    }

    let mut ext_in_s = vec![0.0_f64; nl];
    let mut r_nodes: Vec<u32> = Vec::new();
    let mut sub_stamp = VisitStamp::new(nl);
    let mut sub_edge_weights = vec![0.0_f64; nl];
    let mut touched_subs: Vec<u32> = Vec::new();
    let mut candidates: Vec<(u32, f64)> = Vec::new();

    for (s_id, members) in outer_groups.iter().enumerate() {
        if members.len() <= 1 {
            continue;
        }

        let mut total_s_size = 0.0_f64;
        for &u_u32 in members {
            let u = u_u32 as usize;
            total_s_size += level.node_sizes[u];
            let mut sum_s = 0.0_f64;
            let start = level.indptr[u] as usize;
            let end = level.indptr[u + 1] as usize;
            for pos in start..end {
                let w = level.weights[pos];
                if w <= 0.0 {
                    continue;
                }
                let v = level.indices[pos] as usize;
                if comm[v] as usize == s_id {
                    sum_s += w;
                }
            }
            ext_in_s[u] = sum_s;
        }

        r_nodes.clear();
        for &u_u32 in members {
            let u = u_u32 as usize;
            let size_u = level.node_sizes[u];
            let rhs = gamma_eff * size_u * (total_s_size - size_u).max(0.0);
            if ext_in_s[u] + 1e-12 * rhs.abs() >= rhs {
                r_nodes.push(u_u32);
            }
        }

        rng.shuffle(&mut r_nodes);

        for &v_u32 in &r_nodes {
            let v = v_u32 as usize;
            let c_v = sub_comm[v] as usize;
            if sub_members[c_v] != 1 {
                continue;
            }

            sub_stamp.clear();
            touched_subs.clear();
            let start = level.indptr[v] as usize;
            let end = level.indptr[v + 1] as usize;
            for pos in start..end {
                let w = level.weights[pos];
                if w <= 0.0 {
                    continue;
                }
                let nbr = level.indices[pos] as usize;
                if comm[nbr] as usize != s_id {
                    continue;
                }
                let cn = sub_comm[nbr];
                if cn as usize == c_v {
                    continue;
                }
                if sub_stamp.visit(cn as usize) {
                    touched_subs.push(cn);
                }
                sub_edge_weights[cn as usize] += w;
            }

            if touched_subs.is_empty() {
                continue;
            }

            touched_subs.sort_unstable();
            candidates.clear();
            let size_v = level.node_sizes[v];
            let mut max_gain = 0.0_f64;

            for &cand_c in &touched_subs {
                let c_idx = cand_c as usize;
                let e_v_c = sub_edge_weights[c_idx];
                if e_v_c <= 0.0 {
                    continue;
                }
                let size_c = sub_size[c_idx];
                let rhs = gamma_eff * size_c * (total_s_size - size_c).max(0.0);
                if ext_in_s[c_idx] + 1e-12 * rhs.abs() < rhs {
                    continue;
                }
                let penalty = gamma_eff * size_v * size_c;
                let gain = e_v_c - penalty;
                let tol = 1e-12 * (e_v_c.abs() + penalty.abs());
                if gain >= -tol {
                    let clamped_gain = gain.max(0.0);
                    if clamped_gain > max_gain {
                        max_gain = clamped_gain;
                    }
                    candidates.push((cand_c, clamped_gain));
                }
            }

            if !candidates.is_empty() {
                // Include staying singleton ({v}, 0.0) and apply log-sum-exp max-subtraction.
                let mut total_weight = (-max_gain / randomness).exp();
                for &(_, g) in &candidates {
                    total_weight += ((g - max_gain) / randomness).exp();
                }

                let target_draw = rng.next_f64() * total_weight;
                let mut cum = 0.0_f64;
                let mut chosen_c = c_v as u32;
                for &(cand_c, g) in &candidates {
                    cum += ((g - max_gain) / randomness).exp();
                    if target_draw < cum {
                        chosen_c = cand_c;
                        break;
                    }
                }

                if chosen_c != c_v as u32 {
                    let chosen_idx = chosen_c as usize;
                    let e_v_chosen = sub_edge_weights[chosen_idx];
                    sub_comm[v] = chosen_c;
                    sub_members[c_v] = 0;
                    sub_members[chosen_idx] += 1;
                    sub_size[chosen_idx] += size_v;
                    ext_in_s[chosen_idx] =
                        (ext_in_s[chosen_idx] + ext_in_s[v] - 2.0 * e_v_chosen).max(0.0);
                }
            }

            for &cand_c in &touched_subs {
                sub_edge_weights[cand_c as usize] = 0.0;
            }
        }
    }

    let num_refined = canonicalize_labels(&mut sub_comm, nl);
    (sub_comm, num_refined)
}

/// Splits any disconnected community into its positive-weight connected components in $O(V + E)$
/// and returns the canonical community count `0..num_communities`.
fn split_disconnected_communities<W: Scalar>(graph: &CsrGraph<W>, labels: &mut [u32]) -> usize {
    let n = graph.num_nodes();
    if n == 0 {
        return 0;
    }

    let mut visited = vec![false; n];
    let mut new_labels = vec![0u32; n];
    let mut next_comm = 0u32;
    let mut queue = VecDeque::new();
    let weights = graph.weights();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let target_label = labels[start];
        let comp_id = next_comm;
        next_comm += 1;
        new_labels[start] = comp_id;
        queue.push_back(start as u32);

        while let Some(u_u32) = queue.pop_front() {
            let u = u_u32 as usize;
            let row_start = graph.indptr()[u] as usize;
            let row_end = graph.indptr()[u + 1] as usize;
            for pos in row_start..row_end {
                if let Some(ws) = weights
                    && ws[pos].to_f64() <= 0.0
                {
                    continue;
                }
                let v = graph.indices()[pos] as usize;
                if !visited[v] && labels[v] == target_label {
                    visited[v] = true;
                    new_labels[v] = comp_id;
                    queue.push_back(v as u32);
                }
            }
        }
    }

    labels.copy_from_slice(&new_labels);
    next_comm as usize
}

/// Computes exact `(quality, modularity)` directly on the base graph for canonical `labels`.
fn evaluate_base_partition<W: Scalar>(
    graph: &CsrGraph<W>,
    labels: &[u32],
    num_communities: usize,
    objective: LeidenObjective,
    resolution: f64,
) -> (f64, f64) {
    let n = graph.num_nodes();
    let mut sigma_in = vec![0.0_f64; num_communities];
    let mut sigma_tot = vec![0.0_f64; num_communities];
    let mut comm_counts = vec![0usize; num_communities];
    let mut total_weight_2m = 0.0_f64;

    if let Some(ws) = graph.weights() {
        for u in 0..n {
            let cu = labels[u] as usize;
            comm_counts[cu] += 1;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let wf = ws[pos].to_f64();
                sigma_tot[cu] += wf;
                total_weight_2m += wf;
                if labels[graph.indices()[pos] as usize] as usize == cu {
                    sigma_in[cu] += wf;
                }
            }
        }
    } else {
        for u in 0..n {
            let cu = labels[u] as usize;
            comm_counts[cu] += 1;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let deg = (end - start) as f64;
            sigma_tot[cu] += deg;
            total_weight_2m += deg;
            for pos in start..end {
                if labels[graph.indices()[pos] as usize] as usize == cu {
                    sigma_in[cu] += 1.0;
                }
            }
        }
    }

    let modularity = if total_weight_2m <= 0.0 {
        0.0
    } else {
        let mut q = 0.0_f64;
        for c in 0..num_communities {
            let st = sigma_tot[c];
            if st > 0.0 || sigma_in[c] > 0.0 {
                q += (sigma_in[c] / total_weight_2m) - resolution * (st / total_weight_2m).powi(2);
            }
        }
        q
    };

    let quality = match objective {
        LeidenObjective::Modularity => modularity,
        LeidenObjective::Cpm => {
            let mut h = 0.0_f64;
            for c in 0..num_communities {
                let sz = comm_counts[c] as f64;
                h += 0.5 * sigma_in[c] - resolution * (sz * (sz - 1.0) * 0.5);
            }
            h
        }
    };

    (quality, modularity)
}

/// Runs multi-level Leiden community detection on an undirected [`CsrGraph`] following
/// Traag, Waltman, and van Eck (2019), Algorithm A.2.
///
/// Supports both [`LeidenObjective::Modularity`] and [`LeidenObjective::Cpm`], guarantees that
/// every returned community's positive-weight induced subgraph is internally connected, and
/// returns canonical `0..num_communities` labels in first-appearance vertex order.
pub fn leiden<W: Scalar>(graph: &CsrGraph<W>, config: &LeidenConfig) -> CoreResult<LeidenResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "leiden requires an undirected graph",
        ));
    }
    if config.max_passes == 0 {
        return Err(CoreError::invalid_input("max_passes must be >= 1"));
    }
    if !config.resolution.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::resolution"));
    }
    if config.resolution < 0.0 {
        return Err(CoreError::invalid_input("resolution must be >= 0.0"));
    }
    if !config.randomness.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::randomness"));
    }
    if config.randomness <= 0.0 {
        return Err(CoreError::invalid_input("randomness must be > 0.0"));
    }
    if !config.min_quality_gain.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::min_quality_gain"));
    }
    if config.min_quality_gain < 0.0 {
        return Err(CoreError::invalid_input("min_quality_gain must be >= 0.0"));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(LeidenResult {
            labels: Vec::new(),
            num_communities: 0,
            quality: 0.0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let base_level = LeidenLevelGraph::from_csr(graph, config.objective)?;
    let mut best_labels: Vec<u32> = (0..n as u32).collect();

    if base_level.total_weight_2m <= 0.0 {
        return Ok(LeidenResult {
            labels: best_labels,
            num_communities: n,
            quality: 0.0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let gamma_eff = match config.objective {
        LeidenObjective::Modularity => config.resolution / base_level.total_weight_2m,
        LeidenObjective::Cpm => config.resolution,
    };
    if !gamma_eff.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "effective resolution overflows f64",
        ));
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let mut best_quality =
        base_level.evaluate_quality(&best_labels, n, config.objective, config.resolution);
    if !best_quality.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "initial leiden quality is not finite",
        ));
    }
    let mut passes = 0usize;
    let mut total_steps = 0usize;
    let mut outer_iter = 0usize;

    while total_steps < config.max_passes {
        let outer_start_quality = best_quality;
        let mut level = base_level.clone();
        let mut base_to_super: Vec<u32> = (0..n as u32).collect();
        let mut comm = best_labels.clone();
        let mut current_q = best_quality;
        let mut level_depth = 0usize;

        while total_steps < config.max_passes {
            let nl = level.num_nodes;
            if nl <= 1 {
                break;
            }

            total_steps += 1;
            let any_moved = move_nodes_fast(&level, &mut comm, gamma_eff, &mut rng);
            let mut num_outer = canonicalize_labels(&mut comm, nl);

            if any_moved {
                let next_q =
                    level.evaluate_quality(&comm, num_outer, config.objective, config.resolution);
                if !next_q.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "leiden intermediate quality is not finite",
                    ));
                }
                if next_q - current_q > config.min_quality_gain {
                    current_q = next_q;
                    passes += 1;
                    best_quality = next_q;
                    for i in 0..n {
                        best_labels[i] = comm[base_to_super[i] as usize];
                    }
                } else if level_depth > 0 || outer_iter == 0 {
                    break;
                } else if next_q > best_quality {
                    current_q = next_q;
                    best_quality = next_q;
                    for i in 0..n {
                        best_labels[i] = comm[base_to_super[i] as usize];
                    }
                } else {
                    comm.copy_from_slice(&best_labels);
                    num_outer = canonicalize_labels(&mut comm, nl);
                }
            } else if level_depth > 0 || outer_iter == 0 {
                break;
            }

            if num_outer == nl || total_steps >= config.max_passes {
                break;
            }

            let (refined_comm, num_refined) = refine_partition(
                &level,
                &comm,
                num_outer,
                gamma_eff,
                config.randomness,
                &mut rng,
            );
            if num_refined == nl {
                break;
            }

            let mut lifted_comm = vec![0u32; num_refined];
            for u in 0..nl {
                lifted_comm[refined_comm[u] as usize] = comm[u];
            }
            for s in &mut base_to_super {
                *s = refined_comm[*s as usize];
            }

            level = level.aggregate(&refined_comm, num_refined);
            comm = lifted_comm;
            level_depth += 1;
        }

        if best_quality - outer_start_quality <= config.min_quality_gain {
            break;
        }
        outer_iter += 1;
    }

    let num_communities = split_disconnected_communities(graph, &mut best_labels);
    let (quality, modularity) = evaluate_base_partition(
        graph,
        &best_labels,
        num_communities,
        config.objective,
        config.resolution,
    );
    if !quality.is_finite() || !modularity.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "leiden quality or modularity is not finite",
        ));
    }

    Ok(LeidenResult {
        labels: best_labels,
        num_communities,
        quality,
        modularity,
        passes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KARATE_CSV: &str = include_str!("../../dataset/src/data/karate.csv");
    const LESMIS_CSV: &str = include_str!("../../dataset/src/data/lesmis.csv");
    const BUS_1138_CSV: &str = include_str!("../../dataset/src/data/1138_bus.csv");

    fn parse_test_csv(input: &str) -> CsrGraph<f64> {
        let mut tokens = input.split_ascii_whitespace();
        let header = tokens.next().unwrap();
        let mut hparts = header.split(',');
        let n: usize = hparts.next().unwrap().parse().unwrap();
        let m: usize = hparts.next().unwrap().parse().unwrap();
        let mut edges = Vec::with_capacity(m);
        for row in tokens {
            let mut parts = row.split(',');
            let u: u32 = parts.next().unwrap().parse().unwrap();
            let v: u32 = parts.next().unwrap().parse().unwrap();
            edges.push((u, v));
        }
        assert_eq!(edges.len(), m);
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_two_cliques_barbell(clique_size: usize) -> CsrGraph<f64> {
        let n = clique_size * 2;
        let mut edges = Vec::new();
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push((i as u32, j as u32));
                edges.push(((clique_size + i) as u32, (clique_size + j) as u32));
            }
        }
        edges.push(((clique_size - 1) as u32, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_ring_of_cliques(num_cliques: usize, clique_size: usize) -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for c in 0..num_cliques {
            let base = (c * clique_size) as u32;
            for i in 0..clique_size as u32 {
                for j in (i + 1)..clique_size as u32 {
                    edges.push((base + i, base + j));
                }
            }
            let next_base = (((c + 1) % num_cliques) * clique_size) as u32;
            edges.push((base + clique_size as u32 - 1, next_base));
        }
        CsrGraph::from_edges(num_cliques * clique_size, &edges, None, false, false).unwrap()
    }

    fn oracle_quality_and_modularity<W: Scalar>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        num_communities: usize,
        objective: LeidenObjective,
        resolution: f64,
    ) -> (f64, f64) {
        let n = graph.num_nodes();
        let mut two_e_c = vec![0.0_f64; num_communities];
        let mut k_c = vec![0.0_f64; num_communities];
        let mut n_c = vec![0usize; num_communities];
        let mut two_m = 0.0_f64;

        for u in 0..n {
            let cu = labels[u] as usize;
            n_c[cu] += 1;
            let row_start = graph.indptr()[u] as usize;
            let row_end = graph.indptr()[u + 1] as usize;
            for pos in row_start..row_end {
                let v = graph.indices()[pos] as usize;
                let w = graph.weights().map_or(1.0, |ws| ws[pos].to_f64());
                k_c[cu] += w;
                two_m += w;
                if labels[v] as usize == cu {
                    two_e_c[cu] += w;
                }
            }
        }

        let modularity = if two_m <= 0.0 {
            0.0
        } else {
            let mut q = 0.0_f64;
            for c in 0..num_communities {
                q += two_e_c[c] / two_m - resolution * (k_c[c] / two_m).powi(2);
            }
            q
        };

        let quality = match objective {
            LeidenObjective::Modularity => modularity,
            LeidenObjective::Cpm => {
                let mut h = 0.0_f64;
                for c in 0..num_communities {
                    let sz = n_c[c] as f64;
                    h += 0.5 * two_e_c[c] - resolution * (sz * (sz - 1.0) * 0.5);
                }
                h
            }
        };

        (quality, modularity)
    }

    fn assert_canonical_and_connected<W: Scalar>(graph: &CsrGraph<W>, res: &LeidenResult) {
        let n = graph.num_nodes();
        assert_eq!(res.labels.len(), n);
        if n == 0 {
            assert_eq!(res.num_communities, 0);
            return;
        }

        // Verify canonical 0..num_communities order of first appearance.
        let mut next_expected = 0u32;
        let mut seen = vec![false; res.num_communities];
        for &l in &res.labels {
            let idx = l as usize;
            assert!(idx < res.num_communities);
            if !seen[idx] {
                assert_eq!(l, next_expected);
                seen[idx] = true;
                next_expected += 1;
            }
        }
        assert_eq!(next_expected as usize, res.num_communities);

        // Verify every community induces a single connected component via BFS over positive edges.
        let mut visited = vec![false; n];
        let mut comm_started = vec![false; res.num_communities];
        let mut queue = VecDeque::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let c = res.labels[start] as usize;
            assert!(
                !comm_started[c],
                "community {c} is disconnected (multiple components)"
            );
            comm_started[c] = true;
            visited[start] = true;
            queue.push_back(start as u32);
            while let Some(u_u32) = queue.pop_front() {
                let u = u_u32 as usize;
                let row_start = graph.indptr()[u] as usize;
                let row_end = graph.indptr()[u + 1] as usize;
                for pos in row_start..row_end {
                    if let Some(ws) = graph.weights()
                        && ws[pos].to_f64() <= 0.0
                    {
                        continue;
                    }
                    let v = graph.indices()[pos] as usize;
                    if !visited[v] && res.labels[v] as usize == c {
                        visited[v] = true;
                        queue.push_back(v as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn test_leiden_barbell_ring_and_weighted_f32() {
        let barbell = make_two_cliques_barbell(6);
        let res = leiden(&barbell, &LeidenConfig::default()).unwrap();
        assert_canonical_and_connected(&barbell, &res);
        assert_eq!(res.num_communities, 2);
        assert_eq!(res.labels[..6], [0; 6]);
        assert_eq!(res.labels[6..], [1; 6]);
        assert!((res.quality - res.modularity).abs() < 1e-12);
        assert!((res.modularity - 0.467_741_935_483_870_94).abs() < 1e-10);
        assert!(res.passes >= 1);

        let res_cpm = leiden(
            &barbell,
            &LeidenConfig {
                objective: LeidenObjective::Cpm,
                resolution: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&barbell, &res_cpm);
        assert_eq!(res_cpm.num_communities, 2);
        assert!((res_cpm.quality - 15.0).abs() < 1e-12);

        // Ring of 8 cliques of size 5.
        let ring = make_ring_of_cliques(8, 5);
        let res_ring = leiden(&ring, &LeidenConfig::default()).unwrap();
        assert_canonical_and_connected(&ring, &res_ring);
        assert_eq!(res_ring.num_communities, 8);
        assert!((res_ring.modularity - 0.784_090_909_090_909_1).abs() < 1e-10);
        for c in 0..8 {
            let first = res_ring.labels[c * 5];
            assert!(
                res_ring.labels[c * 5..(c + 1) * 5]
                    .iter()
                    .all(|&l| l == first)
            );
        }

        let res_ring_cpm = leiden(
            &ring,
            &LeidenConfig {
                objective: LeidenObjective::Cpm,
                resolution: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&ring, &res_ring_cpm);
        assert_eq!(res_ring_cpm.num_communities, 8);

        // Weighted f32 barbell + low-resolution merge.
        let mut w_edges = Vec::new();
        let mut w_vals = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                w_edges.push((i, j));
                w_vals.push(8.0_f32);
                w_edges.push((4 + i, 4 + j));
                w_vals.push(8.0_f32);
            }
        }
        w_edges.push((3, 4));
        w_vals.push(0.1_f32);
        let wg = CsrGraph::from_edges(8, &w_edges, Some(&w_vals), false, false).unwrap();

        let w_res = leiden(&wg, &LeidenConfig::default()).unwrap();
        assert_canonical_and_connected(&wg, &w_res);
        assert_eq!(w_res.num_communities, 2);
        assert_eq!(w_res.labels[..4], [0; 4]);
        assert_eq!(w_res.labels[4..], [1; 4]);
        let (oq, om) = oracle_quality_and_modularity(
            &wg,
            &w_res.labels,
            w_res.num_communities,
            LeidenObjective::Modularity,
            1.0,
        );
        assert!((w_res.quality - oq).abs() < 1e-12);
        assert!((w_res.modularity - om).abs() < 1e-12);

        let low_gamma = leiden(
            &wg,
            &LeidenConfig {
                resolution: 0.001,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&wg, &low_gamma);
        assert_eq!(low_gamma.num_communities, 1);
    }

    #[test]
    fn test_leiden_real_datasets_connectivity_and_oracle() {
        let datasets = [
            ("karate", parse_test_csv(KARATE_CSV)),
            ("lesmis", parse_test_csv(LESMIS_CSV)),
            ("1138_bus", parse_test_csv(BUS_1138_CSV)),
        ];

        for (name, graph) in &datasets {
            for &(objective, resolution, max_passes) in &[
                (LeidenObjective::Modularity, 1.0, 100),
                (LeidenObjective::Modularity, 0.5, 100),
                (LeidenObjective::Modularity, 2.0, 100),
                (LeidenObjective::Modularity, 1.0, 1),
                (LeidenObjective::Cpm, 0.0, 100),
                (LeidenObjective::Cpm, 0.05, 100),
                (LeidenObjective::Cpm, 0.25, 100),
                (LeidenObjective::Cpm, 0.5, 100),
                (LeidenObjective::Cpm, 1.0, 100),
                (LeidenObjective::Cpm, 0.1, 1),
            ] {
                let cfg = LeidenConfig {
                    objective,
                    resolution,
                    max_passes,
                    ..Default::default()
                };
                let res = leiden(graph, &cfg).unwrap();
                assert_canonical_and_connected(graph, &res);
                let (oq, om) = oracle_quality_and_modularity(
                    graph,
                    &res.labels,
                    res.num_communities,
                    objective,
                    resolution,
                );
                assert!(
                    (res.quality - oq).abs() < 1e-12,
                    "{name} quality mismatch: {} vs {oq}",
                    res.quality
                );
                assert!(
                    (res.modularity - om).abs() < 1e-12,
                    "{name} modularity mismatch: {} vs {om}",
                    res.modularity
                );
                assert!(res.passes <= max_passes);
            }

            let karate_mod = leiden(graph, &LeidenConfig::default()).unwrap();
            if *name == "karate" {
                assert!(karate_mod.modularity > 0.41);
                assert!(karate_mod.num_communities >= 2);
            }
        }
    }

    #[test]
    fn test_leiden_cpm_resolution_monotonicity() {
        let graphs = [
            parse_test_csv(KARATE_CSV),
            parse_test_csv(LESMIS_CSV),
            make_ring_of_cliques(8, 5),
        ];
        let gammas = [0.0, 0.05, 0.2, 0.5, 1.0];

        for g in &graphs {
            let mut prev_k = 0usize;
            for &gamma in &gammas {
                let res = leiden(
                    g,
                    &LeidenConfig {
                        objective: LeidenObjective::Cpm,
                        resolution: gamma,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_canonical_and_connected(g, &res);
                assert!(
                    res.num_communities >= prev_k,
                    "CPM community count decreased from {prev_k} to {} at gamma={gamma}",
                    res.num_communities
                );
                prev_k = res.num_communities;
            }
        }
    }

    #[test]
    fn test_leiden_adversarial_bridge_connectivity_across_seeds_and_single_pass() {
        // Bridge vertex 0 connects two K_4 subgraphs (1..=4, 5..=8) and a stronger K_5 (9..=13)
        // that pulls vertex 0 away after Pass 0, testing Leiden refinement and BFS guard.
        let mut edges = Vec::new();
        for i in 1..=4u32 {
            for j in (i + 1)..=4 {
                edges.push((i, j));
            }
        }
        for i in 5..=8u32 {
            for j in (i + 1)..=8 {
                edges.push((i, j));
            }
        }
        for i in 9..=13u32 {
            for j in (i + 1)..=13 {
                edges.push((i, j));
            }
        }
        edges.extend_from_slice(&[(0, 1), (0, 2), (0, 5), (0, 6), (0, 9), (0, 10), (0, 11)]);
        let mut weights = vec![1.0_f64; edges.len()];
        let len = weights.len();
        weights[len - 3] = 2.0;
        weights[len - 2] = 2.0;
        weights[len - 1] = 2.0;
        let bridge_graph = CsrGraph::from_edges(14, &edges, Some(&weights), false, false).unwrap();

        for seed in 0..20u64 {
            for max_passes in [1usize, 100] {
                for &(objective, resolution) in &[
                    (LeidenObjective::Modularity, 1.0),
                    (LeidenObjective::Cpm, 0.25),
                    (LeidenObjective::Cpm, 0.0),
                ] {
                    let res = leiden(
                        &bridge_graph,
                        &LeidenConfig {
                            objective,
                            resolution,
                            max_passes,
                            seed,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert_canonical_and_connected(&bridge_graph, &res);
                    let (oq, om) = oracle_quality_and_modularity(
                        &bridge_graph,
                        &res.labels,
                        res.num_communities,
                        objective,
                        resolution,
                    );
                    assert!((res.quality - oq).abs() < 1e-12);
                    assert!((res.modularity - om).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_leiden_log_sum_exp_extreme_weights_and_scales() {
        let mut edges = Vec::new();
        let mut weights_large = Vec::new();
        let mut weights_small = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                edges.push((i, j));
                weights_large.push(1e9_f64);
                weights_small.push(1e-9_f64);
                edges.push((4 + i, 4 + j));
                weights_large.push(1e9_f64);
                weights_small.push(1e-9_f64);
            }
        }
        edges.push((3, 4));
        weights_large.push(1e6_f64);
        weights_small.push(1e-12_f64);

        let g_large = CsrGraph::from_edges(8, &edges, Some(&weights_large), false, false).unwrap();
        let g_small = CsrGraph::from_edges(8, &edges, Some(&weights_small), false, false).unwrap();

        for (g, cpm_gamma, cpm_theta, min_gain) in [
            (&g_large, 1e8_f64, 1e-6_f64, 1e-7_f64),
            (&g_small, 1e-10_f64, 1e-15_f64, 0.0_f64),
        ] {
            for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
                let (resolution, randomness, min_quality_gain) = match objective {
                    LeidenObjective::Modularity => (1.0, 1e-6, 1e-7),
                    LeidenObjective::Cpm => (cpm_gamma, cpm_theta, min_gain),
                };
                let res = leiden(
                    g,
                    &LeidenConfig {
                        objective,
                        resolution,
                        randomness,
                        min_quality_gain,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(res.quality.is_finite());
                assert!(res.modularity.is_finite());
                assert_canonical_and_connected(g, &res);
                assert_eq!(res.num_communities, 2);
                let (oq, om) = oracle_quality_and_modularity(
                    g,
                    &res.labels,
                    res.num_communities,
                    objective,
                    resolution,
                );
                assert!((res.quality - oq).abs() <= 1e-12 * (1.0 + oq.abs()));
                assert!((res.modularity - om).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn test_leiden_determinism_degenerate_and_disconnected_cases() {
        let barbell = make_two_cliques_barbell(5);
        let cfg = LeidenConfig {
            seed: 1234,
            ..Default::default()
        };
        assert_eq!(
            leiden(&barbell, &cfg).unwrap(),
            leiden(&barbell, &cfg).unwrap()
        );

        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let re = leiden(&empty, &cfg).unwrap();
        assert!(re.labels.is_empty());
        assert_eq!(re.num_communities, 0);
        assert_eq!(re.quality, 0.0);
        assert_eq!(re.modularity, 0.0);
        assert_eq!(re.passes, 0);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let rs = leiden(&single, &cfg).unwrap();
        assert_eq!(rs.labels, vec![0]);
        assert_eq!(rs.num_communities, 1);
        assert_eq!(rs.quality, 0.0);
        assert_eq!(rs.modularity, 0.0);

        let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let ri = leiden(&edgeless, &cfg).unwrap();
        assert_eq!(ri.labels, vec![0, 1, 2, 3]);
        assert_eq!(ri.num_communities, 4);
        assert_eq!(ri.quality, 0.0);
        assert_eq!(ri.modularity, 0.0);

        // Zero-weight edges behave identically to edgeless.
        let zero_w =
            CsrGraph::from_edges(3, &[(0, 1), (1, 2)], Some(&[0.0, 0.0]), false, false).unwrap();
        let rz = leiden(&zero_w, &cfg).unwrap();
        assert_eq!(rz.labels, vec![0, 1, 2]);
        assert_eq!(rz.num_communities, 3);
        assert_eq!(rz.modularity, 0.0);

        // Disconnected two triangles + isolated node: never merge even at resolution = 0.0.
        let disc = CsrGraph::<f64>::from_edges(
            7,
            &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
            None,
            false,
            false,
        )
        .unwrap();
        for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
            for resolution in [0.0, 0.5, 1.0] {
                let rd = leiden(
                    &disc,
                    &LeidenConfig {
                        objective,
                        resolution,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_canonical_and_connected(&disc, &rd);
                assert!(rd.num_communities >= 3);
                if resolution <= 0.5 {
                    assert_eq!(rd.labels, vec![0, 0, 0, 1, 1, 1, 2]);
                }
            }
        }
    }

    #[test]
    fn test_leiden_validation_errors() {
        let cfg = LeidenConfig::default();
        let directed =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
        assert!(matches!(
            leiden(&directed, &cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let g = make_two_cliques_barbell(3);
        for bad_cfg in [
            LeidenConfig {
                max_passes: 0,
                ..Default::default()
            },
            LeidenConfig {
                resolution: -0.5,
                ..Default::default()
            },
            LeidenConfig {
                randomness: 0.0,
                ..Default::default()
            },
            LeidenConfig {
                randomness: -0.01,
                ..Default::default()
            },
            LeidenConfig {
                min_quality_gain: -1e-6,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                leiden(&g, &bad_cfg),
                Err(CoreError::InvalidInput(_))
            ));
        }

        for non_finite_cfg in [
            LeidenConfig {
                resolution: f64::NAN,
                ..Default::default()
            },
            LeidenConfig {
                resolution: f64::INFINITY,
                ..Default::default()
            },
            LeidenConfig {
                randomness: f64::NAN,
                ..Default::default()
            },
            LeidenConfig {
                randomness: f64::INFINITY,
                ..Default::default()
            },
            LeidenConfig {
                min_quality_gain: f64::NAN,
                ..Default::default()
            },
            LeidenConfig {
                min_quality_gain: f64::INFINITY,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                leiden(&g, &non_finite_cfg),
                Err(CoreError::NonFiniteValue { .. })
            ));
        }

        let neg_w = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0_f64]), false, false).unwrap();
        assert!(matches!(
            leiden(&neg_w, &cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_leiden_outer_restarts_convergence_and_max_passes_bound() {
        let karate = parse_test_csv(KARATE_CSV);
        let lesmis = parse_test_csv(LESMIS_CSV);
        let bus = parse_test_csv(BUS_1138_CSV);

        // Converged outer-restart Modularity matches igraph(n_iterations=-1) on karate and lesmis.
        for seed in 0..10u64 {
            let rk = leiden(
                &karate,
                &LeidenConfig {
                    seed,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&karate, &rk);
            assert!(
                rk.modularity > 0.4195,
                "karate seed={seed} Q={} below converged threshold",
                rk.modularity
            );
            assert_eq!(rk.num_communities, 4);

            let rl = leiden(
                &lesmis,
                &LeidenConfig {
                    seed,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&lesmis, &rl);
            assert!(
                rl.modularity > 0.5595,
                "lesmis seed={seed} Q={} below converged threshold",
                rl.modularity
            );
            assert_eq!(rl.num_communities, 6);
        }

        // Converged 1138_bus Modularity and CPM outperform single-pass bounds and match igraph(-1).
        let mut bus_q_sum = 0.0_f64;
        let mut bus_h_sum = 0.0_f64;
        for seed in 0..5u64 {
            let rb_mod = leiden(
                &bus,
                &LeidenConfig {
                    seed,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&bus, &rb_mod);
            assert!(rb_mod.passes <= 100);
            bus_q_sum += rb_mod.modularity;

            let rb_cpm = leiden(
                &bus,
                &LeidenConfig {
                    objective: LeidenObjective::Cpm,
                    resolution: 0.25,
                    seed,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&bus, &rb_cpm);
            assert!(rb_cpm.passes <= 100);
            bus_h_sum += rb_cpm.quality;
        }
        let mean_bus_q = bus_q_sum / 5.0;
        let mean_bus_h = bus_h_sum / 5.0;
        assert!(
            mean_bus_q > 0.8985,
            "1138_bus mean Modularity Q={mean_bus_q} did not converge above 0.8985"
        );
        assert!(
            mean_bus_h > 503.0,
            "1138_bus mean CPM(0.25) H={mean_bus_h} did not converge above 503.0"
        );

        // Verify max_passes strictly bounds committed passes across tight budgets.
        for &max_passes in &[1usize, 2, 5] {
            let r1 = leiden(
                &bus,
                &LeidenConfig {
                    max_passes,
                    seed: 42,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&bus, &r1);
            assert!(r1.passes <= max_passes);
        }
    }

    #[test]
    fn test_leiden_extreme_weights_never_return_nan_or_inf() {
        let g_subnormal = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            Some(&[1e-312_f64; 3]),
            false,
            false,
        )
        .unwrap();
        let res_sub_mod = leiden(&g_subnormal, &LeidenConfig::default());
        assert!(matches!(res_sub_mod, Err(CoreError::NumericalBreakdown(_))));

        for (resolution, min_quality_gain, expected_comms) in
            [(1.0_f64, 1e-7_f64, 4usize), (0.0_f64, 0.0_f64, 1usize)]
        {
            let res_sub_cpm = leiden(
                &g_subnormal,
                &LeidenConfig {
                    objective: LeidenObjective::Cpm,
                    resolution,
                    min_quality_gain,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(res_sub_cpm.quality.is_finite());
            assert!(res_sub_cpm.modularity.is_finite());
            assert_canonical_and_connected(&g_subnormal, &res_sub_cpm);
            assert_eq!(res_sub_cpm.num_communities, expected_comms);
        }

        let g_overflow = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            Some(&[1e308_f64; 3]),
            false,
            false,
        )
        .unwrap();
        for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
            let res = leiden(
                &g_overflow,
                &LeidenConfig {
                    objective,
                    ..Default::default()
                },
            );
            assert!(matches!(res, Err(CoreError::NumericalBreakdown(_))));
        }

        // Single edge of 1e308 overflows total_weight_2m = 2e308 even when each degree is 1e308.
        let g_pair_overflow =
            CsrGraph::from_edges(2, &[(0, 1)], Some(&[1e308_f64]), false, false).unwrap();
        for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
            let res = leiden(
                &g_pair_overflow,
                &LeidenConfig {
                    objective,
                    ..Default::default()
                },
            );
            assert!(matches!(res, Err(CoreError::NumericalBreakdown(_))));
        }

        // Huge resolution on small total_weight_2m overflows gamma_eff cleanly without NaN.
        let g_tiny_normal = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            Some(&[1e-305_f64; 3]),
            false,
            false,
        )
        .unwrap();
        let res_gamma_overflow = leiden(
            &g_tiny_normal,
            &LeidenConfig {
                objective: LeidenObjective::Modularity,
                resolution: 1e10,
                ..Default::default()
            },
        );
        assert!(matches!(
            res_gamma_overflow,
            Err(CoreError::NumericalBreakdown(_))
        ));

        // Huge f64::MAX resolution must never return Ok with NaN or Inf.
        let g_unit = make_two_cliques_barbell(3);
        for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
            match leiden(
                &g_unit,
                &LeidenConfig {
                    objective,
                    resolution: f64::MAX,
                    ..Default::default()
                },
            ) {
                Ok(res) => {
                    assert!(res.quality.is_finite());
                    assert!(res.modularity.is_finite());
                }
                Err(CoreError::NumericalBreakdown(_)) => {}
                Err(other) => panic!("unexpected error variant: {other:?}"),
            }
        }
    }
}
