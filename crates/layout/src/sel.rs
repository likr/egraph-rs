//! Structural Entropy Layout (`SEL`) solver.
//!
//! Implements Xue et al. (2025), "Structural Entropy Based Visualization of Social Networks"
//! (*Computer Graphics Forum / Pacific Graphics 2025*) and the reference defaults from
//! `IDEAS-Laboratory/SEL` (`master`):
//! - $k$-dimensional coding tree (`mode = 'v2'`, default `tree_depth = 5`, `+1` Laplace smoothing)
//!   with virtual global root handling for disconnected graphs and isolated vertices.
//! - Algorithm 1 Minimum Entropy Priority Queue (`pq_max_pops = 10`, unassigned sentinel `100.0`)
//!   assigning cross-partition structural entropy scores $W[i, j]$ to edges.
//! - Elbow threshold selection bounded by $\mathrm{SE}_{\mathrm{target}} \le 0.5 H_{\max}$.
//! - Five edge-weight modulation mappings (`Sigmoid`, `Linear`, `Tanh`, `CustomLogistic`, `Binary`).
//! - Cooperative layout optimization delegating to [`FruchtermanReingoldSolver`] (`Fr`) or
//!   [`FieldLayoutSolver::t_fdp`] (`TFdp`).

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus, VisitStamp,
};

use crate::field_layout::{FieldLayoutSolver, TFdpConfig};
use crate::fr::{FruchtermanReingoldConfig, FruchtermanReingoldSolver};

const UNASSIGNED_ENTROPY_SENTINEL: f64 = 100.0;

/// Edge-weight modulation function applied to structural-entropy edge scores $W[i, j]$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelWeightMapping {
    /// $\sigma(1 - \mathrm{SE}_{\mathrm{target}} / W)$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0` (`default`).
    #[default]
    Sigmoid,
    /// Linear $1 - \mathrm{SE}_{\mathrm{target}} / W$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Linear,
    /// $(\tanh(1 - \mathrm{SE}_{\mathrm{target}} / W) + 1) / 2$ when $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Tanh,
    /// Decaying logistic $(1 + \exp(10(z - 0.8)))^{-1}$ matching `IDEAS-Laboratory/SEL` `custom_logistic`.
    CustomLogistic,
    /// Binary `1.0` if $W > \mathrm{SE}_{\mathrm{target}}$, else `0.0`.
    Binary,
}

/// Underlying force-directed layout backend driven by SEL-modulated edge weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelLayoutBackend {
    /// Classic Fruchterman–Reingold solver ([`FruchtermanReingoldSolver`], `default`).
    #[default]
    Fr,
    /// Bounded Student-$t$ force-directed placement ([`FieldLayoutSolver::t_fdp`]).
    TFdp,
}

/// Hyperparameters for [`SelLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelConfig {
    /// Target height of the $k$-dimensional coding tree (`default: 5`).
    pub tree_depth: usize,
    /// Maximum priority-queue pops in Algorithm 1 (`default: 10`).
    pub pq_max_pops: usize,
    /// Edge-weight mapping function (`default: SelWeightMapping::Sigmoid`).
    pub mapping: SelWeightMapping,
    /// Force-directed layout backend (`default: SelLayoutBackend::Fr`).
    pub layout_backend: SelLayoutBackend,
    /// Number of iterations forwarded to the backend solver (`default: 200`).
    pub iterations: usize,
    /// Explicit deterministic RNG seed forwarded to the backend solver (`default: 42`).
    pub seed: u64,
}

impl Default for SelConfig {
    fn default() -> Self {
        Self {
            tree_depth: 5,
            pq_max_pops: 10,
            mapping: SelWeightMapping::Sigmoid,
            layout_backend: SelLayoutBackend::Fr,
            iterations: 200,
            seed: 42,
        }
    }
}

impl SelConfig {
    /// Validates all SEL hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.tree_depth == 0 {
            return Err(CoreError::invalid_input("tree_depth must be >= 1"));
        }
        if self.pq_max_pops == 0 {
            return Err(CoreError::invalid_input("pq_max_pops must be >= 1"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum SelBackendSolver<S: Scalar> {
    Fr(Box<FruchtermanReingoldSolver<S>>),
    TFdp(Box<FieldLayoutSolver<S>>),
}

/// Cooperative Structural Entropy Layout (`SEL`) solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct SelLayoutSolver<S: Scalar = f64> {
    config: SelConfig,
    num_nodes: usize,
    modulated_weights: Vec<f64>,
    se_target: f64,
    backend: SelBackendSolver<S>,
}

impl<S: Scalar> SelLayoutSolver<S> {
    /// Constructs a [`SelLayoutSolver`] from a [`CsrGraph`] and [`SelConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: SelConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_uv, orig_weights) = extract_canonical_undirected_edges(graph)?;
        let m = edges_uv.len();

        let (se_target, modulated_weights) = if n <= 1 || m == 0 {
            (0.0, vec![0.0; m])
        } else {
            let mut node_vol = vec![0.0_f64; n];
            let mut total_vol = 0.0_f64;
            for (&(u, v), &w) in edges_uv.iter().zip(orig_weights.iter()) {
                node_vol[u as usize] += w;
                node_vol[v as usize] += w;
                total_vol += 2.0 * w;
            }
            if total_vol <= 0.0 {
                (0.0, vec![0.0; m])
            } else {
                compute_sel_weights(n, &edges_uv, &orig_weights, &node_vol, total_vol, &config)
            }
        };

        let mod_graph = CsrGraph::from_edges(n, &edges_uv, Some(&modulated_weights), false, false)?;

        let backend = match config.layout_backend {
            SelLayoutBackend::Fr => {
                let fr_cfg = FruchtermanReingoldConfig {
                    iterations: config.iterations,
                    seed: config.seed,
                    ..FruchtermanReingoldConfig::default()
                };
                SelBackendSolver::Fr(Box::new(FruchtermanReingoldSolver::new(
                    &mod_graph, fr_cfg,
                )?))
            }
            SelLayoutBackend::TFdp => {
                let tfdp_cfg = TFdpConfig {
                    total_iterations: config.iterations,
                    seed: config.seed,
                    ..TFdpConfig::default()
                };
                SelBackendSolver::TFdp(Box::new(FieldLayoutSolver::t_fdp(&mod_graph, tfdp_cfg)?))
            }
        };

        Ok(Self {
            config,
            num_nodes: n,
            modulated_weights,
            se_target,
            backend,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &SelConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the SEL-modulated edge weights aligned with canonical sorted undirected edges `(u, v)` (`u < v`).
    #[inline]
    pub fn modulated_weights(&self) -> &[f64] {
        &self.modulated_weights
    }

    /// Returns the elbow-selected structural entropy threshold $\mathrm{SE}_{\mathrm{target}}$.
    #[inline]
    pub fn se_target(&self) -> f64 {
        self.se_target
    }
}

impl<S: Scalar> IterativeSolver<S> for SelLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SelLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        match &mut self.backend {
            SelBackendSolver::Fr(solver) => solver.step(layout),
            SelBackendSolver::TFdp(solver) => solver.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.status(),
            SelBackendSolver::TFdp(solver) => solver.status(),
        }
    }

    #[inline]
    fn iteration(&self) -> usize {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.iteration(),
            SelBackendSolver::TFdp(solver) => solver.iteration(),
        }
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        match &self.backend {
            SelBackendSolver::Fr(solver) => solver.total_iterations(),
            SelBackendSolver::TFdp(solver) => solver.total_iterations(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

type SubTreeExpansion = (usize, Vec<Option<CodingTreeNode>>);
type LeafUpMapping = (usize, Option<SubTreeExpansion>);

#[derive(Debug, Clone)]
struct CodingTreeNode {
    parent: Option<usize>,
    children: Vec<usize>,
    partition: Vec<u32>,
    vol: f64,
    g: f64,
    merged: bool,
    child_h: usize,
    child_cut: f64,
    depth: usize,
    e_increase: f64,
    a_increase: f64,
}

#[inline]
fn combine_delta(n1: &CodingTreeNode, n2: &CodingTreeNode, cut_v: f64, g_vol: f64) -> f64 {
    if g_vol <= 0.0 {
        return 0.0;
    }
    let v1 = n1.vol + 1.0;
    let v2 = n2.vol + 1.0;
    let g1 = n1.g + 1.0;
    let g2 = n2.g + 1.0;
    let v12 = v1 + v2;
    let t1 = (v1 - g1) * (v12 / v1).log2();
    let t2 = (v2 - g2) * (v12 / v2).log2();
    let t3 = 2.0 * cut_v * (g_vol / v12).max(1e-300).log2();
    (t1 + t2 - t3) / g_vol
}

#[inline]
fn compress_delta(node: &CodingTreeNode, parent: &CodingTreeNode) -> f64 {
    let v1 = node.vol + 1.0;
    let v2 = parent.vol + 1.0;
    node.child_cut * (v2 / v1).log2()
}

fn compute_sel_weights(
    n: usize,
    edges_uv: &[(u32, u32)],
    orig_weights: &[f64],
    node_vol: &[f64],
    total_vol: f64,
    config: &SelConfig,
) -> (f64, Vec<f64>) {
    let m = edges_uv.len();

    // Build incident canonical edge lookup per vertex for O(subtree deg) PQ scans.
    let mut vert_edge_indptr = vec![0usize; n + 1];
    for &(u, v) in edges_uv {
        vert_edge_indptr[u as usize + 1] += 1;
        vert_edge_indptr[v as usize + 1] += 1;
    }
    for i in 0..n {
        vert_edge_indptr[i + 1] += vert_edge_indptr[i];
    }
    let mut cursor = vert_edge_indptr[..n].to_vec();
    let mut vert_edge_nbr = vec![0u32; m * 2];
    let mut vert_edge_idx = vec![0usize; m * 2];
    let mut vert_edge_w = vec![0.0_f64; m * 2];
    for (e_idx, (&(u, v), &w)) in edges_uv.iter().zip(orig_weights.iter()).enumerate() {
        let pu = cursor[u as usize];
        cursor[u as usize] += 1;
        vert_edge_nbr[pu] = v;
        vert_edge_idx[pu] = e_idx;
        vert_edge_w[pu] = w;

        let pv = cursor[v as usize];
        cursor[v as usize] += 1;
        vert_edge_nbr[pv] = u;
        vert_edge_idx[pv] = e_idx;
        vert_edge_w[pv] = w;
    }

    let (tree_nodes, root_id) = build_coding_tree(
        n,
        edges_uv,
        orig_weights,
        node_vol,
        total_vol,
        &vert_edge_indptr,
        &vert_edge_nbr,
        &vert_edge_w,
        config.tree_depth,
    );

    let h_max = tree_structural_entropy(&tree_nodes, total_vol);

    // Algorithm 1: Minimum Entropy Priority Queue (SPE_layout.py:860-911).
    let mut edge_entropy = vec![UNASSIGNED_ENTROPY_SENTINEL; m];
    let mut in_frontier = vec![false; tree_nodes.len()];
    in_frontier[root_id] = true;

    let mut pq: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();
    pq.push(Reverse((OrdF64(0.0), root_id)));

    let mut child_bucket = vec![usize::MAX; n];
    let mut leaf_stamp = VisitStamp::new(n);
    let mut entropy_curve: Vec<f64> = Vec::new();
    let mut pops = 0usize;

    while let Some(Reverse((acc_ent, node_id))) = pq.pop() {
        if acc_ent.0 >= UNASSIGNED_ENTROPY_SENTINEL {
            break;
        }
        let Some(node) = tree_nodes[node_id].as_ref() else {
            continue;
        };

        if node.partition.len() > 1 {
            let cur_e = frontier_structural_entropy(&tree_nodes, &in_frontier, total_vol);

            leaf_stamp.clear();
            for &cid in &node.children {
                if let Some(child) = tree_nodes[cid].as_ref() {
                    for &v in &child.partition {
                        leaf_stamp.mark(v as usize);
                        child_bucket[v as usize] = cid;
                    }
                }
            }

            for &u_u32 in &node.partition {
                let u = u_u32 as usize;
                let bu = child_bucket[u];
                for pos in vert_edge_indptr[u]..vert_edge_indptr[u + 1] {
                    let v = vert_edge_nbr[pos] as usize;
                    if u < v && leaf_stamp.is_visited(v) && bu != child_bucket[v] {
                        let e_idx = vert_edge_idx[pos];
                        edge_entropy[e_idx] = cur_e;
                    }
                }
            }

            let has_only_non_leaf_children = !node.children.is_empty()
                && node.children.iter().all(|&cid| {
                    tree_nodes[cid]
                        .as_ref()
                        .is_some_and(|c| c.partition.len() > 1)
                });

            if !has_only_non_leaf_children {
                pq.push(Reverse((OrdF64(UNASSIGNED_ENTROPY_SENTINEL), node_id)));
            } else {
                for &cid in &node.children {
                    in_frontier[cid] = true;
                    if let Some(child) = tree_nodes[cid].as_ref()
                        && !child.children.is_empty()
                    {
                        pq.push(Reverse((OrdF64(child.a_increase), cid)));
                    }
                }
                if !entropy_curve.iter().any(|&x| (x - cur_e).abs() <= 1e-15) {
                    entropy_curve.push(cur_e);
                }
            }
        }

        pops += 1;
        if pops >= config.pq_max_pops
            || pq
                .peek()
                .is_none_or(|Reverse((e, _))| e.0 >= UNASSIGNED_ENTROPY_SENTINEL)
        {
            break;
        }
    }

    let se_target = select_se_target(&entropy_curve, &edge_entropy, h_max);
    let mut modulated = Vec::with_capacity(m);
    for &ent in &edge_entropy {
        modulated.push(map_entropy_weight(ent, se_target, config.mapping));
    }

    (se_target, modulated)
}

#[allow(clippy::too_many_arguments)]
fn build_coding_tree(
    n: usize,
    edges_uv: &[(u32, u32)],
    orig_weights: &[f64],
    node_vol: &[f64],
    total_vol: f64,
    vert_edge_indptr: &[usize],
    vert_edge_nbr: &[u32],
    vert_edge_w: &[f64],
    tree_depth: usize,
) -> (Vec<Option<CodingTreeNode>>, usize) {
    let mut nodes: Vec<Option<CodingTreeNode>> = Vec::with_capacity(n * 3);
    for (v, &vol) in node_vol.iter().enumerate() {
        nodes.push(Some(CodingTreeNode {
            parent: None,
            children: Vec::new(),
            partition: vec![v as u32],
            vol,
            g: vol,
            merged: false,
            child_h: 0,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        }));
    }

    if tree_depth == 1 {
        let root_id = nodes.len();
        let all_leaves: Vec<usize> = (0..n).collect();
        for leaf in &all_leaves {
            if let Some(node) = nodes[*leaf].as_mut() {
                node.parent = Some(root_id);
            }
        }
        nodes.push(Some(CodingTreeNode {
            parent: None,
            children: all_leaves,
            partition: (0..n as u32).collect(),
            vol: total_vol,
            g: 0.0,
            merged: false,
            child_h: 1,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        }));
        compute_tree_entropies(&mut nodes, root_id, tree_depth, total_vol);
        return (nodes, root_id);
    }

    // Build initial adjacency cuts for leaves 0..n.
    let mut global_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (&(u, v), &w) in edges_uv.iter().zip(orig_weights.iter()) {
        if w > 0.0 {
            global_adj[u as usize].push((v as usize, w));
            global_adj[v as usize].push((u as usize, w));
        }
    }
    for row in &mut global_adj {
        row.sort_unstable_by_key(|x| x.0);
    }

    let active_leaves: Vec<usize> = (0..n).collect();
    let root_id = build_k_tree(
        &mut nodes,
        &active_leaves,
        &mut global_adj,
        total_vol,
        Some(2),
    );

    check_balance(&mut nodes, root_id);
    if let Some(root) = nodes[root_id].as_mut()
        && root.child_h < 2
    {
        root.child_h = 2;
    }

    // `mode = 'v2'`: expand tree depth from 2 up to `tree_depth` choosing min-entropy delta.
    let mut leaf_to_tree = vec![0usize; n];
    let mut flag = 0u8;
    let mut cached_leaf_up: Option<(f64, Vec<LeafUpMapping>)> = None;
    let mut cached_root_down: Option<(f64, Option<SubTreeExpansion>)> = None;

    while nodes[root_id]
        .as_ref()
        .is_some_and(|r| r.child_h < tree_depth)
    {
        if flag == 0 || flag == 1 {
            cached_leaf_up = Some(compute_leaf_up(
                &nodes,
                n,
                total_vol,
                vert_edge_indptr,
                vert_edge_nbr,
                vert_edge_w,
                &mut leaf_to_tree,
            ));
        }
        if flag == 0 || flag == 2 {
            cached_root_down = compute_root_down(
                &nodes,
                root_id,
                n,
                vert_edge_indptr,
                vert_edge_nbr,
                vert_edge_w,
                &mut leaf_to_tree,
            );
        }

        let leaf_delta = cached_leaf_up.as_ref().map_or(0.0, |x| x.0);
        let root_delta = cached_root_down.as_ref().map_or(0.0, |x| x.0);
        let has_root_down = cached_root_down.as_ref().is_some_and(|x| x.1.is_some());

        if has_root_down && leaf_delta < root_delta {
            flag = 2;
            if let Some((_, Some((sub_root, sub_nodes)))) = cached_root_down.take() {
                apply_root_down(&mut nodes, root_id, sub_root, sub_nodes);
            }
        } else {
            flag = 1;
            if let Some((_, mappings)) = cached_leaf_up.take() {
                apply_leaf_up(&mut nodes, root_id, mappings);
            } else {
                break;
            }
        }
    }

    // Recompute child_h consistently bottom-up and set E_increase / A_increase.
    let _ = refresh_child_h(&mut nodes, root_id);
    compute_tree_entropies(&mut nodes, root_id, tree_depth, total_vol);

    // Ensure root_id itself has parent == None.
    if let Some(root) = nodes[root_id].as_mut() {
        root.parent = None;
    }

    (nodes, root_id)
}

fn build_k_tree(
    nodes: &mut Vec<Option<CodingTreeNode>>,
    active_ids: &[usize],
    adj_cuts: &mut Vec<Vec<(usize, f64)>>,
    g_vol: f64,
    k_limit: Option<usize>,
) -> usize {
    if active_ids.is_empty() {
        let id = nodes.len();
        nodes.push(Some(CodingTreeNode {
            parent: None,
            children: Vec::new(),
            partition: Vec::new(),
            vol: g_vol,
            g: 0.0,
            merged: false,
            child_h: 0,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        }));
        return id;
    }

    if active_ids.len() == 1 {
        return active_ids[0];
    }

    let mut min_heap: BinaryHeap<Reverse<(OrdF64, usize, usize, OrdF64)>> = BinaryHeap::new();
    let mut cmp_heap: BinaryHeap<Reverse<(OrdF64, usize, usize)>> = BinaryHeap::new();

    for &i in active_ids {
        let Some(ni) = nodes[i].as_ref() else {
            continue;
        };
        for &(j, cut_v) in &adj_cuts[i] {
            if j > i
                && let Some(nj) = nodes[j].as_ref()
            {
                let diff = combine_delta(ni, nj, cut_v, g_vol);
                min_heap.push(Reverse((OrdF64(diff), i, j, OrdF64(cut_v))));
            }
        }
    }

    let mut unmerged_count = active_ids.len();
    let mut last_new_id = active_ids[0];

    while unmerged_count > 1 {
        let Some(Reverse((_, id1, id2, OrdF64(cut_v)))) = min_heap.pop() else {
            break;
        };
        let is_merged_1 = nodes[id1].as_ref().is_none_or(|n| n.merged);
        let is_merged_2 = nodes[id2].as_ref().is_none_or(|n| n.merged);
        if is_merged_1 || is_merged_2 {
            continue;
        }

        let new_id = nodes.len();
        let (part1, v1, g1, h1) = {
            let n1 = match nodes[id1].as_mut() {
                Some(n) => n,
                None => continue,
            };
            n1.merged = true;
            n1.parent = Some(new_id);
            (n1.partition.clone(), n1.vol, n1.g, n1.child_h)
        };
        let (part2, v2, g2, h2) = {
            let n2 = match nodes[id2].as_mut() {
                Some(n) => n,
                None => continue,
            };
            n2.merged = true;
            n2.parent = Some(new_id);
            (n2.partition.clone(), n2.vol, n2.g, n2.child_h)
        };

        let mut merged_part = Vec::with_capacity(part1.len() + part2.len());
        merged_part.extend_from_slice(&part1);
        merged_part.extend_from_slice(&part2);
        merged_part.sort_unstable();

        let new_node = CodingTreeNode {
            parent: None,
            children: vec![id1, id2],
            partition: merged_part,
            vol: v1 + v2,
            g: (g1 + g2 - 2.0 * cut_v).max(0.0),
            merged: false,
            child_h: h1.max(h2) + 1,
            child_cut: cut_v,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        };

        if h1 > 0
            && let Some(n1) = nodes[id1].as_ref()
        {
            cmp_heap.push(Reverse((
                OrdF64(compress_delta(n1, &new_node)),
                id1,
                new_id,
            )));
        }
        if h2 > 0
            && let Some(n2) = nodes[id2].as_ref()
        {
            cmp_heap.push(Reverse((
                OrdF64(compress_delta(n2, &new_node)),
                id2,
                new_id,
            )));
        }
        nodes.push(Some(new_node));

        // Merge sorted neighbor cut lists of `id1` and `id2` into `adj_cuts[new_id]`.
        let merged_adj = merge_cut_lists(&adj_cuts[id1], &adj_cuts[id2], id1, id2, nodes);
        while adj_cuts.len() <= new_id {
            adj_cuts.push(Vec::new());
        }
        for &(nbr_id, c_new) in &merged_adj {
            adj_cuts[nbr_id].push((new_id, c_new));
            if let (Some(n_nbr), Some(n_new)) = (nodes[nbr_id].as_ref(), nodes[new_id].as_ref()) {
                let diff = combine_delta(n_nbr, n_new, c_new, g_vol);
                min_heap.push(Reverse((OrdF64(diff), nbr_id, new_id, OrdF64(c_new))));
            }
        }
        adj_cuts[new_id] = merged_adj;
        adj_cuts[id1].clear();
        adj_cuts[id2].clear();

        last_new_id = new_id;
        unmerged_count -= 1;
    }

    let mut root = last_new_id;

    // Handle disconnected components / isolated vertices under a virtual global root.
    if unmerged_count > 1 {
        let mut unmerged_nodes = Vec::new();
        let mut all_part = Vec::new();
        let mut max_ch = 0usize;
        for &id in active_ids {
            if let Some(node) = nodes[id].as_ref()
                && !node.merged
            {
                unmerged_nodes.push(id);
                all_part.extend_from_slice(&node.partition);
                max_ch = max_ch.max(node.child_h);
            }
        }
        let start_extra = active_ids.last().map_or(0, |&x| x + 1);
        for (id, node_opt) in nodes.iter().enumerate().skip(start_extra) {
            if let Some(node) = node_opt.as_ref()
                && !node.merged
                && !unmerged_nodes.contains(&id)
            {
                unmerged_nodes.push(id);
                all_part.extend_from_slice(&node.partition);
                max_ch = max_ch.max(node.child_h);
            }
        }
        all_part.sort_unstable();
        unmerged_nodes.sort_unstable();

        let new_id = nodes.len();
        let root_node = CodingTreeNode {
            parent: None,
            children: unmerged_nodes.clone(),
            partition: all_part,
            vol: g_vol,
            g: 0.0,
            merged: false,
            child_h: max_ch + 1,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        };
        for &cid in &unmerged_nodes {
            if let Some(child) = nodes[cid].as_mut() {
                child.merged = true;
                child.parent = Some(new_id);
                if child.child_h > 0 {
                    cmp_heap.push(Reverse((
                        OrdF64(compress_delta(child, &root_node)),
                        cid,
                        new_id,
                    )));
                }
            }
        }
        nodes.push(Some(root_node));
        root = new_id;
    }

    if let Some(k) = k_limit {
        while nodes[root].as_ref().is_some_and(|r| r.child_h > k) {
            let Some(Reverse((_, node_id, p_id))) = cmp_heap.pop() else {
                break;
            };
            let valid = nodes[node_id]
                .as_ref()
                .is_some_and(|n| n.parent == Some(p_id) && n.child_h > 0)
                && nodes[p_id].is_some();
            if !valid {
                continue;
            }
            if subtree_full_depth(nodes, node_id) <= k {
                continue;
            }

            let compressed_children = compress_node(nodes, node_id, p_id);
            if nodes[root].as_ref().is_some_and(|r| r.child_h <= k) {
                break;
            }

            if let Some(p_node) = nodes[p_id].as_ref()
                && let Some(gp_id) = p_node.parent
                && let Some(gp_node) = nodes[gp_id].as_ref()
                && subtree_full_depth(nodes, p_id) > k
            {
                cmp_heap.push(Reverse((
                    OrdF64(compress_delta(p_node, gp_node)),
                    p_id,
                    gp_id,
                )));
            }
            for &cid in &compressed_children {
                if let (Some(c_node), Some(p_node)) = (nodes[cid].as_ref(), nodes[p_id].as_ref())
                    && c_node.child_h > 0
                    && subtree_full_depth(nodes, cid) > k
                {
                    cmp_heap.push(Reverse((OrdF64(compress_delta(c_node, p_node)), cid, p_id)));
                }
            }
        }
    }

    root
}

fn merge_cut_lists(
    a: &[(usize, f64)],
    b: &[(usize, f64)],
    skip1: usize,
    skip2: usize,
    nodes: &[Option<CodingTreeNode>],
) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        let (ia, wa) = a[i];
        let (ib, wb) = b[j];
        if ia == ib {
            if ia != skip1 && ia != skip2 && nodes[ia].as_ref().is_some_and(|n| !n.merged) {
                out.push((ia, wa + wb));
            }
            i += 1;
            j += 1;
        } else if ia < ib {
            if ia != skip1 && ia != skip2 && nodes[ia].as_ref().is_some_and(|n| !n.merged) {
                out.push((ia, wa));
            }
            i += 1;
        } else {
            if ib != skip1 && ib != skip2 && nodes[ib].as_ref().is_some_and(|n| !n.merged) {
                out.push((ib, wb));
            }
            j += 1;
        }
    }
    while i < a.len() {
        let (ia, wa) = a[i];
        if ia != skip1 && ia != skip2 && nodes[ia].as_ref().is_some_and(|n| !n.merged) {
            out.push((ia, wa));
        }
        i += 1;
    }
    while j < b.len() {
        let (ib, wb) = b[j];
        if ib != skip1 && ib != skip2 && nodes[ib].as_ref().is_some_and(|n| !n.merged) {
            out.push((ib, wb));
        }
        j += 1;
    }
    out
}

fn subtree_full_depth(nodes: &[Option<CodingTreeNode>], nid: usize) -> usize {
    let Some(node) = nodes[nid].as_ref() else {
        return 0;
    };
    let mut depth = node.child_h;
    let mut cur = node.parent;
    while let Some(pid) = cur {
        depth += 1;
        cur = nodes[pid].as_ref().and_then(|p| p.parent);
    }
    depth
}

fn compress_node(
    nodes: &mut [Option<CodingTreeNode>],
    node_id: usize,
    parent_id: usize,
) -> Vec<usize> {
    let Some(removed) = nodes[node_id].take() else {
        return Vec::new();
    };
    let node_children = removed.children;
    let com_child_h = removed.child_h;

    for &c in &node_children {
        if let Some(child) = nodes[c].as_mut() {
            child.parent = Some(parent_id);
        }
    }

    let p_child_h = {
        let Some(parent) = nodes[parent_id].as_mut() else {
            return node_children;
        };
        let old_h = parent.child_h;
        parent.child_cut += removed.child_cut;
        parent.children.retain(|&id| id != node_id);
        parent.children.extend(node_children.iter().copied());
        parent.children.sort_unstable();
        parent.children.dedup();
        old_h
    };

    if p_child_h.saturating_sub(com_child_h) == 1 {
        let mut cur = Some(parent_id);
        while let Some(pid) = cur {
            let max_ch = nodes[pid]
                .as_ref()
                .map(|p| {
                    p.children
                        .iter()
                        .filter_map(|&cid| nodes[cid].as_ref().map(|c| c.child_h))
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let next_parent = if let Some(p) = nodes[pid].as_mut() {
                if p.child_h == max_ch + 1 {
                    break;
                }
                p.child_h = max_ch + 1;
                p.parent
            } else {
                None
            };
            cur = next_parent;
        }
    }

    node_children
}

fn check_balance(nodes: &mut Vec<Option<CodingTreeNode>>, root_id: usize) {
    let root_children = nodes[root_id]
        .as_ref()
        .map(|r| r.children.clone())
        .unwrap_or_default();
    for cid in root_children {
        if nodes[cid].as_ref().is_some_and(|c| c.child_h == 0) {
            single_up(nodes, cid);
        }
    }
}

fn single_up(nodes: &mut Vec<Option<CodingTreeNode>>, node_id: usize) {
    let Some((p_id, part, vol, g, ch)) = nodes[node_id].as_ref().and_then(|n| {
        n.parent
            .map(|p| (p, n.partition.clone(), n.vol, n.g, n.child_h))
    }) else {
        return;
    };
    let new_id = nodes.len();
    let grow_node = CodingTreeNode {
        parent: Some(p_id),
        children: vec![node_id],
        partition: part,
        vol,
        g,
        merged: false,
        child_h: ch + 1,
        child_cut: 0.0,
        depth: 0,
        e_increase: 0.0,
        a_increase: 0.0,
    };
    nodes.push(Some(grow_node));
    if let Some(child) = nodes[node_id].as_mut() {
        child.parent = Some(new_id);
    }
    if let Some(parent) = nodes[p_id].as_mut() {
        for slot in &mut parent.children {
            if *slot == node_id {
                *slot = new_id;
                break;
            }
        }
        parent.children.sort_unstable();
    }
}

#[allow(clippy::type_complexity)]
fn compute_leaf_up(
    nodes: &[Option<CodingTreeNode>],
    n: usize,
    total_vol: f64,
    vert_edge_indptr: &[usize],
    vert_edge_nbr: &[u32],
    vert_edge_w: &[f64],
    leaf_to_tree: &mut [usize],
) -> (f64, Vec<LeafUpMapping>) {
    let mut h1_ids: Vec<usize> = Vec::new();
    for node_opt in nodes.iter().take(n) {
        if let Some(p_id) = node_opt.as_ref().and_then(|node| node.parent) {
            h1_ids.push(p_id);
        }
    }
    h1_ids.sort_unstable();
    h1_ids.dedup();

    let mut delta = 0.0_f64;
    let mut mappings = Vec::with_capacity(h1_ids.len());

    for &h1_id in &h1_ids {
        let Some(cand) = nodes[h1_id].as_ref() else {
            continue;
        };
        if cand.partition.len() < 3 {
            mappings.push((h1_id, None));
            continue;
        }

        let sub_g_vol = cand.vol - cand.g;
        if sub_g_vol <= 0.0 {
            mappings.push((h1_id, None));
            continue;
        }

        let m_sub = cand.partition.len();
        for (local_i, &v_u32) in cand.partition.iter().enumerate() {
            leaf_to_tree[v_u32 as usize] = local_i;
        }

        let mut ori_ent = 0.0_f64;
        let mut sub_nodes: Vec<Option<CodingTreeNode>> = Vec::with_capacity(m_sub * 3);
        let mut sub_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_sub];

        for (local_u, &u_u32) in cand.partition.iter().enumerate() {
            let u = u_u32 as usize;
            if let Some(orig_leaf) = nodes[u].as_ref() {
                ori_ent +=
                    -(orig_leaf.g / total_vol) * ((orig_leaf.vol + 1.0) / (cand.vol + 1.0)).log2();
            }
            let mut sub_vol = 0.0_f64;
            for pos in vert_edge_indptr[u]..vert_edge_indptr[u + 1] {
                let v = vert_edge_nbr[pos] as usize;
                let w = vert_edge_w[pos];
                if w > 0.0 && cand.partition.binary_search(&(v as u32)).is_ok() {
                    let local_v = leaf_to_tree[v];
                    sub_vol += w;
                    sub_adj[local_u].push((local_v, w));
                }
            }
            sub_adj[local_u].sort_unstable_by_key(|x| x.0);
            sub_nodes.push(Some(CodingTreeNode {
                parent: None,
                children: Vec::new(),
                partition: vec![u_u32],
                vol: sub_vol,
                g: sub_vol,
                merged: false,
                child_h: 0,
                child_cut: 0.0,
                depth: 0,
                e_increase: 0.0,
                a_increase: 0.0,
            }));
        }

        let active: Vec<usize> = (0..m_sub).collect();
        let sub_root = build_k_tree(&mut sub_nodes, &active, &mut sub_adj, sub_g_vol, Some(2));
        check_balance(&mut sub_nodes, sub_root);

        let new_ent = leaf_up_entropy(nodes, &mut sub_nodes, sub_root, cand.vol, cand.g, total_vol);
        delta += ori_ent - new_ent;
        mappings.push((h1_id, Some((sub_root, sub_nodes))));
    }

    delta /= n.max(1) as f64;
    (delta, mappings)
}

fn leaf_up_entropy(
    orig_nodes: &[Option<CodingTreeNode>],
    sub_nodes: &mut [Option<CodingTreeNode>],
    sub_root: usize,
    cand_vol: f64,
    cand_g: f64,
    total_vol: f64,
) -> f64 {
    let mut ent = 0.0_f64;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(sub_root);
    while let Some(sid) = queue.pop_front() {
        let (children, ch, part) = match sub_nodes[sid].as_ref() {
            Some(s) => (s.children.clone(), s.child_h, s.partition.clone()),
            None => continue,
        };
        for &c in &children {
            queue.push_back(c);
        }
        if sid == sub_root {
            if let Some(s) = sub_nodes[sid].as_mut() {
                s.vol = cand_vol;
                s.g = cand_g;
            }
            continue;
        }
        if ch == 1 {
            let (inner_vol, orig_vol) = {
                let s = sub_nodes[sid].as_ref();
                let inner = s.map_or(0.0, |x| x.vol - x.g);
                let ov: f64 = part
                    .iter()
                    .filter_map(|&u| orig_nodes[u as usize].as_ref().map(|n| n.vol))
                    .sum();
                (inner, ov)
            };
            let orig_g = (orig_vol - inner_vol).max(0.0);
            if let Some(s) = sub_nodes[sid].as_mut() {
                s.vol = orig_vol;
                s.g = orig_g;
            }
        } else if ch == 0 && part.len() == 1 {
            let u = part[0] as usize;
            if let (Some(orig), Some(s)) = (orig_nodes[u].as_ref(), sub_nodes[sid].as_mut()) {
                s.vol = orig.vol;
                s.g = orig.g;
            }
        }

        if let Some(s) = sub_nodes[sid].as_ref()
            && let Some(pid) = s.parent
            && let Some(p) = sub_nodes[pid].as_ref()
        {
            ent += -(s.g / total_vol) * ((s.vol + 1.0) / (p.vol + 1.0)).log2();
        }
    }
    ent
}

fn apply_leaf_up(
    nodes: &mut Vec<Option<CodingTreeNode>>,
    root_id: usize,
    mappings: Vec<LeafUpMapping>,
) {
    for (h1_id, sub_opt) in mappings {
        match sub_opt {
            None => {
                let children = nodes[h1_id]
                    .as_ref()
                    .map(|n| n.children.clone())
                    .unwrap_or_default();
                for c in children {
                    single_up(nodes, c);
                }
            }
            Some((sub_root, mut sub_nodes)) => {
                // Re-index non-leaf subtree nodes above `nodes.len()`, and map leaf nodes back to original `u32` IDs.
                let mut id_remap = vec![usize::MAX; sub_nodes.len()];
                let mut next_global = nodes.len();
                for (sid, s_opt) in sub_nodes.iter().enumerate() {
                    if sid == sub_root {
                        id_remap[sid] = h1_id;
                        continue;
                    }
                    let Some(s) = s_opt.as_ref() else {
                        continue;
                    };
                    if s.child_h == 0 && s.partition.len() == 1 {
                        id_remap[sid] = s.partition[0] as usize;
                    } else {
                        id_remap[sid] = next_global;
                        next_global += 1;
                    }
                }
                nodes.resize(next_global, None);

                let sub_root_children = sub_nodes[sub_root]
                    .as_ref()
                    .map(|r| r.children.clone())
                    .unwrap_or_default();

                for (sid, s_opt) in sub_nodes.iter_mut().enumerate() {
                    if sid == sub_root {
                        continue;
                    }
                    let Some(mut s) = s_opt.take() else {
                        continue;
                    };
                    let gid = id_remap[sid];
                    if s.child_h == 0 && s.partition.len() == 1 {
                        if let Some(orig) = nodes[gid].as_mut() {
                            orig.parent = s.parent.map(|p| id_remap[p]);
                        }
                    } else {
                        s.parent = s.parent.map(|p| id_remap[p]);
                        for c in &mut s.children {
                            *c = id_remap[*c];
                        }
                        s.children.sort_unstable();
                        nodes[gid] = Some(s);
                    }
                }

                if let Some(h1_node) = nodes[h1_id].as_mut() {
                    h1_node.children = sub_root_children.into_iter().map(|c| id_remap[c]).collect();
                    h1_node.children.sort_unstable();
                }
            }
        }
    }
    if let Some(root) = nodes[root_id].as_mut() {
        root.child_h += 1;
    }
}

#[allow(clippy::type_complexity)]
fn compute_root_down(
    nodes: &[Option<CodingTreeNode>],
    root_id: usize,
    n: usize,
    vert_edge_indptr: &[usize],
    vert_edge_nbr: &[u32],
    vert_edge_w: &[f64],
    leaf_to_tree: &mut [usize],
) -> Option<(f64, Option<(usize, Vec<Option<CodingTreeNode>>)>)> {
    let root = nodes[root_id].as_ref()?;
    if root.children.len() < 3 {
        return Some((0.0, None));
    }

    let k_c = root.children.len();
    let g_vol = root.vol;
    if g_vol <= 0.0 {
        return Some((0.0, None));
    }

    for (local_c, &cid) in root.children.iter().enumerate() {
        if let Some(c) = nodes[cid].as_ref() {
            for &u in &c.partition {
                leaf_to_tree[u as usize] = local_c;
            }
        }
    }

    let mut ori_ent = 0.0_f64;
    let mut sub_nodes: Vec<Option<CodingTreeNode>> = Vec::with_capacity(k_c * 3);
    let mut sub_adj_pairs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k_c];

    for (local_c, &cid) in root.children.iter().enumerate() {
        let Some(c) = nodes[cid].as_ref() else {
            return Some((0.0, None));
        };
        ori_ent += -(c.g / g_vol) * ((c.vol + 1.0) / g_vol).log2();
        for &u_u32 in &c.partition {
            let u = u_u32 as usize;
            for pos in vert_edge_indptr[u]..vert_edge_indptr[u + 1] {
                let v = vert_edge_nbr[pos] as usize;
                let w = vert_edge_w[pos];
                let local_v = leaf_to_tree[v];
                if w > 0.0 && local_v != local_c {
                    sub_adj_pairs[local_c].push((local_v, w));
                }
            }
        }
        sub_adj_pairs[local_c].sort_unstable_by_key(|x| x.0);
        // Aggregate parallel edges between distinct root children.
        let mut merged_row: Vec<(usize, f64)> = Vec::new();
        for &(nbr_c, w) in &sub_adj_pairs[local_c] {
            if let Some(last) = merged_row.last_mut()
                && last.0 == nbr_c
            {
                last.1 += w;
            } else {
                merged_row.push((nbr_c, w));
            }
        }
        sub_adj_pairs[local_c] = merged_row;

        sub_nodes.push(Some(CodingTreeNode {
            parent: None,
            children: vec![cid],
            partition: c.partition.clone(),
            vol: c.vol,
            g: c.g,
            merged: false,
            child_h: 0,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        }));
    }

    let _ = n;
    let active: Vec<usize> = (0..k_c).collect();
    let sub_root = build_k_tree(&mut sub_nodes, &active, &mut sub_adj_pairs, g_vol, Some(2));
    check_balance(&mut sub_nodes, sub_root);

    let new_ent = tree_structural_entropy(&sub_nodes, g_vol);
    let delta = (ori_ent - new_ent) / (k_c as f64);
    Some((delta, Some((sub_root, sub_nodes))))
}

fn apply_root_down(
    nodes: &mut Vec<Option<CodingTreeNode>>,
    root_id: usize,
    sub_root: usize,
    mut sub_nodes: Vec<Option<CodingTreeNode>>,
) {
    let orig_children = nodes[root_id]
        .as_ref()
        .map(|r| r.children.clone())
        .unwrap_or_default();
    let k_c = orig_children.len();

    let mut id_remap = vec![usize::MAX; sub_nodes.len()];
    let mut next_global = nodes.len();
    for sid in 0..sub_nodes.len() {
        if sid == sub_root {
            id_remap[sid] = root_id;
        } else if sid < k_c {
            id_remap[sid] = orig_children[sid];
        } else if sub_nodes[sid].is_some() {
            id_remap[sid] = next_global;
            next_global += 1;
        }
    }
    nodes.resize(next_global, None);

    let sub_root_children = sub_nodes[sub_root]
        .as_ref()
        .map(|r| r.children.clone())
        .unwrap_or_default();

    for (sid, s_opt) in sub_nodes.iter_mut().enumerate() {
        if sid == sub_root {
            continue;
        }
        let Some(mut s) = s_opt.take() else {
            continue;
        };
        let gid = id_remap[sid];
        if sid < k_c {
            if let Some(orig_c) = nodes[gid].as_mut() {
                orig_c.parent = s.parent.map(|p| id_remap[p]);
            }
        } else {
            s.parent = s.parent.map(|p| id_remap[p]);
            for c in &mut s.children {
                *c = id_remap[*c];
            }
            s.children.sort_unstable();
            nodes[gid] = Some(s);
        }
    }

    if let Some(root) = nodes[root_id].as_mut() {
        root.children = sub_root_children.into_iter().map(|c| id_remap[c]).collect();
        root.children.sort_unstable();
        root.child_h += 1;
    }
}

fn refresh_child_h(nodes: &mut [Option<CodingTreeNode>], nid: usize) -> usize {
    let children = nodes[nid]
        .as_ref()
        .map(|n| n.children.clone())
        .unwrap_or_default();
    if children.is_empty() {
        if let Some(n) = nodes[nid].as_mut() {
            n.child_h = 0;
        }
        return 0;
    }
    let mut max_ch = 0usize;
    for c in children {
        max_ch = max_ch.max(refresh_child_h(nodes, c));
    }
    if let Some(n) = nodes[nid].as_mut() {
        n.child_h = max_ch + 1;
    }
    max_ch + 1
}

fn compute_tree_entropies(
    nodes: &mut [Option<CodingTreeNode>],
    root_id: usize,
    max_depth: usize,
    total_vol: f64,
) {
    // BFS from root to assign depth, e_increase, and a_increase top-down.
    let mut queue = std::collections::VecDeque::new();
    if let Some(root) = nodes[root_id].as_mut() {
        root.depth = 0;
    }
    queue.push_back(root_id);

    let mut bfs_order = Vec::new();
    while let Some(nid) = queue.pop_front() {
        bfs_order.push(nid);
        let (depth, children) = match nodes[nid].as_ref() {
            Some(n) => (n.depth, n.children.clone()),
            None => continue,
        };
        for &cid in &children {
            if let Some(c) = nodes[cid].as_mut() {
                c.depth = depth + 1;
            }
            queue.push_back(cid);
        }
    }

    for &nid in &bfs_order {
        let (depth, vol, children) = match nodes[nid].as_ref() {
            Some(n) => (n.depth, n.vol, n.children.clone()),
            None => continue,
        };
        if depth < max_depth && !children.is_empty() {
            let mut e_inc = 0.0_f64;
            for &cid in &children {
                if let Some(c) = nodes[cid].as_ref() {
                    e_inc += -(c.g / total_vol) * ((c.vol + 1.0) / (vol + 1.0)).log2();
                }
            }
            let parent_acc = nodes[nid]
                .as_ref()
                .and_then(|n| n.parent)
                .and_then(|pid| nodes[pid].as_ref())
                .map_or(0.0, |p| p.a_increase);
            if let Some(n) = nodes[nid].as_mut() {
                n.e_increase = e_inc;
                n.a_increase = e_inc + parent_acc;
            }
        }
    }
}

fn tree_structural_entropy(nodes: &[Option<CodingTreeNode>], total_vol: f64) -> f64 {
    if total_vol <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0_f64;
    for node_opt in nodes {
        if let Some(node) = node_opt
            && let Some(pid) = node.parent
            && let Some(p) = nodes[pid].as_ref()
        {
            ent += -(node.g / total_vol) * ((node.vol + 1.0) / (p.vol + 1.0)).log2();
        }
    }
    ent
}

fn frontier_structural_entropy(
    nodes: &[Option<CodingTreeNode>],
    in_frontier: &[bool],
    total_vol: f64,
) -> f64 {
    if total_vol <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0_f64;
    for (nid, &active) in in_frontier.iter().enumerate() {
        if !active {
            continue;
        }
        if let Some(node) = nodes[nid].as_ref()
            && let Some(pid) = node.parent
            && in_frontier[pid]
            && let Some(p) = nodes[pid].as_ref()
        {
            ent += -(node.g / total_vol) * ((node.vol + 1.0) / (p.vol + 1.0)).log2();
        }
    }
    ent
}

fn select_se_target(entropy_curve: &[f64], edge_entropy: &[f64], h_max: f64) -> f64 {
    let upper = (0.5 * h_max).max(0.0);
    if upper <= 0.0 {
        return 0.0;
    }

    let mut candidates: Vec<f64> = entropy_curve
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v > 0.0 && v <= upper)
        .collect();
    if candidates.is_empty() {
        candidates = edge_entropy
            .iter()
            .copied()
            .filter(|&v| v.is_finite() && v > 0.0 && v < UNASSIGNED_ENTROPY_SENTINEL && v <= upper)
            .collect();
    }
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

    if candidates.is_empty() {
        return 0.0;
    }
    if candidates.len() <= 2 {
        return candidates[0];
    }

    // Normalized Kneedle / max-perpendicular-distance elbow on sorted entropy curve <= 0.5 H_max.
    let k = candidates.len();
    let x_span = (k - 1) as f64;
    let y_min = candidates[0];
    let y_max = candidates[k - 1];
    let y_span = (y_max - y_min).max(1e-12);

    let mut best_idx = 0usize;
    let mut best_dist = -1.0_f64;
    for (i, &val) in candidates.iter().enumerate() {
        let xn = (i as f64) / x_span;
        let yn = (val - y_min) / y_span;
        let dist = (xn - yn).abs() * std::f64::consts::FRAC_1_SQRT_2;
        if dist > best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    candidates[best_idx].min(upper)
}

#[inline]
fn map_entropy_weight(entropy: f64, se_target: f64, mapping: SelWeightMapping) -> f64 {
    if !entropy.is_finite() || entropy <= se_target || entropy <= 0.0 {
        return 0.0;
    }
    let z = 1.0 - se_target / entropy;
    match mapping {
        SelWeightMapping::Sigmoid => 1.0 / (1.0 + (-z).exp()),
        SelWeightMapping::Linear => z,
        SelWeightMapping::Tanh => 0.5 * (z.tanh() + 1.0),
        SelWeightMapping::CustomLogistic => {
            if z >= 1.0 {
                0.0
            } else if z <= 0.0 {
                1.0
            } else {
                1.0 / (1.0 + (10.0 * (z - 0.8)).exp())
            }
        }
        SelWeightMapping::Binary => 1.0,
    }
}

#[allow(clippy::type_complexity)]
fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<(u32, u32)>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("SelLayoutSolver::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let nbrs = graph.neighbors(u);
        let ws = graph.neighbor_weights(u);
        for (pos, &v_u32) in nbrs.iter().enumerate() {
            if u32_u == v_u32 {
                continue;
            }
            let (a, b) = if u32_u < v_u32 {
                (u32_u, v_u32)
            } else {
                (v_u32, u32_u)
            };
            raw.push((a, b, ws.map_or(1.0, |w| w[pos].to_f64())));
        }
    }
    raw.sort_unstable_by_key(|x| (x.0, x.1));

    let weighted = graph.is_weighted();
    let mut edges_uv = Vec::new();
    let mut weights = Vec::new();
    let mut idx = 0usize;
    while idx < raw.len() {
        let (a, b, first_w) = raw[idx];
        let mut sum_w = first_w;
        let mut cnt = 1usize;
        idx += 1;
        while idx < raw.len() && raw[idx].0 == a && raw[idx].1 == b {
            sum_w += raw[idx].2;
            cnt += 1;
            idx += 1;
        }
        let merged_w = if weighted { sum_w / (cnt as f64) } else { 1.0 };
        edges_uv.push((a, b));
        weights.push(merged_w);
    }

    Ok((n, edges_uv, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cliques_bridge() -> CsrGraph<f64> {
        // Two weighted K_6 cliques (intra-clique w=10.0) joined by a weak bridge (5, 6) (w=1.0).
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..6u32 {
            for j in (i + 1)..6u32 {
                edges.push((i, j));
                weights.push(10.0);
            }
        }
        for i in 6..12u32 {
            for j in (i + 1)..12u32 {
                edges.push((i, j));
                weights.push(10.0);
            }
        }
        edges.push((5, 6));
        weights.push(1.0);
        CsrGraph::from_edges(12, &edges, Some(&weights), false, false).unwrap()
    }

    #[test]
    fn test_sel_config_validation() {
        let def = SelConfig::default();
        assert!(def.validate().is_ok());
        assert!(
            SelConfig {
                tree_depth: 0,
                ..def
            }
            .validate()
            .is_err()
        );
        assert!(
            SelConfig {
                pq_max_pops: 0,
                ..def
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_sel_five_mappings_and_entropy_attenuation() {
        let g = two_cliques_bridge();
        for mapping in [
            SelWeightMapping::Sigmoid,
            SelWeightMapping::Linear,
            SelWeightMapping::Tanh,
            SelWeightMapping::CustomLogistic,
            SelWeightMapping::Binary,
        ] {
            let cfg = SelConfig {
                mapping,
                iterations: 40,
                ..SelConfig::default()
            };
            let mut solver = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
            assert_eq!(solver.modulated_weights().len(), 31);
            assert!(solver.se_target().is_finite() && solver.se_target() >= 0.0);
            assert!(
                solver
                    .modulated_weights()
                    .iter()
                    .all(|&w| w.is_finite() && w >= 0.0)
            );
            // Weak inter-clique bridge edge (5, 6) [idx 15] is cut at root (0.0 <= se_target)
            // so it must be attenuated to 0.0 while dense intra-clique edges stay > 0.
            assert_eq!(solver.modulated_weights()[15], 0.0);
            assert!(solver.modulated_weights().iter().any(|&w| w > 0.0));
            if mapping != SelWeightMapping::CustomLogistic || solver.se_target() > 0.0 {
                assert!(solver.modulated_weights()[0] > 0.0);
            }

            let mut layout = Layout2d::<f64>::zeros(g.num_nodes());
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
            assert!(layout.is_all_finite());
        }
    }

    #[test]
    fn test_sel_backends_and_step_run_bit_exact_f64_f32() {
        let g = two_cliques_bridge();
        for backend in [SelLayoutBackend::Fr, SelLayoutBackend::TFdp] {
            let cfg = SelConfig {
                layout_backend: backend,
                iterations: 30,
                ..SelConfig::default()
            };

            // f64 parity
            let mut s_run = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
            let mut s_step = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
            let mut l_run = Layout2d::<f64>::zeros(g.num_nodes());
            let mut l_step = Layout2d::<f64>::zeros(g.num_nodes());
            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            assert_eq!(s_step.iteration(), s_run.iteration());
            assert_eq!(
                l_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );

            // f32 parity
            let mut s32_run = SelLayoutSolver::<f32>::new(&g, cfg).unwrap();
            let mut s32_step = SelLayoutSolver::<f32>::new(&g, cfg).unwrap();
            let mut l32_run = Layout2d::<f32>::zeros(g.num_nodes());
            let mut l32_step = Layout2d::<f32>::zeros(g.num_nodes());
            assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
            while !s32_step.status().is_finished() {
                s32_step.step(&mut l32_step).unwrap();
            }
            assert_eq!(
                l32_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l32_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l32_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l32_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_sel_edge_cases_n0_n1_edgeless_disconnected_directed_depth1() {
        let cfg = SelConfig {
            iterations: 20,
            ..SelConfig::default()
        };

        // N = 0
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = SelLayoutSolver::<f64>::new(&g0, cfg).unwrap();
        assert!(s0.modulated_weights().is_empty());
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        // N = 1
        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = SelLayoutSolver::<f64>::new(&g1, cfg).unwrap();
        let mut l1 = Layout2d::<f64>::from_xy(vec![1.5], vec![-2.5]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x()[0], 1.5);

        // Edgeless graph
        let ge = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let mut se = SelLayoutSolver::<f64>::new(&ge, cfg).unwrap();
        let mut le = Layout2d::<f64>::zeros(5);
        assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Completed);
        assert!(le.is_all_finite());

        // Disconnected graph (two triangles + 2 isolated nodes)
        let gd = CsrGraph::<f64>::from_edges(
            8,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let mut sd = SelLayoutSolver::<f64>::new(&gd, cfg).unwrap();
        let mut ld = Layout2d::<f64>::zeros(8);
        assert_eq!(sd.run(&mut ld).unwrap(), SolverStatus::Completed);
        assert!(ld.is_all_finite());

        // Directed graph with reciprocal weights & tree_depth = 1
        let gdir = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
            Some(&[1.0, 2.0, 1.5, 1.0, 1.0]),
            true,
            false,
        )
        .unwrap();
        let mut sdir = SelLayoutSolver::<f64>::new(
            &gdir,
            SelConfig {
                tree_depth: 1,
                ..cfg
            },
        )
        .unwrap();
        let mut ldir = Layout2d::<f64>::zeros(4);
        assert_eq!(sdir.run(&mut ldir).unwrap(), SolverStatus::Completed);
        assert!(ldir.is_all_finite());

        // Non-finite and dimension mismatch guards
        let mut bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
        let mut sdir2 = SelLayoutSolver::<f64>::new(&gdir, cfg).unwrap();
        assert!(sdir2.step(&mut bad).is_err());
        let mut wrong = Layout2d::<f64>::zeros(3);
        assert!(sdir2.step(&mut wrong).is_err());
    }
}
