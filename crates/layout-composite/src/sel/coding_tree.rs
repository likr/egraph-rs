use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use egraph_core::VisitStamp;

use super::entropy::{
    compute_tree_entropies, frontier_structural_entropy, refresh_child_h, select_se_target,
    tree_structural_entropy,
};
use super::k_tree::{build_k_tree, check_balance};
use super::tree_expand::{apply_leaf_up, apply_root_down, compute_leaf_up, compute_root_down};
use super::{SelConfig, map_entropy_weight};

pub(super) const UNASSIGNED_ENTROPY_SENTINEL: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct OrdF64(pub(super) f64);

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

pub(super) type SubTreeExpansion = (usize, Vec<Option<CodingTreeNode>>);
pub(super) type LeafUpMapping = (usize, Option<SubTreeExpansion>);

#[derive(Debug, Clone)]
pub(super) struct CodingTreeNode {
    pub(super) parent: Option<usize>,
    pub(super) children: Vec<usize>,
    pub(super) partition: Vec<u32>,
    pub(super) vol: f64,
    pub(super) g: f64,
    pub(super) merged: bool,
    pub(super) child_h: usize,
    pub(super) child_cut: f64,
    pub(super) depth: usize,
    pub(super) e_increase: f64,
    pub(super) a_increase: f64,
}

#[inline]
pub(super) fn combine_delta(
    n1: &CodingTreeNode,
    n2: &CodingTreeNode,
    cut_v: f64,
    g_vol: f64,
) -> f64 {
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
pub(super) fn compress_delta(node: &CodingTreeNode, parent: &CodingTreeNode) -> f64 {
    let v1 = node.vol + 1.0;
    let v2 = parent.vol + 1.0;
    node.child_cut * (v2 / v1).log2()
}

pub(super) fn compute_sel_weights(
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
