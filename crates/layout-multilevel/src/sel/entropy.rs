use std::collections::VecDeque;

use super::coding_tree::{CodingTreeNode, UNASSIGNED_ENTROPY_SENTINEL};

pub(super) fn refresh_child_h(nodes: &mut [Option<CodingTreeNode>], nid: usize) -> usize {
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

pub(super) fn compute_tree_entropies(
    nodes: &mut [Option<CodingTreeNode>],
    root_id: usize,
    max_depth: usize,
    total_vol: f64,
) {
    // BFS from root to assign depth, e_increase, and a_increase top-down.
    let mut queue = VecDeque::new();
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

pub(super) fn tree_structural_entropy(nodes: &[Option<CodingTreeNode>], total_vol: f64) -> f64 {
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

pub(super) fn frontier_structural_entropy(
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

pub(super) fn select_se_target(entropy_curve: &[f64], edge_entropy: &[f64], h_max: f64) -> f64 {
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
