use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::coding_tree::{CodingTreeNode, OrdF64, combine_delta, compress_delta};

pub(super) fn build_k_tree(
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

pub(super) fn check_balance(nodes: &mut Vec<Option<CodingTreeNode>>, root_id: usize) {
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

pub(super) fn single_up(nodes: &mut Vec<Option<CodingTreeNode>>, node_id: usize) {
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
